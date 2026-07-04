import datetime
import time
import warnings
from functools import partial
from typing import List

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import dist
from sampler import DistInfiniteBatchSampler, worker_init_fn
from utils import arg_util, misc, lamb
from utils.pet import build_dataset_to_pretrain
from utils.lr_control import lr_wd_annealing, get_param_groups
from pet_foundation_model import PETFoundationModel


class LocalDDP(torch.nn.Module):
    def __init__(self, module):
        super(LocalDDP, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def main_pt():
    warnings.filterwarnings("ignore")

    args: arg_util.Args = arg_util.init_dist_and_get_args()
    print(f"initial args:\n{str(args)}")
    args.log_epoch()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -------------------------------------------------
    # default loss weights if not provided in args
    # -------------------------------------------------
    if not hasattr(args, "weight_recon"):
        args.weight_recon = 1.0
    if not hasattr(args, "weight_clip"):
        args.weight_clip = 1.0
    if not hasattr(args, "weight_matching"):
        args.weight_matching = 1.0
    if not hasattr(args, "weight_suvr"):
        args.weight_suvr = 1.0

    if not hasattr(args, "aal_standard_names_json"):
        args.aal_standard_names_json = "aal_standard_names.json"

    if not hasattr(args, "text_pack_path"):
        args.text_pack_path = "aal_text_pack.pt"

    # -------------------------------------------------
    # build data
    # -------------------------------------------------
    print("[build data for pre-training] ...\n")
    dataset_train = build_dataset_to_pretrain(
        dataset_path=args.data_path,
        input_size=args.input_size,
        mim_ratio=args.mim_ratio,
        patch_size=args.patch_size,
        aal_standard_names_json=args.aal_standard_names_json,
        cache_dir="", 
    )

    data_loader_train = DataLoader(
        dataset=dataset_train,
        num_workers=args.dataloader_workers,
        pin_memory=False,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train),
            glb_batch_size=args.glb_batch_size,
            shuffle=True,
            filling=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        ),
        worker_init_fn=worker_init_fn,
        persistent_workers=(args.dataloader_workers > 0),
        prefetch_factor=1 if args.dataloader_workers > 0 else None,
    )

    itrt_train, iters_train = iter(data_loader_train), len(data_loader_train)
    print(f"[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size_per_gpu}, iters_train={iters_train}")

    # -------------------------------------------------
    # build model
    # -------------------------------------------------
    model_without_ddp = PETFoundationModel(
        text_pack_path=args.text_pack_path,
        num_aal=170,
        text_dim=512,
        local_loss=False,
        gather_with_grad=False,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    ).to(args.device)

    print(f"[PT model] model = {model_without_ddp}\n")

    if dist.initialized():
        model: DistributedDataParallel = DistributedDataParallel(
            model_without_ddp,
            device_ids=[dist.get_local_rank()],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    else:
        model = LocalDDP(model_without_ddp)

    # -------------------------------------------------
    # optimizer
    # -------------------------------------------------
    # param_groups: List[dict] = get_param_groups(model_without_ddp, text_lr_scale=0.1)
    param_groups: List[dict] = get_param_groups(model_without_ddp, text_lr_scale=1.0) # text encoder frozen, this scale is effectively unused
    opt_clz = {
        "sgd": partial(torch.optim.SGD, momentum=0.9, nesterov=True),
        "adamw": partial(torch.optim.AdamW, betas=(0.9, args.ada)),
        "lamb": partial(lamb.TheSameAsTimmLAMB, betas=(0.9, args.ada), max_grad_norm=5.0),
    }[args.opt]

    optimizer = opt_clz(params=param_groups, lr=args.lr, weight_decay=0.0)
    print(f"[optimizer] optimizer({opt_clz}) = {optimizer}\n")

    # -------------------------------------------------
    # resume
    # -------------------------------------------------
    ep_start, performance_desc = misc.load_checkpoint(
        args.resume_from,
        model_without_ddp,
        optimizer,
    )

    if ep_start >= args.ep:
        print(f"  [*] [PT already done]    Min/Last Loss: {performance_desc}")
    else:
        tb_lg = misc.TensorboardLogger(args.tb_lg_dir, is_master=dist.is_master(), prefix="pt")
        min_loss = 1e9
        print(f"[PT start] from ep{ep_start}")

        scaler = GradScaler() if args.amp else None
        pt_start_time = time.time()

        for ep in range(ep_start, args.ep):
            ep_start_time = time.time()
            tb_lg.set_step(ep * iters_train)

            if hasattr(itrt_train, "set_epoch"):
                itrt_train.set_epoch(ep)

            stats = pre_train_one_ep(
                ep=ep,
                args=args,
                tb_lg=tb_lg,
                itrt_train=itrt_train,
                iters_train=iters_train,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
            )

            last_loss = stats["last_loss"]
            min_loss = min(min_loss, last_loss)
            performance_desc = f"{min_loss:.4f} {last_loss:.4f}"

            save_every = 10
            if ((ep + 1) % save_every == 0) or (ep == args.ep - 1):
                misc.save_checkpoint(
                    f"{args.model}_still_pretraining.pth",
                    args,
                    ep,
                    performance_desc,
                    model_without_ddp.state_dict(),
                    optimizer.state_dict(),
                )

            if ep % 20 == 0 and ep != 0:
                misc.save_checkpoint(
                    f"{args.model}_{ep}.pth",
                    args,
                    ep,
                    performance_desc,
                    model_without_ddp.state_dict(),
                    optimizer.state_dict(),
                )

            ep_cost = round(time.time() - ep_start_time, 2) + 1
            remain_secs = (args.ep - 1 - ep) * ep_cost
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs))

            print(
                f"  [*] [ep{ep}/{args.ep}]    "
                f"Min/Last Loss {performance_desc},    "
                f"Cost: {ep_cost}s,    Remain: {remain_time},    Finish @ {finish_time}"
            )

            args.cur_ep = f"{ep + 1}/{args.ep}"
            args.remain_time, args.finish_time = str(remain_time), str(finish_time)
            args.last_loss = last_loss
            args.log_epoch()

            tb_lg.update(min_loss=min_loss, head="train", step=ep)
            tb_lg.update(rest_hours=round(remain_secs / 60 / 60, 2), head="z_burnout", step=ep)
            tb_lg.flush()

        tb_lg.update(min_loss=min_loss, head="result", step=ep_start)
        tb_lg.update(min_loss=min_loss, head="result", step=args.ep)
        tb_lg.flush()

        print(f"final args:\n{str(args)}")
        print("\n\n")
        print(
            f"  [*] [PT finished]    Min/Last Loss: {performance_desc},    "
            f"Total Cost: {(time.time() - pt_start_time) / 60 / 60:.1f}h\n"
        )
        print("\n\n")
        tb_lg.close()
        time.sleep(10)

    args.remain_time, args.finish_time = "-", time.strftime("%m-%d %H:%M", time.localtime(time.time()))
    args.log_epoch()


def pre_train_one_ep(
    ep,
    args: arg_util.Args,
    tb_lg: misc.TensorboardLogger,
    itrt_train,
    iters_train,
    model: DistributedDataParallel,
    optimizer,
    scaler,
):
    model.train()

    me = misc.MetricLogger(delimiter="  ")
    me.add_meter("max_lr", misc.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    header = f"[PT] Epoch {ep}:"

    warnings.filterwarnings("ignore")
    optimizer.zero_grad()

    early_clipping = args.clip > 0 and not hasattr(optimizer, "global_grad_norm")
    print("Early Clipping:", early_clipping)
    late_clipping = hasattr(optimizer, "global_grad_norm")

    if early_clipping:
        params_req_grad = [p for p in model.parameters() if p.requires_grad]

    for it, inp in enumerate(me.log_every(iters_train, itrt_train, 100, header)):
        # -------------------------------------------------
        # adjust lr and wd
        # -------------------------------------------------
        min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(
            optimizer,
            args.lr,
            args.wd,
            args.wde,
            it + ep * iters_train,
            args.wp_ep * iters_train,
            args.ep * iters_train,
        )

        # -------------------------------------------------
        # flatten crop list from RandCropByPosNegLabeld(num_samples=2)
        # inp is a list over batch, each item is a list of cropped dicts
        # -------------------------------------------------
        flat_inp = []
        for sample in inp:
            if isinstance(sample, list):
                flat_inp.extend(sample)
            else:
                flat_inp.append(sample)

        image_t = torch.cat([t["image"] for t in flat_inp], dim=0).to(args.device, non_blocking=True)
        mask_t = torch.cat([t["mask"] for t in flat_inp], dim=0).to(args.device, non_blocking=True)
        mask_image_t = torch.cat([t["mask_image"] for t in flat_inp], dim=0).to(args.device, non_blocking=True)

        aal_label_t = torch.cat([t["aal_label"] for t in flat_inp], dim=0).to(args.device, non_blocking=True)
        aal_patch_ratio_t = torch.cat([t["aal_patch_ratio"] for t in flat_inp], dim=0).to(args.device, non_blocking=True)
        aal_cover_ratio_t = torch.cat([t["aal_cover_ratio"] for t in flat_inp], dim=0).to(args.device, non_blocking=True)
        aal_state_t = torch.cat([t["aal_state"] for t in flat_inp], dim=0).to(args.device, non_blocking=True)
        aal_suvr_t = torch.cat([t["aal_suvr"] for t in flat_inp], dim=0).to(args.device, non_blocking=True)
        aal_state_t = aal_state_t.long()

        # -------------------------------------------------
        # forward / backward
        # -------------------------------------------------
        with autocast(enabled=scaler is not None, dtype=torch.bfloat16):
            out = model(
                image=image_t,
                mim_mask=mask_t,
                mask_image=mask_image_t,
                aal_label=aal_label_t,
                aal_patch_ratio=aal_patch_ratio_t,
                aal_cover_ratio=aal_cover_ratio_t,
                aal_state=aal_state_t,
                aal_suvr=aal_suvr_t,
                weight_recon=args.weight_recon,
                weight_clip=args.weight_clip,
                weight_matching=args.weight_matching,
                weight_suvr=args.weight_suvr,
                return_dict=True,
            )

            loss = out["loss"]
            recon_loss = out["recon_loss"]
            clip_loss = out["clip_loss"]
            matching_loss = out["matching_loss"]
            suvr_loss = out["suvr_loss"]

            grad_norm = None

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if early_clipping:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if early_clipping:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
                optimizer.step()
                if late_clipping:
                    grad_norm = optimizer.global_grad_norm

            loss = loss.item()
            recon_loss = recon_loss.item() if torch.is_tensor(recon_loss) else float(recon_loss)
            clip_loss = clip_loss.item() if torch.is_tensor(clip_loss) else float(clip_loss)
            matching_loss = matching_loss.item() if torch.is_tensor(matching_loss) else float(matching_loss)
            suvr_loss = suvr_loss.item() if torch.is_tensor(suvr_loss) else float(suvr_loss)

            optimizer.zero_grad(set_to_none=True)
            # torch.cuda.synchronize()

        # -------------------------------------------------
        # log
        # -------------------------------------------------
        me.update(last_loss=loss)
        me.update(recon_loss=recon_loss)
        me.update(clip_loss=clip_loss)
        me.update(matching_loss=matching_loss)
        me.update(suvr_loss=suvr_loss)
        me.update(max_lr=max_lr)

        tb_lg.update(loss=me.meters["last_loss"].global_avg, head="train_loss")
        tb_lg.update(recon_loss=me.meters["recon_loss"].global_avg, head="train_loss")
        tb_lg.update(clip_loss=me.meters["clip_loss"].global_avg, head="train_loss")
        tb_lg.update(matching_loss=me.meters["matching_loss"].global_avg, head="train_loss")
        tb_lg.update(suvr_loss=me.meters["suvr_loss"].global_avg, head="train_loss")

        tb_lg.update(sche_lr=max_lr, head="train_hp/lr_max")
        tb_lg.update(sche_lr=min_lr, head="train_hp/lr_min")
        tb_lg.update(sche_wd=max_wd, head="train_hp/wd_max")
        tb_lg.update(sche_wd=min_wd, head="train_hp/wd_min")

        if grad_norm is not None:
            me.update(orig_norm=grad_norm)
            tb_lg.update(orig_norm=grad_norm, head="train_hp")

        tb_lg.set_step()

    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}


if __name__ == "__main__":
    main_pt()
