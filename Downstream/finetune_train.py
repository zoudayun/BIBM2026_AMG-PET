import os
import sys
import time
import warnings, argparse, importlib

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader
from focal_loss import FocalLoss
from models.nnunet_mae_double import AV45BinaryClassifier, load_pretrained_mclim_encoder

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def str2bool(x):
    return x if isinstance(x, bool) else x.lower() in ("1", "true", "t", "yes", "y")


def get_args():
    p = argparse.ArgumentParser()

    for k, v, t in [
        ("dataset_name", "adni_av45_1mm", str),
        ("data_path", "/data/zhaomy/downstream/multi_classification/data_list", str),
        ("dataloader_root", "/data/zhaomy/downstream/multi_classification", str),
        ("log_dir", "/data/zhaomy/downstream/multi_classification/results_latest/Finetune_1mm/adni_av45_1mm_finetune", str),
        ("pretrained_ckpt", "", str),
        ("strict_pretrained", True, str2bool),
        ("max_epochs", 200, int),
        ("learning_rate", 5e-6, float),
        ("train_batch_size", 8, int),
        ("val_batch_size", 8, int),
        ("num_workers", 4, int),
        ("eval_every", 1, int),
        ("dropout", 0.2, float),
        ("num_classes", 3, int),
        ("gamma", 0.5, float),
        ("alpha", "0.83,1.09,1.14", str),
        ("input_size_x", 128, int),
        ("input_size_y", 128, int),
        ("input_size_z", 128, int),
        ("fold", 0, int),
        ("num_folds", 5, int),
        ("seed", 42, int),
        ("resume", True, str2bool),
        ("device", "auto", str),
    ]:
        p.add_argument(f"--{k}", default=v, type=t)

    return p.parse_args()


def compute_metrics(y_true, y_prob, num_classes=3):
    """
    y_true: [N], label = 0/1/2
    y_prob: [N, 3], after softmax 
    """

    labels = list(range(num_classes))
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(y_true,y_pred,labels=labels,zero_division=0,)

    specificity = []
    total = cm.sum()

    for c in labels:
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = total - TP - FN - FP

        spe = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificity.append(spe)

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_specificity = sum(specificity) / len(specificity)
    macro_f1 = f1.mean()

    weighted_precision = (precision * support).sum() / max(support.sum(), 1)
    weighted_recall = (recall * support).sum() / max(support.sum(), 1)
    weighted_specificity = sum(s * n for s, n in zip(specificity, support)) / max(support.sum(), 1)
    weighted_f1 = (f1 * support).sum() / max(support.sum(), 1)

    try:
        y_true_onehot = F.one_hot(torch.tensor(y_true).long(),num_classes=num_classes,).numpy()
        auc_macro = roc_auc_score(y_true_onehot,y_prob,average="macro",multi_class="ovr",)
        auc_weighted = roc_auc_score(y_true_onehot,y_prob,average="weighted",multi_class="ovr",)
        auc_per_class = roc_auc_score(y_true_onehot,y_prob,average=None,multi_class="ovr",)

    except Exception:
        auc_macro = 0.0
        auc_weighted = 0.0
        auc_per_class = [0.0, 0.0, 0.0]

    return {
        "acc": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_sensitivity": macro_recall,
        "macro_specificity": macro_specificity,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_sensitivity": weighted_recall,
        "weighted_specificity": weighted_specificity,
        "weighted_f1": weighted_f1,

        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
        "auc_CN": auc_per_class[0],
        "auc_MCI": auc_per_class[1],
        "auc_AD": auc_per_class[2],

        "sen_CN": recall[0],
        "sen_MCI": recall[1],
        "sen_AD": recall[2],

        "spe_CN": specificity[0],
        "spe_MCI": specificity[1],
        "spe_AD": specificity[2],

        "f1_CN": f1[0],
        "f1_MCI": f1[1],
        "f1_AD": f1[2],

        "support_CN": support[0],
        "support_MCI": support[1],
        "support_AD": support[2],

        "cm": cm,
    }


def main():
    warnings.filterwarnings("ignore")
    args = get_args()

    args.log_dir = os.path.join(args.log_dir, f"fold_{args.fold}")
    os.makedirs(args.log_dir, exist_ok=True)

    if args.dataloader_root not in sys.path:
        sys.path.insert(0, args.dataloader_root)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    build_dataset = importlib.import_module(f"dataloader.{args.dataset_name}").build_dataset

    train_ds, val_ds, *_ = build_dataset(args.data_path,input_size_x=args.input_size_x,input_size_y=args.input_size_y,input_size_z=args.input_size_z,fold=args.fold,num_folds=args.num_folds,seed=args.seed,)
    train_loader = DataLoader(train_ds,batch_size=args.train_batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=torch.cuda.is_available(),)
    val_loader = DataLoader(val_ds,batch_size=args.val_batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=torch.cuda.is_available(),)

    model = AV45BinaryClassifier(dropout=args.dropout,num_classes=args.num_classes,).to(device)
    load_pretrained_mclim_encoder(model,args.pretrained_ckpt,device="cpu",strict=args.strict_pretrained,)

    for p in model.encoder.parameters():
        p.requires_grad = True

    print(f"loaded pretrained encoder from: {args.pretrained_ckpt}")

    alpha = [float(x.strip()) for x in args.alpha.split(",")]
    assert len(alpha) == args.num_classes, "alpha 数量必须等于 num_classes，例如三分类: 0.83,1.09,1.14"
    criterion = FocalLoss(gamma=args.gamma,alpha=alpha,reduction="mean",task_type="multi-class",num_classes=args.num_classes,)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=0.01,)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99,)

    train_log_file = os.path.join(args.log_dir, "train.log")
    best_model_path = os.path.join(args.log_dir, "best_model.pth")
    best_ckpt_path = os.path.join(args.log_dir, "best_checkpoint.pth")
    still_path = os.path.join(args.log_dir, "model_still_training.pth")
    final_model_path = os.path.join(args.log_dir, "final_model.pth")
    final_ckpt_path = os.path.join(args.log_dir, "final_checkpoint.pth")

    start_epoch = 0
    best_auc = -1.0
    best_auc_epoch = -1
    global_step = 0
    latest_metrics = {}

    if args.resume and os.path.exists(still_path):
        ckpt = torch.load(still_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", -1.0)
        best_auc_epoch = ckpt.get("best_auc_epoch", -1)
        global_step = ckpt.get("global_step", 0)
        latest_metrics = ckpt.get("latest_metrics", {})

        print(f"Resume training from epoch {start_epoch}")
    else:
        print("Start pretrain-finetune training from epoch 0")

    def save_ckpt(path, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_auc": best_auc,
                "best_auc_epoch": best_auc_epoch,
                "global_step": global_step,
                "latest_metrics": latest_metrics,
                "args": vars(args),
            },
            path,
        )

    writer = SummaryWriter(log_dir=args.log_dir)

    mode = "a" if (
        args.resume
        and os.path.exists(still_path)
        and os.path.exists(train_log_file)
    ) else "w"

    with open(train_log_file, mode) as f:
        if mode == "w":
            f.write("==== PET three-class pretrain-finetune training ====\n")
            f.write(f"dataset_name    : {args.dataset_name}\n")
            f.write(f"data_path       : {args.data_path}\n")
            f.write(f"pretrained_ckpt : {args.pretrained_ckpt}\n")
            f.write(f"strict_pretrained : {args.strict_pretrained}\n")
            f.write(f"learning_rate   : {args.learning_rate}\n")
            f.write("lr_scheduler    : ExponentialLR, gamma=0.99\n")
            f.write(f"input_size      : ({args.input_size_x},{args.input_size_y},{args.input_size_z})\n")
            f.write(f"num_classes     : {args.num_classes}\n")
            f.write(f"focal loss      : gamma={args.gamma}, alpha={args.alpha}, task_type=multi-class\n")
            f.write(f"max_epochs      : {args.max_epochs}\n")
            f.write(f"batch_size      : train={args.train_batch_size}, val={args.val_batch_size}\n")
            f.write("labels          : CN=0, MCI=1, AD=2\n")
            f.write("==========================\n")
            f.flush()

        for epoch in range(start_epoch, args.max_epochs):
            epoch_t = time.time()
            lr_this_epoch = optimizer.param_groups[0]["lr"]
            new_best = False

            print("-" * 20)
            print(f"epoch {epoch + 1}/{args.max_epochs} | dataset={args.dataset_name}")
            print(f"current lr: {lr_this_epoch:.8e}")

            model.train()
            train_loss = 0.0

            for step, batch in enumerate(train_loader, 1):
                global_step += 1

                x = batch["image"].to(device)
                y = batch["label"].to(device).long().view(-1)

                optimizer.zero_grad()

                out = model(x)
                loss = criterion(out, y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), global_step)

            train_loss /= max(len(train_loader), 1)

            latest_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "lr": lr_this_epoch,
            }

            print(f"epoch {epoch + 1} average loss: {train_loss:.4f}")

            if (epoch + 1) % args.eval_every == 0:
                model.eval()

                val_loss = 0.0
                all_prob = []
                all_label = []

                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["image"].to(device)
                        y = batch["label"].to(device).long().view(-1)

                        out = model(x)
                        loss = criterion(out, y)

                        val_loss += loss.item()

                        prob = torch.softmax(out, dim=1)
                        all_prob.append(prob.cpu())
                        all_label.append(y.cpu())

                val_loss /= max(len(val_loader), 1)

                all_prob = torch.cat(all_prob, dim=0).numpy()
                all_label = torch.cat(all_label, dim=0).numpy()

                metrics = compute_metrics(
                    y_true=all_label,
                    y_prob=all_prob,
                    num_classes=args.num_classes,
                )

                auc = metrics["auc_macro"]

                latest_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": metrics["acc"],
                    "val_auc_macro": metrics["auc_macro"],
                    "val_auc_weighted": metrics["auc_weighted"],

                    "val_macro_sensitivity": metrics["macro_sensitivity"],
                    "val_macro_specificity": metrics["macro_specificity"],
                    "val_weighted_sensitivity": metrics["weighted_sensitivity"],
                    "val_weighted_specificity": metrics["weighted_specificity"],

                    "val_macro_f1": metrics["macro_f1"],
                    "val_weighted_f1": metrics["weighted_f1"],

                    "val_sen_CN": metrics["sen_CN"],
                    "val_sen_MCI": metrics["sen_MCI"],
                    "val_sen_AD": metrics["sen_AD"],

                    "val_spe_CN": metrics["spe_CN"],
                    "val_spe_MCI": metrics["spe_MCI"],
                    "val_spe_AD": metrics["spe_AD"],

                    "val_f1_CN": metrics["f1_CN"],
                    "val_f1_MCI": metrics["f1_MCI"],
                    "val_f1_AD": metrics["f1_AD"],

                    "val_auc_CN": metrics["auc_CN"],
                    "val_auc_MCI": metrics["auc_MCI"],
                    "val_auc_AD": metrics["auc_AD"],

                    "support_CN": metrics["support_CN"],
                    "support_MCI": metrics["support_MCI"],
                    "support_AD": metrics["support_AD"],

                    "confusion_matrix": metrics["cm"].tolist(),
                    "lr": lr_this_epoch,
                }

                if auc > best_auc:
                    best_auc = auc
                    best_auc_epoch = epoch + 1
                    new_best = True

                    torch.save(model.state_dict(), best_model_path)

                print(
                    f"current epoch: {epoch + 1} "
                    f"current accuracy: {metrics['acc']:.4f} "
                    f"current macro AUC: {metrics['auc_macro']:.4f} "
                    f"current macro F1: {metrics['macro_f1']:.4f} "
                    f"best macro AUC: {best_auc:.4f} at epoch {best_auc_epoch}"
                )
                print(f"epoch {epoch + 1} validation loss: {val_loss:.4f}")
                print(f"CM={metrics['cm'].tolist()}")
                print(
                    f"sen_CN={metrics['sen_CN']:.4f}, "
                    f"sen_MCI={metrics['sen_MCI']:.4f}, "
                    f"sen_AD={metrics['sen_AD']:.4f}"
                )
                print(
                    f"spe_CN={metrics['spe_CN']:.4f}, "
                    f"spe_MCI={metrics['spe_MCI']:.4f}, "
                    f"spe_AD={metrics['spe_AD']:.4f}"
                )
                print(
                    f"f1_CN={metrics['f1_CN']:.4f}, "
                    f"f1_MCI={metrics['f1_MCI']:.4f}, "
                    f"f1_AD={metrics['f1_AD']:.4f}"
                )

                for k, v in [
                    ("val_loss", val_loss),
                    ("val_acc", metrics["acc"]),
                    ("val_auc_macro", metrics["auc_macro"]),
                    ("val_auc_weighted", metrics["auc_weighted"]),

                    ("val_macro_sensitivity", metrics["macro_sensitivity"]),
                    ("val_macro_specificity", metrics["macro_specificity"]),
                    ("val_weighted_sensitivity", metrics["weighted_sensitivity"]),
                    ("val_weighted_specificity", metrics["weighted_specificity"]),

                    ("val_macro_f1", metrics["macro_f1"]),
                    ("val_weighted_f1", metrics["weighted_f1"]),

                    ("val_sen_CN", metrics["sen_CN"]),
                    ("val_sen_MCI", metrics["sen_MCI"]),
                    ("val_sen_AD", metrics["sen_AD"]),

                    ("val_spe_CN", metrics["spe_CN"]),
                    ("val_spe_MCI", metrics["spe_MCI"]),
                    ("val_spe_AD", metrics["spe_AD"]),

                    ("val_f1_CN", metrics["f1_CN"]),
                    ("val_f1_MCI", metrics["f1_MCI"]),
                    ("val_f1_AD", metrics["f1_AD"]),

                    ("val_auc_CN", metrics["auc_CN"]),
                    ("val_auc_MCI", metrics["auc_MCI"]),
                    ("val_auc_AD", metrics["auc_AD"]),

                    ("learning_rate", lr_this_epoch),
                ]:
                    writer.add_scalar(k, v, epoch + 1)

                log_msg = (
                    f"Epoch {epoch + 1:04d}/{args.max_epochs:04d} | "
                    f"train_loss={train_loss:.6f} | "
                    f"val_loss={val_loss:.6f} | "
                    f"val_acc={metrics['acc']:.4f} | "
                    f"val_auc_macro={metrics['auc_macro']:.4f} | "
                    f"val_auc_weighted={metrics['auc_weighted']:.4f} | "

                    f"macro_sensitivity={metrics['macro_sensitivity']:.4f} | "
                    f"macro_specificity={metrics['macro_specificity']:.4f} | "
                    f"weighted_sensitivity={metrics['weighted_sensitivity']:.4f} | "
                    f"weighted_specificity={metrics['weighted_specificity']:.4f} | "

                    f"macro_F1={metrics['macro_f1']:.4f} | "
                    f"weighted_F1={metrics['weighted_f1']:.4f} | "

                    f"sen_CN={metrics['sen_CN']:.4f}, "
                    f"sen_MCI={metrics['sen_MCI']:.4f}, "
                    f"sen_AD={metrics['sen_AD']:.4f} | "

                    f"spe_CN={metrics['spe_CN']:.4f}, "
                    f"spe_MCI={metrics['spe_MCI']:.4f}, "
                    f"spe_AD={metrics['spe_AD']:.4f} | "

                    f"f1_CN={metrics['f1_CN']:.4f}, "
                    f"f1_MCI={metrics['f1_MCI']:.4f}, "
                    f"f1_AD={metrics['f1_AD']:.4f} | "

                    f"auc_CN={metrics['auc_CN']:.4f}, "
                    f"auc_MCI={metrics['auc_MCI']:.4f}, "
                    f"auc_AD={metrics['auc_AD']:.4f} | "

                    f"CM={metrics['cm'].tolist()} | "
                    f"best_macro_AUC={best_auc:.4f}@epoch{best_auc_epoch} | "
                    f"lr={lr_this_epoch:.8e} | "
                    f"time={time.time() - epoch_t:.3f}s"
                )

            else:
                writer.add_scalar("learning_rate", lr_this_epoch, epoch + 1)

                log_msg = (
                    f"Epoch {epoch + 1:04d}/{args.max_epochs:04d} | "
                    f"train_loss={train_loss:.6f} | "
                    f"lr={lr_this_epoch:.8e} | "
                    f"time={time.time() - epoch_t:.3f}s"
                )

            print(log_msg, flush=True)
            f.write(log_msg + "\n")
            f.flush()

            scheduler.step()
            next_lr = optimizer.param_groups[0]["lr"]
            print(f"next epoch lr: {next_lr:.8e}")

            if new_best:
                save_ckpt(best_ckpt_path, epoch)
                print("saved new best macro AUC model/checkpoint")

            save_every = 10
            if (epoch + 1) % save_every == 0 or (epoch + 1) == args.max_epochs:
                save_ckpt(still_path, epoch)
                print("saved model_still_training checkpoint")

    print(f"train completed, best_macro_AUC: {best_auc:.4f} at epoch: {best_auc_epoch}")

    torch.save(model.state_dict(), final_model_path)
    save_ckpt(final_ckpt_path, args.max_epochs - 1)

    print("saved final model/checkpoint")

    writer.close()


if __name__ == "__main__":
    main()