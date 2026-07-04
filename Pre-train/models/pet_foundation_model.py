import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import open_clip


from models.nnunetv2 import UNetEncoder, UNetDecoder

# =========================================================
# utils
# =========================================================
def safe_normalize(x, dim=1, eps=1e-6):
    norm = x.norm(dim=dim, keepdim=True)
    return x / norm.clamp_min(eps)


def mask_mse(input, recon, mask, eps=1e-8):
    """
    在"非mask区域"算loss: diff算平方误差
    """
    mask = 1 - mask
    mask = mask.to(input.dtype)

    diff = (input - recon).pow(2) * mask

    B = input.shape[0]
    diff = diff.view(B, -1)
    mask = mask.view(B, -1)

    per_sample_loss = diff.sum(dim=1) / mask.sum(dim=1).clamp_min(eps)
    return per_sample_loss.mean()


def masked_mse_vector(pred, target, valid_mask, eps=1e-8):
    """
    SUVR回归：只在有效ROI上计算
    pred:       [B, R]
    target:     [B, R]
    valid_mask: [B, R]  1表示该ROI在patch里，要参与loss；0表示忽略
    """
    valid_mask = valid_mask.to(pred.dtype)
    diff = (pred - target).pow(2) * valid_mask
    per_sample = diff.sum(dim=1) / valid_mask.sum(dim=1).clamp_min(eps)
    return per_sample.mean()


# =========================================================
# distributed gather for CLIP
# =========================================================


# =========================================================
# losses
# =========================================================
class ReconstructLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, origin_image, reconstruct_image, mim_mask):
        return mask_mse(origin_image, reconstruct_image, mim_mask)


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False  # included for signature continuity, though not used below
):
    """
    Gathers image and text features across distributed processes.

    Args:
        image_features: Tensor of shape (batch_size, feature_dim)
        text_features: Tensor of shape (batch_size, feature_dim)
        local_loss: bool, whether to apply local-only loss (exclude other ranks' features in grad)
        gather_with_grad: bool, whether to allow gradient flow through the gather operation
        rank: int, current process rank
        world_size: int, total number of processes
        use_horovod: bool, placeholder (not actively used here)

    Returns:
        all_image_features, all_text_features: concatenated tensors from all processes
    """
    if local_loss and not gather_with_grad:
        raise ValueError("local_loss=True with gather_with_grad=False is not supported safely in this implementation.")
    
    if gather_with_grad:
        # Allows gradients to flow through the gathering operation
        gathered_image = torch.distributed.nn.all_gather(image_features)
        gathered_text = torch.distributed.nn.all_gather(text_features)
        all_image_features = torch.cat(gathered_image, dim=0)
        all_text_features = torch.cat(gathered_text, dim=0)
    else:
        # No gradient through gather
        gathered_image = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image, image_features)
        dist.all_gather(gathered_text, text_features)
        
        if not local_loss:
            # Ensure local rank's originals are included even if grads aren't flowing
            gathered_image[rank] = image_features
            gathered_text[rank] = text_features

        all_image_features = torch.cat(gathered_image, dim=0)
        all_text_features = torch.cat(gathered_text, dim=0)
        
    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            # print('BS: '+str(all_image_features.shape[0]))
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale=20):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels, ignore_index=-100) +
            F.cross_entropy(logits_per_text, labels, ignore_index=-100)
        ) / 2

        return total_loss


class MatchingLoss(nn.Module):
    """
    对每个ROI的4种状态做显式matching。
    logits:  [B, R, S]
    targets: [B, R, S]
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        targets = targets.float()

        valid_target = targets.view(-1)
        num_pos = valid_target.sum()
        num_neg = valid_target.numel() - num_pos
        pos_weight = torch.clamp(num_neg / (num_pos + 1e-8), min=1.0, max=20.0)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
        )
        return loss


# class SUVRRegressionLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, pred_suvr, gt_suvr, aal_label):
#         return masked_mse_vector(pred_suvr, gt_suvr, aal_label)
    

class SUVRClassificationLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred_logits, gt_state, aal_label):
        """
        pred_logits: [B, R, 4]
        gt_state:    [B, R]   0/1/2/3
        aal_label:   [B, R]   1=valid ROI, 0=ignore
        """
        gt_state = gt_state.long()
        valid_mask = aal_label > 0
        targets = torch.full_like(gt_state, fill_value=self.ignore_index) # 全部设为ignore
        cls_mask = valid_mask & (gt_state >= 1) & (gt_state <= 3) # valid ROI 且 gt_state in {1,2,3} 的位置参与分类
        targets[cls_mask] = gt_state[cls_mask] - 1

        pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])   # [B*156, 3]
        targets = targets.reshape(-1)                                  # [B*156]

        return F.cross_entropy(pred_logits, targets, ignore_index=self.ignore_index)

# =========================================================
# model
# =========================================================
class PETFoundationModel(nn.Module):
    def __init__(
        self,
        # load token_id tensor(after tokenizer)
        text_pack_path="/data/junyan/LiangJ/PET_Foundation_Model/aal_text_outputs/aal_text_pack.pt",
        num_aal=170,
        text_dim=512,
        proj_dim=512,
        text_local_batch_size=6,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()

        self.num_aal = num_aal
        self.num_states = 4
        self.suvr_num_states = 3
        self.stable_roi_idx = [94, 95, 96, 97, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]  
        self.suvr_roi_idx = [i for i in range(self.num_aal) if i not in self.stable_roi_idx] # suvr classification（except stable roi）
        self.num_suvr_roi = len(self.suvr_roi_idx)   # 156     
        self.text_dim = text_dim
        self.proj_dim = proj_dim
        self.text_local_batch_size = text_local_batch_size

        # -------------------------
        # image encoder / decoder
        # -------------------------
        self.image_encoder = UNetEncoder(dims=[32, 64, 128, 256, 512])
        self.image_decoder = UNetDecoder(dims=[32, 64, 128, 256, 512])

        # reconstruction head
        self.proj_intensity = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # global pooling on deepest feature
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # image -> text embedding for CLIP / matching
        # self.image_proj = nn.Linear(512, text_dim)
        self.image_proj = nn.Sequential(nn.Linear(512, text_dim),nn.GELU(),nn.Linear(text_dim, text_dim),)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, text_dim),nn.GELU(),nn.Linear(text_dim, text_dim),)

        # # suvr regression head
        # self.suvr_head = nn.Linear(512, num_aal)

        # suvr classification head
        # self.suvr_head = nn.Linear(512, num_aal * self.num_states)
        self.suvr_head = nn.Sequential(
            nn.LayerNorm(512),                          # [B, 512]
            nn.Linear(512, 512),                # [B, hidden_dim]
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512 // 2),    # [B, hidden_dim//2]
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512 // 2, self.num_suvr_roi * self.suvr_num_states)   # [B, 156*3]
        )

        # -------------------------
        # build BiomedCLIP model
        # -------------------------
        # def load_biomedclip_local(path):
        #     import os, json, open_clip
        #     from open_clip.factory import _MODEL_CONFIGS

        #     with open(os.path.join(path, "open_clip_config.json"), "r") as f:
        #         cfg = json.load(f)

        #     model_name = "biomedclip_local"
        #     if model_name not in _MODEL_CONFIGS:
        #         _MODEL_CONFIGS[model_name] = cfg["model_cfg"]

        #     model, _, preprocess = open_clip.create_model_and_transforms(
        #         model_name=model_name,
        #         pretrained=os.path.join(path, "open_clip_pytorch_model.bin"),
        #         **{f"image_{k}": v for k, v in cfg["preprocess_cfg"].items()},
        #     )

        #     return model, preprocess
        
        self.biomedclip, self.biomedclip_preprocess = open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.text_encoder = self.biomedclip.text
        
        # freeze text_encoder
        for p in self.biomedclip.parameters():
            p.requires_grad = False
       
        # CLIP用于对齐：projection后的embedding/text encoder直接输出最终embedding/无projection无output
        if hasattr(self.biomedclip, "text_projection") and self.biomedclip.text_projection is not None:
            actual_text_dim = int(self.biomedclip.text_projection.shape[-1])
        elif hasattr(self.text_encoder, "output_dim"):
            actual_text_dim = int(self.text_encoder.output_dim)
        else:
            actual_text_dim = text_dim

        if actual_text_dim != text_dim:
            raise ValueError(
                f"text_dim mismatch: model text dim is {actual_text_dim}, but got text_dim={text_dim}"
            )

        self.context_length = int(self.biomedclip.context_length)
        self.max_text_positions = getattr(getattr(self.text_encoder, "config", None),"max_position_embeddings",self.context_length)

        # -------------------------
        # load tokenized text pack
        # expected shape: [4, 170, L]
        # order of states:
        #   0 stable uptake
        #   1 low uptake
        #   2 medium uptake
        #   3 high uptake
        # -------------------------
        pack = torch.load(text_pack_path, map_location="cpu")
        if "tokens" not in pack:
            raise KeyError(f"'tokens' not found in text pack: {text_pack_path}")

        text_tokens = pack["tokens"]
        if not torch.is_tensor(text_tokens):
            text_tokens = torch.as_tensor(text_tokens)

        if text_tokens.ndim != 3:
            raise ValueError(f"text_tokens should be 3D, got shape={tuple(text_tokens.shape)}")

        if text_tokens.shape[0] != self.num_states or text_tokens.shape[1] != self.num_aal:
            raise ValueError(
                f"text_tokens shape mismatch. Expected first two dims ({self.num_states}, {self.num_aal}), "
                f"got {tuple(text_tokens.shape)}"
            )
        
        self.text_token_length = int(text_tokens.shape[2])

        if self.text_token_length > self.max_text_positions:
            raise ValueError(
                f"text token length too long. Max allowed={self.max_text_positions}, "
                f"got {self.text_token_length}"
            )

        text_tokens = text_tokens.long()
        self.register_buffer("text_tokens_all", text_tokens, persistent=True)
        # text_tokens_all: [4, 170, L] token id tensor


        # losses
        self.reconstruct_loss = ReconstructLoss()
        self.clip_loss = ClipLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            rank=rank,
            world_size=world_size,
        )
        self.matching_loss = MatchingLoss()
        self.suvr_loss = SUVRClassificationLoss()
        self._cached_text_features_raw = None

    def train(self, mode=True):
        super().train(mode)
        self.biomedclip.eval()
        return self

    # -----------------------------------------------------
    # helpers
    # -----------------------------------------------------
    # def _encode_all_texts(self):
    #     """
    #     整个text直接加载
    #     把固定的 [4, 170, L] tokens 送入可训练 text encoder，
    #     得到 [4, 170, 512] 的文本特征
    #     """
    #     flat_tokens = self.text_tokens_all.reshape(-1, self.text_token_length).long()  # [4*170, L]
    #     text_features = self.biomedclip.encode_text(flat_tokens)

    #     if isinstance(text_features, (tuple, list)):
    #         text_features = text_features[0]

    #     if text_features.shape[-1] != self.text_dim:
    #         raise ValueError(
    #             f"text feature dim mismatch: expected {self.text_dim}, got {text_features.shape[-1]}"
    #         )

    #     text_features = text_features.reshape(self.num_states, self.num_aal, self.text_dim)
    #     text_features = safe_normalize(text_features, dim=-1)
    #     return text_features
        
    def _encode_all_texts(self):
        """
        把固定的 [4, 170, L] tokens 分 batch
        送入可训练 text encoder，得到 [4, 170, 512] 的文本特征
        冻结 text encoder，只训练 text projection head
        """
        self.biomedclip.eval()
        flat_tokens = self.text_tokens_all.reshape(-1, self.text_token_length).long()  # [4*170, L]
        num_texts = flat_tokens.shape[0]
        encoded_text_list = []

        for start in range(0, num_texts, self.text_local_batch_size):
            text_tokens_batch = flat_tokens[start:start + self.text_local_batch_size]

            with torch.no_grad():
                text_features_batch = self.biomedclip.encode_text(text_tokens_batch)

            if isinstance(text_features_batch, (tuple, list)):
                text_features_batch = text_features_batch[0]

            text_features_batch = self.text_proj(text_features_batch)
            encoded_text_list.append(text_features_batch)

        text_features = torch.cat(encoded_text_list, dim=0)

        if text_features.shape[-1] != self.text_dim:
            raise ValueError(
                f"text feature dim mismatch: expected {self.text_dim}, got {text_features.shape[-1]}"
            )

        text_features = text_features.reshape(self.num_states, self.num_aal, self.text_dim)
        text_features = safe_normalize(text_features, dim=-1)
        return text_features
    
    
    # 控制image-text相似度在softmax中的“温度”，从而影响模型区分正负样本的能力
    def _get_clip_logit_scale(self):
        if hasattr(self.biomedclip, "logit_scale"):
            return self.biomedclip.logit_scale.exp().clamp(max=100.0)
        return torch.tensor(20.0, device=self.text_tokens_all.device)

    def _get_global_image_feature(self, image_features_pyramid_reversed):
        """
        reversed pyramid:
            [deepest, ..., shallowest]
        deepest feature expected shape [B, 512, d, h, w]
        """
        deepest_feature = image_features_pyramid_reversed[0]
        pooled = self.global_pool(deepest_feature).flatten(1)  # [B, 512]
        return pooled

    def _aggregate_patch_text_feature(self, aal_weight, aal_state, text_features_all):
        """
        根据 patch 内有哪些ROI + 每个ROI对应哪个吸收状态，
        从 self.text_features_all[4, 170, 512] 中聚合出 patch-level 正文本特征

        aal_weight: [B, 170]，可以是0/1，也可以是归一化后的soft weight
        aal_state: [B, 170], 0/1/2/3
        return:
            patch_text_feature: [B, 512]
        """
        bsz, _ = aal_weight.shape
        # device = aal_label.device

        state_idx = aal_state.long().clamp(min=0, max=self.num_states - 1)  # [B, R]

        # text_features_all: [4, 170, 512]
        # text_bank = self.text_features_all  # [4, 170, 512]

        # 取出每个样本、每个ROI对应state的文本特征
        # 先转成 [R, 4, 512] 再方便 gather（每个ROI都有4个状态向量）
        # 拓展batch维：给每个样本准备一份同样的文本库
        text_bank_r = text_features_all.permute(1, 0, 2).contiguous()      # [R, 4, 512]
        text_bank_r = text_bank_r.unsqueeze(0).expand(bsz, -1, -1, -1)       # [B, R, 4, 512]
        
        # 第4维 512 要展开出来，这样每个 feature 位置都用同一个状态索引(state要复制到 512 个特征维上)
        # selected [B,170,512]:当前 batch 中，每个样本、每个 ROI 已选好状态后的文本向量
        gather_idx = state_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.text_dim)  # [B, R, 1, 512]
        selected = torch.gather(text_bank_r, dim=2, index=gather_idx).squeeze(2)  # [B, R, 512]

        # 只对 patch 内有的ROI做平均
        weight = aal_weight.float().unsqueeze(-1)  # [B, R, 1]
        pooled = (selected * weight).sum(dim=1)  # [B, 512]

        pooled = safe_normalize(pooled, dim=-1)
        return pooled

    def _build_matching_targets(self, aal_label, aal_state):
        """
        对每个样本、每个ROI，只在“该ROI存在且对应的真实状态”那个位置标1，其它位置全是0，得到一个 [B,170,4] 的one-hot标签
        构造 matching 的 target:
        target[b, r, s] = 1 当且仅当
            aal_label[b, r] == 1 且 aal_state[b, r] == s
        否则为0
        """
        bsz, num_roi = aal_label.shape
        device = aal_label.device

        targets = torch.zeros((bsz, num_roi, self.num_states), device=device, dtype=torch.float32)

        state_idx = aal_state.long().clamp(min=0, max=self.num_states - 1).unsqueeze(-1)  # [B, R, 1]
        pos_mask = aal_label.float().unsqueeze(-1)  # [B, R, 1]

        targets.scatter_(2, state_idx, 1.0)
        targets = targets * pos_mask
        return targets

    def _compute_matching_logits(self, image_embed, text_features_all):
        """
        计算每个图像patch与所有ROI-状态文本语义之间的相似度
        image_embed: [B, 512]
        text_features_all: [4, 170, 512]

        输出:
            logits: [B, 170, 4]
        """
        # [4, 170, 512] -> [170, 4, 512]
        text_bank = text_features_all.permute(1, 0, 2).contiguous()  # [R, S, D]
        text_bank = safe_normalize(text_bank,dim=-1)
        logits = torch.einsum("bd,rsd->brs", image_embed, text_bank)  # [B, R, S]
        return logits

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------
    def forward(
        self,
        image,
        mim_mask=None,
        mask_image=None,
        aal_label=None,
        aal_patch_ratio=None,
        aal_cover_ratio=None,
        aal_state=None,
        aal_suvr=None,
        weight_recon=1.0,
        weight_clip=1.0,
        weight_matching=1.0,
        weight_suvr=1.0,
        return_dict=True,
    ):
        """
        inputs:
            image:      [B, 1, H, W, D]
            mim_mask:   [B, 1, H, W, D]
            mask_image: [B, 1, H, W, D]
            aal_label:  [B, 170]
            aal_patch_ratio: [B, 170]
            aal_cover_ratio: [B, 170]
            aal_state:  [B, 170]
            aal_suvr:   [B, 170]
        """
        # -------------------------
        # encode image
        # -------------------------
        encoder_input = mask_image if mask_image is not None else image
        image_features_pyramid = self.image_encoder(encoder_input)
        image_features_pyramid.reverse()

        # deepest pooled feature
        pooled_feat = self._get_global_image_feature(image_features_pyramid)  # [B, 512]

        # projected image embedding for clip / matching
        # image_embed = self.image_proj(pooled_feat)  # [B, 512]

        # image_embed normalize
        # image_embed = safe_normalize(pooled_feat, dim=-1)
        image_embed = safe_normalize(self.image_proj(pooled_feat), dim=-1)

        need_text_features = (
            (aal_label is not None)
            and (aal_state is not None)
            and ((weight_clip > 0) or (weight_matching > 0))
        )

        if need_text_features:
            text_features_all = self._encode_all_texts()   # [4, 170, 512]
        else:
            text_features_all = None

        # -------------------------
        # 1) reconstruction
        # -------------------------
        if (mask_image is not None) and (mim_mask is not None) and (weight_recon > 0):
            decode_feature = self.image_decoder(image_features_pyramid)
            reconstruct_img = self.proj_intensity(decode_feature)
            recon_loss = self.reconstruct_loss(image, reconstruct_img, mim_mask)
        else:
            reconstruct_img = None
            recon_loss = torch.tensor(0.0, device=image.device)

        # -------------------------
        # 2) clip alignment
        # -------------------------
        if (aal_label is not None) and (aal_state is not None) and (weight_clip > 0):
            if aal_patch_ratio is None or aal_cover_ratio is None:
                raise ValueError(
                    "aal_patch_ratio and aal_cover_ratio must be provided when weight_clip > 0"
                )

            aal_label_float = aal_label.float()
            aal_patch_ratio = aal_patch_ratio.float()
            aal_cover_ratio = aal_cover_ratio.float()
            
            # 计算patch中ROI的权重
            weights_aal = aal_label_float * torch.sqrt(
                torch.clamp(aal_patch_ratio, min=1e-8)
                * torch.clamp(aal_cover_ratio, min=1e-8)
            )
            weights_aal = weights_aal / (weights_aal.sum(dim=1, keepdim=True) + 1e-8) # 归一化变成概率分布（和为1）

            # ----------------------
            # input：weights_aal：[B, 170]、aal_state：[B, 170]（0~3）、text_features_all：[4, 170, 512]
            # output: patch_text_feature: [B, 512]
            # ----------------------
            patch_text_feature = self._aggregate_patch_text_feature(
                weights_aal,
                aal_state,
                text_features_all,
            )
            clip_loss = self.clip_loss(
                image_embed,
                patch_text_feature,
                self._get_clip_logit_scale(),
            )
        else:
            patch_text_feature = None
            clip_loss = torch.tensor(0.0, device=image.device)

        # -------------------------
        # 3) matching
        # -------------------------
        if (aal_label is not None) and (aal_state is not None) and (weight_matching > 0):
            matching_logits = self._compute_matching_logits(image_embed, text_features_all)  # [B, 170, 4]
            matching_targets = self._build_matching_targets(aal_label, aal_state)  # [B, 170, 4]
            matching_loss = self.matching_loss(matching_logits, matching_targets)
        else:
            matching_logits = None
            matching_targets = None
            matching_loss = torch.tensor(0.0, device=image.device)

        # # -------------------------
        # # 4) suvr regression
        # # -------------------------
        # if (aal_label is not None) and (aal_suvr is not None) and (weight_suvr > 0):
        #     pred_suvr = self.suvr_head(pooled_feat)  # [B, 170]
        #     suvr_loss = self.suvr_loss(pred_suvr, aal_suvr, aal_label)
        # else:
        #     pred_suvr = None
        #     suvr_loss = torch.tensor(0.0, device=image.device)

        # -------------------------
        # 4) suvr classification
        # -------------------------            
        if (aal_label is not None) and (aal_state is not None) and (weight_suvr > 0):
            keep_idx = torch.as_tensor(self.suvr_roi_idx, device=aal_label.device, dtype=torch.long)
            aal_label_suvr = torch.index_select(aal_label, dim=1, index=keep_idx)   # [B,156]
            aal_state_suvr = torch.index_select(aal_state, dim=1, index=keep_idx)   # [B,156]

            pred_logits = self.suvr_head(pooled_feat).view(-1, self.num_suvr_roi, self.suvr_num_states)  # [B,156,3]
            suvr_loss = self.suvr_loss(pred_logits, aal_state_suvr, aal_label_suvr)
        else:
            pred_logits = None
            suvr_loss = torch.tensor(0.0, device=image.device)

        total_loss = (
            weight_recon * recon_loss
            + weight_clip * clip_loss
            + weight_matching * matching_loss
            + weight_suvr * suvr_loss
        )

        if not return_dict:
            return total_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "clip_loss": clip_loss,
            "matching_loss": matching_loss,
            "suvr_loss": suvr_loss,
        }


