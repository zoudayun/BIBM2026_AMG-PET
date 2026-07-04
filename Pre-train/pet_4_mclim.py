import os
import json
import numpy as np
from monai.data import Dataset, PersistentDataset
import os
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    ScaleIntensityRangePercentilesd,
    ToTensord,
)
from monai.transforms.transform import MapTransform


class Mask_Origin_Img(MapTransform):
    """
    Mask the input image for MIM.
    """

    def __init__(self, keys, img_size, mask_ratio, patch_size, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_num_per_dim = int(img_size // patch_size)
        self.len_keep = round(
            self.patch_num_per_dim
            * self.patch_num_per_dim
            * self.patch_num_per_dim
            * (1 - self.mask_ratio)
        )

    def __call__(self, data):
        d = dict(data)
        if self.mask_ratio > 0:
            f = self.patch_num_per_dim
            idx = np.random.rand(f * f * f).argsort()
            idx = idx[: self.len_keep]

            msk = np.array(list(range(f * f * f)))
            msk = np.where(np.isin(msk, idx), 1, 0)

            img = d["image"]
            mask = np.zeros_like(img, dtype=np.float32)

            for i in range(self.patch_num_per_dim):
                for j in range(self.patch_num_per_dim):
                    for k in range(self.patch_num_per_dim):
                        patch_idx = (
                            i * self.patch_num_per_dim * self.patch_num_per_dim
                            + j * self.patch_num_per_dim
                            + k
                        )
                        mask[
                            :,
                            i * self.patch_size : (i + 1) * self.patch_size,
                            j * self.patch_size : (j + 1) * self.patch_size,
                            k * self.patch_size : (k + 1) * self.patch_size,
                        ] = msk[patch_idx]

            d["mask_image"] = img * mask
            d["mask"] = mask

        return d


class PatchAALTargets(MapTransform):
    """
    从 crop 后的 AAL patch 中提取：
    1) patch 内包含哪些脑区 -> aal_label (170,)
    2) patch 内脑区的离散吸收状态 -> aal_state (170,)
    3) patch 内脑区的真实 SUVR -> aal_suvr (170,)
    4) patch 内每个脑区占 patch 所有非零AAL体素的比例 -> aal_patch_ratio (170,)
    5) patch 内每个脑区覆盖原脑区的比例 -> aal_cover_ratio (170,)

    注意：
    roi_size_full 现在是每个样本个性化的，从 d["roi_size_full"] 读取
    """

    def __init__(self, keys, num_aal=170, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.num_aal = num_aal

    def __call__(self, data):
        d = dict(data)

        # randcrop后的aal (C,H,W,D)
        aal = d["aal"]
        if aal.ndim == 4:
            aal_arr = aal[0]
        else:
            aal_arr = aal

        aal_arr = np.asarray(aal_arr).astype(np.int32)

        unique_labels, counts = np.unique(aal_arr, return_counts=True)

        # 去掉背景0
        fg_mask = unique_labels > 0
        unique_labels = unique_labels[fg_mask]
        counts = counts[fg_mask].astype(np.float32)

        # 个性化信息
        roi_state_full = np.asarray(d["roi_state_full"], dtype=np.int64)
        roi_suvr_full = np.asarray(d["roi_suvr_full"], dtype=np.float32)
        roi_size_full = np.asarray(d["roi_size_full"], dtype=np.float32)

        if roi_size_full.ndim != 1 or roi_size_full.shape[0] != self.num_aal:
            raise ValueError(
                f"roi_size_full should have shape ({self.num_aal},), got {roi_size_full.shape}"
            )

        # outputs
        aal_label = np.zeros((self.num_aal,), dtype=np.float32)
        aal_state = np.zeros((self.num_aal,), dtype=np.int64)
        aal_suvr = np.zeros((self.num_aal,), dtype=np.float32)
        aal_patch_ratio = np.zeros((self.num_aal,), dtype=np.float32)
        aal_cover_ratio = np.zeros((self.num_aal,), dtype=np.float32)

        # atlas label 假定从 1 开始，对应 index = label - 1
        valid_idx = unique_labels - 1
        valid_mask = (valid_idx >= 0) & (valid_idx < self.num_aal)
        
        # 过滤label=0或>170的值
        valid_idx = valid_idx[valid_mask]
        valid_counts = counts[valid_mask]

        # 1) label / state / suvr
        aal_label[valid_idx] = 1.0
        aal_state[valid_idx] = roi_state_full[valid_idx]
        aal_suvr[valid_idx] = roi_suvr_full[valid_idx]

        # 2) patch内各脑区占 patch 全部非零标签体素的比例
        total_fg_voxels = valid_counts.sum()
        if total_fg_voxels > 0:
            aal_patch_ratio[valid_idx] = valid_counts / total_fg_voxels

        # 3) patch内体素数 / 该样本原始ROI体素数
        denom = roi_size_full[valid_idx]
        nz = denom > 0
        if np.any(nz):
            idx_nz = valid_idx[nz]
            cnt_nz = valid_counts[nz]
            denom_nz = denom[nz]
            aal_cover_ratio[idx_nz] = cnt_nz / denom_nz

        d["aal_label"] = aal_label
        d["aal_state"] = aal_state
        d["aal_suvr"] = aal_suvr
        d["aal_patch_ratio"] = aal_patch_ratio
        d["aal_cover_ratio"] = aal_cover_ratio

        return d


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_state_and_suvr_vectors(
    aal_pet_text_json_path,
    roi_keys_in_order,
):
    """
    读取单个样本的 aal_PET_text.json，
    构造两个长度为 num_aal 的向量：
      - roi_state_full: 0/1/2/3
      - roi_suvr_full: float

    状态编码顺序按你的要求：
      stable uptake -> 0
      low uptake    -> 1
      medium uptake -> 2
      high uptake   -> 3
    """
    state_map = {
        "stable uptake": 0,
        "low uptake": 1,
        "medium uptake": 2,
        "high uptake": 3,
    }

    data = load_json(aal_pet_text_json_path)
    num_aal = len(roi_keys_in_order)

    roi_state_full = np.zeros((num_aal,), dtype=np.int64)
    roi_suvr_full = np.zeros((num_aal,), dtype=np.float32)

    for idx, roi_key in enumerate(roi_keys_in_order):
        if roi_key not in data:
            # 若某个 key 缺失，保持默认 0
            continue

        info = data[roi_key]

        # suvr
        suvr = info.get("suvr", 0.0)
        try:
            roi_suvr_full[idx] = float(suvr)
        except Exception:
            roi_suvr_full[idx] = 0.0

        # discrete state
        state_str = info.get("SUVR_Discrete_State", "stable uptake")
        state_str = str(state_str).strip().lower()
        roi_state_full[idx] = state_map.get(state_str, 0)

    return roi_state_full, roi_suvr_full


def build_dataset_to_pretrain(
    dataset_path,
    input_size,
    mim_ratio,
    patch_size,
    aal_standard_names_json="/data/junyan/LiangJ/PET_Foundation_Model/aal_text_outputs/aal_standard_name.json",
    cache_dir=None,
) -> Dataset:
    """
    输出的每个 sample（每个随机 crop）会包含：
      - image: 原图 patch
      - mask_image: mask后的 patch
      - mask: MIM mask
      - aal_label: (170,) 0/1，patch里有哪些脑区
      - aal_state: (170,) 0/1/2/3，patch外统一0
      - aal_suvr:  (170,) float，patch外统一0

    其中：
      - aal_state 要和 aal_label 一起用，因为 0 同时可能表示 stable 或 absent
      - aal_suvr 可直接配合 aal_label 做回归 loss mask
    """

    # 用 standard_names 的 key 顺序作为全局固定脑区顺序
    aal_standard_names = load_json(aal_standard_names_json)
    roi_keys_in_order = list(aal_standard_names.keys())
    num_aal = len(roi_keys_in_order)

    print(f"Using {num_aal} AAL ROIs from: {aal_standard_names_json}")

    tr_transforms = Compose(
        [
            LoadImaged(keys=["image", "brain_mask", "aal"], image_only=True),
            EnsureChannelFirstd(keys=["image", "brain_mask", "aal"]),
            Orientationd(keys=["image", "brain_mask", "aal"], axcodes="RAS"),
            CropForegroundd(
                keys=["image", "brain_mask", "aal"],
                source_key="brain_mask",
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=1,
                upper=99,
                b_min=0,
                b_max=1,
            ),
            NormalizeIntensityd(keys=["image"]),
            SpatialPadd(
                keys=["image", "brain_mask", "aal"],
                spatial_size=(input_size, input_size, input_size),
                mode=("minimum", "constant", "constant"),
            ),
            RandCropByPosNegLabeld(
                keys=["image", "brain_mask", "aal"],
                label_key="brain_mask",
                spatial_size=(input_size, input_size, input_size),
                pos=1,
                neg=0.1,
                num_samples=1,
            ),
            Mask_Origin_Img(
                keys=["image"],
                img_size=input_size,
                mask_ratio=mim_ratio,
                patch_size=patch_size,
            ),
            PatchAALTargets(keys=["aal"], num_aal=num_aal),
            ToTensord(
                keys=[
                    "image",
                    "mask_image",
                    "mask",
                    "aal_label",
                    "aal_state",
                    "aal_suvr",
                    "aal_patch_ratio",
                    "aal_cover_ratio",
                    "roi_state_full",
                    "roi_suvr_full",
                ],
                track_meta=False,
            ),
        ]
    )

    datalist = []

    scan_list_pet = sorted(os.listdir(dataset_path))

    for scan in scan_list_pet:
        scan_dir = os.path.join(dataset_path, scan)
        if not os.path.isdir(scan_dir):
            continue

        image_path = os.path.join(scan_dir, f"{scan}_rigid_Warped.nii.gz")
        brain_mask_path = os.path.join(scan_dir, f"{scan}_mask_rigid.nii.gz")
        aal_path = os.path.join(scan_dir, f"{scan}_aal_mask_warp_rigid.nii.gz")
        aal_pet_text_path = os.path.join(scan_dir, "aal_PET_text.json")
        roi_size_path = os.path.join(scan_dir, "aal_roi_size_full.npy")

        if not os.path.exists(image_path):
            print(f"[Skip] missing image: {image_path}")
            continue
        if not os.path.exists(brain_mask_path):
            print(f"[Skip] missing brain_mask: {brain_mask_path}")
            continue
        if not os.path.exists(aal_path):
            print(f"[Skip] missing AAL atlas: {aal_path}")
            continue
        if not os.path.exists(aal_pet_text_path):
            print(f"[Skip] missing aal_PET_text.json: {aal_pet_text_path}")
            continue

        roi_state_full, roi_suvr_full = build_state_and_suvr_vectors(
            aal_pet_text_json_path=aal_pet_text_path,
            roi_keys_in_order=roi_keys_in_order,
        )

        roi_size_full = np.load(roi_size_path).astype(np.float32)
        if roi_size_full.ndim != 1 or roi_size_full.shape[0] != num_aal:
            print(f"[Skip] invalid roi_size_full shape at {roi_size_path}: {roi_size_full.shape}")
            continue

        datalist.append(
            {
                "image": image_path,
                "brain_mask": brain_mask_path,
                "aal": aal_path,
                "roi_size_full": roi_size_full,
                "roi_state_full": roi_state_full,
                "roi_suvr_full": roi_suvr_full,
                "scan_id": scan,
            }
        )

    print("Dataset all training: number of data: {}".format(len(datalist)))

    # dataset_train = Dataset(data=datalist, transform=tr_transforms)    

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        dataset_train = PersistentDataset(
            data=datalist,
            transform=tr_transforms,
            cache_dir=cache_dir,
        )
    else:
        dataset_train = Dataset(
            data=datalist,
            transform=tr_transforms,
        )

    return dataset_train


# if __name__ == "__main__":
#     dataset_path = "/data/junyan/PET_MNI_1mm"

#     dataset = build_dataset_to_pretrain(
#         dataset_path=dataset_path,
#         input_size=64,
#         mim_ratio=0.75,
#         patch_size=8,
#     )

#     print("\n==== Dataset Basic Info ====")
#     print("Dataset length:", len(dataset))

#     num_samples_to_check = min(6, len(dataset))
#     print(f"\n==== Checking first {num_samples_to_check} samples ====")

#     for i in range(num_samples_to_check):
#         print("\n" + "=" * 100)
#         print(f"Sample Index: {i}")
#         print("=" * 100)

#         sample = dataset[i]

#         print("\n==== Raw Sample Type ====")
#         print(type(sample))

#         # RandCropByPosNegLabeld(num_samples=2) -> 返回 list
#         if isinstance(sample, list):
#             print("This sample contains multiple crops:", len(sample))
#             crops = sample
#         else:
#             crops = [sample]

#         for j, crop in enumerate(crops):
#             print("\n" + "-" * 80)
#             print(f"Crop Index: {j}")
#             print("-" * 80)

#             print("\n==== Keys ====")
#             print(crop.keys())

#             print("\n==== Tensor Shapes ====")
#             for k in [
#                 "image",
#                 "mask_image",
#                 "mask",
#                 "aal_label",
#                 "aal_state",
#                 "aal_suvr",
#                 "aal_patch_ratio",
#                 "aal_cover_ratio",
#             ]:
#                 if k in crop:
#                     print(f"{k}: {crop[k].shape}")

#             print("\n==== ROI Debug Info ====")
#             aal_label = crop["aal_label"].numpy()
#             aal_state = crop["aal_state"].numpy()
#             aal_suvr = crop["aal_suvr"].numpy()
#             aal_patch_ratio = crop["aal_patch_ratio"].numpy()
#             aal_cover_ratio = crop["aal_cover_ratio"].numpy()

#             present_idx = np.where(aal_label > 0)[0]

#             print("Number of ROIs in this patch:", len(present_idx))
#             print("First 10 ROI indices (present):", present_idx[:10])

#             print("\n==== Example ROI Values (first 5 present ROIs) ====")
#             for idx in present_idx[:5]:
#                 print(
#                     f"ROI {idx+1}: "
#                     f"state={aal_state[idx]}, "
#                     f"suvr={aal_suvr[idx]:.3f}, "
#                     f"patch_ratio={aal_patch_ratio[idx]:.3f}, "
#                     f"cover_ratio={aal_cover_ratio[idx]:.3f}"
#                 )

#             print("\n==== Sanity Checks ====")
#             print("aal_label sum:", aal_label.sum())
#             print("patch_ratio sum (should be ~1):", aal_patch_ratio.sum())
#             print("cover_ratio max:", aal_cover_ratio.max())