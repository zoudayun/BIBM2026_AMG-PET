import os
import json
import numpy as np
from collections import defaultdict

ROOT = "/data/junyan/PET_MNI_1mm"
JSON_NAME = "aal_PET_text.json"
OUT_GROUP_STATS = os.path.join(ROOT, "aal_suvr_group_thresholds.json")

# ---- 可调参数 ----
MIN_SUBJECTS = 10   # 某脑区至少有多少样本
MIN_IQR = 0.05      # IQR太小则不适合硬分三类
MIN_STD = 0.05      # std太小则不适合硬分三类

STABLE_LABEL = "stable uptake"
LOW_LABEL = "low uptake"
MEDIUM_LABEL = "medium uptake"
HIGH_LABEL = "high uptake"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


# 收集所有subject的ROI-SUVR
def collect_group_suvr(root):
    roi2values = defaultdict(list)
    subject_json_paths = []

    subject_dirs = sorted([
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    for subject_dir in subject_dirs:
        json_path = os.path.join(subject_dir, JSON_NAME)
        if not os.path.exists(json_path):
            continue

        try:
            data = load_json(json_path)
        except Exception as e:
            print(f"[Skip] Failed loading {json_path}: {e}")
            continue

        if not isinstance(data, dict):
            print(f"[Skip] Invalid json structure: {json_path}")
            continue

        valid = False
        for roi_name, roi_info in data.items():
            if not isinstance(roi_info, dict):
                continue
            suvr = roi_info.get("suvr", None)
            if suvr is None:
                continue
            try:
                suvr = float(suvr)
            except Exception:
                continue

            roi2values[roi_name].append(suvr)
            valid = True

        if valid:
            subject_json_paths.append(json_path)

    return roi2values, subject_json_paths


def build_group_thresholds(roi2values):
    thresholds = {}

    for roi_name, values in roi2values.items():
        arr = np.asarray(values, dtype=np.float32)
        n = len(arr)

        if n == 0:
            continue

        mean = float(np.mean(arr))
        std = float(np.std(arr))
        p33 = float(np.percentile(arr, 33.3333))
        p50 = float(np.percentile(arr, 50))
        p67 = float(np.percentile(arr, 66.6667))
        iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))

        use_3class = True
        reason = "sufficient_variation"

        if n < MIN_SUBJECTS:
            use_3class = False
            reason = f"too_few_subjects(n={n})"
        elif iqr < MIN_IQR:
            use_3class = False
            reason = f"iqr_too_small({iqr:.6f})"
        elif std < MIN_STD:
            use_3class = False
            reason = f"std_too_small({std:.6f})"

        thresholds[roi_name] = {
            "n_subjects": n,
            "mean": mean,
            "std": std,
            "min": vmin,
            "p33": p33,
            "p50": p50,
            "p67": p67,
            "max": vmax,
            "iqr": iqr,
            "use_3class": use_3class,
            "reason": reason,
            "labels": {
                "low": LOW_LABEL,
                "medium": MEDIUM_LABEL,
                "high": HIGH_LABEL,
                "stable": STABLE_LABEL
            }
        }

    return thresholds


def assign_state(suvr, stats):
    if not stats["use_3class"]:
        return STABLE_LABEL

    p33 = stats["p33"]
    p67 = stats["p67"]
    if suvr < p33:
        return LOW_LABEL
    elif suvr < p67:
        return MEDIUM_LABEL
    else:
        return HIGH_LABEL


def update_subject_jsons(subject_json_paths, thresholds):
    updated_count = 0

    for json_path in subject_json_paths:
        try:
            data = load_json(json_path)
        except Exception as e:
            print(f"[Skip] Failed loading {json_path}: {e}")
            continue

        changed = False

        for roi_name, roi_info in data.items():
            if not isinstance(roi_info, dict):
                continue
            if "suvr" not in roi_info:
                continue
            if roi_name not in thresholds:
                continue

            try:
                suvr = float(roi_info["suvr"])
            except Exception:
                continue

            # 不动 PET_Signal，只新增这个字段
            roi_info["SUVR_Discrete_State"] = assign_state(suvr, thresholds[roi_name])
            changed = True

        if changed:
            save_json(data, json_path)
            updated_count += 1

    return updated_count


def print_summary(thresholds):
    roi_all = sorted(thresholds.keys())
    roi_3class = sorted([k for k, v in thresholds.items() if v["use_3class"]])
    roi_stable = sorted([k for k, v in thresholds.items() if not v["use_3class"]])

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total ROIs found: {len(roi_all)}")
    print(f"  < p33           -> {LOW_LABEL}")
    print(f"  [p33, p67)      -> {MEDIUM_LABEL}")
    print(f"  >= p67          -> {HIGH_LABEL}")
    print(f"ROIs using 3 classes: {len(roi_3class)}")
    print(f"ROIs using stable uptake only: {len(roi_stable)}")

    print("\n[ROIs using 3 classes]")
    for roi in roi_3class:
        s = thresholds[roi]
        print(
            f"{roi}: n={s['n_subjects']}, mean={s['mean']:.6f}, std={s['std']:.6f}, "
            f"p33={s['p33']:.6f}, p67={s['p67']:.6f}, iqr={s['iqr']:.6f}"
        )

    print("\n[ROIs NOT using 3 classes]")
    for roi in roi_stable:
        s = thresholds[roi]
        print(
            f"{roi}: n={s['n_subjects']}, mean={s['mean']:.6f}, std={s['std']:.6f}, "
            f"p33={s['p33']:.6f}, p67={s['p67']:.6f}, iqr={s['iqr']:.6f}, reason={s['reason']}"
        )

    print("=" * 80 + "\n")


def main():
    print("Step 1: Collecting group SUVR values ...")
    roi2values, subject_json_paths = collect_group_suvr(ROOT)
    print(f"Found {len(subject_json_paths)} subject json files")
    print(f"Found {len(roi2values)} ROI names with SUVR values")

    print("Step 2: Building group thresholds ...")
    thresholds = build_group_thresholds(roi2values)
    save_json(thresholds, OUT_GROUP_STATS)
    print(f"[OK] Group threshold file saved to: {OUT_GROUP_STATS}")

    print_summary(thresholds)

    print("Step 3: Updating subject json files ...")
    updated_count = update_subject_jsons(subject_json_paths, thresholds)
    print(f"[Done] Updated {updated_count} json files")


if __name__ == "__main__":
    main()