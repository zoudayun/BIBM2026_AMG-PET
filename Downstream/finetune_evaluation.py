import os
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from monai.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dataloader.adni_T1_1mm import build_dataset
from models.nnunet_mae_double import AV45BinaryClassifier


def compute_metrics(y_true, y_prob, num_classes=3):
    labels = list(range(num_classes))
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    specificity = []
    total = cm.sum()

    for c in labels:
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = total - TP - FN - FP

        spe = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificity.append(spe)

    specificity = np.array(specificity)

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_specificity = specificity.mean()
    macro_f1 = f1.mean()

    weighted_precision = (precision * support).sum() / max(support.sum(), 1)
    weighted_recall = (recall * support).sum() / max(support.sum(), 1)
    weighted_specificity = (specificity * support).sum() / max(support.sum(), 1)
    weighted_f1 = (f1 * support).sum() / max(support.sum(), 1)

    try:
        y_true_onehot = F.one_hot(
            torch.tensor(y_true).long(),
            num_classes=num_classes,
        ).numpy()

        auc_macro = roc_auc_score(
            y_true_onehot,
            y_prob,
            average="macro",
            multi_class="ovr",
        )

        auc_weighted = roc_auc_score(
            y_true_onehot,
            y_prob,
            average="weighted",
            multi_class="ovr",
        )

        auc_per_class = roc_auc_score(
            y_true_onehot,
            y_prob,
            average=None,
            multi_class="ovr",
        )

    except Exception:
        auc_macro = float("nan")
        auc_weighted = float("nan")
        auc_per_class = [float("nan")] * num_classes

    return {
        "acc": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_specificity": macro_specificity,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
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

    data_path = "/data/zhaomy/downstream/multi_classification/data_list"
    save_dir = "/data/zhaomy/downstream/multi_classification/ablation_results/recon_suvr/adni_T1_1mm_finetune/fold_4"

    model_path = os.path.join(save_dir, "best_model.pth")
    ckpt_path = os.path.join(save_dir, "best_checkpoint.pth")
    save_path = os.path.join(save_dir, "test_result_bestmodel.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_ds, *_ = build_dataset(
        data_path,
        input_size_x=128,
        input_size_y=128,
        input_size_z=128,
        fold=4,
        num_folds=5,
        seed=42,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    ckpt = torch.load(ckpt_path, map_location=device)

    best_epoch = ckpt.get("epoch", -1) + 1
    best_auc = ckpt.get("best_auc", None)
    best_auc_epoch = ckpt.get("best_auc_epoch", None)
    latest_metrics = ckpt.get("latest_metrics", {})

    print(f"Loaded best checkpoint from epoch {best_epoch}")
    print(f"Best AUC: {best_auc}")
    print(f"Best AUC epoch: {best_auc_epoch}")
    print(f"Validation macro AUC: {latest_metrics.get('val_auc_macro', None)}")
    print(f"Validation weighted AUC: {latest_metrics.get('val_auc_weighted', None)}")

    model = AV45BinaryClassifier(
        dropout=0.2,
        num_classes=3,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_prob = []
    all_label = []

    with torch.no_grad():
        for test_data in test_loader:
            test_images = test_data["image"].to(device)
            test_labels = test_data["label"].to(device).long().view(-1)

            test_outputs = model(test_images)
            test_prob = torch.softmax(test_outputs, dim=1)

            all_prob.append(test_prob.cpu())
            all_label.append(test_labels.cpu())

    all_prob = torch.cat(all_prob, dim=0).numpy()
    all_label = torch.cat(all_label, dim=0).numpy()

    metrics = compute_metrics(
        y_true=all_label,
        y_prob=all_prob,
        num_classes=3,
    )

    content = (
        f"==== Three-class finetune evaluation ====\n"
        f"Checkpoint epoch: {best_epoch}\n"
        f"Best AUC: {best_auc}\n"
        f"Best AUC epoch: {best_auc_epoch}\n"
        f"Validation macro AUC: {latest_metrics.get('val_auc_macro', None)}\n"
        f"Validation weighted AUC: {latest_metrics.get('val_auc_weighted', None)}\n"
        f"\n"
        f"Accuracy: {metrics['acc']:.4f}\n"
        f"AUC macro: {metrics['auc_macro']:.4f}\n"
        f"AUC weighted: {metrics['auc_weighted']:.4f}\n"
        f"\n"
        f"Macro Precision: {metrics['macro_precision']:.4f}\n"
        f"Macro Sensitivity/Recall: {metrics['macro_recall']:.4f}\n"
        f"Macro Specificity: {metrics['macro_specificity']:.4f}\n"
        f"Macro F1-score: {metrics['macro_f1']:.4f}\n"
        f"\n"
        f"Weighted Precision: {metrics['weighted_precision']:.4f}\n"
        f"Weighted Sensitivity/Recall: {metrics['weighted_recall']:.4f}\n"
        f"Weighted Specificity: {metrics['weighted_specificity']:.4f}\n"
        f"Weighted F1-score: {metrics['weighted_f1']:.4f}\n"
        f"\n"
        f"Confusion Matrix rows=true, cols=pred:\n"
        f"{metrics['cm'].tolist()}\n"
        f"\n"
        f"CN:  AUC={metrics['auc_CN']:.4f}, "
        f"Sensitivity={metrics['sen_CN']:.4f}, "
        f"Specificity={metrics['spe_CN']:.4f}, "
        f"F1={metrics['f1_CN']:.4f}, "
        f"Support={metrics['support_CN']}\n"
        f"MCI: AUC={metrics['auc_MCI']:.4f}, "
        f"Sensitivity={metrics['sen_MCI']:.4f}, "
        f"Specificity={metrics['spe_MCI']:.4f}, "
        f"F1={metrics['f1_MCI']:.4f}, "
        f"Support={metrics['support_MCI']}\n"
        f"AD:  AUC={metrics['auc_AD']:.4f}, "
        f"Sensitivity={metrics['sen_AD']:.4f}, "
        f"Specificity={metrics['spe_AD']:.4f}, "
        f"F1={metrics['f1_AD']:.4f}, "
        f"Support={metrics['support_AD']}\n"
    )

    print(content)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()