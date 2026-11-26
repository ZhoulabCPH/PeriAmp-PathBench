# metrics.py
import os
import numpy as np
from typing import Tuple
import matplotlib
matplotlib.use('Agg')  # Use a noninteractive backend.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    balanced_accuracy_score
)
try:
    from scipy import stats
except Exception:  # Support environments without SciPy.
    stats = None

# Configure Matplotlib for batch plotting.
plt.rcParams['figure.max_open_warning'] = 100  # Raise the open-figure warning threshold.
plt.ioff()  # Disable interactive display.

def cleanup_matplotlib():
    """Close Matplotlib figures to release memory."""
    plt.close('all')  # Close all figures.
    plt.clf()  # Clear the current figure.
    plt.cla()  # Clear the current axes.


def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix='', save_path=None):
    """Plot the binary or multiclass confusion matrix."""
    try:
        # Validate aligned inputs.
        # Confusion matrix with raw counts.
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cmap='Blues')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(f'Confusion Matrix {title_suffix}')
        if save_path:
            base, _ = os.path.splitext(save_path)
            plt.savefig(base + '.pdf', bbox_inches='tight')
        plt.close()

        # Normalize rows by true-class totals and annotate the counts.
        row_sums = np.sum(cm, axis=1, keepdims=True).astype(float)
        safe_row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        proportion_matrix = cm.astype(float) / safe_row_sums
        # Display the within-row percentage and raw count in each cell.
        annot = np.empty_like(proportion_matrix, dtype=object)
        for i in range(proportion_matrix.shape[0]):
            for j in range(proportion_matrix.shape[1]):
                percent = proportion_matrix[i, j] * 100.0
                annot[i, j] = f"{percent:.1f}%\n(n={int(cm[i, j])})"

        plt.figure(figsize=(10, 8))
        sns.heatmap(proportion_matrix, annot=annot, fmt='',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cmap='Blues', vmin=0.0, vmax=1.0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(f'Confusion Matrix (Row-normalized %) {title_suffix}')
        if save_path:
            base, _ = os.path.splitext(save_path)
            plt.savefig(base + '_row_norm.pdf', bbox_inches='tight')
        plt.close()

        # Plot a binary confusion matrix for each class.
        for idx, cls in enumerate(class_names):
            binary_true = (np.array(y_true) == idx).astype(int)
            binary_pred = (np.array(y_pred) == idx).astype(int)
            cm = confusion_matrix(binary_true, binary_pred)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d',
                        xticklabels=['Negative', cls],
                        yticklabels=['Negative', cls],
                        cmap='Blues')
            plt.title(f'Binary CM for {cls} {title_suffix}')

            if save_path:
                base_name = os.path.splitext(save_path)[0]
                plt.savefig(f"{base_name}_binary_{cls}.pdf", bbox_inches='tight')
            plt.close()  # Close each class figure after saving.
    finally:
        cleanup_matplotlib()  # Release Matplotlib resources.


def compute_overall_metrics(y_true, y_pred, y_score, class_names):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    # Compute sensitivity and specificity.
    sensitivities, specificities = [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'sensitivity_macro': float(np.mean(sensitivities)),
        'specificity_macro': float(np.mean(specificities)),
        'class_names': class_names,
        'true': y_true,
        'pred': y_pred
    }

    # Compute the existing AUC metric.
    try:
        metrics['auc_macro'] = float(roc_auc_score(y_true, y_score, multi_class='ovr', average='macro'))
        metrics['auc_micro'] = float(roc_auc_score(y_true, y_score, multi_class='ovr', average='micro'))
    except:
        metrics['auc_macro'] = 0.0
        metrics['auc_micro'] = 0.0

    return metrics


def compute_classwise_metrics(y_true, y_pred, y_score, class_names):
    """Compute per-class metrics for multiclass predictions."""
    metrics_dict = {}
    cm = confusion_matrix(y_true, y_pred)

    for i, cls in enumerate(class_names):
        # Derive true negatives from the confusion matrix totals.
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)  # Exclude the current class row and column.

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Handle undefined per-class AUC.
        try:
            cls_auc = roc_auc_score((np.array(y_true) == i), y_score[:, i])
        except:
            cls_auc = 0.0

        # Compute per-class F1.
        cls_f1 = f1_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)

        metrics_dict[cls] = {
            'auc': float(cls_auc),
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1': float(cls_f1),
            'support': int(np.sum(np.array(y_true) == i))  # Convert the count to a Python integer.
        }
    return metrics_dict


def _compute_delong_auc_ci(binary_true: np.ndarray, pred_scores: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Return AUC and lower and upper confidence limits using DeLong's method."""
    # Normalize input types.
    binary_true = np.asarray(binary_true).astype(int)
    pred_scores = np.asarray(pred_scores).astype(float)

    n1 = int(np.sum(binary_true == 1))
    n2 = int(np.sum(binary_true == 0))
    if n1 == 0 or n2 == 0:
        auc_value = float('nan')
        return auc_value, 0.0, 1.0

    # Compute ranks from sorted scores.
    order = np.argsort(pred_scores)
    ranked_pred = np.zeros_like(pred_scores, dtype=float)
    for rank, idx in enumerate(order):
        ranked_pred[idx] = rank + 1

    pos_ranks = ranked_pred[binary_true == 1]
    neg_ranks = ranked_pred[binary_true == 0]

    U1 = np.sum(pos_ranks) - n1 * (n1 + 1) / 2.0
    auc_value = U1 / (n1 * n2)

    # Estimate the variance.
    V10 = np.zeros(n2, dtype=float)
    V01 = np.zeros(n1, dtype=float)
    for i in range(n1):
        V01[i] = np.sum(neg_ranks < pos_ranks[i]) / n2
    for j in range(n2):
        V10[j] = np.sum(pos_ranks > neg_ranks[j]) / n1
    # Avoid an invalid sample variance when there is only one observation.
    S10 = np.var(V10, ddof=1) if n2 > 1 else 0.0
    S01 = np.var(V01, ddof=1) if n1 > 1 else 0.0
    var_auc = S10 / n2 + S01 / n1
    std_auc = float(np.sqrt(max(var_auc, 1e-12)))

    if stats is not None:
        z = float(stats.norm.ppf(1 - alpha / 2.0))
    else:
        # Use the normal approximation for the 95% interval.
        z = 1.959963984540054
    lower = max(0.0, auc_value - z * std_auc)
    upper = min(1.0, auc_value + z * std_auc)
    return float(auc_value), float(lower), float(upper)


def _bootstrap_tpr_band(binary_true: np.ndarray, pred_scores: np.ndarray, base_fpr: np.ndarray, n_bootstraps: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the 2.5th to 97.5th percentile TPR band by bootstrap."""
    rng = np.random.RandomState(seed)
    binary_true = np.asarray(binary_true).astype(int)
    pred_scores = np.asarray(pred_scores).astype(float)
    pos_idx = np.where(binary_true == 1)[0]
    neg_idx = np.where(binary_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.zeros_like(base_fpr), np.ones_like(base_fpr)

    tprs = []
    for _ in range(n_bootstraps):
        res_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        res_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        res_idx = np.concatenate([res_pos, res_neg])
        fpr_b, tpr_b, _ = roc_curve(binary_true[res_idx], pred_scores[res_idx])
        tpr_interp = np.interp(base_fpr, fpr_b, tpr_b)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
    tprs = np.asarray(tprs)
    lower = np.percentile(tprs, 2.5, axis=0)
    upper = np.percentile(tprs, 97.5, axis=0)
    return lower, upper


def _save_pdf(fig, path_no_ext: str):
    """Save the figure as a PDF."""
    fig.savefig(path_no_ext + '.pdf', bbox_inches='tight', dpi=300)


def plot_roc_curves(y_true, y_score, class_names, title_suffix='', save_dir=None):
    """Plot one-vs-rest ROC curves with DeLong AUC intervals and bootstrap TPR bands. Save an overall ROC and export figures as PDF with PNG backups."""
    try:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        y_true_arr = np.asarray(y_true)
        y_score_arr = np.asarray(y_score)
        base_fpr = np.linspace(0, 1, 101)

        # Plot one-vs-rest ROC curves.
        fig = plt.figure(figsize=(10, 8))
        for i, cls in enumerate(class_names):
            binary_true = (y_true_arr == i).astype(int)
            pred_scores = y_score_arr[:, i]

            fpr, tpr, _ = roc_curve(binary_true, pred_scores)
            roc_auc = auc(fpr, tpr)

            # DeLong AUC confidence interval
            auc_val, ci_lo, ci_hi = _compute_delong_auc_ci(binary_true, pred_scores)

            # Bootstrap TPR confidence band
            band_lo, band_hi = _bootstrap_tpr_band(binary_true, pred_scores, base_fpr)

            # Draw the confidence band.
            plt.fill_between(base_fpr, band_lo, band_hi, alpha=0.15)

            # Draw the ROC as a step curve.
            plt.step(fpr, tpr, where='post', lw=2, label=f'{cls} (AUC = {roc_auc:.3f} [{ci_lo:.3f}-{ci_hi:.3f}])')

        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves {title_suffix}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_dir:
            _save_pdf(fig, os.path.join(save_dir, f'roc_curves_{title_suffix}'))
        plt.close(fig)

        # 2) overall（Macro-average）ROC
        n_classes = len(class_names)
        # Interpolate per-class TPRs and average them to obtain the macro ROC.
        class_tprs_interp = []
        class_aucs = []
        for i in range(n_classes):
            binary_true = (y_true_arr == i).astype(int)
            pred_scores = y_score_arr[:, i]
            fpr_i, tpr_i, _ = roc_curve(binary_true, pred_scores)
            class_tprs_interp.append(np.interp(base_fpr, fpr_i, tpr_i))
            class_tprs_interp[-1][0] = 0.0
            try:
                class_aucs.append(roc_auc_score(binary_true, pred_scores))
            except Exception:
                class_aucs.append(0.0)
        macro_tpr = np.mean(np.vstack(class_tprs_interp), axis=0) if class_tprs_interp else np.zeros_like(base_fpr)
        # Compute macro AUC using the scikit-learn convention.
        try:
            macro_auc_val = roc_auc_score(y_true_arr, y_score_arr, multi_class='ovr', average='macro')
        except Exception:
            macro_auc_val = float(np.mean(class_aucs)) if class_aucs else 0.0

        # Bootstrap the macro TPR band and AUC interval.
        rng = np.random.RandomState(42)
        n_boot = 2000
        macro_tprs_boot = []
        macro_auc_boot = []
        # Sample within classes to preserve class sizes.
        per_class_indices = [np.where(y_true_arr == i)[0] for i in range(n_classes)]
        valid_classes = [idxs for idxs in per_class_indices if len(idxs) > 0]
        for _ in range(n_boot):
            if not valid_classes:
                break
            resampled_indices = []
            for idxs in per_class_indices:
                if len(idxs) == 0:
                    continue
                resampled_indices.append(rng.choice(idxs, size=len(idxs), replace=True))
            resampled_indices = np.concatenate(resampled_indices) if resampled_indices else np.array([], dtype=int)
            if resampled_indices.size == 0:
                continue
            y_true_b = y_true_arr[resampled_indices]
            y_score_b = y_score_arr[resampled_indices]

            # Interpolate each class and average the curves.
            tprs_b = []
            for i in range(n_classes):
                cls_mask = (y_true_b == i)
                if np.sum(cls_mask) == 0:
                    continue
                fpr_b, tpr_b, _ = roc_curve((y_true_b == i).astype(int), y_score_b[:, i])
                interp_b = np.interp(base_fpr, fpr_b, tpr_b)
                interp_b[0] = 0.0
                tprs_b.append(interp_b)
            if tprs_b:
                macro_tprs_boot.append(np.mean(np.vstack(tprs_b), axis=0))
            # Bootstrap macro AUC.
            try:
                macro_auc_boot.append(roc_auc_score(y_true_b, y_score_b, multi_class='ovr', average='macro'))
            except Exception:
                # Fall back to the mean of per-class AUCs.
                aucs_b = []
                for i in range(n_classes):
                    try:
                        aucs_b.append(roc_auc_score((y_true_b == i).astype(int), y_score_b[:, i]))
                    except Exception:
                        pass
                if aucs_b:
                    macro_auc_boot.append(float(np.mean(aucs_b)))

        if macro_tprs_boot:
            band_lo = np.percentile(np.vstack(macro_tprs_boot), 2.5, axis=0)
            band_hi = np.percentile(np.vstack(macro_tprs_boot), 97.5, axis=0)
        else:
            band_lo = np.zeros_like(base_fpr)
            band_hi = np.ones_like(base_fpr)
        if macro_auc_boot:
            ci_lo = float(np.percentile(macro_auc_boot, 2.5))
            ci_hi = float(np.percentile(macro_auc_boot, 97.5))
        else:
            ci_lo, ci_hi = 0.0, 1.0

        fig2 = plt.figure(figsize=(8, 6))
        plt.fill_between(base_fpr, band_lo, band_hi, color='#2E86AB', alpha=0.2, label='95% CI')
        plt.step(base_fpr, macro_tpr, where='post', color='#2E86AB', lw=2,
                 label=f'Macro-average ROC (AUC = {macro_auc_val:.3f} [{ci_lo:.3f}-{ci_hi:.3f}])')
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Overall ROC (Macro-average) {title_suffix}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        if save_dir:
            _save_pdf(fig2, os.path.join(save_dir, f'roc_overall_{title_suffix}'))
        plt.close(fig2)
    finally:
        cleanup_matplotlib()  # Release resources after plotting.