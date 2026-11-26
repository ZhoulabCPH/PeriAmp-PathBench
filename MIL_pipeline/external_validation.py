# external_validation.py
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from dataSet.data_loader import H5Dataset, collate_fn
from metrics import compute_overall_metrics, compute_classwise_metrics, plot_confusion_matrix, plot_roc_curves
from collections import Counter
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score
)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32,
        np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

class ExternalValidator:
    def __init__(self, model_class, model_params, model_path,
                 h5_dir, excel_file, class_names, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self._initialize_model_only = False  # Allow model initialization without loading weights.

        # Import and construct the configured model.
        self.model_class = model_class
        self.model_params = model_params
        self.model = self._load_model(model_class, model_params, model_path)

        # Load the validation dataset.
        self.dataset = H5Dataset(
            h5_dir=h5_dir,
            excel_file=excel_file,
            class_names=class_names
        )
        self.loader = DataLoader(self.dataset, batch_size=1, collate_fn=collate_fn)

        # State used to aggregate fold predictions.
        self.ensemble_predictions = []  # Predicted classes from each model
        self.ensemble_scores = []  # Predicted scores from each model
        self.file_ids = []  # Case identifiers
        self.true_labels = []  # Ground-truth labels

    def _load_model(self, model_class, model_params, model_path):
        """Load the configured model and weights."""
        try:
            model = model_class(**model_params)

            # Load weights when a checkpoint was supplied.
            if model_path and not self._initialize_model_only:
                state_dict = torch.load(model_path, map_location=self.device)

                # Normalize checkpoint parameter names.
                if 'module.' in list(state_dict.keys())[0]:
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                model.load_state_dict(state_dict)

            model = model.to(self.device).eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Could not load the model: {str(e)}")

    def evaluate(self, output_dir='external_results', config_params=None, save_results=True):
        """Evaluate an external set and optionally save results. Disable save_results for per-epoch evaluation."""
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        if save_results:
            os.makedirs(output_dir, exist_ok=True)

        all_true, all_pred, all_scores, all_logits, all_file_ids = [], [], [], [], []
        metrics_params = (config_params or {}).get('metrics', {'average': 'macro'})

        # Configure the external validation loader.
        batch_size = config_params.get('batch_size', 4) if config_params else 4
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,  # Use a single process for HDF5 reads on Windows to surface native errors.
            pin_memory=False,
            persistent_workers=False,
        )

        with torch.no_grad():
            progress = tqdm(
                loader,
                desc="External Validation",
                unit="batch",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar:20}{r_bar}"
            )

            for batch in progress:
                # Unpack the collate_fn output.
                features, _, labels, _, masks, file_ids = batch

                # Move inputs to the compute device.
                features = features.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                labels = labels.to(self.device)

                # Evaluate the validation loss.
                outputs = self.model(features, masks)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Record raw logits.
                logits = outputs.cpu().numpy()
                # Compute softmax probabilities.
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                # Collect the batch outputs.
                all_true.extend(labels.detach().cpu().numpy().tolist())
                all_pred.extend(preds.tolist())
                all_scores.append(probs)
                all_logits.append(logits)
                all_file_ids.extend(file_ids)

        # Compute the full metric set.
        all_scores_array = np.vstack(all_scores) if len(all_scores) > 0 else np.array([])
        all_logits_array = np.vstack(all_logits) if len(all_logits) > 0 else np.array([])
        metrics = {
            "overall": compute_overall_metrics(
                all_true,
                all_pred,
                all_scores_array,
                self.class_names
            ),
            "class_wise": compute_classwise_metrics(
                all_true,
                all_pred,
                all_scores_array,
                self.class_names
            ),
            'accuracy': float(accuracy_score(all_true, all_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(all_true, all_pred))
        }
        avg_loss = total_loss / len(loader)
        metrics['overall']['loss'] = avg_loss

        # Write results only when requested.
        if save_results:
            self._save_results(output_dir, metrics, config_params, all_scores_array)

        return {
            "overall": metrics.get("overall", {}),
            "class_wise": metrics.get("class_wise", {}),
            'y_true': all_true,
            'y_pred': all_pred,
            'scores': all_scores_array,
            'logits': all_logits_array,
            'file_ids': all_file_ids,
            "error": None
        }

    def evaluate_for_ensemble(self, fold_idx=None):
        """Collect predictions for ensemble evaluation using the optional fold index."""
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        all_true, all_pred, all_scores, all_logits, all_file_ids = [], [], [], [], []

        with torch.no_grad():
            progress = tqdm(
                self.loader,
                desc=f"External Validation Fold {fold_idx}" if fold_idx is not None else "External Validation",
                unit="batch",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar:20}{r_bar}"
            )

            for batch in progress:
                # Unpack the collate_fn output.
                features, _, labels, _, masks, file_ids = batch

                # Move inputs to the compute device.
                features = features.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                labels = labels.to(self.device)

                # Evaluate the validation loss.
                outputs = self.model(features, masks)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Record raw logits.
                logits = outputs.cpu().numpy()
                # Compute softmax probabilities.
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                # Collect the batch outputs.
                all_true.extend(labels.detach().cpu().numpy().tolist())
                all_pred.extend(preds.tolist())
                all_scores.append(probs)
                all_logits.append(logits)
                all_file_ids.extend(file_ids)

        # Compute the full metric set.
        all_scores_array = np.vstack(all_scores) if len(all_scores) > 0 else np.array([])
        all_logits_array = np.vstack(all_logits) if len(all_logits) > 0 else np.array([])
        
        # Compute performance metrics.
        metrics = {
            "overall": compute_overall_metrics(
                all_true,
                all_pred,
                all_scores_array,
                self.class_names
            ),
            "class_wise": compute_classwise_metrics(
                all_true,
                all_pred,
                all_scores_array,
                self.class_names
            ),
            'accuracy': float(accuracy_score(all_true, all_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(all_true, all_pred))
        }
        avg_loss = total_loss / len(self.loader)
        metrics['overall']['loss'] = avg_loss

        # Record values for the fold ensemble.
        self.collect_ensemble_data(all_true, all_pred, all_scores_array, all_file_ids, fold_idx)

        return {
            "overall": metrics.get("overall", {}),
            "class_wise": metrics.get("class_wise", {}),
            'y_true': all_true,
            'y_pred': all_pred,
            'scores': all_scores_array,
            'logits': all_logits_array,
            'file_ids': all_file_ids,
            'predictions': all_pred,  # Used by the fold ensemble.
            'probabilities': all_scores_array,  # Used by the fold ensemble.
            'labels': all_true,  # Used by the fold ensemble.
            "error": None
        }

    def collect_ensemble_data(self, true_labels, predictions, scores, file_ids, fold_idx=None):
        """Record true labels, predictions, scores, and file IDs for the given fold."""
        # Save labels and IDs when collecting the first fold.
        if len(self.true_labels) == 0:
            self.true_labels = true_labels.copy()
            self.file_ids = file_ids.copy()
        
        # Append predictions and scores for this fold.
        self.ensemble_predictions.append(predictions)
        self.ensemble_scores.append(scores)
        
        print(f"Collected fold {fold_idx} predictions for {len(predictions)} samples.")

    def compute_ensemble_predictions(self):
        """Compute and return ensemble predictions across folds."""
        if len(self.ensemble_scores) == 0:
            print('No predictions are available for ensemble evaluation.')
            return None
        
        # Stack probabilities as (n_folds, n_samples, n_classes).
        scores_array = np.array(self.ensemble_scores)
        
        # Average over folds.
        ensemble_scores = np.mean(scores_array, axis=0)
        
        # Select the predicted classes from the mean probabilities.
        ensemble_predictions = np.argmax(ensemble_scores, axis=1)
        
        # Compute performance metrics.
        accuracy = accuracy_score(self.true_labels, ensemble_predictions)
        balanced_accuracy = balanced_accuracy_score(self.true_labels, ensemble_predictions)
        
        # Compute the full metric set.
        metrics = {
            "overall": compute_overall_metrics(
                self.true_labels,
                ensemble_predictions,
                ensemble_scores,
                self.class_names
            ),
            "class_wise": compute_classwise_metrics(
                self.true_labels,
                ensemble_predictions,
                ensemble_scores,
                self.class_names
            ),
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy)
        }
        
        # Measure agreement between fold predictions.
        individual_predictions = np.array(self.ensemble_predictions)  # (n_folds, n_samples)
        consistency_scores = []
        
        for sample_idx in range(individual_predictions.shape[1]):
            sample_predictions = individual_predictions[:, sample_idx]
            # Measure the fraction voting for the most frequent class.
            unique, counts = np.unique(sample_predictions, return_counts=True)
            max_agreement = np.max(counts) / len(sample_predictions)
            consistency_scores.append(max_agreement)
        
        # Summarize prediction confidence.
        max_confidences = np.max(ensemble_scores, axis=1)
        
        ensemble_result = {
            'method': 'ensemble_average_probability',
            'ensemble_predictions': ensemble_predictions.tolist(),
            'ensemble_scores': ensemble_scores,
            'true_labels': self.true_labels,
            'file_ids': self.file_ids,
            'individual_predictions': individual_predictions.tolist(),
            'consistency_scores': consistency_scores,
            'confidence_scores': max_confidences.tolist(),
            'metrics': metrics,
            'n_folds': len(self.ensemble_scores),
            'n_samples': len(self.true_labels),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"Computed ensemble predictions for {len(self.ensemble_scores)} models and {len(self.true_labels)} samples.")
        print(f"Ensemble accuracy: {accuracy:.4f}, balanced accuracy: {balanced_accuracy:.4f}")
        
        return ensemble_result

    def reset_ensemble_data(self):
        """Clear the stored predictions for ensemble evaluation."""
        self.ensemble_predictions = []
        self.ensemble_scores = []
        self.file_ids = []
        self.true_labels = []
        print('Cleared ensemble prediction data.')

    def _save_results(self, output_dir, metrics, config_params, y_score):
        """Write the external validation results to disk."""

        # Save confusion matrices.
        matrix_dir = os.path.join(output_dir, 'confusion_matrix')
        os.makedirs(matrix_dir, exist_ok=True)
        plot_confusion_matrix(
            y_true=metrics['overall']['true'],
            y_pred=metrics['overall']['pred'],
            class_names=self.class_names,
            title_suffix='External Validation',
            save_path=os.path.join(matrix_dir, 'confusion_matrix.pdf')
        )

        # Write detailed evaluation metrics.
        report_path = os.path.join(output_dir, 'validation_report.txt')
        with open(report_path, 'w') as f:
            # Report header
            f.write("=== External Validation Report ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overall metrics
            f.write("[Overall Metrics]\n")
            overall_keys = [
                ('accuracy', 'Accuracy'),
                ('balanced_accuracy', 'Balanced Accuracy'),
                ('auc_macro', 'AUC-Macro'),
                ('auc_micro', 'AUC-Micro'),
                ('f1_macro', 'F1-Macro'),  # Macro-averaged F1
                ('loss', 'Loss')
            ]
            for metric, display_name in overall_keys:
                value = metrics['overall'].get(metric, 0.0)
                f.write(f"{display_name}: {value:.4f}\n")

            # Per-class metrics
            f.write("\n[Class-wise Metrics]\n")
            class_wise_metrics = metrics.get('class_wise', {})
            for cls in self.class_names:
                cls_metrics = class_wise_metrics.get(cls, {})
                support = cls_metrics.get('support', 0)

                f.write(f"\n* Class: {cls} (n={support}) *\n")
                for metric in ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1']:
                    value = cls_metrics.get(metric, 0.0)
                    f.write(f"{metric.upper():<12}: {value:.4f}\n")

            # Sample statistics
            f.write("\n[Dataset Statistics]\n")
            f.write(f"Total samples: {len(self.dataset)}\n")

            class_dist = Counter(self.dataset.label_codes)  # Use encoded labels.
            for cls_code, count in class_dist.items():
                class_name = self.class_names[cls_code]  # Resolve the class name from its integer index.
                f.write(f"{class_name}: {count} samples\n")

            # Save the ROC curves.
            roc_dir = os.path.join(output_dir, 'roc_curves')
            os.makedirs(roc_dir, exist_ok=True)
            plot_roc_curves(
                y_true=metrics['overall']['true'],
                y_score=y_score,
                class_names=self.class_names,
                title_suffix='External Validation',
                save_dir=roc_dir
            )

        # Save machine-readable outputs.
        # Convert NumPy values to native Python values.
        metrics = convert_numpy_types(metrics)
        config_params = convert_numpy_types(config_params)
        # Save evaluation metrics.
        with open(os.path.join(output_dir, 'external_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save the evaluation configuration.
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config_params, f, indent=2)

    def update_model_weights(self, new_state_dict):
        """Update weights on the existing model instance."""
        self.model.load_state_dict(new_state_dict)
        self.model.eval()