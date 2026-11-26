"""Save, record, and aggregate experiment results."""
import os
import json
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.cell import Cell
from openpyxl.styles import Alignment
import matplotlib
matplotlib.use('Agg')  # Use a noninteractive backend.
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Matplotlib for batch plots.
plt.rcParams['figure.max_open_warning'] = 100
plt.ioff()  # Disable interactive display.

def cleanup_matplotlib():
    """Close Matplotlib figures to release memory."""
    plt.close('all')  # Close all figures.
    plt.clf()  # Clear the current figure.
    plt.cla()  # Clear the current axes.
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                           classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve, f1_score)

from metrics import compute_overall_metrics, compute_classwise_metrics, plot_confusion_matrix, plot_roc_curves


class ResultSaver:
    """Save and aggregate experiment results."""
    METRIC_MAPPING = {
        'acc': 'accuracy',
        'balanced_acc': 'balanced_accuracy',
        'auc': 'auc_macro',
        'loss': 'loss'
    }

    def __init__(self, config, device, save_root, timestamp=None):
        """Initialize output paths and the computation device from the experiment configuration. Generate a timestamp when one is omitted."""
        self.config = config
        self.device = device
        self.save_root = save_root
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Store internal cross-validation predictions.
        self.all_best_internal_true = []  # Ground-truth labels from each fold's validation split
        self.all_best_internal_pred = []  # Predictions from the best model in each fold
        self.all_best_internal_scores = []  # Prediction scores from the best model in each fold
        self.all_records = []  # Records for all processed samples

        # Map each fold to its best epoch.
        self.best_epochs = {}
        # Hold fold-averaged metrics for each external dataset.
        self.per_set_ensemble_results = {}

    def generate_visualizations(self, y_true, y_pred, scores, fold_idx, epoch=None, data_type='Internal'):
        """Generate the configured evaluation plots."""
        # Generate confusion matrices.
        matrix_dir = os.path.join(self.save_root, f"fold_{fold_idx}", "confusion_matrix")
        if epoch is not None:
            matrix_dir = os.path.join(matrix_dir, f"epoch_{epoch}")
        os.makedirs(matrix_dir, exist_ok=True)

        plot_confusion_matrix(
            y_true, y_pred, self.config['class_names'],
            title_suffix=f'{data_type} Fold {fold_idx} Epoch {epoch}' if epoch else f'{data_type} Fold {fold_idx}',
            save_path=os.path.join(matrix_dir, 'confusion.pdf')
        )

        # Generate ROC curves.
        roc_dir = os.path.join(self.save_root, f"fold_{fold_idx}", "roc_curves")
        roc_save_dir = os.path.join(roc_dir, f'epoch_{epoch}') if epoch else roc_dir
        plot_roc_curves(
            y_true, scores,
            self.config['class_names'],
            title_suffix=f'Fold {fold_idx} Epoch {epoch}' if epoch else f'Fold {fold_idx}',
            save_dir=roc_save_dir
        )

    def write_metrics(self, save_dir, metrics, filename, is_external):
        """Write an internal or external evaluation metric dictionary to a text file."""
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'w') as f:
            # Omit private path fields from the report.
            f.write("=== Experiment Configuration ===\n")
            filtered_config = {k: v for k, v in self.config.items()
                               if k not in ['external_h5_dir', 'external_excel']}
            f.write(json.dumps(filtered_config, indent=2, default=str) + "\n")

            # Identify the validation split.
            val_type = "EXTERNAL VALIDATION" if is_external else "INTERNAL VALIDATION"
            f.write(f"\n=== {val_type} RESULTS ===\n")

            f.write("\n[Overall Metrics]\n")
            overall = metrics.get('overall', {})
            f.write(f"Accuracy: {overall.get('accuracy', 0.0):.4f}\n")
            f.write(f"Balanced Accuracy: {overall.get('balanced_accuracy', 0.0):.4f}\n")
            f.write(f"AUC-Macro: {overall.get('auc_macro', 0.0):.4f}\n")
            f.write(f"AUC-Micro: {overall.get('auc_micro', 0.0):.4f}\n")
            f.write(f"F1-Macro: {overall.get('f1_macro', 0.0):.4f}\n")
            f.write(f"Loss: {overall.get('loss', 0.0):.4f}\n\n")

            f.write("\n[Class-wise Metrics]\n")
            class_wise_metrics = metrics.get('class_wise', {})  # Allow missing metrics.
            for cls in self.config['class_names']:
                cls_metrics = class_wise_metrics.get(cls, {})  # Allow missing metrics.
                support = cls_metrics.get('support', 0)

                f.write(f"\n* Class: {cls} (n={support}) *\n")
                for metric in ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1']:
                    value = cls_metrics.get(metric, 0.0)
                    f.write(f"{metric.upper():<12}: {value:.4f}\n")

    def save_fold_results(self, metrics, fold_idx):
        """Save evaluation metrics for a single fold."""
        internal_dir = os.path.join(self.save_root, f"fold_{fold_idx}")
        os.makedirs(internal_dir, exist_ok=True)

        # Save internal validation outputs.
        self.write_metrics(
            save_dir=internal_dir,
            metrics=metrics,
            filename="internal_results.txt",
            is_external=False
        )

    def save_fold_best_results(self, metrics, fold_idx, best_epoch):
        """Save metrics and plots for the best epoch of the current fold."""
        # Update the best epoch for this fold.
        self.best_epochs[fold_idx] = best_epoch

        # Collect predictions.
        self.all_best_internal_true.extend(metrics['y_true'])
        self.all_best_internal_pred.extend(metrics['y_pred'])
        self.all_best_internal_scores.append(metrics['scores'])

        # Centralize best-epoch plots under Validation_Summary.

    def save_combined_confusion_matrix(self):
        """Save the confusion matrix for all internal validation samples."""
        if len(self.all_best_internal_true) == 0:
            return
        combin_path = os.path.join(self.save_root, "Internal_Validation_Combin")
        os.makedirs(combin_path, exist_ok=True)
        plot_confusion_matrix(
            y_true=self.all_best_internal_true,
            y_pred=self.all_best_internal_pred,
            class_names=self.config['class_names'],
            title_suffix='Combined Best Models internal-Validation',
            save_path=os.path.join(combin_path, 'internal combined_best_confusion.png')
        )

    def style_excel_report(self, filepath):
        """Format the Excel report at the supplied path."""
        # Workbook fill styles
        internal_fill = PatternFill(
            start_color='FFC7CE',  # Light red
            end_color='FFC7CE',
            fill_type='solid'
        )
        external_fill = PatternFill(
            start_color='C6EFCE',  # Light green
            end_color='C6EFCE',
            fill_type='solid'
        )

        wb = load_workbook(filepath)
        ws = wb.active

        # Apply styles by row.
        for row_idx, row in enumerate(ws.iter_rows(), 1):
            cell_value = ws.cell(row=row_idx, column=1).value

            if "AGGREGATED_INTERNAL" in str(cell_value):
                # Update the row label.
                ws.cell(row=row_idx, column=1).value = "AGGREGATED_INTERNAL"

                # Apply the fill to every cell in the row.
                for cell in row:
                    cell.fill = internal_fill

            elif "AGGREGATED_EXTERNAL" in str(cell_value):
                for cell in row:
                    cell.fill = external_fill

        wb.save(filepath)

    def record_validation_data(self, metrics, fold_idx, dataset_type='internal'):
        """Record validation logits and probabilities for an internal or external fold."""
        file_ids = metrics.get('file_ids', [])
        true_labels = metrics.get('y_true', [])
        pred_labels = metrics.get('y_pred', [])
        scores = metrics.get('scores', np.array([]))
        logits = metrics.get('logits', np.array([]))

        # Validate aligned prediction arrays.
        assert len(file_ids) == len(true_labels) == len(pred_labels), 'Labels, predictions, and IDs have different lengths.'
        if len(scores) > 0:
            assert len(file_ids) == len(scores), 'Scores and labels have different lengths.'
        if len(logits) > 0:
            assert len(file_ids) == len(logits), 'Logits and labels have different lengths.'

        for idx in range(len(file_ids)):
            record = {
                'file_id': file_ids[idx],
                'fold': fold_idx + 1,
                'dataset_type': dataset_type.upper(),
                'true_label': self.config['class_names'][true_labels[idx]],
                'pred_label': self.config['class_names'][pred_labels[idx]],
                'is_correct': true_labels[idx] == pred_labels[idx],
            }
            
            # Add one probability column per class.
            for cls_idx, cls_name in enumerate(self.config['class_names']):
                if len(scores) > 0:
                    record[f'{cls_name}_prob'] = scores[idx][cls_idx].item()  # Convert NumPy values to native Python values.
                if len(logits) > 0:
                    record[f'{cls_name}_logit'] = logits[idx][cls_idx].item()  # Convert NumPy values to native Python values.

            self.all_records.append(record)

    def save_validation_details(self):
        """Export probability scores to a formatted Excel workbook."""
        df = pd.DataFrame(self.all_records)

        # Keep IDs as text to prevent Excel scientific notation.
        if 'file_id' in df.columns:
            df['file_id'] = df['file_id'].astype(str)

        # Sort IDs with or without a numeric component.
        df.sort_values(
            by=['file_id', 'fold'],
            key=lambda x: x.map(lambda v:
                                int(''.join(filter(str.isdigit, str(v))) or '0')
                                if isinstance(v, (str, int)) else 0
                                ),
            inplace=True
        )

        # Define the output column order.
        columns = ['file_id', 'fold', 'dataset_type']
        
        # Add logit and probability columns.
        logits_columns = [f'{cls}_logit' for cls in self.config['class_names']]
        probs_columns = [f'{cls}_prob' for cls in self.config['class_names']]
        
        # List logits before probabilities for each class.
        columns += logits_columns + probs_columns
        columns += ['pred_label', 'true_label', 'is_correct']
        
        # Keep the available columns.
        existing_columns = [col for col in columns if col in df.columns]
        df = df[existing_columns]

        # Save a formatted workbook with openpyxl.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(self.save_root, f"validation_details_{timestamp}.xlsx")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Keep the underlying floating-point precision.
            df.to_excel(writer, index=False, sheet_name='Validation Details', float_format="%.10f")

            # Get the worksheet object.
            workbook = writer.book
            worksheet = writer.sheets['Validation Details']

            # Size columns to fit their contents.
            for col_idx, column in enumerate(df.columns, 1):
                column_letter = get_column_letter(col_idx)

                # Account for decimal places when estimating cell widths.
                if column.endswith('_prob') or column.endswith('_logit'):
                    # Handle logit and probability columns separately.
                    max_len = len("0.0000000000")  # Reserve width for ten decimal places.
                    content_samples = df[column].astype(str).str.split('.', expand=True)[1]
                    max_decimal = content_samples.str.len().max()
                    if pd.notnull(max_decimal):
                        max_len = 3 + max_decimal  # Count the decimal prefix and fractional digits.
                else:
                    # Use standard widths for the other columns.
                    max_len = max(
                        len(str(column)) * 1.2,  # Include the header length in the estimate.
                        df[column].astype(str).map(len).max() * 1.1  # Include data values in the estimate.
                    )

                # Limit widths between 10 and 120, with extra room for case IDs.
                if column == 'file_id':
                    worksheet.column_dimensions[column_letter].width = min(max(12, max_len), 120)
                else:
                    worksheet.column_dimensions[column_letter].width = min(max(10, max_len), 120)

                # Set cell alignment.
                for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row, min_col=col_idx, max_col=col_idx):
                    for cell in row:
                        # Center cells by default.
                        cell.alignment = Alignment(horizontal='center')

                        # Format logit and probability values as numbers.
                        if column.endswith('_prob') or column.endswith('_logit'):
                            # Apply the custom decimal number format.
                            cell.number_format = '0.####################'  # Show up to 20 decimal places.
                        # Format IDs as text.
                        if column == 'file_id':
                            cell.number_format = '@'
                            cell.alignment = Alignment(horizontal='left')

        print(f"Saved validation details to {excel_path}")

    def record_training_data(self, train_data, fold_idx):
        """Record training logits, probabilities, labels, and file IDs for the given fold."""
        if not train_data or 'logits' not in train_data:
            return
            
        logits = train_data.get('logits', np.array([]))
        probs = train_data.get('probs', np.array([]))
        labels = train_data.get('labels', [])
        file_ids = train_data.get('file_ids', [])
        
        if len(logits) == 0 or len(probs) == 0:
            return
            
        # Derive predicted labels from the probabilities.
        pred_labels = np.argmax(probs, axis=1)
        
        for idx in range(len(file_ids)):
            record = {
                'file_id': file_ids[idx],
                'fold': fold_idx + 1,
                'dataset_type': 'TRAIN',
                'true_label': self.config['class_names'][labels[idx]],
                'pred_label': self.config['class_names'][pred_labels[idx]],
                'is_correct': labels[idx] == pred_labels[idx],
            }
            
            # Add logits and probabilities for each class.
            for cls_idx, cls_name in enumerate(self.config['class_names']):
                record[f'{cls_name}_prob'] = probs[idx][cls_idx].item()
                record[f'{cls_name}_logit'] = logits[idx][cls_idx].item()
                
            self.all_records.append(record)

    def save_comprehensive_csv(self):
        """Export training and validation predictions to a combined CSV."""
        if not self.all_records:
            print('No records are available to save as CSV.')
            return
            
        df = pd.DataFrame(self.all_records)
        
        # Keep IDs as strings before exporting.
        if 'file_id' in df.columns:
            df['file_id'] = df['file_id'].astype(str)

        # Sort the rows.
        df.sort_values(
            by=['dataset_type', 'fold', 'file_id'],
            key=lambda x: x.map(lambda v:
                                int(''.join(filter(str.isdigit, str(v))) or '0')
                                if isinstance(v, (str, int)) else 0
                                ),
            inplace=True
        )
        
        # Define the output column order.
        columns = ['file_id', 'fold', 'dataset_type']
        
        # Add logit and probability columns.
        logits_columns = [f'{cls}_logit' for cls in self.config['class_names']]
        probs_columns = [f'{cls}_prob' for cls in self.config['class_names']]
        
        # List logits before probabilities for each class.
        columns += logits_columns + probs_columns
        columns += ['pred_label', 'true_label', 'is_correct']
        
        # Keep the available columns.
        existing_columns = [col for col in columns if col in df.columns]
        df = df[existing_columns]
        
        # Resolve the CSV output path.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.save_root, f"comprehensive_logits_probs_{timestamp}.csv")
        
        # Export CSV with high-precision scores.
        df.to_csv(csv_path, index=False, float_format='%.10f')
        
        print(f"Saved combined logits and probabilities to {csv_path}")
        
        # Also save the workbook with column sizing and text-formatted IDs.
        try:
            excel_path = csv_path.replace('.csv', '.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Comprehensive', float_format='%.10f')

                workbook = writer.book
                worksheet = writer.sheets['Comprehensive']

                # Set the width and format of each column.
                for col_idx, column in enumerate(df.columns, 1):
                    column_letter = get_column_letter(col_idx)

                    # Estimate the maximum content width.
                    if column.endswith('_prob') or column.endswith('_logit'):
                        max_len = len("0.0000000000")
                        content_samples = df[column].astype(str).str.split('.', expand=True)[1]
                        max_decimal = content_samples.str.len().max()
                        if pd.notnull(max_decimal):
                            max_len = 3 + max_decimal
                    else:
                        max_len = max(
                            len(str(column)) * 1.2,
                            df[column].astype(str).map(len).max() * 1.1
                        )

                    # Allow extra width for the ID column.
                    if column == 'file_id':
                        worksheet.column_dimensions[column_letter].width = min(max(12, max_len), 120)
                    else:
                        worksheet.column_dimensions[column_letter].width = min(max(10, max_len), 120)

                    # Set cell formats and alignment.
                    for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row, min_col=col_idx, max_col=col_idx):
                        for cell in row:
                            # Center cells by default.
                            cell.alignment = Alignment(horizontal='center')
                            if column.endswith('_prob') or column.endswith('_logit'):
                                cell.number_format = '0.####################'
                            if column == 'file_id':
                                # Keep IDs left-aligned and formatted as text.
                                cell.number_format = '@'
                                cell.alignment = Alignment(horizontal='left')

            print(f"Saved the formatted logits and probabilities workbook to {excel_path}")
        except Exception as e:
            print(f"Could not save the formatted workbook: {e}")

        # Write the prediction data summary.
        self._generate_data_statistics_report(df, csv_path)
        
    def _generate_data_statistics_report(self, df, csv_path):
        """Generate a report summarizing the prediction records."""
        report_path = csv_path.replace('.csv', '_statistics.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('=== Combined prediction summary ===\n')
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall counts
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Total folds: {df['fold'].nunique()}\n")
            f.write(f"Dataset types: {', '.join(df['dataset_type'].unique())}\n\n")
            
            # Statistics by dataset type
            f.write('=== Statistics by dataset type ===\n')
            for dataset_type in df['dataset_type'].unique():
                subset = df[df['dataset_type'] == dataset_type]
                f.write(f"{dataset_type}:\n")
                f.write(f"  Samples: {len(subset)}\n")
                f.write(f"  Folds: {subset['fold'].nunique()}\n")
                f.write(f"  Accuracy: {subset['is_correct'].mean():.4f}\n")
                
                # Count ground-truth labels by class.
                f.write('  Ground-truth label distribution:\n')
                for label, count in subset['true_label'].value_counts().items():
                    f.write(f"    {label}: {count}\n")
                f.write("\n")
            
            # Statistics by fold
            f.write('=== Statistics by fold ===\n')
            for fold in sorted(df['fold'].unique()):
                subset = df[df['fold'] == fold]
                f.write(f"Fold {fold}:\n")
                f.write(f"  Samples: {len(subset)}\n")
                for dataset_type in subset['dataset_type'].unique():
                    type_subset = subset[subset['dataset_type'] == dataset_type]
                    f.write(f"  {dataset_type}: {len(type_subset)} samples\n")
                f.write("\n")
        
        print(f"Saved the data summary to {report_path}")

    def generate_excel_report(self, report_data):
        """Build the validation workbook from the report rows."""
        # Define overall and per-class metric columns.
        overall_metrics = [
            'Accuracy',
            'Balanced Accuracy',
            'AUC(Macro)',
            'F1(Macro)',
            'Sensitivity(Macro)',
            'Specificity(Macro)'
        ]
        classwise_metrics = ['Accuracy', 'AUC', 'F1', 'Sensitivity', 'Specificity']  # Per-class metric fields
        
        # Construct the multilevel column index.
        overall_columns = [('Overall', metric) for metric in overall_metrics]
        classwise_columns = []
        for cls in self.config['class_names']:
            for metric in classwise_metrics:
                classwise_columns.append((cls, metric))
        
        columns = pd.MultiIndex.from_tuples(overall_columns + classwise_columns)

        # Build rows and row labels.
        rows = []
        index_labels = []
        for data in report_data:
            row = []

            # Overall metrics
            overall = data['overall']
            row.extend([
                overall.get('accuracy', ''),
                overall.get('balanced_accuracy', ''),
                overall.get('auc_macro', ''),
                overall.get('f1_macro', ''),
                overall.get('sensitivity_macro', ''),
                overall.get('specificity_macro', '')
            ])

            # Per-class metrics
            for cls in self.config['class_names']:
                cls_metrics = data['class_wise'][cls]
                
                # Normalize the supported per-class result layouts.
                if 'accuracy' in cls_metrics:
                    # Standard classwise_metrics layout
                    row.extend([
                        cls_metrics.get('accuracy', 0.0),
                        cls_metrics.get('auc', 0.0),
                        cls_metrics.get('f1', 0.0),
                        cls_metrics.get('sensitivity', 0.0),
                        cls_metrics.get('specificity', 0.0)
                    ])
                elif 'precision' in cls_metrics:
                    # Classification report layout used by the ensemble
                    # Read values from the classification report.
                    precision = cls_metrics.get('precision', 0.0)
                    recall = cls_metrics.get('recall', 0.0)
                    f1_score = cls_metrics.get('f1-score', 0.0)
                    support = cls_metrics.get('support', 0)
                    
                    # Reuse precomputed specificity and accuracy.
                    specificity = cls_metrics.get('specificity', precision)  # Prefer precomputed specificity.
                    accuracy = cls_metrics.get('accuracy', recall)  # Prefer precomputed accuracy.
                    
                    # Map the ensemble metrics into report columns.
                    row.extend([
                        accuracy,         # Use the computed accuracy.
                        cls_metrics.get('auc', f1_score),  # Prefer the computed AUC, then use the existing F1 fallback.
                        f1_score,         # f1
                        recall,           # sensitivity (recall)
                        specificity       # Use the computed specificity.
                    ])
                else:
                    # Use defaults for unknown layouts.
                    row.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            rows.append(row)

            # Build the row label.
            if data['type'] == 'aggregated_internal':
                index_labels.append("AGGREGATED_INTERNAL")
            elif data['type'] == 'external' and data['fold'] == 'ALL':
                index_labels.append("AGGREGATED_External")
            else:
                dtype = data['type']
                if dtype == 'external':
                    index_labels.append(f"EXTERNAL_FOLD_{data['fold']}")
                elif dtype == 'average_external':
                    # Name fold-averaged probability results.
                    # Use Average_External for pooled cohorts or the supplied cohort name.
                    index_labels.append("Average_External")
                elif dtype.startswith('average_external_set_'):
                    set_id = dtype.split('_')[-1]
                    index_labels.append(f"Average_External{set_id}")
                elif dtype.startswith('external_set_'):
                    set_id = dtype.split('_')[-1]
                    index_labels.append(f"EXTERNAL_SET_{set_id}_FOLD_{data['fold']}")
                elif dtype.startswith('aggregated_external_set_'):
                    set_id = dtype.split('_')[-1]
                    index_labels.append(f"AGGREGATED_EXTERNAL_SET_{set_id}")
                else:
                    index_labels.append(f"{dtype.upper()}_FOLD_{data['fold']}")

        # Build the report table.
        df = pd.DataFrame(
            rows,
            columns=columns,
            index=index_labels
        )

        # Use a timestamped filename.
        report_filename = f"validation_report_{self.timestamp}.xlsx"
        report_path = os.path.join(self.save_root, report_filename)

        # Workbook styles
        style_config = {
            'internal_fill': PatternFill(start_color='FFC7CE', fill_type='solid'),  # Light red
            'external_fill': PatternFill(start_color='C6EFCE', fill_type='solid'),  # Light green
            'aggregate_internal_fill': PatternFill(start_color='FFD700', fill_type='solid'),  # Gold
            'aggregate_external_fill': PatternFill(start_color='B4C6E7', fill_type='solid'),  # Light blue
        }

        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # Write the main report sheet.
            df.to_excel(writer, float_format="%.4f", sheet_name='Validation Report')

            # Get workbook and worksheet handles.
            workbook = writer.book
            worksheet = writer.sheets['Validation Report']

            # Size columns to fit contents.
            for col_idx, column_cells in enumerate(worksheet.columns, 1):
                max_length = 0
                column_letter = get_column_letter(col_idx)

                for cell in column_cells:
                    if isinstance(cell, Cell) and cell.value:
                        try:
                            length = len(str(cell.value))
                            if length > max_length:
                                max_length = length
                        except:
                            continue

                adjusted_width = (max_length + 2) * 1.2
                worksheet.column_dimensions[column_letter].width = adjusted_width

            # Add a metric glossary sheet.
            notes_df = pd.DataFrame({
                'Metric glossary': [
                    'Accuracy: fraction of correctly classified samples',
                    'Balanced Accuracy: mean recall across classes',
                    'AUC: macro-average across classes for Overall, or individual class AUC',
                    'F1: macro-average across classes for Overall, or individual class F1',
                    'Sensitivity: recall',
                    'Specificity: true negative rate'
                ]
            })
            notes_df.to_excel(writer, sheet_name='Metric glossary', index=False)

            # Apply the appropriate background fill to each row.
            for row_idx in range(3, 3 + len(index_labels)):  # The two header rows precede the data.
                label_cell = worksheet.cell(row=row_idx, column=1)
                label = str(label_cell.value).strip().upper()

                if label == "AGGREGATED_INTERNAL":
                    fill_color = style_config['aggregate_internal_fill']
                elif label == "AGGREGATED_EXTERNAL" or label == "AGGREGATED_EXTERNAL_SET" or label == "AGGREGATED_EXTERNAL_SET_":
                    fill_color = style_config['aggregate_external_fill']
                elif label.startswith("AGGREGATED_EXTERNAL_SET_"):
                    fill_color = style_config['aggregate_external_fill']
                elif label.startswith("AGGREGATED_EXTERNAL"):
                    fill_color = style_config['aggregate_external_fill']
                elif label.startswith("INTERNAL_FOLD"):
                    fill_color = style_config['internal_fill']
                elif label.startswith("EXTERNAL_FOLD"):
                    fill_color = style_config['external_fill']
                elif label.startswith("EXTERNAL_SET_"):
                    fill_color = style_config['external_fill']
                elif label.startswith("AVERAGE_EXTERNAL"):
                    fill_color = style_config['aggregate_external_fill']
                else:
                    fill_color = None

                if fill_color:
                    for col_idx in range(1, worksheet.max_column + 1):
                        worksheet.cell(row=row_idx, column=col_idx).fill = fill_color

        print(f"\nSaved the validation report to {report_path}")

    def aggregate_results(self, fold_results):
        """Aggregate internal and external results across the supplied folds."""
        # Collect results for aggregation.
        # Initialize internal validation accumulators.
        all_internal_true = []
        all_internal_pred = []
        all_internal_scores = []
        internal_file_ids = []

        # Initialize pooled external accumulators.
        all_external_true = []
        all_external_pred = []
        all_external_scores = []
        external_file_ids = []

        # Initialize per-cohort external accumulators.
        external_sets_data = {}  # key: set_idx -> {'true':[], 'pred':[], 'scores':[], 'file_ids':[]}

        report_data = []

        # Process cross-validation folds.
        for fold_idx, fold_metric in enumerate(fold_results):
            # Internal validation
            internal_metrics = fold_metric.get('internal', {})
            if internal_metrics:
                # Add a row to the report.
                report_data.append({
                    'type': 'internal',
                    'fold': fold_idx + 1,
                    'overall': internal_metrics['overall'],
                    'class_wise': internal_metrics['class_wise']
                })

                # Accumulate internal results.
                all_internal_true.extend(internal_metrics['y_true'])
                all_internal_pred.extend(internal_metrics['y_pred'])
                all_internal_scores.append(internal_metrics['scores'])
                internal_file_ids.extend(internal_metrics.get('file_ids', []))

            # Single or pooled external validation
            external_metrics = fold_metric.get('external', {})
            if external_metrics:
                # Add a row to the report.
                report_data.append({
                    'type': 'external',
                    'fold': fold_idx + 1,
                    'overall': external_metrics['overall'],
                    'class_wise': external_metrics['class_wise']
                })

                # Accumulate external results.
                all_external_true.extend(external_metrics['y_true'])
                all_external_pred.extend(external_metrics['y_pred'])
                all_external_scores.append(external_metrics['scores'])
                external_file_ids.extend(external_metrics.get('file_ids', []))

            # Per-cohort external validation
            external_sets = fold_metric.get('external_sets', [])
            if external_sets:
                for set_idx, m in enumerate(external_sets, start=1):
                    if not m:
                        continue
                    # Add a row for this fold and cohort.
                    report_data.append({
                        'type': f'external_set_{set_idx}',
                        'fold': fold_idx + 1,
                        'overall': m['overall'],
                        'class_wise': m['class_wise']
                    })

                    # Accumulate results for this cohort.
                    if set_idx not in external_sets_data:
                        external_sets_data[set_idx] = {
                            'true': [], 'pred': [], 'scores': [], 'file_ids': []
                        }
                    external_sets_data[set_idx]['true'].extend(m.get('y_true', []))
                    external_sets_data[set_idx]['pred'].extend(m.get('y_pred', []))
                    if isinstance(m.get('scores'), np.ndarray) and m['scores'].size > 0:
                        external_sets_data[set_idx]['scores'].append(m['scores'])
                    external_sets_data[set_idx]['file_ids'].extend(m.get('file_ids', []))

        # Aggregate cross-validation metrics.
        # Pooled internal metrics
        agg_internal = {}
        if all_internal_true:
            all_internal_scores = np.vstack(all_internal_scores)
            agg_internal = {
                'overall': compute_overall_metrics(
                    all_internal_true, all_internal_pred, all_internal_scores,
                    self.config['class_names']
                ),
                'class_wise': compute_classwise_metrics(
                    all_internal_true, all_internal_pred, all_internal_scores,
                    self.config['class_names']
                )
            }
            report_data.append({
                'type': 'aggregated_internal',
                'fold': 'ALL',
                **agg_internal
            })

        # Single-cohort or pooled external metrics
        agg_external = {}
        if all_external_true:
            all_external_scores = np.vstack(all_external_scores)
            agg_external = {
                'overall': compute_overall_metrics(
                    all_external_true, all_external_pred, all_external_scores,
                    self.config['class_names']
                ),
                'class_wise': compute_classwise_metrics(
                    all_external_true, all_external_pred, all_external_scores,
                    self.config['class_names']
                )
            }
            report_data.append({
                'type': 'aggregated_external',
                'fold': 'ALL',
                **agg_external
            })

        # Per-cohort external metrics
        if external_sets_data:
            for set_idx, pack in external_sets_data.items():
                if not pack['true']:
                    continue
                scores = np.vstack(pack['scores']) if len(pack['scores']) > 0 else np.array([])
                agg = {
                    'overall': compute_overall_metrics(pack['true'], pack['pred'], scores, self.config['class_names']),
                    'class_wise': compute_classwise_metrics(pack['true'], pack['pred'], scores, self.config['class_names'])
                }
                report_data.append({
                    'type': f'aggregated_external_set_{set_idx}',
                    'fold': 'ALL',
                    **agg
                })

        # Evaluate the fold ensemble.
        # Compute fold-averaged predictions.
        ensemble_results = self.evaluate_ensemble_predictions(fold_results)
        if ensemble_results and 'error' not in ensemble_results:
            # Add the pooled ensemble result as Average_External.
            report_data.append({
                'type': 'average_external',
                'fold': 'ENSEMBLE',
                'overall': {
                    'accuracy': ensemble_results['accuracy'],
                    'balanced_accuracy': ensemble_results['balanced_accuracy'],
                    'auc_macro': ensemble_results['macro_auc'],
                    'auc_weighted': ensemble_results['weighted_auc'],
                    'f1_macro': ensemble_results['f1_macro'],
                    'sensitivity_macro': ensemble_results['sensitivity_macro'],
                    'specificity_macro': ensemble_results['specificity_macro']
                },
                'class_wise': ensemble_results['classification_report']
            })

            # Add per-cohort ensemble metrics when available.
            if getattr(self, 'per_set_ensemble_results', None):
                for set_idx, set_ens in self.per_set_ensemble_results.items():
                    report_data.append({
                        'type': f'average_external_set_{set_idx}',
                        'fold': 'ENSEMBLE',
                        'overall': {
                            'accuracy': set_ens['accuracy'],
                            'balanced_accuracy': set_ens['balanced_accuracy'],
                            'auc_macro': set_ens['macro_auc'],
                            'auc_weighted': set_ens.get('weighted_auc', 0.0),
                            'f1_macro': set_ens['f1_macro'],
                            'sensitivity_macro': set_ens['sensitivity_macro'],
                            'specificity_macro': set_ens['specificity_macro']
                        },
                        'class_wise': set_ens['classification_report']
                    })

        # Write final reports.
        # Excel workbook
        self.generate_excel_report(report_data)

        # JSON summary
        combin_path = os.path.join(self.save_root, "Validation_Summary")
        os.makedirs(combin_path, exist_ok=True)

        final_report = {
            'internal': {
                'aggregated_metrics': agg_internal,
                'data_info': {
                    'total_samples': len(all_internal_true),
                    'class_distribution': dict(Counter(all_internal_true)),
                    'prediction_distribution': dict(Counter(all_internal_pred)),
                    'file_ids': internal_file_ids
                }
            },
            'external': {
                'aggregated_metrics': agg_external,
                'data_info': {
                    'total_samples': len(all_external_true),
                    'class_distribution': dict(Counter(all_external_true)),
                    'prediction_distribution': dict(Counter(all_external_pred)),
                    'file_ids': external_file_ids
                } if all_external_true else None
            },
            'ensemble': ensemble_results if ensemble_results and 'error' not in ensemble_results else None,
            'config': {
                k: v for k, v in self.config.items()
                if k not in ['external_h5_dir', 'external_excel']  # Exclude private path fields.
            }
        }

        # Serialize the JSON report.
        with open(os.path.join(combin_path, 'cross_validation_report.json'), 'w') as f:
            json.dump(final_report, f, indent=2, default=str, ensure_ascii=False)

        # Generate aggregate figures and organize outputs.
        # Separate internal and external ROC and confusion plots under Validation_Summary.
        roc_root = os.path.join(combin_path, 'ROC')
        conf_root = os.path.join(combin_path, 'Confusion')
        os.makedirs(roc_root, exist_ok=True)
        os.makedirs(conf_root, exist_ok=True)

        # Write pooled internal confusion and ROC plots.
        if all_internal_true:
            conf_int_dir = os.path.join(conf_root, 'Internal')
            os.makedirs(conf_int_dir, exist_ok=True)
            plot_confusion_matrix(
                all_internal_true, all_internal_pred,
                self.config['class_names'],
                title_suffix='Combined Internal Validation',
                save_path=os.path.join(conf_int_dir, 'internal_combined_confusion.pdf')
            )

            # Include class-level and overall ROC curves.
            roc_int_dir = os.path.join(roc_root, 'Internal')
            os.makedirs(roc_int_dir, exist_ok=True)
            plot_roc_curves(
                all_internal_true,
                all_internal_scores,
                self.config['class_names'],
                title_suffix='Internal Validation (Aggregated)',
                save_dir=roc_int_dir
            )

        # Write pooled and individual cohort plots for external validation.
        roc_ext_root = os.path.join(roc_root, 'External')
        conf_ext_root = os.path.join(conf_root, 'External')
        os.makedirs(roc_ext_root, exist_ok=True)
        os.makedirs(conf_ext_root, exist_ok=True)

        # Pooled external results
        if all_external_true:
            agg_dir_roc = os.path.join(roc_ext_root, 'AGGREGATED_External')
            os.makedirs(agg_dir_roc, exist_ok=True)
            plot_roc_curves(
                all_external_true,
                all_external_scores,
                self.config['class_names'],
                title_suffix='External (Aggregated)',
                save_dir=agg_dir_roc
            )

            agg_dir_conf = os.path.join(conf_ext_root, 'AGGREGATED_External')
            os.makedirs(agg_dir_conf, exist_ok=True)
            plot_confusion_matrix(
                all_external_true,
                all_external_pred,
                self.config['class_names'],
                title_suffix='External (Aggregated)',
                save_path=os.path.join(agg_dir_conf, 'confusion.pdf')
            )

        # Individual external cohorts
        if external_sets_data:
            for set_idx, pack in external_sets_data.items():
                if not pack['true']:
                    continue
                set_scores = np.vstack(pack['scores']) if len(pack['scores']) > 0 else np.array([])
                set_dir_roc = os.path.join(roc_ext_root, f'set_{set_idx}')
                os.makedirs(set_dir_roc, exist_ok=True)
                plot_roc_curves(
                    pack['true'],
                    set_scores,
                    self.config['class_names'],
                    title_suffix=f'External Set {set_idx} (Aggregated)',
                    save_dir=set_dir_roc
                )

                set_dir_conf = os.path.join(conf_ext_root, f'set_{set_idx}')
                os.makedirs(set_dir_conf, exist_ok=True)
                if 'pred' in pack and pack['pred']:
                    set_pred = pack['pred']
                else:
                    try:
                        set_pred = list(np.argmax(set_scores, axis=1))
                    except Exception:
                        set_pred = []
                plot_confusion_matrix(
                    pack['true'],
                    set_pred,
                    self.config['class_names'],
                    title_suffix=f'External Set {set_idx} (Aggregated)',
                    save_path=os.path.join(set_dir_conf, 'confusion.pdf')
                )

        # Return ensemble results for the validation report.
        return ensemble_results

    def evaluate_ensemble_predictions(self, fold_results):
        """Evaluate the ensemble across the five cross-validation folds."""
        # Collect external predictions across folds.
        external_scores = []
        external_labels = []
        external_file_ids = []
        
        for fold_result in fold_results:
            if 'external' in fold_result:
                external_data = fold_result['external']
                if 'scores' in external_data and 'y_true' in external_data:
                    external_scores.append(external_data['scores'])
                    if len(external_labels) == 0:  # Use one copy of labels and IDs to align fold results.
                        external_labels = external_data['y_true']
                        external_file_ids = external_data.get('file_ids', [])
        
        if not external_scores or len(external_labels) == 0:
            print('There are too few external predictions for ensemble evaluation.')
            return {'error': 'No sufficient external predictions found for ensemble evaluation'}
        
        # Compute ensemble predictions for the pooled external set.
        ensemble_results = self._compute_ensemble_predictions(external_scores, external_labels)
        
        # Save ensemble metrics.
        self._save_ensemble_results(ensemble_results)
        
        # Generate ensemble plots.
        self._generate_ensemble_visualizations(ensemble_results)

        # Record fold-averaged probabilities.
        # For several cohorts, write one Average_External entry per cohort.
        # For a single cohort, use Average_External1.
        try:
            per_set_scores_map = {}
            per_set_label_map = {}
            for fold_result in fold_results:
                if 'external_sets' in fold_result and isinstance(fold_result['external_sets'], list):
                    for set_idx, m in enumerate(fold_result['external_sets'], start=1):
                        if not m or 'scores' not in m or 'y_true' not in m or 'file_ids' not in m:
                            continue
                        scores_arr = m['scores']
                        file_ids_list = m['file_ids']
                        y_true_list = m['y_true']
                        if len(scores_arr) != len(file_ids_list) or len(scores_arr) != len(y_true_list):
                            continue
                        if set_idx not in per_set_scores_map:
                            per_set_scores_map[set_idx] = {}
                            per_set_label_map[set_idx] = {}
                        for i, fid in enumerate(file_ids_list):
                            per_set_scores_map[set_idx].setdefault(fid, []).append(scores_arr[i])
                            if fid not in per_set_label_map[set_idx]:
                                per_set_label_map[set_idx][fid] = y_true_list[i]

            if per_set_scores_map:  # Write one probability table, metrics file, and set of plots per cohort.
                # Clear the per-cohort summary.
                self.per_set_ensemble_results = {}
                for set_idx, fid_to_scores in per_set_scores_map.items():
                    fids, labels, avg_scores = [], [], []
                    for fid, score_list in fid_to_scores.items():
                        try:
                            avg = np.mean(np.stack(score_list, axis=0), axis=0)
                        except Exception:
                            avg = np.mean(np.array(score_list), axis=0)
                        fids.append(fid)
                        labels.append(per_set_label_map[set_idx].get(fid, 0))
                        avg_scores.append(avg)
                    if fids:
                        avg_scores_arr = np.vstack(avg_scores)
                        # Record per-sample outputs.
                        self._record_average_probabilities(
                            file_ids=fids,
                            true_labels=labels,
                            scores=avg_scores_arr,
                            dataset_type_label=f'Average_External{set_idx}'
                        )
                        # Compute and write the cohort's ensemble evaluation.
                        set_ensemble = self._compute_ensemble_predictions([avg_scores_arr], labels)
                        # Store the result for Validation_Summary.
                        self.per_set_ensemble_results[set_idx] = set_ensemble
                        self._save_set_ensemble_results(set_idx, set_ensemble)
                        self._generate_set_ensemble_visualizations(set_idx, set_ensemble)
            else:  # Write the single external cohort as Average_External1.
                if 'ensemble_scores' in ensemble_results and len(external_labels) == len(external_file_ids) == len(ensemble_results['ensemble_scores']):
                    self._record_average_probabilities(
                        file_ids=external_file_ids,
                        true_labels=external_labels,
                        scores=ensemble_results['ensemble_scores'],
                        dataset_type_label='Average_External1'
                    )
        except Exception as e:
            print(f"Could not record Average_External probabilities: {e}")

        return ensemble_results

    def _compute_ensemble_predictions(self, scores_list, true_labels):
        """Combine per-fold probabilities and compute ensemble predictions."""
        # Stack scores as (n_folds, n_samples, n_classes).
        scores_array = np.array(scores_list)
        
        # Average probabilities across folds.
        ensemble_scores = np.mean(scores_array, axis=0)
        
        # Choose classes from the mean probabilities.
        ensemble_predictions = np.argmax(ensemble_scores, axis=1)
        
        # Convert labels to the expected format.
        true_labels = np.array(true_labels)
        
        # Compute overall metrics.
        overall_metrics = compute_overall_metrics(true_labels, ensemble_predictions, ensemble_scores, self.config['class_names'])
        
        # Compute the confusion matrix.
        conf_matrix = confusion_matrix(true_labels, ensemble_predictions)
        
        # Compute per-class metrics.
        classwise_metrics = compute_classwise_metrics(true_labels, ensemble_predictions, ensemble_scores, self.config['class_names'])
        
        # Compute the classification report.
        class_report = classification_report(true_labels, ensemble_predictions, 
                                           target_names=self.config['class_names'], output_dict=True,
                                           zero_division=0)
        
        # Add per-class metrics to the classification report.
        for class_name in self.config['class_names']:
            if class_name in class_report and class_name in classwise_metrics:
                class_report[class_name].update(classwise_metrics[class_name])
        
        # Compute multiclass AUC.
        try:
            macro_auc = roc_auc_score(true_labels, ensemble_scores, 
                                    multi_class='ovr', average='macro')
            weighted_auc = roc_auc_score(true_labels, ensemble_scores, 
                                       multi_class='ovr', average='weighted')
        except:
            macro_auc = 0.0
            weighted_auc = 0.0
        
        # Compute per-class ROC curves.
        roc_data = {}
        for i, class_name in enumerate(self.config['class_names']):
            try:
                # Convert one class to a one-vs-rest target.
                binary_labels = (true_labels == i).astype(int)
                class_scores = ensemble_scores[:, i]
                
                fpr, tpr, thresholds = roc_curve(binary_labels, class_scores)
                class_auc = roc_auc_score(binary_labels, class_scores)
                
                roc_data[class_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'auc': class_auc
                }
            except:
                roc_data[class_name] = {
                    'fpr': [],
                    'tpr': [],
                    'thresholds': [],
                    'auc': 0.0
                }
        
        # Summarize prediction confidence.
        max_scores = np.max(ensemble_scores, axis=1)
        confidence_stats = {
            'mean_confidence': float(np.mean(max_scores)),
            'std_confidence': float(np.std(max_scores)),
            'min_confidence': float(np.min(max_scores)),
            'max_confidence': float(np.max(max_scores))
        }
        
        # Measure agreement among fold predictions.
        individual_predictions = np.argmax(scores_array, axis=2)  # (n_folds, n_samples)
        consistency_scores = []
        
        for sample_idx in range(individual_predictions.shape[1]):
            sample_predictions = individual_predictions[:, sample_idx]
            # Compute the fraction voting for the most frequent class.
            unique, counts = np.unique(sample_predictions, return_counts=True)
            max_agreement = np.max(counts) / len(sample_predictions)
            consistency_scores.append(max_agreement)
        
        consistency_stats = {
            'mean_consistency': float(np.mean(consistency_scores)),
            'std_consistency': float(np.std(consistency_scores))
        }
        
        return {
            'method': 'ensemble_average_probability',
            'accuracy': overall_metrics['accuracy'],
            'balanced_accuracy': overall_metrics['balanced_accuracy'],
            'macro_auc': macro_auc,
            'weighted_auc': weighted_auc,
            'f1_macro': overall_metrics['f1_macro'],
            'sensitivity_macro': overall_metrics['sensitivity_macro'],
            'specificity_macro': overall_metrics['specificity_macro'],
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'roc_data': roc_data,
            'confidence_stats': confidence_stats,
            'consistency_stats': consistency_stats,
            'ensemble_scores': ensemble_scores,
            'ensemble_predictions': ensemble_predictions.tolist(),
            'true_labels': true_labels.tolist(),
            'timestamp': datetime.now().isoformat()
        }

    def _record_average_probabilities(self, file_ids, true_labels, scores, dataset_type_label):
        """Add average probabilities to all_records with fold zero and the supplied dataset label, such as Average_External."""
        assert len(file_ids) == len(true_labels) == len(scores), 'Fold-averaged probabilities and labels have different lengths.'
        for idx in range(len(file_ids)):
            record = {
                'file_id': str(file_ids[idx]),
                'fold': 0,
                'dataset_type': dataset_type_label,
                'true_label': self.config['class_names'][true_labels[idx]],
                'pred_label': self.config['class_names'][int(np.argmax(scores[idx]))],
                'is_correct': int(np.argmax(scores[idx])) == int(true_labels[idx]),
            }
            for cls_idx, cls_name in enumerate(self.config['class_names']):
                record[f'{cls_name}_prob'] = float(scores[idx][cls_idx])
            self.all_records.append(record)

    def _save_ensemble_results(self, ensemble_results):
        """Save the results from averaging five-fold probabilities to fold_average."""
        # Create the fold-average output directory.
        ensemble_dir = os.path.join(self.save_root, "fold_average")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # Save detailed results.
        model_name = self.config.get('feature_model', 'model')
        results_file = os.path.join(ensemble_dir, f"{model_name}_ensemble_evaluation.json")
        
        # Exclude arrays from the JSON summary.
        save_data = {k: v for k, v in ensemble_results.items() 
                    if k not in ['ensemble_scores']}
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # Save the confusion matrix.
        confusion_file = os.path.join(ensemble_dir, f"{model_name}_confusion_matrix.json")
        with open(confusion_file, 'w', encoding='utf-8') as f:
            json.dump({
                'confusion_matrix': ensemble_results['confusion_matrix'],
                'class_names': self.config['class_names']
            }, f, indent=2, ensure_ascii=False)
        
        # Save ROC curve data.
        roc_file = os.path.join(ensemble_dir, f"{model_name}_roc_curves.json")
        with open(roc_file, 'w', encoding='utf-8') as f:
            json.dump(ensemble_results['roc_data'], f, indent=2, ensure_ascii=False)
        
        print(f"Saved fold-averaged probabilities to {ensemble_dir}")

    def _generate_ensemble_visualizations(self, ensemble_results):
        """Plot the results from averaging five-fold probabilities under fold_average."""
        # Create the fold-average plot directory.
        viz_dir = os.path.join(self.save_root, "fold_average")
        os.makedirs(viz_dir, exist_ok=True)
        
        model_name = self.config.get('feature_model', 'model')
        
        # Configure plotting fonts.
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Plot the confusion matrix.
        self._plot_ensemble_confusion_matrix(ensemble_results, viz_dir, model_name)
        
        # Plot ROC curves.
        self._plot_ensemble_roc_curves(ensemble_results, viz_dir, model_name)
        
        # Plot confidence and agreement.
        self._plot_confidence_distribution(ensemble_results, viz_dir, model_name)
        
        print(f"Saved fold-average plots to {viz_dir}")

    def _save_set_ensemble_results(self, set_idx, ensemble_results):
        """Write ensemble metrics for one external cohort under fold_average/set_{idx}."""
        base_dir = os.path.join(self.save_root, "fold_average", f"set_{set_idx}")
        os.makedirs(base_dir, exist_ok=True)

        model_name = self.config.get('feature_model', 'model')
        results_file = os.path.join(base_dir, f"{model_name}_ensemble_evaluation.json")
        save_data = {k: v for k, v in ensemble_results.items() if k not in ['ensemble_scores']}
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        confusion_file = os.path.join(base_dir, f"{model_name}_confusion_matrix.json")
        with open(confusion_file, 'w', encoding='utf-8') as f:
            json.dump({
                'confusion_matrix': ensemble_results['confusion_matrix'],
                'class_names': self.config['class_names']
            }, f, indent=2, ensure_ascii=False)

        roc_file = os.path.join(base_dir, f"{model_name}_roc_curves.json")
        with open(roc_file, 'w', encoding='utf-8') as f:
            json.dump(ensemble_results['roc_data'], f, indent=2, ensure_ascii=False)

    def _generate_set_ensemble_visualizations(self, set_idx, ensemble_results):
        """Write ensemble plots for one external cohort under fold_average/set_{idx}."""
        viz_dir = os.path.join(self.save_root, "fold_average", f"set_{set_idx}")
        os.makedirs(viz_dir, exist_ok=True)

        model_name = self.config.get('feature_model', 'model')
        self._plot_ensemble_confusion_matrix(ensemble_results, viz_dir, f"{model_name}_set_{set_idx}")
        self._plot_ensemble_roc_curves(ensemble_results, viz_dir, f"{model_name}_set_{set_idx}")
        self._plot_confidence_distribution(ensemble_results, viz_dir, f"{model_name}_set_{set_idx}")

    def _plot_ensemble_confusion_matrix(self, ensemble_results, viz_dir, model_name):
        """Plot the ensemble confusion matrix."""
        try:
            plt.figure(figsize=(10, 8))
            
            conf_matrix = np.array(ensemble_results['confusion_matrix'])
            
            # Normalize counts to percentages.
            conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            
            # Draw the confusion heatmap.
            sns.heatmap(conf_matrix_percent, annot=True, fmt='.2%', 
                       xticklabels=self.config['class_names'], 
                       yticklabels=self.config['class_names'],
                       cmap='Blues', cbar_kws={'label': 'Prediction ratio'})
            
            plt.title(f'Ensemble confusion matrix - {model_name}')
            plt.xlabel('Predicted class')
            plt.ylabel('True class')
            plt.tight_layout()
            
            save_path = os.path.join(viz_dir, f"{model_name}_ensemble_confusion_matrix.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
        finally:
            cleanup_matplotlib()

    def _plot_ensemble_roc_curves(self, ensemble_results, viz_dir, model_name):
        """Plot ensemble ROC curves."""
        try:
            plt.figure(figsize=(10, 8))
            
            colors = ['red', 'green', 'blue', 'orange']
            
            for i, (class_name, roc_info) in enumerate(ensemble_results['roc_data'].items()):
                if roc_info['fpr'] and roc_info['tpr']:
                    plt.plot(roc_info['fpr'], roc_info['tpr'], 
                            color=colors[i % len(colors)], lw=2,
                            label=f'{class_name} (AUC = {roc_info["auc"]:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False positive rate (FPR)')
            plt.ylabel('True positive rate (TPR)')
            plt.title(f'Ensemble ROC curves - {model_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(viz_dir, f"{model_name}_ensemble_roc_curves.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
        finally:
            cleanup_matplotlib()

    def _plot_confidence_distribution(self, ensemble_results, viz_dir, model_name):
        """Plot the distribution of prediction confidence."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Compute each sample's maximum predicted probability.
            ensemble_scores = ensemble_results['ensemble_scores']
            max_confidences = np.max(ensemble_scores, axis=1)
            
            # Create plot panels.
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Confidence histogram
            ax1.hist(max_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Prediction confidence')
            ax1.set_ylabel('Sample count')
            ax1.set_title('Ensemble confidence distribution')
            ax1.grid(True, alpha=0.3)
            
            # Confidence summary
            stats = ensemble_results['confidence_stats']
            ax2.bar(['Mean', 'Std. dev.', 'Minimum', 'Maximum'], 
                   [stats['mean_confidence'], stats['std_confidence'], 
                    stats['min_confidence'], stats['max_confidence']], 
                   color=['red', 'green', 'blue', 'orange'])
            ax2.set_ylabel('Confidence')
            ax2.set_title('Confidence statistics')
            
            # Prediction agreement
            consistency_stats = ensemble_results['consistency_stats']
            ax3.bar(['Mean agreement', 'Agreement std. dev.'], 
                   [consistency_stats['mean_consistency'], consistency_stats['std_consistency']], 
                   color=['purple', 'brown'])
            ax3.set_ylabel('Agreement')
            ax3.set_title('Agreement among fold models')
            
            # Performance metrics
            metrics = ['Accuracy', 'Balanced accuracy', 'Macro AUC', 'Weighted AUC']
            values = [ensemble_results['accuracy'], ensemble_results['balanced_accuracy'],
                     ensemble_results['macro_auc'], ensemble_results['weighted_auc']]
            ax4.bar(metrics, values, color=['lightcoral', 'lightgreen', 'lightblue', 'lightyellow'])
            ax4.set_ylabel('Metric value')
            ax4.set_title('Ensemble performance')
            ax4.set_ylim(0, 1)
            
            plt.suptitle(f'Ensemble prediction analysis - {model_name}', fontsize=16)
            plt.tight_layout()
            
            save_path = os.path.join(viz_dir, f"{model_name}_ensemble_analysis.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
        finally:
            cleanup_matplotlib()

    def generate_ensemble_validation_report_line(self, ensemble_results):
        """Format the ensemble evaluation row for the validation report."""
        if not ensemble_results or 'error' in ensemble_results:
            return None
        
        model_name = self.config.get('feature_model', 'model')
        
        # Format the ensemble report row.
        ensemble_line = (
            f"Average_External\t{model_name}\t"
            f"{ensemble_results['accuracy']:.4f}\t"
            f"{ensemble_results['balanced_accuracy']:.4f}\t"
            f"{ensemble_results['macro_auc']:.4f}\t"
            f"{ensemble_results['f1_macro']:.4f}\t"
            f"{ensemble_results['sensitivity_macro']:.4f}\t"
            f"{ensemble_results['specificity_macro']:.4f}\t"
            f"Fold average\t"
            f"Confidence: {ensemble_results['confidence_stats']['mean_confidence']:.3f}±"
            f"{ensemble_results['confidence_stats']['std_confidence']:.3f}\t"
            f"Agreement: {ensemble_results['consistency_stats']['mean_consistency']:.3f}±"
            f"{ensemble_results['consistency_stats']['std_consistency']:.3f}"
        )
        
        return ensemble_line

    def log_error(self, error_type, fold_idx, message):
        """Log an error category, fold index, and message."""
        try:
            # Create the error log directory.
            error_log_dir = os.path.join(self.save_root, "error_logs")
            os.makedirs(error_log_dir, exist_ok=True)
            
            # Construct the error log filename.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_log_file = os.path.join(error_log_dir, f"error_log_{timestamp}.txt")
            
            # Write the error details.
            with open(error_log_file, 'w', encoding='utf-8') as f:
                f.write(f"Error type: {error_type}\n")
                f.write(f"Fold index: {fold_idx}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Error message: {message}\n")
                f.write("-" * 50 + "\n")
            
            print(f"Logged the error to {error_log_file}")
            
        except Exception as e:
            print(f"Could not write the error log: {str(e)}")
