# cross_validation_All.py
import copy
import json
import os
import random
from collections import Counter

import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
# from model.AttentionMIL import MILModel
from torch.utils.tensorboard.writer import SummaryWriter
from dataSet.data_loader import H5Dataset, collate_fn
from metrics import compute_overall_metrics, compute_classwise_metrics, plot_confusion_matrix, plot_roc_curves
from external_validation import ExternalValidator
from datetime import datetime
from result_saver import ResultSaver

'Seed DataLoader workers independently.'


def seed_worker(worker_id):
    # Derive an independent seed for each worker.
    worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Map abbreviated metrics to their dictionary keys.
class CrossValidator:
    METRIC_MAPPING = {
        'acc': 'accuracy',
        'balanced_acc': 'balanced_accuracy',
        'auc': 'auc_macro',
        'loss': 'loss'
    }
    
    # Feature encoder configurations
    FEATURE_MODEL_CONFIGS = {
        'H-optimus-0': {
            'dim': 1536,
            'h5_dir': r"data/features/H-optimus-0/internal",
            'external_h5_dir': [r"data/features/H-optimus-0/external_a", r"data/features/H-optimus-0/external_b"]
        },
        'H-optimus-1': {
            'dim': 1536,
            'h5_dir': r"data/features/H-optimus-1/internal",
            'external_h5_dir': [r"data/features/H-optimus-1/external_a", r"data/features/H-optimus-1/external_b"]
        },
        'gigapath': {
            'dim': 1536,
            'h5_dir': r"data/features/gigapath/internal/tile_h5",
            'external_h5_dir': [r"data/features/gigapath/external_a/tile_h5", r"data/features/gigapath/external_b/tile"]
        },
        'UNI2': {
            'dim': 1536,
            'h5_dir': r"data/features/UNI2/internal",
            'external_h5_dir': [r"data/features/UNI2/external_a", r"data/features/UNI2/external_b"]
        },
        'ctranspath': {
            'dim': 768,
            'h5_dir': r"data/features/Ctranspath(new patch)/internal",
            'external_h5_dir': [r"data/features/Ctranspath(new patch)/external_a", r"data/features/Ctranspath(new patch)/external_b_20x"]
        },
        'CHIEF': {
            'dim': 768,
            'h5_dir': r"data/features/CHIEF/internal",
            'external_h5_dir': [r"data/features/CHIEF/external_a", r"data/features/CHIEF/external_b"]
        },
        'CONCH1.5': {
            'dim': 768,
            'h5_dir': r"data/features/TITAN_CONCH1.5/internal",
            'external_h5_dir': [r"data/features/TITAN_CONCH1.5/external_a", r"data/features/TITAN_CONCH1.5/external_b"]
        },
        'UNI': {
            'dim': 1024,
            'h5_dir': r"data/features/UNI1/internal",
            'external_h5_dir': [r"data/features/UNI1/external_a", r"data/features/UNI1/external_b"]
        },
        'CONCH': {
            'dim': 512,
            'h5_dir': r"data/features/CONCH/internal",
            'external_h5_dir': [r"data/features/CONCH/external_a", r"data/features/CONCH/external_b"]
        },
        'MUSK': {
            'dim': 2048,
            'h5_dir': r"data/features/MUSK/internal",
            'external_h5_dir': [r"data/features/MUSK/external_a", r"data/features/MUSK/external_b"]
        },
        'Virchow2': {
            'dim': 2560,
            'h5_dir': r"data/features/Virchow2/internal",
            'external_h5_dir': [r"data/features/Virchow2/external_a", r"data/features/Virchow2/external_b"]
        },
        'Phikon-v2': {
            'dim': 1024,
            'h5_dir': r"data/features/Phikon-v2/internal",
            'external_h5_dir': [r"data/features/Phikon-v2/external_a", r"data/features/Phikon-v2/external_b"]
        },
        'ResNet-50': {
            'dim': 2048,
            'h5_dir': r"data/features/ResNet50/internal",
            'external_h5_dir': [r"data/features/ResNet50/external_a", r"data/features/ResNet50/external_b"]
        },
        'finetune_Virchow2_S_ep1': {
            'dim': 2560,
            'h5_dir': r"data/features/finetun_virchow2/ep1/student/internal",
            'external_h5_dir': [r"data/features/finetun_virchow2/ep1/student/external_a",
                                r"data/features/finetun_virchow2/ep1/student/external_b"]
        },
        'finetune_Virchow2_T_ep1': {
            'dim': 2560,
            'h5_dir': r"data/features/finetun_virchow2/ep1/teacher/internal",
            'external_h5_dir': [r"data/features/finetun_virchow2/ep1/teacher/external_a",
                                r"data/features/finetun_virchow2/ep1/teacher/external_b"]
        },
        'finetune_Virchow2_S_ep8': {
            'dim': 2560,
            'h5_dir': r"data/features/finetun_virchow2/ep8/student/internal",
            'external_h5_dir': [r"data/features/finetun_virchow2/ep8/student/external_a",
                                r"data/features/finetun_virchow2/ep8/student/external_b"]
        },
        'finetune_Virchow2_T_ep8': {
            'dim': 2560,
            'h5_dir': r"data/features/finetun_virchow2/ep1/teacher/internal",
            'external_h5_dir': [r"data/features/finetun_virchow2/ep8/teacher/external_a",
                                r"data/features/finetun_virchow2/ep8/teacher/external_b"]
        },
        'finetune_Virchow2_S_ep19': {
            'dim': 2560,
            'h5_dir': r"data/features/finetun_virchow2/ep19/student/internal",
            'external_h5_dir': [r"data/features/finetun_virchow2/ep19/student/external_a",
                                r"data/features/finetun_virchow2/ep19/student/external_b"]
        },
        'finetune_Virchow2_T_ep19': {
            'dim': 2560,
            'h5_dir': r"data/features/finetun_virchow2/ep1/teacher/internal",
            'external_h5_dir': [r"data/features/finetun_virchow2/ep19/teacher/external_a",
                                r"data/features/finetun_virchow2/ep19/teacher/external_b"]
        },
        'finetune_gigapath_S_early': {
            'dim': 1536,
            'h5_dir': r"data/features/finetune_gigapath/Early/student/internal",
            'external_h5_dir': [r"data/features/finetune_gigapath/Early/student/external_a",
                                r"data/features/finetune_gigapath/Early/student/external_b"]
        },
        'finetune_gigapath_T_early': {
            'dim': 1536,
            'h5_dir': r"data/features/finetune_gigapath/Early/teacher/internal",
            'external_h5_dir': [r"data/features/finetune_gigapath/Early/teacher/external_a",
                                r"data/features/finetune_gigapath/Early/teacher/external_b"]
        },
        'finetune_gigapath_S_mid': {
            'dim': 1536,
            'h5_dir': r"data/features/finetune_gigapath/Mid/student/internal",
            'external_h5_dir': [r"data/features/finetune_gigapath/Mid/student/external_a",
                                r"data/features/finetune_gigapath/Mid/student/external_b"]
        },
        'finetune_gigapath_T_mid': {
            'dim': 1536,
            'h5_dir': r"data/features/finetune_gigapath/Mid/teacher/internal",
            'external_h5_dir': [r"data/features/finetune_gigapath/Mid/teacher/external_a",
                                r"data/features/finetune_gigapath/Mid/teacher/external_b"]
        },
        'finetune_gigapath_S_Late': {
            'dim': 1536,
            'h5_dir': r"data/features/finetune_gigapath/Late/student/internal",
            'external_h5_dir': [r"data/features/finetune_gigapath/Late/student/external_a",
                                r"data/features/finetune_gigapath/Late/student/external_b"]
        },
        'finetune_gigapath_T_Late': {
            'dim': 1536,
            'h5_dir': r"data/features/finetune_gigapath/Late/teacher/internal",
            'external_h5_dir': [r"data/features/finetune_gigapath/Late/teacher/external_a",
                                r"data/features/finetune_gigapath/Late/teacher/external_b"]
        }
    }
    
    # MIL aggregator configurations
    MIL_MODEL_CONFIGS = {
        'WiKG': {
            'class': 'model.WiKG.WiKG',
            'params': {
                'n_classes': 4,
                'in_dim': 1024,
                'hidden_dim': 512,
                'topk': 6,
                'agg_type': 'bi-interaction',
                'dropout': 0.3,
                'pool': 'attn'
            }
        },
        'TransMIL': {
            'class': 'model.TransMIL.TransMIL',
            'params': {
                'n_classes': 4,
                'in_dim': 1024,
                'hidden_dim': 1024,
                'num_heads': 8,
                'dim_head': 64,
                'num_landmarks': 256,
                'pinv_iterations': 6,
                'attn_dropout': 0.3,
                'fc_dropout': 0.3
            }
        },
        'CLAM': {
            'class': 'model.CLAM.CLAM',
            'params': {
                'n_classes': 4,
                'in_dim': 1024,
                'hidden_dim': 512,
                'variant': 'mb',
                'gate': True,  # Use gated attention.
                'size_arg': 'small',  # Attention network size: small or big
                'clam_dropout': 0.25,  # Dropout within the bag network
                'fc_dropout': 0.1,  # Dropout after the input projection
                'k_sample': 8,  # Number of samples for instance-level learning
                'subtyping': False  # Enable subtype-specific instance learning.
            }
        },
        'ABMIL': {
            'class': 'model.ABMIL.ABMIL',
            'params': {
                'n_classes': 4,
                'in_dim': 1024,
                'hidden_dim': 1024,
                'dropout': 0.3,
                'is_norm': True
            }
        }
    }

    def __init__(self, config):
        # Use the named MIL preset or the explicit model configuration.
        self._auto_configure_mil_model(config)
        # Apply the selected encoder's settings.
        self._auto_configure_feature_model(config)
        
        # Configure deterministic execution.
        self._set_deterministic(config['seed'])
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestamp = config.get('run_timestamp') or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Read the warmup settings.
        self.warmup_config = config.get('warmup', {'enabled': False})
        self.warmup_enabled = self.warmup_config.get('enabled', False)
        self.warmup_epochs = self.warmup_config.get('warmup_epochs', 5)
        self.warmup_start_lr = self.warmup_config.get('warmup_start_lr', 1e-6)
        self.warmup_method = self.warmup_config.get('warmup_method', 'linear')

        # Build the configured model architecture.
        try:
            self.model_class = self._parse_model_class(config['model']['class'])
            self.model_params = config['model']['params']
        except (ValueError, KeyError) as e:
            error_msg = f"Model configuration error: {str(e)}"
            print(f"[Critical Error] {error_msg}")
            raise ValueError(error_msg) from e

        # Initialize the feature dataset.
        self.dataset = H5Dataset(
            h5_dir=self.config['h5_dir'],
            excel_file=self.config['excel_file'],
            class_names=self.config['class_names']
        )

        # Create run output directories.
        append_timestamp_subdir = config.get('append_timestamp_subdir', True)
        if append_timestamp_subdir:
            self.save_root = os.path.join(config['save_dir'], self.timestamp)
            self.log_root = os.path.join(config['log_dir'], self.timestamp)
        else:
            self.save_root = config['save_dir']
            self.log_root = config['log_dir']
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.log_root, exist_ok=True)

        # Open the TensorBoard writer.
        self.writer = SummaryWriter(log_dir=self.log_root)

        # Save the resolved experiment configuration.
        with open(os.path.join(self.save_root, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, default=str)

        # Initialize class weights.
        self.init_class_weights = self._calculate_class_weights()
        self.class_weights = self.init_class_weights.clone()

        # Initialize result storage.
        self.result_saver = ResultSaver(
            config=config,
            device=self.device,  # Record the compute device.
            save_root=self.save_root,
            timestamp=self.timestamp
        )

        # Configure gradient clipping.
        self.grad_clip = config.get('grad_clip', 1.0)  # Default clipping norm

        # Read label-smoothing options.
        self.label_smoothing_config = config.get('label_smoothing', {'enabled': False})
        self.label_smoothing_enabled = self.label_smoothing_config.get('enabled', False)
        self.smoothing_epsilon = self.label_smoothing_config.get('epsilon', 0.1)
        self.adaptive_smoothing = self.label_smoothing_config.get('adaptive', False)
        self.adaptive_method = self.label_smoothing_config.get('adaptive_method', 'confidence')
        self.min_epsilon = self.label_smoothing_config.get('min_epsilon', 0.05)
        self.max_epsilon = self.label_smoothing_config.get('max_epsilon', 0.2)

        # Initialize adaptive smoothing state.
        if self.adaptive_smoothing:
            self.class_confidence_history = {cls: [] for cls in self.config['class_names']}
            self.adaptive_epsilon_per_class = {cls: self.smoothing_epsilon for cls in self.config['class_names']}

        # Accept one or more external validation sets.
        self.external_sets, self.external_sets_independent = self._parse_external_sets(self.config)

    def _auto_configure_mil_model(self, config):
        """Configure the MIL aggregator selected by mil_model."""
        mil_model = config.get('mil_model')
        if not mil_model:
            if 'model' not in config:
                raise ValueError('Provide mil_model or an explicit model configuration.')
            return

        if mil_model not in self.MIL_MODEL_CONFIGS:
            available_models = list(self.MIL_MODEL_CONFIGS.keys())
            raise ValueError(f"Unsupported mil_model: {mil_model}. Supported models: {available_models}")

        config['model'] = copy.deepcopy(self.MIL_MODEL_CONFIGS[mil_model])
        print(f"[config] Selected MIL aggregator: {mil_model}")

    def _auto_configure_feature_model(self, config):
        """Set the input dimensions and feature paths for the selected encoder."""
        feature_model = config.get('feature_model')
        if not feature_model:
            print('[config] No feature_model was provided. Using the configured feature paths.')
            return
            
        if feature_model not in self.FEATURE_MODEL_CONFIGS:
            available_models = list(self.FEATURE_MODEL_CONFIGS.keys())
            raise ValueError(f"Unsupported feature_model: {feature_model}. Supported encoders: {available_models}")
        
        model_config = self.FEATURE_MODEL_CONFIGS[feature_model]
        
        # Set the input width from the encoder preset.
        if 'model' in config and 'params' in config['model']:
            config['model']['params']['in_dim'] = model_config['dim']
            print(f"[config] Model input width: {model_config['dim']}")
        
        # Use the default training feature path when no override is supplied.
        if not config.get('h5_dir'):
            config['h5_dir'] = model_config['h5_dir']
            print(f"[config] Training HDF5 directory: {model_config['h5_dir']}")
        
        # Use the default external feature paths when no override is supplied.
        if not config.get('external_h5_dir'):
            config['external_h5_dir'] = model_config['external_h5_dir']
            print(f"[config] External HDF5 directories: {model_config['external_h5_dir']}")
        
        print(f"[config] Selected feature encoder: {feature_model}")

    def _parse_external_sets(self, config):
        """Normalize string or list inputs into external feature and label pairs and return the separate-evaluation setting."""
        h5 = config.get('external_h5_dir')
        xl = config.get('external_excel')

        # Evaluate external datasets separately by default.
        independent = config.get('external_sets_independent', True)

        if not h5 or not xl:
            return [], independent

        # Normalize paths to lists.
        if isinstance(h5, str):
            h5_list = [h5]
        else:
            h5_list = list(h5)
        if isinstance(xl, str):
            xl_list = [xl]
        else:
            xl_list = list(xl)

        # Require a label workbook for each external feature path.
        if len(h5_list) != len(xl_list):
            raise ValueError(f"external_h5_dir and external_excel have different lengths: {len(h5_list)} vs {len(xl_list)}")

        sets = []
        for i in range(len(h5_list)):
            sets.append({'h5_dir': h5_list[i], 'excel_file': xl_list[i]})
        return sets, independent

    def _create_label_smoothing_loss(self):
        """Build the configured label-smoothing loss."""
        if not self.label_smoothing_enabled:
            return torch.nn.CrossEntropyLoss(weight=self.class_weights)

        if self.adaptive_smoothing:
            # Use the adaptive smoothing loss.
            return self._adaptive_label_smoothing_loss
        else:
            # Use uniform label smoothing.
            return self._uniform_label_smoothing_loss

    def _uniform_label_smoothing_loss(self, outputs, targets):
        """Compute cross-entropy with uniform label smoothing."""
        log_probs = F.log_softmax(outputs, dim=1)
        num_classes = outputs.size(1)

        # Construct smoothed targets.
        smooth_targets = torch.zeros_like(log_probs).to(self.device)
        smooth_targets.fill_(self.smoothing_epsilon / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing_epsilon)

        # Compute the weighted loss.
        if self.class_weights is not None:
            # Apply the class weight for each sample.
            sample_weights = self.class_weights[targets]
            loss = -(smooth_targets * log_probs).sum(dim=1) * sample_weights
            return loss.mean()
        else:
            return -(smooth_targets * log_probs).sum(dim=1).mean()

    def _adaptive_label_smoothing_loss(self, outputs, targets):
        """Compute cross-entropy with adaptive label smoothing."""
        log_probs = F.log_softmax(outputs, dim=1)
        probs = F.softmax(outputs, dim=1)
        num_classes = outputs.size(1)

        # Compute an adaptive epsilon for each sample.
        adaptive_epsilons = []
        for i, target in enumerate(targets):
            target_class = self.config['class_names'][target.item()]

            if self.adaptive_method == 'confidence':
                # Adjust smoothing by prediction confidence.
                confidence = probs[i, target].item()
                epsilon = self.max_epsilon * (1 - confidence)
            elif self.adaptive_method == 'entropy':
                # Adjust smoothing by predictive entropy.
                entropy = -(probs[i] * torch.log(probs[i] + 1e-8)).sum().item()
                max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32)).item()
                normalized_entropy = entropy / max_entropy
                epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * normalized_entropy
            elif self.adaptive_method == 'loss_based':
                # Use the unsmoothed loss to adjust smoothing.
                original_loss = F.cross_entropy(outputs[i:i + 1], targets[i:i + 1], reduction='none').item()
                epsilon = min(self.max_epsilon, self.min_epsilon + original_loss * 0.1)
            else:
                epsilon = self.smoothing_epsilon

            # Restrict epsilon to the configured bounds.
            epsilon = max(self.min_epsilon, min(self.max_epsilon, epsilon))
            adaptive_epsilons.append(epsilon)

        # Build adaptively smoothed targets.
        smooth_targets = torch.zeros_like(log_probs).to(self.device)
        for i, (target, epsilon) in enumerate(zip(targets, adaptive_epsilons)):
            smooth_targets[i].fill_(epsilon / (num_classes - 1))
            smooth_targets[i, target] = 1.0 - epsilon

        # Compute the weighted loss.
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]
            loss = -(smooth_targets * log_probs).sum(dim=1) * sample_weights
            return loss.mean()
        else:
            return -(smooth_targets * log_probs).sum(dim=1).mean()

    def _update_class_confidence_history(self, outputs, targets):
        """Track class confidence for adaptive label smoothing."""
        if not self.adaptive_smoothing:
            return

        probs = F.softmax(outputs, dim=1)
        for i, target in enumerate(targets):
            class_name = self.config['class_names'][target.item()]
            confidence = probs[i, target].item()

            # Keep a sliding confidence window.
            if len(self.class_confidence_history[class_name]) >= 100:
                self.class_confidence_history[class_name].pop(0)
            self.class_confidence_history[class_name].append(confidence)

            # Update the adaptive epsilon.
            if len(self.class_confidence_history[class_name]) >= 10:
                avg_confidence = np.mean(self.class_confidence_history[class_name][-20:])
                self.adaptive_epsilon_per_class[class_name] = self.max_epsilon * (1 - avg_confidence)
                self.adaptive_epsilon_per_class[class_name] = max(
                    self.min_epsilon,
                    min(self.max_epsilon, self.adaptive_epsilon_per_class[class_name])
                )

    def _record_label_smoothing_metrics(self, fold_idx, epoch):
        """Log label-smoothing metrics to TensorBoard."""
        if not self.label_smoothing_enabled or not self.adaptive_smoothing:
            return

        # Log the per-class smoothing coefficients.
        for class_name in self.config['class_names']:
            if class_name in self.adaptive_epsilon_per_class:
                epsilon_value = self.adaptive_epsilon_per_class[class_name]
                self.writer.add_scalar(
                    f'fold{fold_idx}-adaptive_smoothing/{class_name}/epsilon',
                    epsilon_value,
                    epoch
                )

            # Log the average class confidence.
            if (class_name in self.class_confidence_history and
                    len(self.class_confidence_history[class_name]) > 0):
                avg_confidence = np.mean(self.class_confidence_history[class_name][-10:])
                self.writer.add_scalar(
                    f'fold{fold_idx}-adaptive_smoothing/{class_name}/avg_confidence',
                    avg_confidence,
                    epoch
                )

    def _feature_mixup(self, features):
        """Apply Mixup to the input features."""
        if len(features) < 2:
            return features
        lam = np.random.beta(0.4, 0.4)
        index = torch.randperm(features.size(0)).to(features.device)
        mixed = lam * features + (1 - lam) * features[index]
        return mixed

    def _parse_model_class(self, class_path):
        """Import the configured model class dynamically."""
        try:
            from importlib import import_module
            if not isinstance(class_path, str) or not class_path:
                raise ValueError(f"Invalid class path: {class_path}")

            if '.' not in class_path:
                raise ValueError(f"Class path must contain module and class name: {class_path}")

            module_path, class_name = class_path.rsplit('.', 1)
            module = import_module(module_path)
            model_class = getattr(module, class_name)

            # Validate the imported class.
            if not isinstance(model_class, type):
                raise ValueError(f"'{class_name}' is not a class in module '{module_path}'")

            return model_class
        except (ImportError, AttributeError, ValueError) as e:
            error_msg = f"Failed to parse model class '{class_path}': {str(e)}"
            print(f"[Error] {error_msg}")
            raise ValueError(error_msg) from e

    def _set_deterministic(self, seed):
        """Set random seeds and deterministic runtime options."""
        # Set the base random seed.
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Configure GPU seeding and determinism.
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Some PyTorch versions lack these settings.
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except AttributeError:
                pass
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # Warn when PyTorch uses an operation without a deterministic implementation.
        # WiKG's GlobalAttention uses scatter_reduce, which can trigger this warning.
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _calculate_class_weights(self):
        """Initialize class weights from the inverse class frequencies."""
        labels = self.dataset.label_codes
        class_counts = np.bincount(labels)
        class_counts += 1  # Laplace smoothing
        return torch.tensor(1.0 / class_counts, dtype=torch.float32).to(self.device)

    def _update_dynamic_class_weights(self, epoch, val_metrics):
        """Use fixed weights early, then transition to balanced weights. Keep val_metrics for caller compatibility."""
        early_epoch = self.config.get('dynamic_weight_early_epoch', 5)
        mid_epoch = self.config.get('dynamic_weight_mid_epoch', 10)

        # Keep class weights fixed during the early epochs.
        if epoch < early_epoch:
            self.class_weights = self.init_class_weights.clone()

        # Interpolate to balanced weights, then keep them fixed.
        else:
            decay_factor = max(0.0, 1 - (epoch - early_epoch) / max((mid_epoch - early_epoch), 1))
            self.class_weights = self.init_class_weights * decay_factor + (1.0 - decay_factor)

    def _create_loaders(self, train_idx, val_idx, fold_idx=None):
        train_subset = Subset(self.dataset, train_idx)
        val_subset = Subset(self.dataset, val_idx)

        # DataLoader settings
        num_workers = 0
        pin_memory = True
        persistent_workers = True

        # Use a fold-specific generator so training order is independent of other loaders.
        train_generator = torch.Generator()
        try:
            base_seed = int(self.config.get('seed', 42))
        except Exception:
            base_seed = 42
        if fold_idx is None:
            fold_seed = base_seed
        else:
            fold_seed = base_seed + 1000 * int(fold_idx)
        train_generator.manual_seed(fold_seed)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            generator=train_generator
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return train_loader, val_loader

    def _train_step(self, model, optimizer, batch, collect_data=False):
        features, _, _, _, masks, file_ids = batch
        features = features.to(self.device)
        masks = masks.to(self.device)
        labels = batch[2].to(self.device)

        # Apply feature augmentation when enabled.
        if self.config.get('use_feature_aug', False):
            features = self._feature_mixup(features)

        optimizer.zero_grad()
        # Build the training loss.
        criterion = self._create_label_smoothing_loss()
        
        # Run the forward pass with AMP when enabled.
        if self.config.get('amp_enabled', True):
            try:
                with torch.cuda.amp.autocast():
                    outputs = model(features, masks)
                    if self.label_smoothing_enabled:
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)
            except AttributeError:
                # Use full precision if AMP is unavailable.
                outputs = model(features, masks)
                if self.label_smoothing_enabled:
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)
        else:
            outputs = model(features, masks)
            if self.label_smoothing_enabled:
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
        
        # Update confidence history for adaptive smoothing.
        if self.label_smoothing_enabled and self.adaptive_smoothing:
            with torch.no_grad():
                self._update_class_confidence_history(outputs, labels)

        loss.backward()

        # Clip model gradients when configured.
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        except AttributeError:
            # Fall back to clip_grad_norm for older PyTorch versions.
            torch.nn.utils.clip_grad_norm(model.parameters(), self.grad_clip)
        optimizer.step()
        
        # Optionally return raw logits and probabilities.
        if collect_data:
            with torch.no_grad():
                # Keep the raw logits.
                train_logits = outputs.cpu().numpy()
                # Compute softmax probabilities.
                train_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                train_preds = np.argmax(train_probs, axis=1)
                
                return {
                    'loss': loss.item(),
                    'logits': train_logits,
                    'probs': train_probs,
                    'predictions': train_preds,
                    'labels': labels.cpu().numpy(),
                    'file_ids': file_ids
                }
        
        return loss.item()

    def _validate(self, model, data_loader, fold_idx, is_internal=True, epoch=None):
        """Evaluate the model on the validation loader."""
        model.eval()
        all_true, all_pred, all_scores, all_logits, all_file_ids = [], [], [], [], []
        total_loss = 0.0

        # Use standard cross-entropy for validation.
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)

        # Initialize validation progress.
        val_iter = tqdm(
            data_loader,
            desc="Internal Validating" if is_internal else "External Validating",
            unit="batch",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar:20}{r_bar}",
            leave=False
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iter):
                # Load the validation batch.
                features, coords, labels, lens, masks, file_ids = batch
                features = features.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                # Run a forward pass.
                outputs = model(features, masks)
                loss = criterion(outputs, labels)  # Validation uses standard cross-entropy.
                total_loss += loss.item()

                # Keep the raw logits.
                logits = outputs.cpu().numpy()
                # Compute softmax probabilities.
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                # Collect validation results.
                all_true.extend(labels.cpu().numpy().tolist())
                all_pred.extend(preds.tolist())
                all_scores.append(probs)
                all_logits.append(logits)
                all_file_ids.extend(file_ids)

                # Update validation progress.
                val_iter.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
                })

        # Convert predictions to NumPy arrays.
        all_true = [int(x) for x in all_true]
        all_pred = [int(x) for x in all_pred]
        all_scores = np.vstack(all_scores)
        all_logits = np.vstack(all_logits)
        avg_loss = total_loss / len(data_loader)

        # Compute validation metrics.
        overall_metrics = compute_overall_metrics(
            all_true, all_pred, all_scores,
            self.config['class_names']
        )
        overall_metrics['loss'] = avg_loss

        classwise_metrics = compute_classwise_metrics(
            all_true, all_pred, all_scores,
            self.config['class_names']
        )

        # Save plots for the best epoch or after a fold to reduce training I/O.

        return {
            'overall': overall_metrics,
            'class_wise': classwise_metrics,
            'y_true': all_true,
            'y_pred': all_pred,
            'file_ids': all_file_ids,
            'scores': all_scores,
            'logits': all_logits,
            # Include the data required for ensemble evaluation.
            'predictions': all_pred,
            'probabilities': all_scores,
            'labels': all_true
        }

    def _train_fold(self, model, optimizer, scheduler, train_loader, val_loader, fold_idx):
        """Train one cross-validation fold."""
        raw_metric = self.config.get('early_stop_metric', 'auc')
        early_stop_metric = self.METRIC_MAPPING.get(raw_metric, 'auc_macro')  # Early-stopping metric

        # Initialize the metric comparison direction.
        if early_stop_metric == 'loss':
            best_metric_value = float('inf')
            compare_op = lambda x, y: x < y  # Lower is better.
        else:
            best_metric_value = 0.0
            compare_op = lambda x, y: x > y  # Higher is better.

        epochs_no_improve = 0
        best_model_state = None
        best_val_metrics = None
        best_ext_metrics = None  # Track the best external results.
        best_epoch = -1

        # Prepare external validation when configured.
        external_validator = None
        has_external_validation = len(self.external_sets) > 0

        # Set up per-epoch external evaluation.
        if has_external_validation and len(self.external_sets) == 1:
            # Evaluate a single external cohort per epoch to bound training overhead.
            external_validator = self._prepare_external_validator(fold_idx, self.external_sets[0])

        # Log the label-smoothing mode.
        if self.label_smoothing_enabled:
            if self.adaptive_smoothing:
                print(
                    f"Adaptive label smoothing with method {self.adaptive_method} and range [{self.min_epsilon:.3f}, {self.max_epsilon:.3f}]")
            else:
                print(f"Uniform label smoothing with epsilon {self.smoothing_epsilon:.3f}")
        else:
            print('Label smoothing is disabled.')

        if has_external_validation:
            print('Running internal and external validation after each epoch.')

        for epoch in range(self.config['max_epochs']):
            # Update the warmup learning rate.
            if self.warmup_enabled and epoch < self.warmup_epochs:
                self._update_warmup_lr(optimizer, epoch)
                print(f"Warm-up Epoch {epoch + 1}/{self.warmup_epochs}, LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Training
            model.train()
            train_loss = 0.0

            # Initialize the batch progress bar.
            train_iter = tqdm(
                train_loader,
                desc=f"Train Epoch {epoch + 1}/{self.config['max_epochs']}",
                unit="batch",
                dynamic_ncols=True,
                leave=False
            )

            for batch_idx, batch in enumerate(train_iter):
                # Train on one batch without collecting predictions.
                loss = self._train_step(model, optimizer, batch)
                train_loss += loss

                # Update the batch progress display.
                train_iter.set_postfix({
                    'batch_loss': f"{loss:.4f}",
                    'avg_loss': f"{train_loss / (batch_idx + 1):.4f}"
                })

            # Log training loss to TensorBoard.
            train_loss_avg = train_loss / len(train_loader)
            self.writer.add_scalar(f'fold{fold_idx}-train/loss', train_loss_avg, epoch + 1)

            # Log the current learning rate.
            current_lr = optimizer.param_groups[0]['lr']
            self.writer.add_scalar(f'fold{fold_idx}-train/loss', train_loss_avg, epoch + 1)
            self.writer.add_scalar(f'fold{fold_idx}-val/lr', current_lr, epoch + 1)

            # Mark epochs that are still in warmup.
            if self.warmup_enabled and epoch < self.warmup_epochs:
                self.writer.add_scalar(f'fold{fold_idx}-warmup/lr', current_lr, epoch + 1)

            # Log label-smoothing metrics.
            self._record_label_smoothing_metrics(fold_idx, epoch + 1)
            # Internal validation
            val_metrics = self._validate(model, val_loader, fold_idx, epoch=epoch, is_internal=True)
            # Log internal validation metrics.
            self._record_tensorboard_metrics(fold_idx, val_metrics, epoch + 1, prefix='val')

            # External validation
            external_metrics = None
            if has_external_validation and external_validator:
                try:
                    # Run the existing external validator.
                    external_metrics = self._validate_external_per_epoch(external_validator, model, fold_idx, epoch)

                    # Log external validation metrics.
                    if external_metrics:
                        self._record_tensorboard_metrics(fold_idx, external_metrics, epoch + 1, prefix='ext')
                except Exception as e:
                    print(f"Per-epoch external validation failed: {str(e)}")
                    external_metrics = None

            current_metric = val_metrics['overall'].get(early_stop_metric, 0.0)
            train_iter.close()

            # Update dynamic class weights.
            if self.config.get('use_dynamic_class_weight', False):
                self._update_dynamic_class_weights(epoch, val_metrics)

            # Print the epoch summary.
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_loss_avg:.4f}")
            print(f"Val Loss: {val_metrics['overall']['loss']:.4f}")
            for cls in self.config['class_names']:
                auc = val_metrics['class_wise'][cls].get('auc', 0.0)
                print(f"{cls} AUC: {auc:.4f}")
            # Report overall metrics.
            print(f"Overall Balanced Accuracy: {val_metrics['overall']['balanced_accuracy']:.4f}")
            print(f"Overall Accuracy: {val_metrics['overall'].get('accuracy', 0.0):.4f}")
            print(f"Overall AUC Macro: {val_metrics['overall']['auc_macro']:.4f}")

            # Early stopping
            is_in_warmup = self.warmup_enabled and epoch < self.warmup_epochs

            if compare_op(current_metric, best_metric_value):
                best_metric_value = current_metric
                epochs_no_improve = 0
                best_val_metrics = val_metrics.copy()
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
                if has_external_validation and external_metrics:
                    best_ext_metrics = external_metrics.copy()

                # Record the best epoch.
                if self.result_saver:
                    self.result_saver.best_epochs[fold_idx] = best_epoch + 1

                # Save the best model checkpoint.
                torch.save(best_model_state, os.path.join(self.save_root, f"fold_{fold_idx}_best.pth"))
            else:
                if not is_in_warmup:  # Count epochs toward early stopping only after warmup.
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.config['patience']:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        model.load_state_dict(best_model_state)
                        best_val_metrics = self._validate(model, val_loader, fold_idx)

                        # Save the selected checkpoint before terminating the fold.
                        torch.save(best_model_state, os.path.join(self.save_root, f"fold_{fold_idx}_best.pth"))

                        # Evaluate the external data again.
                        if has_external_validation and external_validator:
                            best_ext_metrics = self._validate_external_per_epoch(
                                external_validator, model, fold_idx, best_epoch)
                        break

            # Update the learning-rate schedule.
            if not is_in_warmup:
                self._update_scheduler(scheduler, best_val_metrics, epoch, fold_idx)

        # Finalize the completed fold.
        print(f"\n=== Fold {fold_idx} Best Model ===")
        if self.result_saver:
            print(f"Best Epoch: {self.result_saver.best_epochs[fold_idx]}")
        else:
            print(f"Best Epoch: {best_epoch + 1}")
        if best_val_metrics is not None:
            print(f"Best Val {early_stop_metric}: {best_val_metrics['overall'][early_stop_metric]:.4f}")
            print(f"Val AUC: {best_val_metrics['overall']['auc_macro']:.4f}")
        print(f"Final Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save results from the best epoch.
        if self.result_saver:
            self.result_saver.save_fold_best_results(
                metrics=best_val_metrics,
                fold_idx=fold_idx,
                best_epoch=self.result_saver.best_epochs[fold_idx]
            )

            # Record validation predictions.
            self.result_saver.record_validation_data(
                metrics=best_val_metrics,
                fold_idx=fold_idx,
                dataset_type='internal'
            )

        # Predict the training set again using the best checkpoint.
        train_data = {}
        if best_model_state is not None:
            print(f"Predicting the training set with the best checkpoint (epoch {best_epoch + 1})...")
            model.load_state_dict(best_model_state)
            train_data = self._predict_with_best_model(model, train_loader, fold_idx, dataset_type='train')

        if best_val_metrics is not None:
            return {
                'overall': best_val_metrics['overall'],
                'class_wise': best_val_metrics['class_wise'],
                'y_true': best_val_metrics['y_true'],
                'y_pred': best_val_metrics['y_pred'],
                'scores': best_val_metrics['scores'],
                'logits': best_val_metrics['logits'],
                'file_ids': best_val_metrics['file_ids'],
                'train_data': train_data
            }
        else:
            # Return an empty result if there are no validation metrics.
            return {
                'overall': {},
                'class_wise': {},
                'y_true': [],
                'y_pred': [],
                'scores': np.array([]),
                'logits': np.array([]),
                'file_ids': [],
                'train_data': train_data
            }

    # External validator initialization
    def _prepare_external_validator(self, fold_idx, set_pair=None):
        """Initialize an external validator for per-epoch evaluation on the requested set."""
        try:
            # Create the validator for the selected external cohort.
            external_validator = ExternalValidator(
                model_class=self.model_class,
                model_params=self.model_params,
                model_path=None,  # Initialize the model without restoring a checkpoint.
                h5_dir=(set_pair['h5_dir'] if set_pair else self.config['external_h5_dir']),
                excel_file=(set_pair['excel_file'] if set_pair else self.config['external_excel']),
                class_names=self.config['class_names'],
                device=self.device.type
            )

            # Mark the validator as initialized without checkpoint weights.
            external_validator._initialize_model_only = True

            return external_validator
        except Exception as e:
            print(f"Could not initialize external validation: {str(e)}")
            return None

    # Per-epoch external validation
    def _validate_external_per_epoch(self, external_validator, model, fold_idx, epoch):
        """Evaluate the external sets for the current epoch without saving artifacts."""
        try:
            model.eval()  # Set the model to evaluation mode.

            # Update the existing validator with the current weights.
            external_validator.update_model_weights(model.state_dict())

            # Disable training behavior such as dropout and batch normalization updates.
            try:
                external_validator.model.eval()
            except Exception:
                pass

            # Use a temporary output path for this evaluation.
            temp_dir = os.path.join(self.save_root, "temp_external")
            os.makedirs(temp_dir, exist_ok=True)

            # Disable artifact writes and isolate random state from training.
            fork_devices = None
            if torch.cuda.is_available():
                try:
                    fork_devices = [torch.cuda.current_device()]
                except Exception:
                    fork_devices = None
            with torch.random.fork_rng(devices=fork_devices, enabled=True):
                with torch.no_grad():
                    all_true, all_pred, all_scores, all_logits, all_file_ids = [], [], [], [], []
                    total_loss = 0.0
                    criterion = torch.nn.CrossEntropyLoss()

                    # Use the external validation loader.
                    loader = external_validator.loader

                    progress = tqdm(
                        loader,
                        desc=f"Fold {fold_idx} Epoch {epoch + 1} External Validation",
                        unit="batch",
                        dynamic_ncols=True,
                        bar_format="{l_bar}{bar:20}{r_bar}",
                        leave=False
                    )

                    for batch in progress:
                        try:
                            # Unpack the batch.
                            features, _, labels, _, masks, file_ids = batch

                            # Move the inputs to the compute device.
                            features = features.to(self.device, non_blocking=True)
                            masks = masks.to(self.device, non_blocking=True)
                            labels = labels.to(self.device, non_blocking=True)

                            # Compute the loss.
                            outputs = external_validator.model(features, masks)
                            loss = criterion(outputs, labels)
                            total_loss += loss.item()

                            # Keep raw logits.
                            logits = outputs.cpu().numpy()
                            # Compute softmax probabilities.
                            probs = torch.softmax(outputs, dim=1).cpu().numpy()
                            preds = np.argmax(probs, axis=1)

                            # Collect this batch's outputs.
                            all_true.extend(labels.cpu().numpy().tolist())
                            all_pred.extend(preds.tolist())
                            all_scores.append(probs)
                            all_logits.append(logits)
                            all_file_ids.extend(file_ids)

                            # Update the progress bar.
                            progress.set_postfix({
                                'ext_loss': f"{loss.item():.4f}"
                            })

                            # Release batch intermediates.
                            del features, masks, labels, outputs, probs
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        except (RuntimeError, MemoryError) as mem_error:
                            print(f"Out of memory during external validation: {str(mem_error)}")
                            raise  # Propagate the exception.

                    # Compute external validation metrics.
                    if len(all_scores) > 0:
                        all_scores_array = np.vstack(all_scores)
                        all_logits_array = np.vstack(all_logits)

                        overall_metrics = compute_overall_metrics(
                            all_true, all_pred, all_scores_array,
                            self.config['class_names']
                        )
                        overall_metrics['loss'] = total_loss / len(external_validator.loader)

                        classwise_metrics = compute_classwise_metrics(
                            all_true, all_pred, all_scores_array,
                            self.config['class_names']
                        )

                        return {
                            'overall': overall_metrics,
                            'class_wise': classwise_metrics,
                            'y_true': all_true,
                            'y_pred': all_pred,
                            'scores': all_scores_array,
                            'logits': all_logits_array,
                            'file_ids': all_file_ids
                        }

                    return None

        except Exception as e:
            print(f"Per-epoch external validation failed: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    def _run_external_validation(self, fold_idx):
        """Evaluate the best model against external cohorts, separately or together as configured."""
        try:
            print(f"\n=== Running external validation (fold {fold_idx + 1}/{self.config['n_splits']}) ===")

            # Check that the model checkpoint exists.
            model_path = os.path.join(self.save_root, f"fold_{fold_idx}_best.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

            # Return early when no external dataset is configured.
            if len(self.external_sets) == 0:
                return None

            # Save the single dataset under external_validation/set_1.
            if len(self.external_sets) == 1:

                external_validator = ExternalValidator(
                    model_class=self.model_class,
                    model_params=self.model_params,
                    model_path=model_path,
                    h5_dir=self.external_sets[0]['h5_dir'],
                    excel_file=self.external_sets[0]['excel_file'],
                    class_names=self.config['class_names'],
                    device=self.device.type
                )

                external_metrics = external_validator.evaluate(
                    output_dir=None,
                    config_params=self.config,
                    save_results=False
                )

                required_keys = ['overall', 'class_wise', 'y_true', 'y_pred', 'scores', 'logits', 'file_ids']
                if not external_metrics or any(k not in external_metrics for k in required_keys):
                    error_msg = f"Fold {fold_idx} returned incomplete external validation results."
                    print(f"[Error] {error_msg}")
                    raise ValueError(error_msg)

                if self.result_saver:
                    self.result_saver.record_validation_data(
                        metrics={
                            'y_true': external_metrics.get('y_true', []),
                            'y_pred': external_metrics.get('y_pred', []),
                            'scores': external_metrics.get('scores', np.array([])),
                            'logits': external_metrics.get('logits', np.array([])),
                            'file_ids': external_metrics.get('file_ids', [])
                        },
                        fold_idx=fold_idx,
                        dataset_type='external'
                    )

                return {
                    'overall': external_metrics['overall'],
                    'class_wise': external_metrics['class_wise'],
                    'y_true': external_metrics['y_true'],
                    'y_pred': external_metrics['y_pred'],
                    'scores': external_metrics['scores'],
                    'file_ids': external_metrics['file_ids']
                }

            # Write per-cohort results and pooled results for multiple external cohorts.
            results = []

            # Collect outputs for pooled evaluation.
            all_true, all_pred, all_scores, all_logits, all_file_ids = [], [], [], [], []

            for idx, set_pair in enumerate(self.external_sets, start=1):
                

                external_validator = ExternalValidator(
                    model_class=self.model_class,
                    model_params=self.model_params,
                    model_path=model_path,
                    h5_dir=set_pair['h5_dir'],
                    excel_file=set_pair['excel_file'],
                    class_names=self.config['class_names'],
                    device=self.device.type
                )

                metrics_i = external_validator.evaluate(
                    output_dir=None,
                    config_params=self.config,
                    save_results=False
                )

                required_keys = ['overall', 'class_wise', 'y_true', 'y_pred', 'scores', 'logits', 'file_ids']
                if not metrics_i or any(k not in metrics_i for k in required_keys):
                    print(f"[warning] External dataset {idx} returned incomplete results. Skipping.")
                    results.append(None)
                    continue

                # Record the dataset identity with each result.
                if self.result_saver:
                    self.result_saver.record_validation_data(
                        metrics={
                            'y_true': metrics_i.get('y_true', []),
                            'y_pred': metrics_i.get('y_pred', []),
                            'scores': metrics_i.get('scores', np.array([])),
                            'logits': metrics_i.get('logits', np.array([])),
                            'file_ids': metrics_i.get('file_ids', [])
                        },
                        fold_idx=fold_idx,
                        dataset_type=f'external_set_{idx}'
                    )

                results.append({
                    'overall': metrics_i['overall'],
                    'class_wise': metrics_i['class_wise'],
                    'y_true': metrics_i['y_true'],
                    'y_pred': metrics_i['y_pred'],
                    'scores': metrics_i['scores'],
                    'file_ids': metrics_i['file_ids']
                })

                # Accumulate predictions for pooling.
                if metrics_i.get('y_true'):
                    all_true.extend(metrics_i['y_true'])
                    all_pred.extend(metrics_i['y_pred'])
                    if isinstance(metrics_i.get('scores'), np.ndarray) and metrics_i['scores'].size > 0:
                        all_scores.append(metrics_i['scores'])
                    if isinstance(metrics_i.get('logits'), np.ndarray) and metrics_i['logits'].size > 0:
                        all_logits.append(metrics_i['logits'])
                    all_file_ids.extend(metrics_i.get('file_ids', []))

            # Compute and save pooled metrics.
            aggregated = None
            if len(all_true) > 0:
                scores_concat = np.vstack(all_scores) if len(all_scores) > 0 else np.array([])
                logits_concat = np.vstack(all_logits) if len(all_logits) > 0 else np.array([])
                if isinstance(logits_concat, np.ndarray) and logits_concat.size > 0 and logits_concat.shape[0] != len(all_file_ids):
                    logits_concat = np.array([])

                overall_metrics = compute_overall_metrics(all_true, all_pred, scores_concat, self.config['class_names'])
                classwise_metrics = compute_classwise_metrics(all_true, all_pred, scores_concat, self.config['class_names'])

                if self.result_saver:
                    self.result_saver.record_validation_data(
                        metrics={
                            'y_true': all_true,
                            'y_pred': all_pred,
                            'scores': scores_concat,
                            'logits': logits_concat,
                            'file_ids': all_file_ids
                        },
                        fold_idx=fold_idx,
                        dataset_type='external_merged'
                    )

                aggregated = {
                    'overall': overall_metrics,
                    'class_wise': classwise_metrics,
                    'y_true': all_true,
                    'y_pred': all_pred,
                    'scores': scores_concat,
                    'file_ids': all_file_ids
                }

            return {
                'external': aggregated if aggregated else None,
                'external_sets': results
            }

        except FileNotFoundError as fnf_error:
            error_msg = f"Model checkpoint missing: {str(fnf_error)}"
        except ValueError as ve:
            error_msg = f"Data validation failed: {str(ve)}"
        except Exception as e:
            error_msg = f"Unhandled error: {str(e)}"
            import traceback
            error_msg += f"\nStack Trace:\n{traceback.format_exc()}"
        else:
            # Continue when validation succeeds.
            pass

        # Handle any captured validation error.
        if 'error_msg' in locals():
            # Use a common error-handling path.
            print(f"\n[Critical Error] {error_msg}")

            # Log the error to the experiment outputs.
            if self.result_saver:
                try:
                    self.result_saver.log_error(
                        error_type="External Validation",
                        fold_idx=fold_idx,
                        message=error_msg
                    )
                except AttributeError:
                    # Use console output when no result logger is available.
                    print(f"[Error] External Validation failed for fold {fold_idx}: {error_msg}")

            return None

    def _record_tensorboard_metrics(self, fold_idx, metrics, epoch, prefix='val'):
        """Record internal and external validation metrics in separate TensorBoard groups."""
        # Select the TensorBoard group from the prefix.
        tab_name = f"fold{fold_idx}-{prefix}"

        # Log overall metrics.
        self.writer.add_scalar(f'{tab_name}/loss', metrics['overall']['loss'], epoch)
        self.writer.add_scalar(f'{tab_name}/accuracy', metrics['overall'].get('accuracy', 0.0), epoch)
        self.writer.add_scalar(f'{tab_name}/balanced_accuracy', metrics['overall']['balanced_accuracy'], epoch)
        self.writer.add_scalar(f'{tab_name}/auc_macro', metrics['overall']['auc_macro'], epoch)
        self.writer.add_scalar(f'{tab_name}/auc_micro', metrics['overall']['auc_micro'], epoch)

        # Log the remaining overall metrics.
        self.writer.add_scalar(f'{tab_name}/f1_macro', metrics['overall']['f1_macro'], epoch)
        self.writer.add_scalar(f'{tab_name}/precision_macro', metrics['overall']['precision_macro'], epoch)
        self.writer.add_scalar(f'{tab_name}/recall_macro', metrics['overall']['recall_macro'], epoch)
        self.writer.add_scalar(f'{tab_name}/sensitivity_macro', metrics['overall']['sensitivity_macro'], epoch)
        self.writer.add_scalar(f'{tab_name}/specificity_macro', metrics['overall']['specificity_macro'], epoch)

        # Log per-class AUC.
        for cls in self.config['class_names']:
            cls_auc = metrics['class_wise'][cls].get('auc', 0.0)
            self.writer.add_scalar(f'{tab_name}/{cls}/auc', cls_auc, epoch)

    def _initialize_scheduler(self, optimizer):
        """Initialize the learning-rate scheduler from the experiment settings."""
        scheduler_config = self.config.get('scheduler', {'type': 'none'})
        scheduler_type = scheduler_config.get('type', 'none').lower()
        params = scheduler_config.get('params', {})

        if scheduler_type == 'cosine':
            # Cosine annealing
            # Adjust T_max when warmup is enabled.
            t_max = params.get('T_max', self.config['max_epochs'])
            if self.warmup_enabled:
                # Start cosine annealing after warmup.
                t_max = max(1, t_max - self.warmup_epochs)
            eta_min = params.get('eta_min', 1e-6)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min
            )
        elif scheduler_type == 'exponential':
            # Exponential decay
            gamma = params.get('gamma', 0.95)
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )
        elif scheduler_type == 'plateau':
            # Validation-driven plateau scheduling
            mode = params.get('mode', 'min')
            factor = params.get('factor', 0.1)
            patience = params.get('patience', 3)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience,
                verbose=True
            )
        else:
            # No learning-rate scheduler
            return None

    def _get_warmup_lr(self, epoch, base_lr):
        """Compute the learning rate for the zero-based warmup epoch and target rate."""
        if not self.warmup_enabled or epoch >= self.warmup_epochs:
            return base_lr

        progress = epoch / self.warmup_epochs  # Warmup fraction [0, 1)

        if self.warmup_method == 'linear':
            # Increase the learning rate linearly from start_lr to base_lr.
            lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
        elif self.warmup_method == 'cosine':
            # Increase the learning rate on a cosine curve.
            lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (1 - np.cos(progress * np.pi)) / 2
        else:
            # Use linear warmup by default.
            lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress

        return lr

    def _update_warmup_lr(self, optimizer, epoch):
        """Set the PyTorch optimizer's learning rate for the zero-based warmup epoch."""
        if not self.warmup_enabled or epoch >= self.warmup_epochs:
            return

        base_lr = self.config['lr']  # Target learning rate
        new_lr = self._get_warmup_lr(epoch, base_lr)

        # Update the learning rate for all parameter groups.
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def _update_scheduler(self, scheduler, val_metrics, epoch, fold_idx):
        """Update the scheduler using the current epoch, fold, and validation metrics."""
        # Skip scheduler updates during warmup.
        if self.warmup_enabled and epoch < self.warmup_epochs:
            return

        # Pass a metric when using ReduceLROnPlateau.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # Read the monitored validation metric.
            raw_scheduler_metric = self.config.get('scheduler', {}).get('params', {}).get('metric', 'loss')
            plateau_metric = self.METRIC_MAPPING.get(raw_scheduler_metric, 'loss')
            metric_value = val_metrics['overall'][plateau_metric]
            scheduler.step(metric_value)
        else:
            # Update epoch-based schedulers.
            # if hasattr(scheduler, 'last_epoch'):
            #     # Log the current learning rate for debugging.
            #     current_lr = scheduler.get_last_lr()[0]
            #     self.writer.add_scalar(f'fold{fold_idx}-val/learning_rate', current_lr, epoch)

            # Update the scheduler.
            scheduler.step()

    def _run_cross_validation_core(self):
        """Run the shared cross-validation loop."""
        import traceback
        import psutil
        import gc

        try:
            skf = StratifiedKFold(n_splits=self.config['n_splits'],
                                  shuffle=True,
                                  random_state=self.config['seed'])
            fold_results = []
            labels = self.dataset.label_codes

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(self.dataset)), labels)):
                print(f"\n=== Fold {fold_idx + 1}/{self.config['n_splits']} ===")

                # Record resource usage at the start of this fold.
                try:
                    if psutil:
                        memory_info = psutil.virtual_memory()
                        print(
                            f"Fold {fold_idx + 1} started. Memory usage: {memory_info.percent:.1f}%, available: {memory_info.available / (1024 ** 3):.1f}GB")
                except Exception:
                    pass

                try:
                    # Initialize the fold model.
                    model = self.model_class(**self.model_params).to(self.device)
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=self.config['lr'],
                        weight_decay=self.config.get('weight_decay', 1e-5),
                        betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999)),
                        eps=self.config.get('eps', 1e-8)
                    )

                    # Initialize the training scheduler.
                    scheduler = self._initialize_scheduler(optimizer)

                    # Create loaders with fold-specific sampling seeds.
                    train_loader, val_loader = self._create_loaders(train_idx, val_idx, fold_idx=fold_idx)

                    # Train the fold and collect internal validation results.
                    internal_metrics = self._train_fold(model, optimizer, scheduler, train_loader, val_loader, fold_idx)
                    fold_metrics = {'internal': internal_metrics}

                    # Run external validation when configured.
                    if len(self.external_sets) > 0:
                        try:
                            external_metrics = self._run_external_validation(fold_idx)

                            # Normalize the supported result formats.
                            # A list contains separate dataset results.
                            # A dictionary may contain a single external result.
                            # A combined dictionary contains external and external_sets.
                            if isinstance(external_metrics, list):
                                fold_metrics['external_sets'] = external_metrics
                            elif isinstance(external_metrics, dict):
                                if 'external' in external_metrics or 'external_sets' in external_metrics:
                                    if external_metrics.get('external'):
                                        fold_metrics['external'] = external_metrics['external']
                                    if external_metrics.get('external_sets'):
                                        fold_metrics['external_sets'] = external_metrics['external_sets']
                                elif all(key in external_metrics for key in ['overall', 'class_wise', 'y_true', 'y_pred', 'scores']):
                                    fold_metrics['external'] = external_metrics
                                else:
                                    print(f"[warning] Fold {fold_idx + 1} returned an unknown external_metrics format. Skipping.")
                            else:
                                print(f"[warning] Fold {fold_idx + 1} returned incomplete external_metrics. Skipping.")
                        except Exception as e:
                            error_msg = str(e)
                            print(f"External validation failed: {error_msg}")
                            print(f"Exception details: {traceback.format_exc()}")

                            # Record diagnostics for external validation failures.
                            if 'CUDA' in error_msg or 'GPU' in error_msg:
                                print(f"[diagnostic] GPU error. Check GPU memory usage.")
                            elif 'Memory' in error_msg or 'memory' in error_msg:
                                print(f"[diagnostic] Memory error. Check available RAM and batch_size.")
                            elif 'File' in error_msg or 'file' in error_msg:
                                print(f"[diagnostic] File access error. Check the external data paths.")

                    # Save the fold results.
                    fold_results.append(fold_metrics)
                    # Record training-set predictions.
                    if self.result_saver:
                        if 'internal' in fold_metrics and 'train_data' in fold_metrics['internal']:
                            self.result_saver.record_training_data(
                                fold_metrics['internal']['train_data'], fold_idx
                            )

                    # Release resources after the fold.
                    try:
                        del model, optimizer, scheduler, train_loader, val_loader
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                except Exception as fold_error:
                    error_msg = str(fold_error) if str(
                        fold_error).strip() else f"{type(fold_error).__name__}: no exception detail available"
                    print(f"[error] Fold {fold_idx + 1} failed: {error_msg}")
                    print(f"[details] {traceback.format_exc()}")

                    # Provide diagnostics based on the error type.
                    if 'CUDA' in error_msg or 'GPU' in error_msg:
                        print(f"[diagnostic] GPU error. Check device memory or retry on the CPU.")
                    elif 'Memory' in error_msg or 'OutOfMemory' in error_msg:
                        print(f"[diagnostic] Out of memory. Check batch_size, hidden_dim, and RAM.")
                    elif 'FileNotFound' in error_msg or 'No such file' in error_msg:
                        print(f"[diagnostic] File not found. Check the configured data paths.")
                    elif 'Permission' in error_msg or 'Access' in error_msg:
                        print(f"[diagnostic] Permission error. Check file access.")

                    # Record resource usage for diagnosis.
                    try:
                        if psutil:
                            memory_info = psutil.virtual_memory()
                            print(
                                f"[system] Memory usage: {memory_info.percent:.1f}%, available: {memory_info.available / (1024 ** 3):.1f}GB")
                            if torch.cuda.is_available():
                                print(
                                    f"[GPU] Memory usage: {torch.cuda.memory_allocated() / (1024 ** 3):.1f}GB / {torch.cuda.max_memory_allocated() / (1024 ** 3):.1f}GB")
                    except Exception:
                        pass

                    # Propagate the exception.
                    raise fold_error

            return fold_results

        except Exception as cv_error:
            error_msg = str(cv_error)
            print(f"[fatal] Cross-validation failed: {error_msg}")
            print(f"[details] {traceback.format_exc()}")

            # Classify the error for diagnostics.
            if 'CUDA' in error_msg:
                print(f"[diagnostic] CUDA error. Check the GPU driver or retry on the CPU.")
            elif 'Memory' in error_msg or 'OutOfMemory' in error_msg:
                print(f"[diagnostic] Out of system memory. Check free RAM and model size.")
            elif 'module' in error_msg and 'not found' in error_msg:
                print(f"[diagnostic] Module import failed. Check the project dependencies.")
            elif 'config' in error_msg.lower():
                print(f"[diagnostic] Configuration error. Check argument values and file format.")

            # Report final resource usage.
            try:
                if psutil:
                    memory_info = psutil.virtual_memory()
                    print(
                        f"[final system state] Memory usage: {memory_info.percent:.1f}%, available: {memory_info.available / (1024 ** 3):.1f}GB")
            except Exception:
                pass

            # Propagate the exception to the caller.
            raise cv_error

    def _predict_with_best_model(self, model, data_loader, fold_idx, dataset_type='train'):
        """Return logits, probabilities, labels, and file IDs from the model with the best weights for the requested fold and split."""
        model.eval()
        all_logits, all_probs, all_labels, all_file_ids = [], [], [], []
        
        print(f"Predicting the {dataset_type} set again (fold {fold_idx + 1}）...")
        
        with torch.no_grad():
            progress = tqdm(
                data_loader,
                desc=f"Best Model Prediction ({dataset_type})",
                unit="batch",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar:20}{r_bar}",
                leave=False
            )
            
            for batch in progress:
                # Unpack the batch.
                features, coords, labels, lens, masks, file_ids = batch
                features = features.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                
                # Run a forward pass.
                outputs = model(features, masks)
                
                # Extract logits and probabilities.
                logits = outputs.cpu().numpy()
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                # Collect prediction records.
                all_logits.append(logits)
                all_probs.append(probs)
                all_labels.extend(labels.cpu().numpy().tolist())
                all_file_ids.extend(file_ids)
        
        # Convert collected arrays to NumPy.
        all_logits = np.vstack(all_logits) if all_logits else np.array([])
        all_probs = np.vstack(all_probs) if all_probs else np.array([])
        
        print(f"{dataset_type} predictions completed for {len(all_labels)} samples.")
        
        return {
            'logits': all_logits,
            'probs': all_probs,
            'labels': all_labels,
            'file_ids': all_file_ids
        }

    def run(self):
        """Run cross-validation."""
        fold_results = self._run_cross_validation_core()

        # Write the final evaluation outputs.
        if self.result_saver:
            # Aggregate folds and evaluate the ensemble.
            ensemble_results = self.result_saver.aggregate_results(fold_results)

            # Save detailed validation results.
            self.result_saver.save_validation_details()
            
            # Save the combined logits and probabilities.
            self.result_saver.save_comprehensive_csv()

            # Append ensemble metrics when available.
            if ensemble_results and 'error' not in ensemble_results:
                ensemble_line = self.result_saver.generate_ensemble_validation_report_line(ensemble_results)
                if ensemble_line:
                    self._update_validation_report_with_ensemble(ensemble_line)

        # Close the TensorBoard writer.
        if self.writer:
            self.writer.close()

        return fold_results

    def _update_validation_report_with_ensemble(self, ensemble_line):
        """Append an ensemble evaluation row to the validation report."""
        try:
            # Resolve the report in the current run directory.
            report_file = os.path.join(self.save_root, 'validation_report.txt')

            if os.path.exists(report_file):
                # Read the existing report.
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Append the ensemble evaluation row.
                updated_content = content + '\n' + ensemble_line + '\n'

                # Write the updated report.
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(updated_content)

                print(f"Appended ensemble results to validation_report.")
            else:
                # Create the report when it is missing.
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(ensemble_line + '\n')

                print(f"Created validation_report: {report_file}")

        except Exception as e:
            print(f"Warning: Could not update validation_report: {e}")


if __name__ == "__main__":

    config = {
        # Model and encoder selection
        # Select an encoder from FEATURE_MODEL_CONFIGS.
        'feature_model': 'ResNet-50',
        # MIL aggregators: WiKG, TransMIL, CLAM, or ABMIL
        # mil_model loads the corresponding model parameters from MIL_MODEL_CONFIGS.
        'mil_model': 'WiKG',
        # Learning-rate schedule
        'scheduler': {
            'type': 'cosine',  # Choose cosine, exponential, plateau, or none.
            'params': {
                'T_max': 45,  # Cosine annealing period in epochs
                'eta_min': 1e-06,  # Cosine annealing's minimum learning rate

                'gamma': 0.95,  # Exponential factor applied at each epoch

                'mode': 'max',  # Use min for loss or max for accuracy and AUC.
                'metric': 'auc',  # Monitor loss, auc, acc, or balanced_acc.
                'patience': 9,  # Epochs without improvement before reducing the learning rate
                'factor': 0.5  # Learning-rate multiplier after a plateau
            }
        },
        # Learning-rate warmup
        'warmup': {
            'enabled': True,  # Enable warmup.
            'warmup_epochs': 3,  # Warmup length in epochs
            'warmup_start_lr': 1e-06,  # Initial warmup learning rate
            'warmup_method': 'cosine',  # Use a linear or cosine warmup.
        },
        # Label smoothing
        'label_smoothing': {
            'enabled': True,  # Enable label smoothing.
            'epsilon': 0.1,  # Uniform label-smoothing strength
            'adaptive': True,  # Enable adaptive label smoothing.
            'adaptive_method': 'confidence',  # Adjust smoothing by prediction confidence, predictive entropy, or loss.
            'min_epsilon': 0.02,  # Minimum adaptive epsilon
            'max_epsilon': 0.2,  # Maximum adaptive epsilon
        },

        # The encoder configuration supplies default HDF5 feature directories.

        'external_sets_independent': True,  # Evaluate external cohorts separately or pool them into one dataset.

        # Label workbooks in the order of the configured feature directories.
        'excel_file': r"data/labels/internal.xlsx",
        'external_excel': [r"data/labels/external_a.xlsx", r"data/labels/external_b.xlsx"],

        'class_names': ["Cholangiocarcinoma", "Ampullary", "Duodenal", "Pancreatic"],

        'batch_size': 1,
        'grad_clip': 2.5,  # Gradient clipping norm. Use None to disable clipping.
        'lr': 5e-4,
        'n_splits': 5,  # Number of cross-validation folds
        'max_epochs': 60,
        'patience': 20,  # Epochs without improvement before stopping
        'seed': 42,  # Random seed
        'amp_enabled': False,  # Enable automatic mixed precision.
        'log_dir': "cv_logs",
        'save_dir': "cv_results",
        'weight_decay': 1e-5,

        # Optional AdamW settings
        # 'beta1': 0.9,        # First moment decay
        # 'beta2': 0.999,      # Second moment decay
        # 'eps': 1e-8,         # Numerical stability

        'early_stop_metric': 'loss',  # Early-stopping metric
        'use_feature_aug': False,  # Enable feature-level Mixup.
        'use_dynamic_class_weight': True,  # Update class weights during training.
        'dynamic_weight_early_epoch': 10,  # Epoch when fixed early-stage weights end
        'dynamic_weight_mid_epoch': 15  # Epoch when weights finish transitioning to balanced values
    }




    # Evaluate combinations of encoders and MIL aggregators.
    # Disable this setting to run the single model configuration below.
    run_all_feature_models = True

    if run_all_feature_models:
        import time

        feature_models_to_run = [
            # 'H-optimus-0', 'H-optimus-1', 'gigapath', 'UNI2', 'ctranspath',
            # 'CHIEF', 'CONCH1.5', 'UNI', 'CONCH', 'MUSK', 'Virchow2', 'Phikon-v2','ResNet-50',
            # 'finetune_Virchow2_S_ep1', 'finetune_Virchow2_T_ep1','finetune_Virchow2_S_ep8',
            # 'finetune_Virchow2_T_ep8', 'finetune_Virchow2_S_ep19', 'finetune_Virchow2_T_ep19',
            'finetune_gigapath_S_early','finetune_gigapath_T_early', 'finetune_gigapath_S_mid',
            'finetune_gigapath_T_mid', 'finetune_gigapath_S_Late', 'finetune_gigapath_T_Late'
        ]
        mil_models_to_run = ['WiKG', 'TransMIL', 'CLAM', 'ABMIL']
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for mil in mil_models_to_run:
            for fm in feature_models_to_run:
                combo_name = f"{mil}+{fm}"
                print(f"\n================ Evaluating combination: {combo_name} ================" )
                cfg = copy.deepcopy(config)
                cfg['feature_model'] = fm
                cfg['mil_model'] = mil
                # Use a shared batch timestamp above the MIL and encoder directories.
                cfg['save_dir'] = os.path.join(config['save_dir'], batch_timestamp, combo_name)
                cfg['log_dir'] = os.path.join(config['log_dir'], batch_timestamp, combo_name)
                # Use the parent timestamp without adding another run subdirectory.
                cfg['append_timestamp_subdir'] = False
                cfg['run_timestamp'] = batch_timestamp

                os.makedirs(cfg['save_dir'], exist_ok=True)
                os.makedirs(cfg['log_dir'], exist_ok=True)

                validator = CrossValidator(cfg)
                validator.run()

                # Pause briefly between sequential runs.
                time.sleep(1)

        print('Completed all configured encoder and MIL evaluations.')
    else:
        # Run the single model configuration.
        validator = CrossValidator(config)
        validator.run()
