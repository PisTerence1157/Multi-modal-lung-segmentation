#!/usr/bin/env python3
"""
Unified Visualization & Statistical Analysis Script

Generates all tables and figures needed for the dissertation:

Tables:
  1. CXR main comparison table (4 models x 3 datasets, Dice/IoU/Precision/Recall)
  2. Statistical significance tests (paired t-test / Wilcoxon)
  3. LUNA16 2D vs 3D comparison table
  4. Dataset statistics table
  5. Hyperparameter table
  6. Model complexity vs performance table (params, FLOPs, Dice)

Figures:
  1. Qualitative segmentation comparison (multi-model overlay)
  2. Training curves from TensorBoard logs
  3. Dice box plots (per dataset per model)
  4. Parameter count vs performance scatter
  5. CT slice-level visualization (2D vs 3D predictions)
  6. Attention map visualization (SE / CBAM / Attention Gate)

Usage:
    # Generate all tables from existing results (run locally)
    python scripts/visualization_sum.py --mode tables --results-dir hyperion_runs_20260123_163540

    # Generate training curves
    python scripts/visualization_sum.py --mode curves --results-dir hyperion_runs_20260123_163540

    # Generate attention maps (requires checkpoint + data)
    python scripts/visualization_sum.py --mode attention --checkpoint outputs/.../best.pt

    # Generate CXR qualitative figures (requires checkpoint + data + results_cases.csv)
    python scripts/visualization_sum.py --mode qualitative --results-dir hyperion_runs_20260123_163540

    # Generate CT slice visualization (requires saved predictions)
    python scripts/visualization_sum.py --mode ct_slices --results-dir hyperion_runs_20260123_163540

    # Generate everything that can be done locally (tables + curves)
    python scripts/visualization_sum.py --mode all --results-dir hyperion_runs_20260123_163540
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from PIL import Image as PILImage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


# ============================================================
# Utility: Collect all CXR training summary data
# ============================================================

def collect_cxr_summary(results_dir):
    """
    Parse summary_cxr_*.csv files from Hyperion results.
    Returns a DataFrame with columns:
        model, dataset, seed, run_dir, epoch, train_loss, train_dice, val_loss, val_dice
    """
    results_dir = Path(results_dir)
    outputs_dir = results_dir / 'outputs'

    rows = []
    for csv_file in sorted(outputs_dir.glob('summary_cxr_*.csv')):
        df = pd.read_csv(csv_file)
        rows.append(df)

    if not rows:
        print("Warning: no summary_cxr_*.csv files found")
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    # Drop duplicate header rows if they snuck in
    combined = combined[combined['model'] != 'model'].copy()

    # Cast numeric columns
    for col in ['seed', 'epoch', 'train_loss', 'train_dice', 'val_loss', 'val_dice']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')

    return combined


def collect_ct_summary(results_dir):
    """
    Parse summary_metrics_ct.csv from Hyperion results (COVID CT - old, not used).
    """
    results_dir = Path(results_dir)
    csv_path = results_dir / 'outputs' / 'summary_metrics_ct.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def collect_luna16_logs(results_dir):
    """
    Parse LUNA16 training logs from Hyperion results to extract epoch-level metrics.
    
    Expected log format:
        Epoch 1/40
          Train - Loss: 0.0382, Dice: 0.9377
          Val   - Loss: 0.0295, Dice: 0.9638
    
    Returns dict with '2d' and '3d' keys mapping to DataFrames.
    """
    results_dir = Path(results_dir)
    result = {}

    for mode, dirname in [('2d', 'ct - 2d- lung segemantation'),
                           ('3d', 'ct- 3d- lung segemantation')]:
        log_dir = results_dir / dirname / 'logs'
        out_files = list(log_dir.glob('*.out'))
        if not out_files:
            continue

        log_file = out_files[0]
        epochs = []
        with open(log_file, 'r', errors='ignore') as f:
            current_epoch = {}
            for line in f:
                line = line.strip()
                # Match "Epoch 1/40"
                if line.startswith('Epoch ') and '/' in line and 'Epochs' not in line:
                    try:
                        part = line.replace('[', '').replace(']', '')
                        nums = part.split('Epoch ')[1].split('/')
                        current_epoch = {'epoch': int(nums[0])}
                    except (ValueError, IndexError):
                        pass
                # Match "  Train - Loss: 0.0382, Dice: 0.9377"
                elif 'Train - Loss' in line or 'Train -' in line and 'Loss' in line:
                    if current_epoch:
                        try:
                            parts = line.split(',')
                            for p in parts:
                                p = p.strip()
                                if 'Loss' in p:
                                    current_epoch['train_loss'] = float(p.split(':')[-1].strip())
                                elif 'Dice' in p:
                                    current_epoch['train_dice'] = float(p.split(':')[-1].strip())
                        except (ValueError, IndexError):
                            pass
                # Match "  Val   - Loss: 0.0295, Dice: 0.9638"
                elif 'Val' in line and 'Loss' in line and 'Dice' in line:
                    if current_epoch:
                        try:
                            parts = line.split(',')
                            for p in parts:
                                p = p.strip()
                                if 'Loss' in p:
                                    current_epoch['val_loss'] = float(p.split(':')[-1].strip())
                                elif 'Dice' in p:
                                    current_epoch['val_dice'] = float(p.split(':')[-1].strip())
                        except (ValueError, IndexError):
                            pass
                        if 'val_dice' in current_epoch:
                            epochs.append(current_epoch)
                            current_epoch = {}

        if epochs:
            result[mode] = pd.DataFrame(epochs)

    return result


def collect_per_case_results(results_dir):
    """
    Collect all results_cases.csv from evaluation runs.
    """
    results_dir = Path(results_dir)
    all_files = list(results_dir.rglob('results_cases.csv'))
    if not all_files:
        return pd.DataFrame()

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = str(f.relative_to(results_dir))
            dfs.append(df)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# ============================================================
# Table 1: CXR Main Comparison Table
# ============================================================

def generate_cxr_table(cxr_df, output_dir):
    """
    Generate main CXR comparison table.
    Format: model x dataset, showing val_dice mean +/- std (3 seeds).
    If results_cases.csv exists with test dice, uses that instead.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cxr_df.empty:
        print("No CXR data available for table generation")
        return

    # Group by model + dataset, compute mean/std over seeds
    grouped = cxr_df.groupby(['model', 'dataset']).agg(
        val_dice_mean=('val_dice', 'mean'),
        val_dice_std=('val_dice', 'std'),
        train_dice_mean=('train_dice', 'mean'),
        n_seeds=('seed', 'count'),
    ).reset_index()

    # Create pivot table for display
    models = ['unet', 'attention_unet', 'se_unet', 'cbam_unet']
    datasets = ['kaggle', 'montgomery', 'shenzhen']
    model_labels = {
        'unet': 'U-Net',
        'attention_unet': 'Attention U-Net',
        'se_unet': 'SE-UNet',
        'cbam_unet': 'CBAM-UNet',
    }

    # Text table
    lines = []
    lines.append("="*80)
    lines.append("Table 1: CXR Lung Field Segmentation - Validation Dice (mean +/- std, 3 seeds)")
    lines.append("="*80)
    header = f"{'Model':<20} {'Darwin':<20} {'Montgomery':<20} {'Shenzhen':<20}"
    lines.append(header)
    lines.append("-"*80)

    for model in models:
        row_parts = [f"{model_labels.get(model, model):<20}"]
        for dataset in datasets:
            subset = grouped[(grouped['model'] == model) & (grouped['dataset'] == dataset)]
            if len(subset) > 0:
                mean = subset.iloc[0]['val_dice_mean']
                std = subset.iloc[0]['val_dice_std']
                row_parts.append(f"{mean:.4f} +/- {std:.4f} ")
            else:
                row_parts.append(f"{'N/A':<20}")
        lines.append("".join(row_parts))

    lines.append("="*80)

    table_text = "\n".join(lines)
    print(table_text)

    with open(output_dir / 'table1_cxr_comparison.txt', 'w') as f:
        f.write(table_text)

    # Also save as CSV for easy import
    grouped.to_csv(output_dir / 'table1_cxr_comparison.csv', index=False)

    # LaTeX version
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{CXR Lung Field Segmentation Results (Validation Dice Score)}")
    latex_lines.append(r"\label{tab:cxr_comparison}")
    latex_lines.append(r"\begin{tabular}{lccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Model & Darwin & Montgomery & Shenzhen \\")
    latex_lines.append(r"\midrule")

    best_per_dataset = {}
    for dataset in datasets:
        subset = grouped[grouped['dataset'] == dataset]
        if len(subset) > 0:
            best_per_dataset[dataset] = subset['val_dice_mean'].max()

    for model in models:
        parts = [model_labels.get(model, model)]
        for dataset in datasets:
            subset = grouped[(grouped['model'] == model) & (grouped['dataset'] == dataset)]
            if len(subset) > 0:
                mean = subset.iloc[0]['val_dice_mean']
                std = subset.iloc[0]['val_dice_std']
                val_str = f"{mean:.4f} $\\pm$ {std:.4f}"
                if dataset in best_per_dataset and abs(mean - best_per_dataset[dataset]) < 1e-6:
                    val_str = r"\textbf{" + val_str + "}"
                parts.append(val_str)
            else:
                parts.append("--")
        latex_lines.append(" & ".join(parts) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    with open(output_dir / 'table1_cxr_comparison.tex', 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"Saved to {output_dir / 'table1_cxr_comparison.tex'}")


# ============================================================
# Table 2: Statistical Significance Tests
# ============================================================

def generate_significance_table(per_case_df, output_dir):
    """
    Generate paired statistical tests between models.
    Requires per-case results (results_cases.csv).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if per_case_df.empty:
        print("No per-case results available. Run evaluate.py on Hyperion first.")
        print("Skipping statistical significance tests.")
        return

    from scipy.stats import ttest_rel, wilcoxon

    models = per_case_df['model'].unique()
    datasets = per_case_df['dataset'].unique() if 'dataset' in per_case_df.columns else ['all']

    results = []
    for dataset in datasets:
        if dataset != 'all':
            df_d = per_case_df[per_case_df['dataset'] == dataset]
        else:
            df_d = per_case_df

        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                d1 = df_d[df_d['model'] == m1]['dice'].values
                d2 = df_d[df_d['model'] == m2]['dice'].values

                if len(d1) == 0 or len(d2) == 0 or len(d1) != len(d2):
                    continue

                t_stat, t_pval = ttest_rel(d1, d2)
                try:
                    w_stat, w_pval = wilcoxon(d1, d2)
                except ValueError:
                    w_stat, w_pval = np.nan, np.nan

                results.append({
                    'dataset': dataset,
                    'model_1': m1,
                    'model_2': m2,
                    'mean_dice_1': np.mean(d1),
                    'mean_dice_2': np.mean(d2),
                    'diff': np.mean(d1) - np.mean(d2),
                    't_statistic': t_stat,
                    't_p_value': t_pval,
                    'wilcoxon_statistic': w_stat,
                    'wilcoxon_p_value': w_pval,
                    'significant_005': t_pval < 0.05,
                })

    if results:
        df_sig = pd.DataFrame(results)
        df_sig.to_csv(output_dir / 'table2_significance_tests.csv', index=False)
        print(f"Saved significance tests to {output_dir / 'table2_significance_tests.csv'}")
        print(df_sig.to_string(index=False))
    else:
        print("Not enough paired data for significance tests.")


# ============================================================
# Table 3: LUNA16 2D vs 3D Comparison
# ============================================================

def generate_luna16_table(results_dir, output_dir):
    """
    Generate LUNA16 2D vs 3D comparison table.
    Uses training logs for val_dice since evaluate_ct hasn't been run yet.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    luna_logs = collect_luna16_logs(results_dir)

    lines = []
    lines.append("="*60)
    lines.append("Table 3: LUNA16 Lung Field Segmentation - 2D vs 3D")
    lines.append("="*60)

    for mode in ['2d', '3d']:
        if mode in luna_logs:
            df = luna_logs[mode]
            best_epoch = df.loc[df['val_dice'].idxmax()]
            last_epoch = df.iloc[-1]
            lines.append(f"\n{mode.upper()} UNet:")
            lines.append(f"  Total epochs: {len(df)}")
            lines.append(f"  Best val Dice: {best_epoch['val_dice']:.4f} (epoch {int(best_epoch['epoch'])})")
            lines.append(f"  Last val Dice: {last_epoch['val_dice']:.4f} (epoch {int(last_epoch['epoch'])})")
            lines.append(f"  Best train Dice: {best_epoch.get('train_dice', 'N/A')}")
        else:
            lines.append(f"\n{mode.upper()} UNet: no log data found")

    lines.append("="*60)

    # Also load run_meta for details
    for mode, dirname in [('2d', 'ct - 2d- lung segemantation'),
                           ('3d', 'ct- 3d- lung segemantation')]:
        meta_path = Path(results_dir) / dirname / 'outputs' / 'run_meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            lines.append(f"\n{mode.upper()} Run Meta:")
            for k, v in meta.items():
                lines.append(f"  {k}: {v}")

    table_text = "\n".join(lines)
    print(table_text)

    with open(output_dir / 'table3_luna16_2d_vs_3d.txt', 'w') as f:
        f.write(table_text)

    # LaTeX
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{LUNA16 Lung Field Segmentation: 2D vs 3D}")
    latex_lines.append(r"\label{tab:luna16_2d_vs_3d}")
    latex_lines.append(r"\begin{tabular}{lcccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Mode & Model & Best Val Dice & Epochs Trained & Params \\")
    latex_lines.append(r"\midrule")

    for mode in ['2d', '3d']:
        if mode in luna_logs:
            df = luna_logs[mode]
            best = df.loc[df['val_dice'].idxmax()]
            model_name = "U-Net" if mode == '2d' else "3D U-Net"
            params = "13.4M" if mode == '2d' else "3.0M"
            latex_lines.append(
                f"{mode.upper()} & {model_name} & {best['val_dice']:.4f} & {len(df)} & {params} \\\\"
            )

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    with open(output_dir / 'table3_luna16_2d_vs_3d.tex', 'w') as f:
        f.write("\n".join(latex_lines))


# ============================================================
# Table 4: Dataset Statistics
# ============================================================

def generate_dataset_stats_table(output_dir):
    """
    Generate dataset statistics table from index.csv and index_luna16.csv.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("="*80)
    lines.append("Table 4: Dataset Statistics")
    lines.append("="*80)

    # CXR datasets
    index_path = project_root / 'data' / 'index.csv'
    if index_path.exists():
        df = pd.read_csv(index_path)
        lines.append("\nCXR Datasets:")
        lines.append(f"{'Dataset':<15} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8} {'Pos Ratio':<12}")
        lines.append("-"*60)
        for dataset in df['dataset'].unique():
            sub = df[df['dataset'] == dataset]
            lines.append(
                f"{dataset:<15} {len(sub):<8} "
                f"{len(sub[sub['split']=='train']):<8} "
                f"{len(sub[sub['split']=='val']):<8} "
                f"{len(sub[sub['split']=='test']):<8} "
                f"{sub['positive_ratio'].mean():.4f}"
            )
        lines.append(f"\n{'Total':<15} {len(df):<8}")
    else:
        lines.append("CXR index.csv not found")

    # LUNA16
    luna_index = project_root / 'data' / 'index_luna16.csv'
    if luna_index.exists():
        df_luna = pd.read_csv(luna_index)
        lines.append("\n\nLUNA16 Dataset:")
        lines.append(f"{'Split':<10} {'Cases':<8}")
        lines.append("-"*20)
        for split in ['train', 'val', 'test']:
            n = len(df_luna[df_luna['split'] == split])
            lines.append(f"{split:<10} {n:<8}")
        lines.append(f"{'Total':<10} {len(df_luna):<8}")
    else:
        lines.append("\nLUNA16 index not found")

    table_text = "\n".join(lines)
    print(table_text)

    with open(output_dir / 'table4_dataset_stats.txt', 'w') as f:
        f.write(table_text)


# ============================================================
# Table 5: Hyperparameter Table
# ============================================================

def generate_hyperparameter_table(output_dir):
    """
    Generate hyperparameter summary table from base.yaml config.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    config_path = project_root / 'configs' / 'base.yaml'

    lines = []
    lines.append("="*60)
    lines.append("Table 5: Training Hyperparameters")
    lines.append("="*60)

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

        params = {
            'Input Resolution': f"{config.get('dataset', {}).get('image_size', 'N/A')}",
            'Optimizer': config.get('training', {}).get('optimizer', 'N/A'),
            'Initial Learning Rate': config.get('training', {}).get('learning_rate', 'N/A'),
            'Weight Decay': config.get('training', {}).get('weight_decay', 'N/A'),
            'Batch Size': config.get('training', {}).get('batch_size', 'N/A'),
            'Max Epochs': config.get('training', {}).get('num_epochs', 'N/A'),
            'LR Scheduler': config.get('scheduler', {}).get('type', 'N/A'),
            'Scheduler Factor': config.get('scheduler', {}).get('factor', 'N/A'),
            'Scheduler Patience': config.get('scheduler', {}).get('patience', 'N/A'),
            'Early Stopping Patience': config.get('early_stopping', {}).get('patience', 'N/A'),
            'Early Stopping Min Delta': config.get('early_stopping', {}).get('min_delta', 'N/A'),
            'Loss Function': config.get('loss', {}).get('type', 'N/A'),
            'Dice Weight': config.get('loss', {}).get('dice_weight', 'N/A'),
            'BCE Weight': config.get('loss', {}).get('bce_weight', 'N/A'),
            'Random Seed': config.get('training', {}).get('seed', 'N/A'),
            'Augmentation - H.Flip': config.get('augmentation', {}).get('horizontal_flip', 'N/A'),
            'Augmentation - Rotation': config.get('augmentation', {}).get('rotation', 'N/A'),
            'Augmentation - Brightness': config.get('augmentation', {}).get('brightness', 'N/A'),
            'Augmentation - Contrast': config.get('augmentation', {}).get('contrast', 'N/A'),
            'Normalization Mean': config.get('dataset', {}).get('mean', 'N/A'),
            'Normalization Std': config.get('dataset', {}).get('std', 'N/A'),
            'Inference Threshold': config.get('inference', {}).get('threshold', 'N/A'),
        }

        for k, v in params.items():
            lines.append(f"  {k:<35} {v}")
    else:
        lines.append("  base.yaml not found")

    lines.append("="*60)
    table_text = "\n".join(lines)
    print(table_text)

    with open(output_dir / 'table5_hyperparameters.txt', 'w') as f:
        f.write(table_text)


# ============================================================
# Table 6: Model Complexity vs Performance
# ============================================================

def generate_complexity_table(cxr_df, output_dir):
    """
    Generate model complexity (params) vs performance table.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from models import get_model
    except ImportError:
        print("Warning: torch not available. Using hardcoded parameter counts.")
        # Hardcoded from previous runs
        hardcoded = [
            {'model': 'unet', 'model_label': 'U-Net', 'params': 13394177, 'size_mb': 51.11},
            {'model': 'attention_unet', 'model_label': 'Attention U-Net', 'params': 13570497, 'size_mb': 51.78},
            {'model': 'se_unet', 'model_label': 'SE-UNet', 'params': 13432865, 'size_mb': 51.26},
            {'model': 'cbam_unet', 'model_label': 'CBAM-UNet', 'params': 13433019, 'size_mb': 51.26},
        ]
        for item in hardcoded:
            if not cxr_df.empty:
                model_data = cxr_df[cxr_df['model'] == item['model']]
                item['avg_val_dice'] = model_data['val_dice'].mean() if len(model_data) > 0 else None
            else:
                item['avg_val_dice'] = None
        
        lines = []
        lines.append("="*80)
        lines.append("Table 6: Model Complexity vs Performance")
        lines.append("="*80)
        lines.append(f"{'Model':<20} {'Params':<15} {'Size (MB)':<12} {'Avg Val Dice':<15}")
        lines.append("-"*65)
        for item in hardcoded:
            dice_str = f"{item['avg_val_dice']:.4f}" if item['avg_val_dice'] else "N/A"
            lines.append(f"{item['model_label']:<20} {item['params']:>12,} {item['size_mb']:>10.2f} {dice_str:>12}")
        lines.append("="*80)
        print("\n".join(lines))
        with open(output_dir / 'table6_complexity_performance.txt', 'w') as f:
            f.write("\n".join(lines))
        pd.DataFrame(hardcoded).to_csv(output_dir / 'table6_complexity_performance.csv', index=False)
        return hardcoded

    model_configs = {
        'unet': {'model': {'name': 'unet', 'in_channels': 1, 'out_channels': 1,
                           'features': [64, 128, 256, 512], 'dropout': 0.1}},
        'attention_unet': {'model': {'name': 'attention_unet', 'in_channels': 1, 'out_channels': 1,
                                      'features': [64, 128, 256, 512], 'dropout': 0.1}},
        'se_unet': {'model': {'name': 'se_unet', 'in_channels': 1, 'out_channels': 1,
                               'features': [64, 128, 256, 512], 'dropout': 0.1}},
        'cbam_unet': {'model': {'name': 'cbam_unet', 'in_channels': 1, 'out_channels': 1,
                                 'features': [64, 128, 256, 512], 'dropout': 0.1}},
    }

    model_labels = {
        'unet': 'U-Net',
        'attention_unet': 'Attention U-Net',
        'se_unet': 'SE-UNet',
        'cbam_unet': 'CBAM-UNet',
    }

    lines = []
    lines.append("="*80)
    lines.append("Table 6: Model Complexity vs Performance")
    lines.append("="*80)
    lines.append(f"{'Model':<20} {'Params':<15} {'Size (MB)':<12} {'Avg Val Dice':<15}")
    lines.append("-"*65)

    complexity_data = []
    for name, cfg in model_configs.items():
        try:
            model = get_model(cfg)
            total_params = sum(p.numel() for p in model.parameters())
            param_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024**2

            avg_dice = None
            if not cxr_df.empty:
                model_data = cxr_df[cxr_df['model'] == name]
                if len(model_data) > 0:
                    avg_dice = model_data['val_dice'].mean()

            dice_str = f"{avg_dice:.4f}" if avg_dice is not None else "N/A"
            lines.append(f"{model_labels[name]:<20} {total_params:>12,} {param_size_mb:>10.2f} {dice_str:>12}")

            complexity_data.append({
                'model': name,
                'model_label': model_labels[name],
                'params': total_params,
                'size_mb': param_size_mb,
                'avg_val_dice': avg_dice,
            })
        except Exception as e:
            lines.append(f"{model_labels[name]:<20} Error: {e}")

    lines.append("="*80)
    table_text = "\n".join(lines)
    print(table_text)

    with open(output_dir / 'table6_complexity_performance.txt', 'w') as f:
        f.write(table_text)

    if complexity_data:
        pd.DataFrame(complexity_data).to_csv(
            output_dir / 'table6_complexity_performance.csv', index=False
        )

    return complexity_data


# ============================================================
# Figure 1: Training Curves
# ============================================================

def plot_training_curves(cxr_df, luna_logs, output_dir):
    """
    Plot training curves (loss and dice) for CXR and LUNA16.
    Uses summary data (epoch-level) from CXR summaries.
    For detailed curves, reads TensorBoard event files if available.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- LUNA16 Training Curves ---
    if luna_logs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for mode in ['2d', '3d']:
            if mode in luna_logs:
                df = luna_logs[mode]
                label = f"{'2D U-Net' if mode == '2d' else '3D U-Net'}"
                color = '#2196F3' if mode == '2d' else '#FF5722'

                axes[0].plot(df['epoch'], df['train_loss'], '--', color=color, alpha=0.5,
                           label=f'{label} Train')
                axes[0].plot(df['epoch'], df['val_loss'], '-', color=color,
                           label=f'{label} Val')

                axes[1].plot(df['epoch'], df['train_dice'], '--', color=color, alpha=0.5,
                           label=f'{label} Train')
                axes[1].plot(df['epoch'], df['val_dice'], '-', color=color,
                           label=f'{label} Val')

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('LUNA16: Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('LUNA16: Training & Validation Dice')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'fig1_luna16_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'fig1_luna16_training_curves.png'}")


# ============================================================
# Figure 2: Dice Box Plots
# ============================================================

def plot_dice_boxplots(cxr_df, output_dir):
    """
    Plot validation Dice box plots grouped by dataset and model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cxr_df.empty:
        print("No CXR data for box plots")
        return

    models = ['unet', 'attention_unet', 'se_unet', 'cbam_unet']
    model_labels = ['U-Net', 'Attn U-Net', 'SE-UNet', 'CBAM-UNet']
    datasets = ['kaggle', 'montgomery', 'shenzhen']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax_idx, dataset in enumerate(datasets):
        data = []
        for model in models:
            subset = cxr_df[(cxr_df['model'] == model) & (cxr_df['dataset'] == dataset)]
            data.append(subset['val_dice'].values)

        bp = axes[ax_idx].boxplot(data, tick_labels=model_labels, patch_artist=True,
                                   widths=0.6, showmeans=True,
                                   meanprops=dict(marker='D', markerfacecolor='red', markersize=5))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        dataset_labels = {'kaggle': 'Darwin', 'montgomery': 'Montgomery', 'shenzhen': 'Shenzhen'}
        axes[ax_idx].set_title(dataset_labels.get(dataset, dataset.capitalize()))
        axes[ax_idx].set_ylabel('Validation Dice' if ax_idx == 0 else '')
        axes[ax_idx].grid(True, alpha=0.3, axis='y')
        axes[ax_idx].tick_params(axis='x', rotation=30)

    plt.suptitle('CXR Lung Segmentation: Dice Score Distribution (3 seeds)', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_dice_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'fig2_dice_boxplots.png'}")


# ============================================================
# Figure 3: Parameter Count vs Performance
# ============================================================

def plot_params_vs_performance(complexity_data, output_dir):
    """
    Scatter plot of model parameters vs average Dice.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not complexity_data:
        print("No complexity data available")
        return

    df = pd.DataFrame(complexity_data)
    df = df.dropna(subset=['avg_val_dice'])

    if df.empty:
        print("No Dice data available for params vs performance plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    markers = ['o', 's', '^', 'D']

    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row['params'] / 1e6, row['avg_val_dice'],
                  s=120, c=colors[i % len(colors)], marker=markers[i % len(markers)],
                  label=row['model_label'], zorder=5, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Average Validation Dice')
    ax.set_title('Model Complexity vs Segmentation Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_params_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'fig3_params_vs_performance.png'}")


# ============================================================
# Figure 4: Attention Map Visualization
# ============================================================

def visualize_attention_maps(checkpoint_path, data_root, output_dir, num_samples=3):
    """
    Visualize attention maps from a trained attention model.
    Requires a checkpoint with attention model weights.
    """
    import torch
    from models import get_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    model_name = config['model']['name']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load test data
    from datasets.chest_xray_dataset import ChestXrayDataset
    dataset = ChestXrayDataset(
        data_root=data_root,
        index_path=config['data'].get('index_path', 'data/index.csv'),
        split='test',
        dataset=config['data'].get('dataset_filter'),
        image_size=config['dataset']['image_size']
    )

    for sample_idx in range(min(num_samples, len(dataset))):
        sample = dataset[sample_idx]
        image = sample['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            pred = torch.sigmoid(output)

        if not hasattr(model, 'attention_weights') or not model.attention_weights:
            print(f"Model {model_name} has no attention_weights attribute")
            return

        attention_maps = model.attention_weights
        image_np = sample['image'].squeeze().cpu().numpy()
        mask_np = sample['mask'].squeeze().cpu().numpy()
        pred_np = pred.squeeze().cpu().numpy()

        # Denormalize for display
        image_display = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

        if model_name == 'attention_unet':
            # Show 4 attention gate maps
            n_maps = len(attention_maps)
            fig, axes = plt.subplots(2, n_maps + 1, figsize=(4 * (n_maps + 1), 8))

            # Top row: image + attention maps
            axes[0, 0].imshow(image_display, cmap='gray')
            axes[0, 0].set_title('Input')
            axes[0, 0].axis('off')

            for i, amap in enumerate(attention_maps):
                amap_np = amap[0, 0].cpu().numpy()  # [H, W]
                amap_uint8 = (amap_np * 255).astype(np.uint8)
                amap_resized = np.array(
                    PILImage.fromarray(amap_uint8).resize(
                        (image_display.shape[1], image_display.shape[0]),
                        PILImage.BILINEAR
                    )
                ) / 255.0
                axes[0, i + 1].imshow(image_display, cmap='gray', alpha=0.5)
                axes[0, i + 1].imshow(amap_resized, cmap='hot', alpha=0.6)
                axes[0, i + 1].set_title(f'Attn Gate L{i+1}')
                axes[0, i + 1].axis('off')

            # Bottom row: GT + pred + overlay
            axes[1, 0].imshow(mask_np, cmap='gray')
            axes[1, 0].set_title('Ground Truth')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(pred_np > 0.5, cmap='gray')
            axes[1, 1].set_title('Prediction')
            axes[1, 1].axis('off')

            # Overlay
            overlay = np.stack([image_display]*3, axis=-1)
            overlay[mask_np > 0.5] = [0, 1, 0]
            overlay[pred_np > 0.5] = [1, 0.5, 0]
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('Overlay')
            axes[1, 2].axis('off')

            for j in range(3, n_maps + 1):
                axes[1, j].axis('off')

        elif model_name in ['se_unet', 'cbam_unet']:
            # SE: channel attention weights as bar chart
            # CBAM: spatial attention maps
            n_maps = len(attention_maps)
            fig, axes = plt.subplots(1, n_maps + 2, figsize=(4 * (n_maps + 2), 4))

            axes[0].imshow(image_display, cmap='gray')
            axes[0].set_title('Input')
            axes[0].axis('off')

            for i, amap in enumerate(attention_maps):
                if model_name == 'cbam_unet':
                    # Show spatial attention
                    spatial = amap['spatial'][0, 0].cpu().numpy()
                    spatial_uint8 = (spatial * 255).astype(np.uint8)
                    spatial_resized = np.array(
                        PILImage.fromarray(spatial_uint8).resize(
                            (image_display.shape[1], image_display.shape[0]),
                            PILImage.BILINEAR
                        )
                    ) / 255.0
                    axes[i + 1].imshow(image_display, cmap='gray', alpha=0.5)
                    axes[i + 1].imshow(spatial_resized, cmap='hot', alpha=0.6)
                    axes[i + 1].set_title(f'CBAM Spatial L{i+1}')
                    axes[i + 1].axis('off')
                else:
                    # SE: show channel weights as bar
                    se_weights = amap[0, :, 0, 0].cpu().numpy()  # [C]
                    axes[i + 1].barh(range(len(se_weights)), se_weights, color='steelblue')
                    axes[i + 1].set_title(f'SE Weights L{i+1}')
                    axes[i + 1].set_xlabel('Weight')
                    axes[i + 1].set_ylabel('Channel')

            axes[-1].imshow(pred_np > 0.5, cmap='gray')
            axes[-1].set_title('Prediction')
            axes[-1].axis('off')
        else:
            print(f"Attention visualization not supported for {model_name}")
            return

        plt.suptitle(f'{model_name}: {sample["case_id"]}', y=1.02)
        plt.tight_layout()
        save_path = output_dir / f'fig_attention_{model_name}_{sample["case_id"]}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


# ============================================================
# Figure 5: CXR Qualitative Comparison (multi-model)
# ============================================================

def plot_qualitative_comparison(results_dir, data_root, output_dir, n_cases=5):
    """
    Generate multi-model qualitative comparison figure.
    Requires prediction masks saved by evaluate.py --save-predictions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(results_dir)

    # Find all prediction directories
    pred_dirs = list(results_dir.rglob('predictions'))
    if not pred_dirs:
        print("No prediction directories found. Run evaluate.py --save-predictions first.")
        return

    print(f"Found {len(pred_dirs)} prediction directories")
    # This function will be fully functional after evaluate.py is run with --save-predictions
    print("Qualitative comparison will be generated after running evaluate.py with --save-predictions")


# ============================================================
# Figure 6: CT Slice Visualization
# ============================================================

def plot_ct_slices(results_dir, output_dir, n_cases=3):
    """
    Visualize CT prediction slices (2D vs 3D comparison).
    Requires prediction .npz files saved by evaluate_ct.py --save-predictions.
    Auto-discovers prediction directories from eval_outputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(results_dir)

    # Auto-discover prediction directories by searching for npz files
    pred_dirs = {}  # mode -> pred_dir (take latest per mode)
    for pred_path in sorted(results_dir.rglob('predictions')):
        if not pred_path.is_dir():
            continue
        # Check if it contains case subdirectories with npz files
        has_npz = any(pred_path.rglob('*.npz'))
        if not has_npz:
            continue
        parent_name = str(pred_path.parent.name).lower()
        if 'unet2d' in parent_name or '2d' in parent_name:
            pred_dirs['2d'] = pred_path
        elif 'unet3d' in parent_name or '3d' in parent_name:
            pred_dirs['3d'] = pred_path

    if not pred_dirs:
        print("No CT prediction directories found. Run evaluate_ct.py --save-predictions first.")
        return

    # Find common cases between 2D and 3D for side-by-side comparison
    case_sets = {}
    for mode, pred_dir in pred_dirs.items():
        case_sets[mode] = sorted([d.name for d in pred_dir.iterdir() if d.is_dir()])

    # If both 2D and 3D exist, do side-by-side comparison
    if '2d' in pred_dirs and '3d' in pred_dirs:
        common_cases = sorted(set(case_sets['2d']) & set(case_sets['3d']))[:n_cases]
        if common_cases:
            for case_id in common_cases:
                # Get slices that exist in both
                slices_2d = {f.stem: f for f in (pred_dirs['2d'] / case_id).glob('*.npz')}
                slices_3d = {f.stem: f for f in (pred_dirs['3d'] / case_id).glob('*.npz')}
                common_slices = sorted(set(slices_2d.keys()) & set(slices_3d.keys()))[:5]

                if not common_slices:
                    continue

                n_slices = len(common_slices)
                fig, axes = plt.subplots(4, n_slices, figsize=(4 * n_slices, 16))
                if n_slices == 1:
                    axes = axes.reshape(-1, 1)

                for j, slice_name in enumerate(common_slices):
                    data_2d = np.load(slices_2d[slice_name])
                    data_3d = np.load(slices_3d[slice_name])

                    axes[0, j].imshow(data_2d['image'], cmap='gray')
                    axes[0, j].set_title(f'{slice_name}')
                    axes[0, j].axis('off')

                    axes[1, j].imshow(data_2d['mask'], cmap='gray')
                    axes[1, j].set_title('Ground Truth')
                    axes[1, j].axis('off')

                    axes[2, j].imshow(data_2d['pred'] > 0.5, cmap='gray')
                    axes[2, j].set_title('2D UNet Pred')
                    axes[2, j].axis('off')

                    axes[3, j].imshow(data_3d['pred'] > 0.5, cmap='gray')
                    axes[3, j].set_title('3D UNet Pred')
                    axes[3, j].axis('off')

                # Truncate case_id for title
                short_id = case_id[:20] + '...' if len(case_id) > 20 else case_id
                plt.suptitle(f'CT Lung Segmentation: 2D vs 3D - {short_id}', y=1.02)
                plt.tight_layout()
                safe_name = case_id.replace('.', '_')[:40]
                save_path = output_dir / f'fig_ct_2d_vs_3d_{safe_name}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved: {save_path}")
            return

    # Fallback: show each mode separately
    for mode, pred_dir in pred_dirs.items():
        case_dirs = sorted([d for d in pred_dir.iterdir() if d.is_dir()])[:n_cases]
        for case_dir in case_dirs:
            npz_files = sorted(case_dir.glob('*.npz'))[:5]
            if not npz_files:
                continue

            n_slices = len(npz_files)
            fig, axes = plt.subplots(3, n_slices, figsize=(4 * n_slices, 12))
            if n_slices == 1:
                axes = axes.reshape(-1, 1)

            for j, npz_path in enumerate(npz_files):
                data = np.load(npz_path)
                axes[0, j].imshow(data['image'], cmap='gray')
                axes[0, j].set_title(f'{npz_path.stem}')
                axes[0, j].axis('off')

                axes[1, j].imshow(data['mask'], cmap='gray')
                axes[1, j].set_title('Ground Truth')
                axes[1, j].axis('off')

                axes[2, j].imshow(data['pred'] > 0.5, cmap='gray')
                axes[2, j].set_title(f'Pred ({mode.upper()})')
                axes[2, j].axis('off')

            short_id = case_dir.name[:20] + '...' if len(case_dir.name) > 20 else case_dir.name
            plt.suptitle(f'{mode.upper()} UNet - {short_id}', y=1.02)
            plt.tight_layout()
            safe_name = case_dir.name.replace('.', '_')[:40]
            save_path = output_dir / f'fig_ct_{mode}_{safe_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Unified visualization and analysis')

    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'tables', 'curves', 'attention',
                                 'attention_export', 'qualitative', 'ct_slices'],
                        help='What to generate')
    parser.add_argument('--results-dir', type=str,
                        default=str(project_root / 'hyperion_runs_20260123_163540'),
                        help='Path to Hyperion results directory')
    parser.add_argument('--output-dir', type=str,
                        default=str(project_root / 'dissertation_figures'),
                        help='Output directory for all generated figures and tables')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (for single attention map visualization)')
    parser.add_argument('--checkpoints', type=str, nargs='+', default=None,
                        help='Multiple checkpoint paths (for batch attention export)')
    parser.add_argument('--data-root', type=str, default=str(project_root),
                        help='Data root directory')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples for attention visualization')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)

    print("="*60)
    print("Dissertation Visualization & Analysis")
    print("="*60)
    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Mode:        {args.mode}")
    print("="*60)

    # Collect data
    cxr_df = collect_cxr_summary(results_dir)
    luna_logs = collect_luna16_logs(results_dir)
    per_case_df = collect_per_case_results(results_dir)

    if not cxr_df.empty:
        print(f"\nCXR data: {len(cxr_df)} runs loaded")
        print(f"  Models: {cxr_df['model'].unique().tolist()}")
        print(f"  Datasets: {cxr_df['dataset'].unique().tolist()}")

    if luna_logs:
        print(f"\nLUNA16 logs: {list(luna_logs.keys())}")
        for mode, df in luna_logs.items():
            print(f"  {mode}: {len(df)} epochs")

    if not per_case_df.empty:
        print(f"\nPer-case results: {len(per_case_df)} entries")

    # Generate outputs based on mode
    if args.mode in ['all', 'tables']:
        print("\n" + "="*60)
        print("Generating Tables")
        print("="*60)

        generate_cxr_table(cxr_df, output_dir)
        generate_luna16_table(results_dir, output_dir)
        generate_dataset_stats_table(output_dir)
        generate_hyperparameter_table(output_dir)
        complexity_data = generate_complexity_table(cxr_df, output_dir)

        if not per_case_df.empty:
            generate_significance_table(per_case_df, output_dir)

    if args.mode in ['all', 'curves']:
        print("\n" + "="*60)
        print("Generating Training Curves")
        print("="*60)

        plot_training_curves(cxr_df, luna_logs, output_dir)
        plot_dice_boxplots(cxr_df, output_dir)

        if 'complexity_data' not in dir():
            complexity_data = generate_complexity_table(cxr_df, output_dir)
        plot_params_vs_performance(complexity_data, output_dir)

    if args.mode == 'attention':
        if args.checkpoint is None:
            print("Error: --checkpoint required for attention mode")
            return
        print("\n" + "="*60)
        print("Generating Attention Maps")
        print("="*60)
        visualize_attention_maps(args.checkpoint, args.data_root, output_dir,
                                 num_samples=args.num_samples)

    if args.mode == 'attention_export':
        checkpoints = args.checkpoints or []
        if args.checkpoint:
            checkpoints.append(args.checkpoint)
        if not checkpoints:
            # Auto-discover attention model checkpoints from outputs/
            outputs_base = Path(args.data_root) / 'outputs'
            for model_name in ['attention_unet', 'se_unet', 'cbam_unet']:
                for dataset_dir in sorted(outputs_base.iterdir()) if outputs_base.exists() else []:
                    model_dir = dataset_dir / model_name
                    if model_dir.exists():
                        # Take first run (seed42)
                        for run_dir in sorted(model_dir.iterdir()):
                            ckpt = run_dir / 'checkpoints' / 'best.pt'
                            if ckpt.exists():
                                checkpoints.append(str(ckpt))
                                break  # one per model per dataset
        if not checkpoints:
            print("Error: no checkpoints found. Use --checkpoints or --checkpoint.")
            return
        print("\n" + "="*60)
        print(f"Batch Attention Export: {len(checkpoints)} checkpoints")
        print("="*60)
        for ckpt_path in checkpoints:
            print(f"\n--- Processing: {ckpt_path}")
            try:
                visualize_attention_maps(ckpt_path, args.data_root, output_dir,
                                         num_samples=args.num_samples)
            except Exception as e:
                print(f"Error processing {ckpt_path}: {e}")

    if args.mode == 'qualitative':
        print("\n" + "="*60)
        print("Generating Qualitative Comparison")
        print("="*60)
        plot_qualitative_comparison(results_dir, args.data_root, output_dir)

    if args.mode == 'ct_slices':
        print("\n" + "="*60)
        print("Generating CT Slice Visualizations")
        print("="*60)
        plot_ct_slices(results_dir, output_dir)

    print("\n" + "="*60)
    print(f"All outputs saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
