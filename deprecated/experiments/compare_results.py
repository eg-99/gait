"""
Compare results across all experiments.
Generates comprehensive comparison tables and visualizations.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def load_experiment_results(results_dir):
    """
    Load all experiment results from directory structure.
    
    Expected structure:
        results/
            exp1_zero_shot/
                vae_binary.json
                contrastive_binary.json
                vae_multiclass.json
                contrastive_multiclass.json
            exp2_finetune/
                vae/
                    binary_results.json
                    multiclass_results.json
                contrastive/
                    binary_results.json
                    multiclass_results.json
            exp3_from_scratch/
                vae/
                    binary_results.json
                    multiclass_results.json
                contrastive/
                    binary_results.json
                    multiclass_results.json
    """
    results_dir = Path(results_dir)
    all_results = []
    
    # Experiment 1: Zero-Shot
    exp1_dir = results_dir / 'exp1_zero_shot'
    if exp1_dir.exists():
        for model in ['vae', 'contrastive']:
            binary_file = exp1_dir / f'{model}_binary.json'
            multiclass_file = exp1_dir / f'{model}_multiclass.json'
            
            if binary_file.exists():
                with open(binary_file, 'r') as f:
                    result = json.load(f)
                    result['experiment'] = 'Exp1: Zero-Shot'
                    result['model'] = model.upper()
                    all_results.append(result)
            
            if multiclass_file.exists():
                with open(multiclass_file, 'r') as f:
                    result = json.load(f)
                    result['experiment'] = 'Exp1: Zero-Shot'
                    result['model'] = model.upper()
                    all_results.append(result)
    
    # Experiment 2: Fine-Tuning
    exp2_dir = results_dir / 'exp2_finetune'
    if exp2_dir.exists():
        for model in ['vae', 'contrastive']:
            model_dir = exp2_dir / model
            binary_file = model_dir / 'binary_results.json'
            multiclass_file = model_dir / 'multiclass_results.json'
            
            if binary_file.exists():
                with open(binary_file, 'r') as f:
                    result = json.load(f)
                    result['experiment'] = 'Exp2: Fine-Tuned'
                    result['model'] = model.upper()
                    all_results.append(result)
            
            if multiclass_file.exists():
                with open(multiclass_file, 'r') as f:
                    result = json.load(f)
                    result['experiment'] = 'Exp2: Fine-Tuned'
                    result['model'] = model.upper()
                    all_results.append(result)
    
    # Experiment 3: From Scratch
    exp3_dir = results_dir / 'exp3_from_scratch'
    if exp3_dir.exists():
        for model in ['vae', 'contrastive']:
            model_dir = exp3_dir / model
            binary_file = model_dir / 'binary_results.json'
            multiclass_file = model_dir / 'multiclass_results.json'
            
            if binary_file.exists():
                with open(binary_file, 'r') as f:
                    result = json.load(f)
                    result['experiment'] = 'Exp3: From Scratch'
                    result['model'] = model.upper()
                    all_results.append(result)
            
            if multiclass_file.exists():
                with open(multiclass_file, 'r') as f:
                    result = json.load(f)
                    result['experiment'] = 'Exp3: From Scratch'
                    result['model'] = model.upper()
                    all_results.append(result)
    
    return all_results


def create_comparison_table(results, task='binary'):
    """Create comparison table for specific task."""
    filtered = [r for r in results if r.get('task') == task]
    
    if task == 'binary':
        data = []
        for r in filtered:
            data.append({
                'Experiment': r['experiment'],
                'Model': r['model'],
                'Accuracy': r['accuracy'],
                'Precision': r['precision'],
                'Recall': r['recall'],
                'F1': r['f1'],
                'AUC-ROC': r['auc_roc']
            })
        df = pd.DataFrame(data)
    
    elif task == 'multiclass':
        data = []
        for r in filtered:
            data.append({
                'Experiment': r['experiment'],
                'Model': r['model'],
                'Top-1 Acc': r['accuracy'],
                'Top-2 Acc': r.get('top2_accuracy', 0),
                'Macro F1': r['macro_f1'],
                'Num Classes': r['num_classes']
            })
        df = pd.DataFrame(data)
    
    return df


def plot_comparison_bars(df, metric, title, save_path=None):
    """Create bar plot comparing metric across experiments and models."""
    plt.figure(figsize=(12, 6))
    
    # Pivot for grouped bar chart
    pivot_df = df.pivot(index='Experiment', columns='Model', values=metric)
    
    ax = pivot_df.plot(kind='bar', width=0.8, edgecolor='black')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_heatmap(df, metrics, title, save_path=None):
    """Create heatmap of metrics across experiments and models."""
    # Create combined index
    df['Exp+Model'] = df['Experiment'] + ' - ' + df['Model']
    
    # Select metrics
    heatmap_data = df[['Exp+Model'] + metrics].set_index('Exp+Model')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()


def generate_summary_report(results, output_path):
    """Generate comprehensive summary report."""
    report = []
    
    report.append("="*80)
    report.append("GAIT PATHOLOGY DETECTION - EXPERIMENTAL RESULTS SUMMARY")
    report.append("="*80)
    report.append("")
    
    # Binary classification results
    binary_df = create_comparison_table(results, task='binary')
    if not binary_df.empty:
        report.append("BINARY CLASSIFICATION (Normal vs Pathological)")
        report.append("-"*80)
        report.append(binary_df.to_string(index=False))
        report.append("")
        
        # Find best models
        best_acc = binary_df.loc[binary_df['Accuracy'].idxmax()]
        best_auc = binary_df.loc[binary_df['AUC-ROC'].idxmax()]
        
        report.append("Best Accuracy:  {:.4f} ({} - {})".format(
            best_acc['Accuracy'], best_acc['Experiment'], best_acc['Model']
        ))
        report.append("Best AUC-ROC:   {:.4f} ({} - {})".format(
            best_auc['AUC-ROC'], best_auc['Experiment'], best_auc['Model']
        ))
        report.append("")
    
    # Multi-class results
    multiclass_df = create_comparison_table(results, task='multiclass')
    if not multiclass_df.empty:
        report.append("MULTI-CLASS CLASSIFICATION (Specific Conditions)")
        report.append("-"*80)
        report.append(multiclass_df.to_string(index=False))
        report.append("")
        
        # Find best models
        best_top1 = multiclass_df.loc[multiclass_df['Top-1 Acc'].idxmax()]
        best_f1 = multiclass_df.loc[multiclass_df['Macro F1'].idxmax()]
        
        report.append("Best Top-1 Acc: {:.4f} ({} - {})".format(
            best_top1['Top-1 Acc'], best_top1['Experiment'], best_top1['Model']
        ))
        report.append("Best Macro F1:  {:.4f} ({} - {})".format(
            best_f1['Macro F1'], best_f1['Experiment'], best_f1['Model']
        ))
        report.append("")
    
    # Key findings
    report.append("="*80)
    report.append("KEY FINDINGS")
    report.append("="*80)
    report.append("")
    
    if not binary_df.empty:
        # Compare VAE vs Contrastive
        vae_avg = binary_df[binary_df['Model'] == 'VAE']['AUC-ROC'].mean()
        cont_avg = binary_df[binary_df['Model'] == 'CONTRASTIVE']['AUC-ROC'].mean()
        
        report.append("1. VAE vs Contrastive VAE (Binary Classification):")
        report.append(f"   - Average AUC-ROC (VAE):          {vae_avg:.4f}")
        report.append(f"   - Average AUC-ROC (Contrastive):  {cont_avg:.4f}")
        report.append(f"   - Improvement:                    {((cont_avg/vae_avg - 1)*100):.2f}%")
        report.append("")
        
        # Compare experiments
        exp_aucs = binary_df.groupby('Experiment')['AUC-ROC'].mean()
        report.append("2. Experiment Comparison (Average AUC-ROC):")
        for exp, auc in exp_aucs.items():
            report.append(f"   - {exp:<25} {auc:.4f}")
        report.append("")
    
    report.append("="*80)
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    # Also print to console
    print('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description='Compare All Experiment Results')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Root directory containing all experiment results')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                        help='Output directory for comparison visualizations')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading experiment results...")
    results = load_experiment_results(args.results_dir)
    
    if not results:
        print("❌ No results found. Run experiments first.")
        return
    
    print(f"✅ Loaded {len(results)} result files\n")
    
    # Generate comparison tables
    print("="*80)
    print("GENERATING COMPARISON TABLES")
    print("="*80 + "\n")
    
    binary_df = create_comparison_table(results, task='binary')
    multiclass_df = create_comparison_table(results, task='multiclass')
    
    # Save tables as CSV
    if not binary_df.empty:
        binary_df.to_csv(output_dir / 'binary_comparison.csv', index=False)
        print("✅ Binary comparison table saved")
    
    if not multiclass_df.empty:
        multiclass_df.to_csv(output_dir / 'multiclass_comparison.csv', index=False)
        print("✅ Multi-class comparison table saved")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    if not binary_df.empty:
        plot_comparison_bars(
            binary_df, 'AUC-ROC',
            'Binary Classification: AUC-ROC Comparison',
            save_path=output_dir / 'binary_auc_comparison.png'
        )
        
        plot_comparison_bars(
            binary_df, 'F1',
            'Binary Classification: F1 Score Comparison',
            save_path=output_dir / 'binary_f1_comparison.png'
        )
        
        plot_heatmap(
            binary_df, ['Accuracy', 'F1', 'AUC-ROC'],
            'Binary Classification: Metrics Heatmap',
            save_path=output_dir / 'binary_heatmap.png'
        )
    
    if not multiclass_df.empty:
        plot_comparison_bars(
            multiclass_df, 'Top-1 Acc',
            'Multi-class Classification: Top-1 Accuracy Comparison',
            save_path=output_dir / 'multiclass_accuracy_comparison.png'
        )
        
        plot_comparison_bars(
            multiclass_df, 'Macro F1',
            'Multi-class Classification: Macro F1 Comparison',
            save_path=output_dir / 'multiclass_f1_comparison.png'
        )
    
    # Generate summary report
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80 + "\n")
    
    generate_summary_report(results, output_dir / 'summary_report.txt')
    
    print(f"\n✅ Comparison complete! All results saved to {output_dir}")


if __name__ == '__main__':
    main()
