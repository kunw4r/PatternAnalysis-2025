"""
Sequential Experiment Runner with Auto-Prediction and Cleanup
Runs experiments one at a time: Train → Predict → Save Results → Delete Checkpoint

This saves storage by not keeping all checkpoints at once.
Perfect for Rangpur HPC with limited storage (8GB).
"""

import os
import sys
import json
import time
import shutil
import traceback
from datetime import datetime
import pandas as pd

from experiment_configs import EXPERIMENTS, get_experiment_config
from train import train
from dataset import get_dataloaders


class SequentialExperimentRunner:
    """
    Runs experiments sequentially with automatic prediction and cleanup
    """
    
    def __init__(self, results_dir='./results', keep_best_n=1):
        """
        Args:
            results_dir: Directory to save all results
            keep_best_n: Number of best models to keep (by test accuracy)
        """
        self.results_dir = results_dir
        self.keep_best_n = keep_best_n
        self.checkpoints_dir = './checkpoints'
        
        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Results tracking
        self.results_file = os.path.join(results_dir, 'all_experiments_results.json')
        self.results_csv = os.path.join(results_dir, 'all_experiments_results.csv')
        self.results = self.load_results()
        
    def load_results(self):
        """Load previous experiment results"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_results(self):
        """Save experiment results to JSON and CSV"""
        # Save JSON
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV
        if self.results:
            df = pd.DataFrame.from_dict(self.results, orient='index')
            df.to_csv(self.results_csv)
            print(f"\n✓ Results saved to:")
            print(f"  - {self.results_file}")
            print(f"  - {self.results_csv}")
    
    def run_prediction(self, experiment_name, checkpoint_path):
        """
        Run prediction on a trained model
        
        Returns:
            dict: Test results including accuracy and metrics
        """
        print("\n" + "=" * 80)
        print(f"RUNNING PREDICTION: {experiment_name}")
        print("=" * 80)
        
        import torch
        from modules import get_model
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        import numpy as np
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model = get_model(
            model_name=config['model_name'],
            num_classes=2,
            dropout=config['dropout'],
            pretrained=config.get('pretrained', False),
            pretrain_stages=config.get('pretrain_stages', 'all')
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load test data
        _, _, test_loader = get_dataloaders(
            data_dir=config.get('data_dir', '/home/groups/comp3710/ADNI/AD_NC'),
            batch_size=32,
            num_workers=1,
            img_size=config.get('img_size', 224),
            val_split=0.2
        )
        
        # Evaluate
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        # Calculate metrics
        accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
        
        # Per-class accuracy
        nc_mask = all_labels == 0
        ad_mask = all_labels == 1
        nc_acc = 100 * (all_preds[nc_mask] == all_labels[nc_mask]).sum() / nc_mask.sum()
        ad_acc = 100 * (all_preds[ad_mask] == all_labels[ad_mask]).sum() / ad_mask.sum()
        
        # F1, Precision, Recall
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'test_accuracy': float(accuracy),
            'test_nc_accuracy': float(nc_acc),
            'test_ad_accuracy': float(ad_acc),
            'test_f1_nc': float(f1[0]),
            'test_f1_ad': float(f1[1]),
            'test_precision_nc': float(precision[0]),
            'test_precision_ad': float(precision[1]),
            'test_recall_nc': float(recall[0]),
            'test_recall_ad': float(recall[1]),
            'confusion_matrix': cm.tolist(),
        }
        
        print(f"\n{'=' * 80}")
        print(f"TEST RESULTS: {experiment_name}")
        print(f"{'=' * 80}")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"NC Accuracy:      {nc_acc:.2f}%")
        print(f"AD Accuracy:      {ad_acc:.2f}%")
        print(f"F1 Score (NC):    {f1[0]:.4f}")
        print(f"F1 Score (AD):    {f1[1]:.4f}")
        print(f"{'=' * 80}\n")
        
        # Save detailed results
        detailed_results = {
            'experiment_name': experiment_name,
            'checkpoint_info': config,
            'metrics': results
        }
        
        results_path = os.path.join(self.results_dir, f'test_results_{experiment_name}.json')
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"✓ Detailed results saved to {results_path}")
        
        return results
    
    def cleanup_checkpoint(self, checkpoint_path, experiment_name):
        """
        Delete checkpoint file to save space
        
        Args:
            checkpoint_path: Path to checkpoint file
            experiment_name: Name of experiment (for logging)
        """
        try:
            if os.path.exists(checkpoint_path):
                file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                os.remove(checkpoint_path)
                print(f"🗑️  Deleted checkpoint: {checkpoint_path} ({file_size_mb:.1f} MB freed)")
                
                # Also delete config file if it exists
                config_path = checkpoint_path.replace('best_model_', 'config_').replace('.pth', '.json')
                if os.path.exists(config_path):
                    os.remove(config_path)
                    print(f"🗑️  Deleted config: {config_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete {checkpoint_path}: {e}")
    
    def keep_best_models(self):
        """
        Keep only the top N models by test accuracy, delete others
        """
        if not self.results or len(self.results) <= self.keep_best_n:
            return
        
        print(f"\n{'=' * 80}")
        print(f"KEEPING BEST {self.keep_best_n} MODEL(S)")
        print(f"{'=' * 80}")
        
        # Sort by test accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get('test_accuracy', 0),
            reverse=True
        )
        
        # Keep best N
        keep_experiments = [exp_name for exp_name, _ in sorted_results[:self.keep_best_n]]
        
        print(f"\nKeeping:")
        for exp_name in keep_experiments:
            acc = self.results[exp_name].get('test_accuracy', 0)
            print(f"  ✓ {exp_name} ({acc:.2f}%)")
        
        # Delete others
        print(f"\nDeleting:")
        for exp_name, result in sorted_results[self.keep_best_n:]:
            acc = result.get('test_accuracy', 0)
            print(f"  🗑️  {exp_name} ({acc:.2f}%)")
            
            # Delete checkpoint and curves
            checkpoint_path = os.path.join(self.checkpoints_dir, f'best_model_{exp_name}.pth')
            curves_path = os.path.join(self.checkpoints_dir, f'training_curves_{exp_name}.png')
            
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            if os.path.exists(curves_path):
                os.remove(curves_path)
        
        print(f"{'=' * 80}\n")
    
    def run_experiment(self, experiment_name, config):
        """
        Run single experiment: Train → Predict → Cleanup
        
        Returns:
            dict: Combined results from training and testing
        """
        print("\n" + "#" * 80)
        print(f"EXPERIMENT: {experiment_name}")
        print("#" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        start_time = time.time()
        
        try:
            # STEP 1: TRAIN
            print(f"\n{'=' * 80}")
            print(f"STEP 1: TRAINING")
            print(f"{'=' * 80}\n")
            
            # Add experiment name to config
            config['experiment_name'] = experiment_name
            
            # Train model
            model, history = train(**config)
            
            train_time = time.time() - start_time
            
            # Get checkpoint path
            checkpoint_path = os.path.join(self.checkpoints_dir, f'best_model_{experiment_name}.pth')
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # STEP 2: PREDICT
            print(f"\n{'=' * 80}")
            print(f"STEP 2: PREDICTION")
            print(f"{'=' * 80}\n")
            
            test_results = self.run_prediction(experiment_name, checkpoint_path)
            
            # STEP 3: Store checkpoint path for later cleanup
            # Don't delete yet - we'll decide which to keep at the end
            print(f"\n{'=' * 80}")
            print(f"STEP 3: CHECKPOINT SAVED")
            print(f"{'=' * 80}")
            checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"✓ Checkpoint saved: {checkpoint_path} ({checkpoint_size_mb:.1f} MB)")
            print(f"  Will keep best {self.keep_best_n} model(s) at the end\n")
            
            # Compile results
            total_time = time.time() - start_time
            
            result = {
                'experiment_name': experiment_name,
                'status': 'completed',
                'train_time_minutes': train_time / 60,
                'total_time_minutes': total_time / 60,
                'best_val_acc': history['best_val_acc'],
                'best_epoch': history['best_epoch'],
                'total_epochs': len(history['train_accs']),
                'model_name': config['model_name'],
                'scheduler_type': config['scheduler_type'],
                'loss_type': config.get('loss_type', 'label_smoothing'),
                'pretrained': config.get('pretrained', False),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                **test_results  # Add all test metrics
            }
            
            print(f"\n{'=' * 80}")
            print(f"✅ EXPERIMENT COMPLETE: {experiment_name}")
            print(f"{'=' * 80}")
            print(f"Validation Acc: {result['best_val_acc']:.2f}%")
            print(f"Test Acc:       {result['test_accuracy']:.2f}%")
            print(f"Training Time:  {result['train_time_minutes']:.1f} min")
            print(f"Total Time:     {result['total_time_minutes']:.1f} min")
            print(f"{'=' * 80}\n")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            
            print("\n" + "!" * 80)
            print(f"❌ ERROR IN EXPERIMENT: {experiment_name}")
            print("!" * 80)
            print(f"Error: {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print("!" * 80 + "\n")
            
            return {
                'experiment_name': experiment_name,
                'status': 'failed',
                'error': str(e),
                'total_time_minutes': total_time / 60,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
    
    def run_all(self, experiment_names=None, skip_completed=True):
        """
        Run multiple experiments sequentially
        
        Args:
            experiment_names: List of experiment names to run (None = all)
            skip_completed: Skip already completed experiments
        """
        if experiment_names is None:
            experiments_to_run = list(EXPERIMENTS.keys())
        else:
            experiments_to_run = experiment_names
        
        print("\n" + "=" * 80)
        print("SEQUENTIAL EXPERIMENT RUNNER WITH AUTO-CLEANUP")
        print("=" * 80)
        print(f"Total experiments: {len(experiments_to_run)}")
        print(f"Skip completed: {skip_completed}")
        print(f"Keep best N: {self.keep_best_n}")
        print(f"Results directory: {self.results_dir}")
        print("=" * 80 + "\n")
        
        # Filter completed
        if skip_completed:
            pending = [e for e in experiments_to_run 
                      if e not in self.results or self.results[e].get('status') != 'completed']
            print(f"Completed: {len(experiments_to_run) - len(pending)}")
            print(f"Pending: {len(pending)}\n")
            experiments_to_run = pending
        
        if not experiments_to_run:
            print("\n✅ All experiments already completed!")
            self.print_final_summary()
            return
        
        # Run each experiment
        for i, exp_name in enumerate(experiments_to_run, 1):
            print(f"\n{'#' * 80}")
            print(f"EXPERIMENT {i}/{len(experiments_to_run)}")
            print(f"{'#' * 80}\n")
            
            config = get_experiment_config(exp_name)
            result = self.run_experiment(exp_name, config)
            
            self.results[exp_name] = result
            self.save_results()
            
            # Print progress
            self.print_progress()
        
        # Keep only best models
        if self.keep_best_n > 0:
            self.keep_best_models()
        
        self.print_final_summary()
    
    def print_progress(self):
        """Print current progress"""
        completed = [r for r in self.results.values() if r.get('status') == 'completed']
        
        if completed:
            print(f"\n{'=' * 80}")
            print(f"📊 PROGRESS ({len(completed)} completed)")
            print(f"{'=' * 80}")
            
            # Sort by test accuracy
            sorted_results = sorted(completed, key=lambda x: x.get('test_accuracy', 0), reverse=True)
            
            print(f"{'Experiment':<45s} | {'Val Acc':<8s} | {'Test Acc':<8s}")
            print("-" * 80)
            for r in sorted_results[:5]:  # Top 5
                print(f"{r['experiment_name']:<45s} | {r['best_val_acc']:>6.2f}% | {r['test_accuracy']:>6.2f}%")
            
            if len(sorted_results) > 5:
                print(f"  ... and {len(sorted_results) - 5} more")
            
            print(f"{'=' * 80}\n")
    
    def print_final_summary(self):
        """Print final summary"""
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        
        completed = [r for r in self.results.values() if r.get('status') == 'completed']
        failed = [r for r in self.results.values() if r.get('status') == 'failed']
        
        print(f"\nTotal: {len(self.results)}")
        print(f"  ✅ Completed: {len(completed)}")
        print(f"  ❌ Failed: {len(failed)}")
        
        if completed:
            print("\n" + "=" * 80)
            print("🏆 TOP MODELS BY TEST ACCURACY")
            print("=" * 80)
            print(f"{'Experiment':<45s} | {'Val Acc':<8s} | {'Test Acc':<8s} | {'Time':<8s}")
            print("-" * 80)
            
            top_models = sorted(completed, key=lambda x: x.get('test_accuracy', 0), reverse=True)
            for r in top_models:
                print(f"{r['experiment_name']:<45s} | {r['best_val_acc']:>6.2f}% | "
                      f"{r['test_accuracy']:>6.2f}% | {r['total_time_minutes']:>6.1f}m")
            
            # Best model info
            best = top_models[0]
            print("\n" + "=" * 80)
            print("🥇 BEST MODEL:")
            print(f"   {best['experiment_name']}")
            print(f"   Validation: {best['best_val_acc']:.2f}%")
            print(f"   Test:       {best['test_accuracy']:.2f}%")
            print(f"   F1 (AD):    {best['test_f1_ad']:.4f}")
            print(f"   F1 (NC):    {best['test_f1_nc']:.4f}")
        
        if failed:
            print(f"\n❌ FAILED EXPERIMENTS ({len(failed)}):")
            for r in failed:
                print(f"  - {r['experiment_name']}: {r.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 80)
        print(f"Results saved to: {self.results_file}")
        print("=" * 80 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sequential experiment runner with auto-cleanup')
    parser.add_argument('--experiments', type=str, default='all',
                       help='Comma-separated experiment names or "all"')
    parser.add_argument('--skip-completed', action='store_true', default=True,
                       help='Skip already completed experiments')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--keep-best', type=int, default=1,
                       help='Number of best models to keep (0 = keep none)')
    
    args = parser.parse_args()
    
    # Parse experiment names
    if args.experiments.lower() == 'all':
        experiment_names = None
    else:
        experiment_names = [e.strip() for e in args.experiments.split(',')]
    
    # Run experiments
    runner = SequentialExperimentRunner(
        results_dir=args.results_dir,
        keep_best_n=args.keep_best
    )
    runner.run_all(
        experiment_names=experiment_names,
        skip_completed=args.skip_completed
    )


if __name__ == '__main__':
    main()
