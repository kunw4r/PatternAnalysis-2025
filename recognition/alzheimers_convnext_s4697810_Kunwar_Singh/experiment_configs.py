"""
Experiment Configurations for Alzheimer's Disease Classification
Rangpur HPC - Storage-Efficient Sequential Experiments
"""

# All experiments use these common settings
COMMON_CONFIG = {
    'data_dir': '/home/groups/comp3710/ADNI/AD_NC',
    'weight_decay': 0.01,
    'img_size': 224,
    'num_workers': 1,
    'patience': 10,
    'use_wandb': False,
    'save_dir': './checkpoints',
}

# Define experiments
# Each experiment will be run sequentially: Train → Predict → Save Results
EXPERIMENTS = {
    
    # ============================================================================
    # BASELINE EXPERIMENTS (Standard configurations)
    # ============================================================================
    
    '1_tiny_onecycle': {
        'model_name': 'convnext_tiny',
        'dropout': 0.5,
        'learning_rate': 1e-4,
        'scheduler_type': 'onecycle',
        'loss_type': 'label_smoothing',
        'label_smoothing': 0.1,
        'batch_size': 64,
        'num_epochs': 30,  # Increased from 20 - model still improving at epoch 20
        'pretrained': False,
    },
    
    '2_small_onecycle': {
        'model_name': 'convnext_small',
        'dropout': 0.5,
        'learning_rate': 1e-4,
        'scheduler_type': 'onecycle',
        'loss_type': 'label_smoothing',
        'label_smoothing': 0.1,
        'batch_size': 32,
        'num_epochs': 30,  # Increased from 20 - consistent with baselines
        'pretrained': False,
    },
    
    '3_base_onecycle': {
        'model_name': 'convnext_base',
        'dropout': 0.5,
        'learning_rate': 1e-4,
        'scheduler_type': 'onecycle',
        'loss_type': 'label_smoothing',
        'label_smoothing': 0.1,
        'batch_size': 16,
        'num_epochs': 30,  # Increased from 20 - consistent with baselines
        'pretrained': False,
    },
    
    # ============================================================================
    # ADVANCED EXPERIMENTS (Targeting 80%+ test accuracy)
    # ============================================================================
    
    '4_base_pretrained': {
        'model_name': 'convnext_base',
        'dropout': 0.3,
        'learning_rate': 1e-4,
        'scheduler_type': 'onecycle',
        'loss_type': 'focal',
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'batch_size': 16,
        'num_epochs': 30,
        'pretrained': True,          # ImageNet pretrained
        'pretrain_stages': 'all',
    },
    
    '5_base_focal_aggressive': {
        'model_name': 'convnext_base',
        'dropout': 0.3,
        'learning_rate': 1e-4,
        'scheduler_type': 'onecycle',
        'loss_type': 'focal',
        'focal_alpha': 0.5,          # Higher alpha
        'focal_gamma': 3.0,          # Higher gamma
        'batch_size': 16,
        'num_epochs': 30,
        'pretrained': False,
    },
    
    '6_base_cosine_highLR': {
        'model_name': 'convnext_base',
        'dropout': 0.4,
        'learning_rate': 5e-4,       # 5x higher LR
        'scheduler_type': 'cosine',
        'loss_type': 'label_smoothing',
        'label_smoothing': 0.1,
        'batch_size': 16,
        'num_epochs': 30,
        'pretrained': False,
    },
    
    '7_small_mixup': {
        'model_name': 'convnext_small',
        'dropout': 0.5,
        'learning_rate': 1e-4,
        'scheduler_type': 'onecycle',
        'loss_type': 'label_smoothing',
        'label_smoothing': 0.1,
        'batch_size': 32,
        'num_epochs': 30,  # Increased from 20 - consistent with baselines
        'use_mixup': True,           # MixUp augmentation
        'mixup_alpha': 0.4,
        'pretrained': False,
    },
    
    '8_base_crossentropy': {
        'model_name': 'convnext_base',
        'dropout': 0.2,
        'learning_rate': 5e-4,
        'scheduler_type': 'onecycle',
        'loss_type': 'cross_entropy',  # No smoothing
        'batch_size': 16,
        'num_epochs': 40,
        'pretrained': False,
    },
}


def get_experiment_config(exp_name):
    """
    Get full configuration for an experiment
    
    Args:
        exp_name: Name of experiment (e.g., '1_tiny_onecycle')
    
    Returns:
        dict: Complete configuration merging common and experiment-specific settings
    """
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_name}. Available: {list(EXPERIMENTS.keys())}")
    
    # Merge common config with experiment-specific config
    config = {**COMMON_CONFIG, **EXPERIMENTS[exp_name]}
    return config


def list_experiments():
    """Print all available experiments with details"""
    print("\n" + "=" * 80)
    print("ALZHEIMER'S DISEASE CLASSIFICATION - EXPERIMENT CONFIGURATIONS")
    print("=" * 80)
    print(f"\nTotal experiments: {len(EXPERIMENTS)}")
    print("Strategy: Sequential training with auto-cleanup (saves storage)")
    print("\n" + "=" * 80)
    print("EXPERIMENTS:")
    print("=" * 80)
    
    baseline = ['1_tiny_onecycle', '2_small_onecycle', '3_base_onecycle']
    advanced = ['4_base_pretrained', '5_base_focal_aggressive', '6_base_cosine_highLR', 
                '7_small_mixup', '8_base_crossentropy']
    
    print("\n📊 BASELINE EXPERIMENTS (3 - All 30 epochs):")
    for name in baseline:
        if name in EXPERIMENTS:
            config = EXPERIMENTS[name]
            print(f"\n  {name}:")
            print(f"    Model:     {config['model_name']}")
            print(f"    Scheduler: {config['scheduler_type']}")
            print(f"    Loss:      {config['loss_type']}")
            print(f"    Batch:     {config['batch_size']}")
            print(f"    Epochs:    {config['num_epochs']}")
    
    print("\n🚀 ADVANCED EXPERIMENTS (5 - Targeting 80%+ test accuracy):")
    for name in advanced:
        if name in EXPERIMENTS:
            config = EXPERIMENTS[name]
            special = []
            if config.get('pretrained'):
                special.append("PRETRAINED")
            if config.get('use_mixup'):
                special.append("MIXUP")
            if config.get('loss_type') == 'focal':
                special.append(f"Focal(α={config.get('focal_alpha', 0.25)},γ={config.get('focal_gamma', 2.0)})")
            if config.get('learning_rate', 1e-4) > 1e-4:
                special.append(f"HighLR({config.get('learning_rate')})")
            
            print(f"\n  {name}:")
            print(f"    Model:     {config['model_name']}")
            print(f"    Special:   {', '.join(special) if special else 'Standard'}")
            print(f"    Epochs:    {config['num_epochs']}")
    
    print("\n" + "=" * 80)
    print("💡 TIPS:")
    print("  - Run all: python run_experiments_sequential.py --experiments all")
    print("  - Run one: python run_experiments_sequential.py --experiments 4_base_pretrained")
    print("  - Keep best 2: python run_experiments_sequential.py --keep-best 2")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    list_experiments()
