"""Checkpoint utilities for saving and loading model checkpoints."""

import os
import torch


def save_checkpoint(model, optimizer, step, total_training_time, config, checkpoint_dir, filename, val_bpb=None):
    """Save model checkpoint to disk.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        step: Current training step
        total_training_time: Total training time in seconds
        config: Model configuration
        checkpoint_dir: Directory to save checkpoint
        filename: Name of checkpoint file
        val_bpb: Optional validation BPB score
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'total_training_time': total_training_time,
        'config': config,
    }
    if val_bpb is not None:
        data['val_bpb'] = val_bpb
    torch.save(data, checkpoint_path)
    return checkpoint_path
