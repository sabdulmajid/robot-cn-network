import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from robot_cn_network.models import ACTPolicy, ModelConfig
from robot_cn_network.data import RobotDataset, create_train_val_datasets, create_dataloader
from robot_cn_network.utils import (
    setup_logging, load_config, save_config, set_seed, get_device,
    save_model, compute_metrics, init_wandb, log_metrics
)


def main():
    parser = argparse.ArgumentParser(description="Train Robot Imitation Learning Policy")
    
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--output-dir", type=str, default="./outputs/training")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-type", type=str, default="act", choices=["act", "diffusion"])
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="robot-cn-network")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--save-freq", type=int, default=10)
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    device = get_device(args.device)
    
    logger.info("Starting training")
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.dataset_path}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = create_default_config(args)
        logger.info("Using default configuration")
    
    save_config(config, output_dir / "config.yaml")
    
    if args.use_wandb:
        init_wandb(
            project_name=args.wandb_project,
            config=config,
            run_name=args.run_name,
            tags=["imitation_learning", args.policy_type]
        )
    
    logger.info("Loading datasets...")
    train_dataset, val_dataset = create_train_val_datasets(
        data_path=args.dataset_path,
        sequence_length=args.sequence_length,
        action_horizon=args.action_horizon,
        train_ratio=0.8,
        random_seed=args.seed
    )
    
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    logger.info("Creating model...")
    model_config = ModelConfig(
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_encoder_layers=config.model.num_encoder_layers,
        action_dim=config.model.action_dim,
        chunk_size=args.action_horizon
    )
    
    sample_obs = train_dataset[0]['observations'][0]
    state_dim = len(sample_obs.get('agent_pos', []))
    camera_names = list(sample_obs.get('pixels', {}).keys())
    
    if args.policy_type == "act":
        model = ACTPolicy(config=model_config, state_dim=state_dim, camera_names=camera_names)
    else:
        raise NotImplementedError(f"Policy type {args.policy_type} not implemented")
    
    model = model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=config.training.weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    logger.info("Starting training loop...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        
        if epoch % args.val_freq == 0:
            model.eval()
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
            
            metrics.update({
                'val_loss': val_loss,
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(
                    model, output_dir / "best_model.pth", optimizer=optimizer, epoch=epoch,
                    metadata={'val_loss': val_loss, 'train_loss': train_loss}
                )
                logger.info(f"New best model saved (val_loss: {val_loss:.6f})")
        
        if epoch % args.save_freq == 0:
            save_model(
                model, output_dir / f"checkpoint_epoch_{epoch:03d}.pth", optimizer=optimizer, epoch=epoch,
                metadata={'val_loss': val_loss if 'val_loss' in metrics else None}
            )
        
        if args.use_wandb:
            log_metrics(metrics, step=epoch)
        
        logger.info(
            f"Epoch {epoch:3d}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {metrics.get('val_loss', 'N/A')}"
        )
    
    save_model(model, output_dir / "final_model.pth", optimizer=optimizer, epoch=args.num_epochs - 1, metadata={'final_epoch': True})
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Models saved in: {output_dir}")


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device, epoch: int) -> float:
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        observations = batch['observations']
        actions = batch['actions'].to(device)
        
        batch_size = len(observations)
        batch_images = {}
        batch_states = []
        
        for sample_obs_seq in observations:
            last_obs = sample_obs_seq[-1]
            
            if 'pixels' in last_obs:
                for camera_name, image in last_obs['pixels'].items():
                    if camera_name not in batch_images:
                        batch_images[camera_name] = []
                    batch_images[camera_name].append(image)
            
            if 'agent_pos' in last_obs:
                batch_states.append(last_obs['agent_pos'])
        
        for camera_name in batch_images:
            batch_images[camera_name] = torch.stack(batch_images[camera_name]).to(device)
        
        if batch_states:
            batch_states = torch.stack(batch_states).to(device)
        else:
            batch_states = torch.zeros(batch_size, 7).to(device)
        
        predicted_actions = model(batch_images, batch_states)
        loss = criterion(predicted_actions, actions)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            progress = 100.0 * batch_idx / num_batches
            print(f"\rEpoch {epoch} [{batch_idx:3d}/{num_batches}] ({progress:3.0f}%) Loss: {loss.item():.6f}", end='')
    
    print()
    return total_loss / num_batches


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, epoch: int) -> tuple:
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            observations = batch['observations']
            actions = batch['actions'].to(device)
            
            batch_size = len(observations)
            batch_images = {}
            batch_states = []
            
            for sample_obs_seq in observations:
                last_obs = sample_obs_seq[-1]
                
                if 'pixels' in last_obs:
                    for camera_name, image in last_obs['pixels'].items():
                        if camera_name not in batch_images:
                            batch_images[camera_name] = []
                        batch_images[camera_name].append(image)
                
                if 'agent_pos' in last_obs:
                    batch_states.append(last_obs['agent_pos'])
            
            for camera_name in batch_images:
                batch_images[camera_name] = torch.stack(batch_images[camera_name]).to(device)
            
            if batch_states:
                batch_states = torch.stack(batch_states).to(device)
            else:
                batch_states = torch.zeros(batch_size, 7).to(device)
            
            predicted_actions = model(batch_images, batch_states)
            loss = criterion(predicted_actions, actions)
            total_loss += loss.item()
            
            all_predictions.append(predicted_actions.cpu().numpy())
            all_targets.append(actions.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    predictions_flat = predictions.reshape(-1, predictions.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])
    
    metrics = compute_metrics(predictions_flat, targets_flat)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics


def create_default_config(args) -> dict:
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'model': {
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'action_dim': 7,
        },
        'training': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'weight_decay': 1e-4,
            'grad_clip_norm': 1.0,
        },
        'data': {
            'sequence_length': args.sequence_length,
            'action_horizon': args.action_horizon,
        }
    })
    
    return config


if __name__ == "__main__":
    main()
