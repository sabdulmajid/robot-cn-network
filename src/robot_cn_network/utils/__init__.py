import logging
import yaml
import json
import pickle
import os
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    logging_config = {
        'level': getattr(logging, level.upper()),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        logging_config['filename'] = log_file
        logging_config['filemode'] = 'a'
    
    logging.basicConfig(**logging_config)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.create(config_dict)


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    if device == "auto" or device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def save_model(
    model: torch.nn.Module,
    save_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {'model_state_dict': model.state_dict(), 'model_config': getattr(model, 'config', None)}
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, save_path)


def load_model(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch'),
        'metadata': checkpoint.get('metadata', {})
    }


def save_dataset(data: List[Dict[str, Any]], save_path: Union[str, Path]) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def load_dataset(dataset_path: Union[str, Path]) -> List[Dict[str, Any]]:
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    mse_per_dim = np.mean((predictions - targets) ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(predictions - targets), axis=0)
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse)),
    }
    
    for i, (mse_dim, mae_dim) in enumerate(zip(mse_per_dim, mae_per_dim)):
        metrics[f'mse_dim_{i}'] = float(mse_dim)
        metrics[f'mae_dim_{i}'] = float(mae_dim)
    
    return metrics


def init_wandb(
    project_name: str,
    config: DictConfig,
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> None:
    if not WANDB_AVAILABLE:
        print("Warning: wandb not available, skipping initialization")
        return
    wandb.init(
        project=project_name,
        config=OmegaConf.to_container(config, resolve=True),
        name=run_name,
        tags=tags
    )


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)


def create_video_from_frames(
    frames: List[np.ndarray],
    save_path: Union[str, Path],
    fps: int = 20
) -> None:
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not available, cannot create video")
        return
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_frames = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        processed_frames.append(frame)
    
    imageio.mimsave(str(save_path), processed_frames, fps=fps)


def normalize_observations(obs: Dict[str, Any]) -> Dict[str, Any]:
    normalized_obs = obs.copy()
    
    if 'pixels' in obs:
        for camera_name in obs['pixels']:
            image = obs['pixels'][camera_name]
            if image.dtype == np.uint8:
                normalized_obs['pixels'][camera_name] = image.astype(np.float32) / 255.0
    
    if 'agent_pos' in obs:
        normalized_obs['agent_pos'] = obs['agent_pos']
    
    return normalized_obs


class EpisodeBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
    
    def add_step(self, observation: Dict[str, Any], action: np.ndarray, reward: float, done: bool, info: Dict[str, Any]) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
    
    def get_episode(self) -> Dict[str, Any]:
        return {
            'observations': self.observations,
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'infos': self.infos,
            'length': len(self.observations)
        }
    
    def clear(self) -> None:
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []


def print_system_info() -> None:
    import platform
    import torch
    
    print("System Information:")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon): Available")
    else:
        print("No GPU acceleration available")
