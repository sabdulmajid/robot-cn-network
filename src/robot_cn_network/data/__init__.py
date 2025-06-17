import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union
import pickle
from pathlib import Path
import logging
from robot_cn_network.utils import normalize_observations

logger = logging.getLogger(__name__)


class RobotDataset(Dataset):
    def __init__(self, data_path: Union[str, Path], sequence_length: int = 1, action_horizon: int = 1, transform: Optional[callable] = None):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.action_horizon = action_horizon
        self.transform = transform
        
        self.episodes = self._load_episodes()
        self.samples = self._create_samples()
        
        logger.info(f"Loaded {len(self.episodes)} episodes with {len(self.samples)} samples")
    
    def _load_episodes(self) -> List[Dict[str, Any]]:
        episodes = []
        
        if self.data_path.is_file():
            with open(self.data_path, 'rb') as f:
                episodes = pickle.load(f)
        else:
            episode_files = sorted(self.data_path.glob("episode_*.pkl"))
            
            for episode_file in episode_files:
                with open(episode_file, 'rb') as f:
                    episode_data = pickle.load(f)
                    episodes.append(episode_data)
        
        return episodes
    
    def _create_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        for episode_idx, episode in enumerate(self.episodes):
            episode_length = len(episode)
            
            for start_idx in range(episode_length - self.sequence_length - self.action_horizon + 1):
                sample = {
                    'episode_idx': episode_idx,
                    'start_idx': start_idx,
                    'obs_indices': list(range(start_idx, start_idx + self.sequence_length)),
                    'action_indices': list(range(
                        start_idx + self.sequence_length,
                        start_idx + self.sequence_length + self.action_horizon
                    ))
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        episode = self.episodes[sample['episode_idx']]
        
        observations = []
        for obs_idx in sample['obs_indices']:
            obs = episode[obs_idx]['observation']
            if self.transform:
                obs = self.transform(obs)
            else:
                obs = normalize_observations(obs)
            observations.append(obs)
        
        actions = []
        for action_idx in sample['action_indices']:
            actions.append(episode[action_idx]['action'])
        
        return {
            'observations': observations,
            'actions': np.array(actions),
            'rewards': [episode[i]['reward'] for i in sample['action_indices']],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch_obs = []
    for sample in batch:
        obs_sequence = sample['observations']
        
        processed_sequence = []
        for obs in obs_sequence:
            processed_obs = {}
            
            if 'pixels' in obs:
                processed_obs['pixels'] = {}
                for camera_name, image in obs['pixels'].items():
                    if isinstance(image, np.ndarray):
                        if image.ndim == 3 and image.shape[-1] == 3:  # [H, W, C] -> [C, H, W]
                            image = image.transpose(2, 0, 1)
                        image = torch.FloatTensor(image)
                    processed_obs['pixels'][camera_name] = image
            
            if 'agent_pos' in obs:
                processed_obs['agent_pos'] = torch.FloatTensor(obs['agent_pos'])
            
            processed_sequence.append(processed_obs)
        
        batch_obs.append(processed_sequence)
    
    batch_actions = torch.FloatTensor([sample['actions'] for sample in batch])
    batch_rewards = torch.FloatTensor([sample['rewards'] for sample in batch])
    
    return {
        'observations': batch_obs,
        'actions': batch_actions,
        'rewards': batch_rewards
    }


def create_dataloader(
    dataset: RobotDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


class DataCollector:
    def __init__(self, save_dir: Union[str, Path]):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        
    def save_episode(self, episode_data: List[Dict[str, Any]]) -> str:
        filename = f"episode_{self.episode_count:04d}.pkl"
        filepath = self.save_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(episode_data, f)
        
        logger.info(f"Saved episode {self.episode_count} with {len(episode_data)} steps")
        self.episode_count += 1
        
        return str(filepath)
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        episode_files = sorted(self.save_dir.glob("episode_*.pkl"))
        
        if not episode_files:
            return {"num_episodes": 0, "total_steps": 0}
        
        total_steps = 0
        episode_lengths = []
        
        for episode_file in episode_files:
            with open(episode_file, 'rb') as f:
                episode_data = pickle.load(f)
                episode_length = len(episode_data)
                episode_lengths.append(episode_length)
                total_steps += episode_length
        
        return {
            "num_episodes": len(episode_files),
            "total_steps": total_steps,
            "avg_episode_length": np.mean(episode_lengths),
            "min_episode_length": np.min(episode_lengths),
            "max_episode_length": np.max(episode_lengths),
        }


def split_dataset(
    data_path: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    data_path = Path(data_path)
    episode_files = sorted([str(f) for f in data_path.glob("episode_*.pkl")])
    
    if not episode_files:
        raise ValueError(f"No episode files found in {data_path}")
    
    np.random.seed(random_seed)
    np.random.shuffle(episode_files)
    
    n_total = len(episode_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = episode_files[:n_train]
    val_files = episode_files[n_train:n_train + n_val]
    test_files = episode_files[n_train + n_val:]
    
    logger.info(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    return train_files, val_files, test_files


def create_train_val_datasets(
    data_path: Union[str, Path],
    sequence_length: int = 1,
    action_horizon: int = 1,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[RobotDataset, RobotDataset]:
    train_files, val_files, _ = split_dataset(
        data_path, train_ratio, 1 - train_ratio, 0, random_seed
    )
    
    temp_dir = Path(data_path).parent / "temp_split"
    train_dir = temp_dir / "train"
    val_dir = temp_dir / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    
    for i, train_file in enumerate(train_files):
        shutil.copy(train_file, train_dir / f"episode_{i:04d}.pkl")
    
    for i, val_file in enumerate(val_files):
        shutil.copy(val_file, val_dir / f"episode_{i:04d}.pkl")
    
    train_dataset = RobotDataset(
        train_dir, sequence_length=sequence_length, action_horizon=action_horizon
    )
    val_dataset = RobotDataset(
        val_dir, sequence_length=sequence_length, action_horizon=action_horizon
    )
    
    return train_dataset, val_dataset
