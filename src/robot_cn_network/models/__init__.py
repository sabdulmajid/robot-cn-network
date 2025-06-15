import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelConfig:
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None  
    cnn_strides: List[int] = None
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    action_dim: int = 7
    chunk_size: int = 10
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128, 256]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [7, 5, 3, 3]
        if self.cnn_strides is None:
            self.cnn_strides = [2, 2, 2, 2]


class CNNEncoder(nn.Module):
    def __init__(self, config: ModelConfig, input_channels: int = 3):
        super().__init__()
        self.config = config
        
        layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size, stride in zip(
            config.cnn_channels, config.cnn_kernel_sizes, config.cnn_strides
        ):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
            
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = config.cnn_channels[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        pooled = self.pool(features)
        return pooled.flatten(1)


class MultiCameraEncoder(nn.Module):
    def __init__(self, config: ModelConfig, camera_names: List[str]):
        super().__init__()
        self.camera_names = camera_names
        
        self.encoders = nn.ModuleDict({
            name: CNNEncoder(config) for name in camera_names
        })
        
        total_dim = len(camera_names) * config.cnn_channels[-1]
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        
        for name in self.camera_names:
            if name in images:
                cam_features = self.encoders[name](images[name])
                features.append(cam_features)
            else:
                batch_size = list(images.values())[0].shape[0]
                device = list(images.values())[0].device
                zero_features = torch.zeros(
                    batch_size, self.encoders[name].output_dim, device=device
                )
                features.append(zero_features)
                
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class ACTPolicy(nn.Module):
    def __init__(self, config: ModelConfig, state_dim: int, camera_names: List[str]):
        super().__init__()
        self.config = config
        self.camera_names = camera_names
        
        self.vision_encoder = MultiCameraEncoder(config, camera_names)
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.combined_encoder = nn.Sequential(
            nn.Linear(config.d_model + config.d_model // 2, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.pos_encoder = PositionalEncoding(config.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )
        
        self.action_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.action_dim * config.chunk_size)
        )
        
    def forward(self, images: Dict[str, torch.Tensor], states: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        batch_size = states.shape[0]
        
        vision_features = self.vision_encoder(images)
        state_features = self.state_encoder(states)
        
        combined = torch.cat([vision_features, state_features], dim=1)
        combined = self.combined_encoder(combined)
        
        if seq_len is None:
            seq_len = 1
            
        combined = combined.unsqueeze(1).expand(-1, seq_len, -1)
        combined = self.pos_encoder(combined.transpose(0, 1)).transpose(0, 1)
        
        encoded = self.transformer_encoder(combined)
        encoded = encoded[:, -1, :]
        
        actions = self.action_decoder(encoded)
        actions = actions.view(batch_size, self.config.chunk_size, self.config.action_dim)
        
        return actions
        
    def predict_action(self, images: Dict[str, torch.Tensor], states: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            actions = self.forward(images, states)
            return actions[0, 0].cpu().numpy()


class DiffusionPolicy(nn.Module):
    def __init__(self, config: ModelConfig, state_dim: int, camera_names: List[str]):
        super().__init__()
        self.config = config
        
        self.vision_encoder = MultiCameraEncoder(config, camera_names)
        self.state_encoder = nn.Linear(state_dim, config.d_model // 2)
        
        self.noise_pred_net = nn.Sequential(
            nn.Linear(config.d_model + config.action_dim, config.dim_feedforward),
            nn.ReLU(),
            nn.Linear(config.dim_feedforward, config.dim_feedforward),
            nn.ReLU(), 
            nn.Linear(config.dim_feedforward, config.action_dim)
        )
        
    def forward(self, images: Dict[str, torch.Tensor], states: torch.Tensor, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        vision_features = self.vision_encoder(images)
        state_features = self.state_encoder(states)
        
        obs_features = torch.cat([vision_features, state_features], dim=1)
        combined = torch.cat([obs_features, actions], dim=1)
        
        noise_pred = self.noise_pred_net(combined)
        return noise_pred
