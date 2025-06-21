#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import torch
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrainingVisualizer:
    def __init__(self, output_dir: str, experiment_name: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics_file = self.output_dir / f"{self.experiment_name}_metrics.json"
        self.metrics_history = self._load_metrics_history()
        
    def _load_metrics_history(self) -> Dict:
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                return json.load(f)
        return {"train": {}, "val": {}, "test": {}, "epochs": [], "config": {}}
        
    def _save_metrics_history(self):
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def log_config(self, config: Dict[str, Any]):
        self.metrics_history["config"] = config
        self._save_metrics_history()
        
    def log_metrics(self, metrics: Dict[str, float], split: str, epoch: int):
        if epoch not in self.metrics_history["epochs"]:
            self.metrics_history["epochs"].append(epoch)
            
        for key, value in metrics.items():
            if key not in self.metrics_history[split]:
                self.metrics_history[split][key] = []
            
            # Ensure the metric list has the correct length
            current_len = len(self.metrics_history[split][key])
            expected_idx = self.metrics_history["epochs"].index(epoch)
            
            # Fill with None for any missing epochs
            while current_len < expected_idx:
                self.metrics_history[split][key].append(None)
                current_len += 1
                
            # Add value at the right position
            if current_len == expected_idx:
                self.metrics_history[split][key].append(value)
            else:
                self.metrics_history[split][key][expected_idx] = value
                
        self._save_metrics_history()
    
    def plot_learning_curves(self, metrics: List[str] = None, interactive: bool = True):
        if not metrics:
            metrics = []
            for split in ["train", "val"]:
                metrics.extend(list(self.metrics_history[split].keys()))
            metrics = list(set(metrics))
        
        if not metrics:
            return None
            
        if interactive:
            return self._plot_interactive_learning_curves(metrics)
        else:
            return self._plot_static_learning_curves(metrics)
    
    def _plot_interactive_learning_curves(self, metrics: List[str]):
        epochs = self.metrics_history["epochs"]
        
        if not epochs:
            return None
            
        fig = make_subplots(rows=len(metrics), cols=1, 
                            shared_xaxes=True, 
                            subplot_titles=metrics)
        
        for i, metric in enumerate(metrics):
            row = i + 1
            
            for split in ["train", "val"]:
                if metric in self.metrics_history[split]:
                    values = self.metrics_history[split][metric]
                    
                    # Remove None values for plotting
                    valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
                    valid_values = [v for v in values if v is not None]
                    
                    if valid_epochs and valid_values:
                        fig.add_trace(
                            go.Scatter(x=valid_epochs, y=valid_values, mode="lines+markers", 
                                      name=f"{split}_{metric}"),
                            row=row, col=1
                        )
        
        fig.update_layout(height=300*len(metrics), width=800, 
                         title=f"Training Metrics - {self.experiment_name}")
        
        html_path = self.output_dir / f"{self.experiment_name}_learning_curves.html"
        fig.write_html(str(html_path))
        
        return fig
    
    def _plot_static_learning_curves(self, metrics: List[str]):
        epochs = self.metrics_history["epochs"]
        
        if not epochs:
            return None
            
        n_plots = len(metrics)
        fig, axs = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots), sharex=True)
        
        if n_plots == 1:
            axs = [axs]
            
        for i, metric in enumerate(metrics):
            for split in ["train", "val"]:
                if metric in self.metrics_history[split]:
                    values = self.metrics_history[split][metric]
                    
                    # Remove None values for plotting
                    valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
                    valid_values = [v for v in values if v is not None]
                    
                    if valid_epochs and valid_values:
                        axs[i].plot(valid_epochs, valid_values, marker="o", label=f"{split}")
                        
            axs[i].set_title(metric)
            axs[i].legend()
            axs[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.output_dir / f"{self.experiment_name}_learning_curves.png"
        plt.savefig(fig_path)
        
        return fig
    
    def visualize_action_space(self, actions: np.ndarray, labels: Optional[List[str]] = None):
        if actions.size == 0:
            return None
            
        action_dim = actions.shape[1]
        n_samples = min(1000, actions.shape[0])  # Limit to 1000 samples for clarity
        
        if n_samples < actions.shape[0]:
            idx = np.random.choice(actions.shape[0], n_samples, replace=False)
            actions_subset = actions[idx]
        else:
            actions_subset = actions
            
        if action_dim <= 2:
            return self._plot_2d_action_space(actions_subset, labels)
        else:
            return self._plot_high_dim_action_space(actions_subset, labels)
    
    def _plot_2d_action_space(self, actions: np.ndarray, labels: Optional[List[str]] = None):
        fig = go.Figure()
        
        if actions.shape[1] == 1:
            # 1D actions, create a scatter with y=0
            actions_plot = np.hstack([actions, np.zeros((actions.shape[0], 1))])
            fig.add_trace(go.Scatter(
                x=actions_plot[:, 0], 
                y=actions_plot[:, 1],
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.7),
                name='Actions'
            ))
            fig.update_layout(
                title="Action Space Distribution (1D)",
                xaxis_title="Action Value",
                yaxis_title="",
                yaxis_visible=False
            )
        else:
            # 2D actions
            fig.add_trace(go.Scatter(
                x=actions[:, 0], 
                y=actions[:, 1],
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.7),
                name='Actions'
            ))
            fig.update_layout(
                title="Action Space Distribution (2D)",
                xaxis_title="Action Dimension 1",
                yaxis_title="Action Dimension 2"
            )
            
        html_path = self.output_dir / f"{self.experiment_name}_action_space.html"
        fig.write_html(str(html_path))
        
        return fig
    
    def _plot_high_dim_action_space(self, actions: np.ndarray, labels: Optional[List[str]] = None):
        # For high-dimensional actions, use PCA plot or parallel coordinates
        action_dim = actions.shape[1]
        
        # Create a parallel coordinates plot
        if not labels:
            labels = [f"dim_{i}" for i in range(action_dim)]
            
        fig = px.parallel_coordinates(
            pd.DataFrame(actions, columns=labels),
            labels={col: col for col in labels},
            title=f"Action Space Distribution ({action_dim}D)"
        )
        
        html_path = self.output_dir / f"{self.experiment_name}_action_space.html"
        fig.write_html(str(html_path))
        
        return fig
    
    def compare_policies(self, policy_paths: List[str], env_id: str, num_episodes: int = 5, 
                         episode_length: int = 100, seed: int = 42):
        import gymnasium as gym
        from robot_cn_network.models import ACTPolicy
        from robot_cn_network.utils import load_policy
        
        results = {}
        
        for policy_path in policy_paths:
            policy_name = Path(policy_path).stem
            policy = load_policy(policy_path)
            
            env = gym.make(env_id)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            
            episode_returns = []
            success_rate = 0
            
            for episode in range(num_episodes):
                obs, _ = env.reset(seed=seed+episode)
                episode_return = 0
                success = False
                
                for step in range(episode_length):
                    with torch.no_grad():
                        # Convert observation to tensor and add batch dimension
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        action = policy(obs_tensor).detach().cpu().numpy()[0]
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                    
                    if terminated or truncated:
                        if "success" in info and info["success"]:
                            success = True
                        break
                        
                episode_returns.append(episode_return)
                success_rate += int(success)
                
            results[policy_name] = {
                "mean_return": np.mean(episode_returns),
                "std_return": np.std(episode_returns),
                "success_rate": success_rate / num_episodes,
                "episode_returns": episode_returns
            }
            
        # Create visualization
        fig = make_subplots(rows=1, cols=2, 
                            subplot_titles=["Average Return", "Success Rate"])
                            
        policies = list(results.keys())
        mean_returns = [results[p]["mean_return"] for p in policies]
        std_returns = [results[p]["std_return"] for p in policies]
        success_rates = [results[p]["success_rate"] for p in policies]
        
        # Plot mean returns with error bars
        fig.add_trace(
            go.Bar(
                x=policies, 
                y=mean_returns,
                error_y=dict(type="data", array=std_returns),
                name="Average Return"
            ),
            row=1, col=1
        )
        
        # Plot success rates
        fig.add_trace(
            go.Bar(
                x=policies,
                y=success_rates,
                name="Success Rate"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"Policy Comparison - {env_id}",
            height=500,
            width=800
        )
        
        html_path = self.output_dir / "policy_comparison.html"
        fig.write_html(str(html_path))
        
        # Save numerical results
        results_file = self.output_dir / "policy_comparison.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        return fig, results


class RobotTrajectoryVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_end_effector_path(self, positions: np.ndarray, targets: Optional[np.ndarray] = None,
                                   title: str = "End Effector Trajectory"):
        fig = go.Figure()
        
        # Plot start and end positions
        fig.add_trace(go.Scatter3d(
            x=[positions[0, 0]], 
            y=[positions[0, 1]], 
            z=[positions[0, 2]],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Start'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[positions[-1, 0]], 
            y=[positions[-1, 1]], 
            z=[positions[-1, 2]],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='End'
        ))
        
        # Plot trajectory
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0], 
            y=positions[:, 1], 
            z=positions[:, 2],
            mode='lines',
            line=dict(color='purple', width=4),
            name='Trajectory'
        ))
        
        # Plot targets if provided
        if targets is not None:
            fig.add_trace(go.Scatter3d(
                x=targets[:, 0], 
                y=targets[:, 1], 
                z=targets[:, 2],
                mode='markers',
                marker=dict(size=5, color='green', symbol='diamond'),
                name='Targets'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            )
        )
        
        filename = title.lower().replace(" ", "_")
        html_path = self.output_dir / f"{filename}.html"
        fig.write_html(str(html_path))
        
        return fig
    
    def create_trajectory_animation(self, positions: np.ndarray, orientations: Optional[np.ndarray] = None,
                                   cube_positions: Optional[np.ndarray] = None):
        from plotly.subplots import make_subplots
        
        n_frames = min(positions.shape[0], 100)  # Limit to 100 frames for performance
        
        if n_frames < positions.shape[0]:
            # Subsample for smoother animation
            indices = np.linspace(0, positions.shape[0]-1, n_frames).astype(int)
            positions = positions[indices]
            if orientations is not None:
                orientations = orientations[indices]
            if cube_positions is not None:
                cube_positions = cube_positions[indices]
        
        fig = make_subplots(
            rows=1, cols=1, 
            specs=[[{'type': 'scene'}]],
            subplot_titles=["Robot Trajectory Animation"]
        )
        
        # Create frame for each position
        frames = []
        
        for i in range(n_frames):
            frame_data = []
            
            # Add end effector trace for this frame
            frame_data.append(
                go.Scatter3d(
                    x=positions[:i+1, 0],
                    y=positions[:i+1, 1],
                    z=positions[:i+1, 2],
                    mode='lines+markers',
                    line=dict(color='purple', width=4),
                    marker=dict(size=4, color='purple', opacity=0.5),
                    name='End Effector'
                )
            )
            
            # Add current position marker
            frame_data.append(
                go.Scatter3d(
                    x=[positions[i, 0]],
                    y=[positions[i, 1]],
                    z=[positions[i, 2]],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='Current Position'
                )
            )
            
            # Add cube if available
            if cube_positions is not None:
                frame_data.append(
                    go.Scatter3d(
                        x=[cube_positions[i, 0]],
                        y=[cube_positions[i, 1]],
                        z=[cube_positions[i, 2]],
                        mode='markers',
                        marker=dict(size=15, color='green', symbol='square'),
                        name='Cube'
                    )
                )
            
            frames.append(go.Frame(data=frame_data, name=f"frame{i}"))
        
        # Add initial data to figure
        initial_data = frames[0].data
        for trace in initial_data:
            fig.add_trace(trace)
        
        # Set up animation
        fig.frames = frames
        
        fig.update_layout(
            title="Robot Trajectory Animation",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }],
            sliders=[{
                'active': 0,
                'steps': [
                    {
                        'args': [[f.name], {'frame': {'duration': 50, 'redraw': True}, 'mode': 'immediate'}],
                        'label': str(k),
                        'method': 'animate'
                    } 
                    for k, f in enumerate(frames)
                ]
            }]
        )
        
        html_path = self.output_dir / "trajectory_animation.html"
        fig.write_html(str(html_path))
        
        return fig
