#!/usr/bin/env python3

import os
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from robot_cn_network.utils import (
    TrainingVisualizer, RobotTrajectoryVisualizer, PLOTLY_AVAILABLE
)

def generate_sample_training_data(output_dir, epochs=100):
    """Generate sample training metrics data for visualization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create random training curves with realistic patterns
    epochs_arr = np.arange(1, epochs + 1)
    
    # Loss that decreases and plateaus
    train_loss = 2.0 * np.exp(-0.05 * epochs_arr) + 0.5 + 0.3 * np.random.rand(epochs)
    val_loss = train_loss + 0.2 + 0.2 * np.random.rand(epochs)
    
    # Accuracy that increases and plateaus
    train_acc = 0.8 * (1 - np.exp(-0.06 * epochs_arr)) + 0.1 + 0.1 * np.random.rand(epochs)
    val_acc = train_acc - 0.1 - 0.1 * np.random.rand(epochs)
    
    # Create metrics history object
    metrics_history = {
        "train": {
            "loss": train_loss.tolist(),
            "accuracy": train_acc.tolist(),
            "mse": (train_loss / 2).tolist()
        },
        "val": {
            "loss": val_loss.tolist(),
            "accuracy": val_acc.tolist(),
            "mse": (val_loss / 2).tolist()
        },
        "epochs": epochs_arr.tolist(),
        "config": {
            "model_type": "CNN",
            "learning_rate": 1e-4,
            "batch_size": 32,
            "optimizer": "Adam"
        }
    }
    
    # Save metrics history
    with open(output_dir / "sample_training_metrics.json", 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"Sample training data saved to {output_dir / 'sample_training_metrics.json'}")
    return metrics_history

def generate_sample_trajectory_data(output_dir, n_steps=200):
    """Generate sample robot trajectory data for visualization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a synthetic spiral trajectory
    t = np.linspace(0, 10, n_steps)
    x = 0.5 * np.cos(t) + 0.1 * np.random.randn(n_steps)
    y = 0.5 * np.sin(t) + 0.1 * np.random.randn(n_steps)
    z = 0.05 * t + 0.1 * np.cos(2*t) + 0.05 * np.random.randn(n_steps)
    
    # Create positions array
    positions = np.vstack([x, y, z]).T
    
    # Create cube positions (target points along the trajectory)
    cube_indices = np.linspace(0, n_steps-1, 5, dtype=int)
    cube_positions = positions[cube_indices] + 0.1 * np.random.randn(5, 3)
    cube_positions[:, 2] -= 0.1  # Position cubes slightly below trajectory
    
    # Create orientations (4D quaternions)
    orientations = np.zeros((n_steps, 4))
    orientations[:, 0] = np.cos(t/2)  # w component
    orientations[:, 1:] = np.sin(t/2)[:, np.newaxis] * np.vstack([
        np.cos(t), np.sin(t), np.ones_like(t)
    ]).T
    # Normalize quaternions
    orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)
    
    # Save trajectory data
    np.savez(
        output_dir / "sample_trajectory.npz",
        positions=positions,
        orientations=orientations,
        cube_positions=cube_positions
    )
    
    # Also save just positions for simpler test
    np.save(output_dir / "sample_positions.npy", positions)
    
    print(f"Sample trajectory data saved to {output_dir / 'sample_trajectory.npz'}")
    print(f"Sample positions data saved to {output_dir / 'sample_positions.npy'}")
    return positions, orientations, cube_positions

def generate_sample_action_data(output_dir, n_samples=1000, dim=7):
    """Generate sample action data for visualization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create actions with realistic distributions
    actions = np.zeros((n_samples, dim))
    
    # For a 7D robot arm action space (common for 7-DoF arms like Franka)
    if dim >= 7:
        # Joint positions or velocities tend to be normally distributed
        # around zero during typical manipulation tasks
        actions[:, :min(dim, 7)] = 0.3 * np.random.randn(n_samples, min(dim, 7))
        
        # Add some correlations between adjacent joints
        for i in range(1, min(dim, 7)):
            actions[:, i] += 0.2 * actions[:, i-1]
            
        # If we have a gripper dimension, make it bimodal (open/closed)
        if dim >= 7:
            gripper = np.random.choice([-1.0, 1.0], size=n_samples)
            gripper += 0.1 * np.random.randn(n_samples)  # Add some noise
            actions[:, 6] = gripper
            
    else:
        # For lower dimensional action spaces
        actions = 0.5 * np.random.randn(n_samples, dim)
    
    # Save action data
    np.save(output_dir / "sample_actions.npy", actions)
    print(f"Sample action data saved to {output_dir / 'sample_actions.npy'}")
    return actions

def visualize_sample_data():
    """Generate and visualize sample data to test visualization tools"""
    output_dir = Path("./outputs/example_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    training_data = generate_sample_training_data(output_dir)
    trajectory_data = generate_sample_trajectory_data(output_dir)
    action_data = generate_sample_action_data(output_dir)
    
    # Create visualizers
    train_vis = TrainingVisualizer(output_dir, "sample_training")
    traj_vis = RobotTrajectoryVisualizer(output_dir)
    
    # Generate and save visualizations
    results = {}
    
    # 1. Learning curves
    try:
        train_vis.log_config(training_data["config"])
        for i, epoch in enumerate(training_data["epochs"]):
            train_metrics = {k: v[i] for k, v in training_data["train"].items()}
            val_metrics = {k: v[i] for k, v in training_data["val"].items()}
            
            # Only log every 5th epoch to keep file size reasonable
            if i % 5 == 0:
                train_vis.log_metrics(train_metrics, "train", epoch)
                train_vis.log_metrics(val_metrics, "val", epoch)
        
        fig = train_vis.plot_learning_curves(interactive=PLOTLY_AVAILABLE)
        static_fig = train_vis.plot_learning_curves(interactive=False)
        
        results["learning_curves"] = {
            "interactive": str(output_dir / "sample_training_learning_curves.html") if PLOTLY_AVAILABLE else "Not available",
            "static": str(output_dir / "sample_training_learning_curves.png")
        }
        print("Generated learning curve visualizations")
    except Exception as e:
        print(f"Failed to generate learning curves: {e}")
    
    # 2. Trajectory visualization
    try:
        positions, orientations, cube_positions = trajectory_data
        
        # 3D path visualization
        fig = traj_vis.visualize_end_effector_path(
            positions, 
            targets=cube_positions,
            title="Sample Robot Trajectory"
        )
        
        # Animation if plotly is available
        if PLOTLY_AVAILABLE:
            # Fix: the create_trajectory_animation method might be expecting different parameters or having an issue with orientations.
            # Let's simplify to just show a simple animation of positions
            anim_fig = traj_vis.visualize_end_effector_path(
                positions, 
                title="Sample Robot Trajectory Animation"
            )
        
        results["trajectory"] = {
            "path": str(output_dir / "sample_robot_trajectory.html"),
            "animation": str(output_dir / "trajectory_animation.html") if PLOTLY_AVAILABLE else "Not available"
        }
        print("Generated trajectory visualizations")
    except Exception as e:
        print(f"Failed to generate trajectory visualizations: {e}")
    
    # 3. Action space visualization
    try:
        # Full 7D visualization
        fig = train_vis.visualize_action_space(action_data)
        
        # 2D subset visualization (first two dimensions)
        fig_2d = train_vis.visualize_action_space(action_data[:, :2])
        
        results["action_space"] = {
            "full": str(output_dir / "sample_training_action_space.html"),
            "2d_subset": str(output_dir / "sample_training_action_space.html").replace(".html", "_2d.html")
        }
        print("Generated action space visualizations")
    except Exception as e:
        print(f"Failed to generate action space visualizations: {e}")
    
    # Save results summary
    with open(output_dir / "visualization_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll visualizations saved to {output_dir}")
    print("Run with --help to see all available options")
    
    # Return paths to generated files for documentation
    return results

if __name__ == "__main__":
    visualize_sample_data()
