# Robot CN Network: Sim-Based Robotic Teleoperation & Imitation Learning

A comprehensive research project implementing convolutional neural networks for robotic image frame classification and imitation learning using simulation-based teleoperation. This project integrates with LeRobot and gym-hil for Human-In-the-Loop (HIL) reinforcement learning.

## Project Overview

This project focuses on developing sim-based robotic teleoperation capabilities for imitation learning research. Key features include:

- **Simulation-first approach**: Train and test on macOS/Linux with MuJoCo simulation
- **Keyboard & Mouse Control**: Intuitive teleoperation without specialized hardware
- **CNN-based Vision**: Convolutional neural networks for robotic perception

## Key Features

### Teleoperation
- **Keyboard Control**: Use arrow keys, WASD, and spacebar for robot control
- **Mouse Support**: Click-and-drag interface for precise positioning
- **Real-time Feedback**: Live visualization of robot state and actions

### Machine Learning
- **ACT (Action Chunking Transformer)**: State-of-the-art policy architecture
- **CNN Vision Pipeline**: Process visual observations for decision making
- **Dataset Management**: Automatic data collection and organization
- **Wandb Integration**: Experiment tracking and visualization

### Simulation Environment
- **Franka Panda Robot**: High-fidelity MuJoCo simulation
- **Pick-and-Place Tasks**: Cube manipulation challenges
- **Multiple Camera Views**: Front and wrist cameras for comprehensive observation
- **Configurable Scenarios**: Customizable environments and tasks

## Installation

### Quick Start

1. **Clone the repository with submodules:**
```bash
git clone --recursive https://github.com/ayman/robot-cn-network.git
cd robot-cn-network
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. **Install the project:**
```bash
pip install -e .
```

4. **Install LeRobot integration (optional):**
```bash
pip install -e ".[lerobot]"
```

### Development Installation

For development and research:
```bash
pip install -e ".[dev]"
pre-commit install
```

## Usage Guide

### 1. Basic Teleoperation

Start the simulation with keyboard control:
```bash
robot-teleop --mode keyboard --env PandaPickCubeKeyboard-v0
```

Controls:
- **Arrow Keys**: Move in X-Y plane
- **Shift/Ctrl**: Move up/down in Z axis
- **Space**: Enable/disable control
- **Left/Right Ctrl**: Close/open gripper
- **ESC**: Exit

### 2. Data Collection

Collect demonstration data:
```bash
robot-collect --config configs/data_collection.yaml --num-episodes 30
```

### 3. Training

Train an imitation learning policy:
```bash
robot-train --config configs/training.yaml --dataset-path ./data/demonstrations
```

### 4. Evaluation

Evaluate a trained policy:
```bash
robot-eval --policy-path ./outputs/policies/latest --num-episodes 10
```

## Research Applications

This project is designed for researchers interested in:

- **Imitation Learning**: Learning robot behaviors from human demonstrations
- **Human-Robot Interaction**: Studying teleoperation and intervention strategies
- **Vision-Based Robotics**: Developing CNN architectures for robotic perception
- **Sim-to-Real Transfer**: Preparing policies for real-world deployment
- **Reinforcement Learning**: HIL-RL and interactive policy improvement

## Experiment Tracking

The project integrates with Weights & Biases for comprehensive experiment tracking:

- Training metrics and loss curves
- Episode videos and trajectory visualizations
- Hyperparameter sweeps and comparisons
- Model performance analytics

## Teleoperation Guide

### Keyboard Controls
- `↑↓←→`: Move robot end-effector in X-Y plane
- `Shift + ↑↓`: Move up/down in Z axis
- `Space`: Toggle intervention mode
- `Ctrl`: Open/close gripper
- `R`: Reset episode
- `Q`: Quit application

### Advanced Controls
- `1-9`: Switch between preset positions
- `Tab`: Toggle camera view
- `Enter`: Save current episode
- `Backspace`: Delete last action

## Configuration

### Environment Configuration
```yaml
# configs/environment.yaml
environment:
  name: "PandaPickCubeKeyboard-v0"
  render_mode: "human"
  image_obs: true
  control_dt: 0.1
  physics_dt: 0.002
```

### Training Configuration
```yaml
# configs/training.yaml
training:
  policy_type: "act"
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  device: "mps"  # Use Apple Silicon GPU
```

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face's robotics toolkit
- [gym-hil](https://github.com/huggingface/gym-hil) - Human-in-the-loop environments
- [MuJoCo](https://mujoco.org/) - Physics simulation
- [Franka Emika](https://www.franka.de/) - Robot hardware inspiration
