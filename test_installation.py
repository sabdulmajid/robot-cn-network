#!/usr/bin/env python3
"""Simple test to verify robot-cn-network installation."""

import sys
import traceback

def test_imports():
    """Test basic imports without MuJoCo dependencies."""
    try:
        print("Testing robot-cn-network imports...")
        
        # Test basic package import
        import robot_cn_network
        print(f"✓ robot_cn_network version: {robot_cn_network.__version__}")
        
        # Test individual modules
        from robot_cn_network.models import ModelConfig, CNNEncoder
        print("✓ models module imported successfully")
        
        from robot_cn_network.utils import setup_logging, set_seed, get_device
        print("✓ utils module imported successfully")
        
        from robot_cn_network.data import RobotDataset
        print("✓ data module imported successfully")
        
        # Test PyTorch
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
        
        device = get_device()
        print(f"✓ Device: {device}")
        
        # Test basic model creation
        config = ModelConfig()
        encoder = CNNEncoder(config)
        print("✓ CNNEncoder created successfully")
        
        print("\n🎉 All core components working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
