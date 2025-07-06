#!/usr/bin/env python3
"""
Live training status monitor
"""

import time
import subprocess
from pathlib import Path

def check_training_status():
    """Check current training status and progress"""
    print("CNN Training Monitor")
    print("=" * 20)
    
    # Check if process is running
    try:
        result = subprocess.run(['pgrep', '-f', 'minimal_wound_trainer'], 
                              capture_output=True, text=True)
        is_running = result.returncode == 0
        
        if is_running:
            print("Status: TRAINING ACTIVE")
            
            # Check for log file
            log_file = Path("training_live.log")
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                # Get last few lines
                recent_lines = lines[-5:] if len(lines) >= 5 else lines
                print("\nRecent Progress:")
                for line in recent_lines:
                    if line.strip():
                        print(f"  {line.strip()}")
            
            # Check for model files
            model_files = list(Path('.').glob('wound_model_*.pth'))
            if model_files:
                print(f"\nModel Files Created: {len(model_files)}")
                for model in model_files:
                    print(f"  {model.name}")
        else:
            print("Status: NOT RUNNING")
            
            # Check for completed models
            model_files = list(Path('.').glob('wound_model_*.pth'))
            if model_files:
                print(f"Training Complete! Models found:")
                for model in model_files:
                    print(f"  {model.name}")
                return "completed"
            else:
                print("No models found - training may have failed")
                return "failed"
                
    except Exception as e:
        print(f"Error checking status: {e}")
        return "error"
    
    return "running" if is_running else "stopped"

if __name__ == "__main__":
    status = check_training_status()
    print(f"\nStatus: {status}")