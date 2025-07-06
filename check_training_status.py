#!/usr/bin/env python3
"""
Check CNN training status and progress
"""

import subprocess
import time
from pathlib import Path
import re

def check_training_status():
    """Check if training is still running and show progress"""
    print("🔍 Checking CNN Training Status")
    print("=" * 35)
    
    # Check if training process is running
    try:
        result = subprocess.run(['pgrep', '-f', 'simple_wound_trainer'], 
                              capture_output=True, text=True)
        is_running = result.returncode == 0
        
        if is_running:
            print("✅ Training is ACTIVE")
        else:
            print("⏹️  Training process not found")
        
        # Check log file for latest progress
        log_file = Path("training_progress.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Extract latest epoch info
            latest_epoch = None
            latest_loss = None
            latest_acc = None
            
            for line in reversed(lines[-20:]):  # Check last 20 lines
                if "Epoch" in line and "Results:" in line:
                    epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                    if epoch_match:
                        latest_epoch = f"{epoch_match.group(1)}/{epoch_match.group(2)}"
                        break
                elif "Train Loss:" in line:
                    loss_match = re.search(r'Train Loss: ([\d.]+)', line)
                    acc_match = re.search(r'Train Acc: ([\d.]+)%', line)
                    if loss_match:
                        latest_loss = loss_match.group(1)
                    if acc_match:
                        latest_acc = acc_match.group(1)
            
            print(f"\n📊 Latest Progress:")
            if latest_epoch:
                print(f"   Epoch: {latest_epoch}")
            if latest_loss:
                print(f"   Training Loss: {latest_loss}")
            if latest_acc:
                print(f"   Training Accuracy: {latest_acc}%")
        
        # Check for completed model files
        model_files = list(Path('.').glob('best_wound_model_*.pth'))
        if model_files:
            print(f"\n💾 Model Files Found:")
            for model_file in model_files:
                print(f"   {model_file.name}")
        
        # Check if training completed
        if not is_running and model_files:
            print("\n🎉 TRAINING COMPLETED!")
            return True
        elif not is_running and not model_files:
            print("\n❌ Training may have failed - no model files found")
            return False
        else:
            print(f"\n⏳ Training in progress...")
            return False
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False

if __name__ == "__main__":
    check_training_status()