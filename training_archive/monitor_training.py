#!/usr/bin/env python3
"""
Monitor CNN training progress
"""

import time
import os
from pathlib import Path
import subprocess

def monitor_training():
    """Monitor the training process"""
    print("🔄 CNN Training Monitor")
    print("=" * 30)
    
    log_file = "training.log"
    
    # Wait for log file to be created
    print("⏳ Waiting for training to start...")
    while not Path(log_file).exists():
        time.sleep(2)
    
    print("✅ Training started! Monitoring progress...")
    print("=" * 30)
    
    # Monitor log file
    try:
        with open(log_file, 'r') as f:
            # Move to end of file
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    print(line.strip())
                else:
                    time.sleep(1)
                    
                # Check if training is still running
                result = subprocess.run(['pgrep', '-f', 'train_wound_cnn'], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    print("\n🏁 Training process completed!")
                    break
                    
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")
    except Exception as e:
        print(f"❌ Error monitoring: {e}")

if __name__ == "__main__":
    monitor_training()