#!/usr/bin/env python3
"""
Simple real-time training monitor for wound CNN
"""

import time
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def check_training_running():
    """Check if training is currently running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return 'minimal_wound_trainer' in result.stdout
    except:
        return False

def get_latest_log_lines(log_file="training_progress.log", lines=10):
    """Get the latest lines from the training log"""
    if not Path(log_file).exists():
        return []
    
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            return [line.strip() for line in all_lines[-lines:]]
    except:
        return []

def count_saved_models():
    """Count saved model files"""
    model_files = list(Path(".").glob("wound_model_*.pth"))
    return len(model_files)

def start_training():
    """Start training if not running"""
    if not check_training_running():
        print("Starting training...")
        try:
            subprocess.Popen([sys.executable, "minimal_wound_trainer.py"])
            return True
        except Exception as e:
            print(f"Failed to start training: {e}")
            return False
    return True

def monitor_training():
    """Monitor training with simple status updates"""
    print("🔍 WOUND CNN TRAINING MONITOR")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring\n")
    
    last_log_size = 0
    
    try:
        while True:
            # Check if training is running
            is_running = check_training_running()
            status = "RUNNING" if is_running else "STOPPED"
            
            # Get current timestamp
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Check for new log entries
            if Path("training_progress.log").exists():
                current_size = Path("training_progress.log").stat().st_size
                if current_size > last_log_size:
                    # New log entries
                    latest_lines = get_latest_log_lines(lines=5)
                    
                    print(f"\n[{timestamp}] Status: {status}")
                    print("Latest progress:")
                    for line in latest_lines:
                        if any(keyword in line.lower() for keyword in ['epoch', 'loss', 'acc', 'val']):
                            print(f"  {line}")
                    
                    last_log_size = current_size
            else:
                print(f"[{timestamp}] Status: {status} - No log file yet")
            
            # Show model count
            model_count = count_saved_models()
            if model_count > 0:
                print(f"Models saved: {model_count}")
            
            # Auto-restart if needed
            if not is_running:
                print("Training stopped - attempting restart...")
                start_training()
                time.sleep(3)  # Give it time to start
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

def quick_status():
    """Show quick training status"""
    is_running = check_training_running()
    latest_lines = get_latest_log_lines(lines=3)
    model_count = count_saved_models()
    
    print(f"Training Status: {'Running' if is_running else 'Stopped'}")
    print(f"Models Saved: {model_count}")
    
    if latest_lines:
        print("\nLatest Progress:")
        for line in latest_lines:
            if any(keyword in line.lower() for keyword in ['epoch', 'loss', 'acc']):
                print(f"  {line}")
    else:
        print("No training log found")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        monitor_training()
    else:
        quick_status()