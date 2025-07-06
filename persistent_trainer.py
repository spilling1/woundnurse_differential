#!/usr/bin/env python3
"""
Persistent training with automatic restart and session management
"""

import time
import subprocess
import signal
import sys
import os
from pathlib import Path

class PersistentTrainer:
    def __init__(self):
        self.training_process = None
        self.restart_count = 0
        self.max_restarts = 10
        
    def start_training(self):
        """Start the training process"""
        try:
            self.training_process = subprocess.Popen(
                [sys.executable, "robust_wound_trainer.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            print(f"Training started with PID: {self.training_process.pid}")
            return True
        except Exception as e:
            print(f"Failed to start training: {e}")
            return False
    
    def monitor_training(self):
        """Monitor training and restart if needed"""
        print("Starting persistent training monitor...")
        
        while self.restart_count < self.max_restarts:
            if not self.training_process or self.training_process.poll() is not None:
                # Process has stopped, restart it
                self.restart_count += 1
                print(f"Training stopped. Restart attempt {self.restart_count}/{self.max_restarts}")
                
                if self.start_training():
                    print("Training restarted successfully")
                else:
                    print("Failed to restart training")
                    break
            
            # Read and display output
            if self.training_process and self.training_process.stdout:
                try:
                    output = self.training_process.stdout.readline()
                    if output:
                        print(f"TRAINING: {output.strip()}")
                        
                        # Save to log file
                        with open("persistent_training.log", "a") as f:
                            f.write(output)
                except:
                    pass
            
            time.sleep(2)
        
        print("Max restarts reached or training completed")
    
    def stop_training(self):
        """Stop the training process"""
        if self.training_process:
            self.training_process.terminate()
            self.training_process.wait()

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\nShutting down persistent trainer...")
    trainer.stop_training()
    sys.exit(0)

if __name__ == "__main__":
    trainer = PersistentTrainer()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        trainer.monitor_training()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)