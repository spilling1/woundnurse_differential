#!/usr/bin/env python3
"""
Real-time training monitor for wound CNN
"""

import time
import os
import json
from pathlib import Path
from datetime import datetime
import subprocess
import sys

class TrainingMonitor:
    def __init__(self, log_file="training_progress.log", models_dir="."):
        self.log_file = log_file
        self.models_dir = Path(models_dir)
        self.last_position = 0
        self.training_stats = {
            'epochs_completed': 0,
            'current_epoch': 0,
            'current_batch': 0,
            'latest_loss': 0,
            'best_accuracy': 0,
            'training_time': 0,
            'status': 'stopped'
        }
    
    def check_training_process(self):
        """Check if training process is running"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            return 'minimal_wound_trainer' in result.stdout
        except:
            return False
    
    def parse_log_line(self, line):
        """Parse a single log line for training information"""
        if "Epoch" in line and "Batch" in line and "Loss" in line:
            # Extract epoch, batch, and loss from log line
            parts = line.split()
            for i, part in enumerate(parts):
                if "Epoch" in part and i+1 < len(parts):
                    epoch_info = parts[i+1].split('/')
                    if len(epoch_info) == 2:
                        self.training_stats['current_epoch'] = int(epoch_info[0])
                        self.training_stats['total_epochs'] = int(epoch_info[1])
                
                if "Batch" in part and i+1 < len(parts):
                    batch_info = parts[i+1].split('/')
                    if len(batch_info) == 2:
                        self.training_stats['current_batch'] = int(batch_info[0])
                        self.training_stats['total_batches'] = int(batch_info[1])
                
                if "Loss:" in part and i+1 < len(parts):
                    try:
                        self.training_stats['latest_loss'] = float(parts[i+1])
                    except:
                        pass
        
        elif "Val Acc:" in line:
            # Extract validation accuracy
            parts = line.split()
            for i, part in enumerate(parts):
                if "Acc:" in part and i+1 < len(parts):
                    try:
                        acc = float(parts[i+1].replace('%', ''))
                        self.training_stats['best_accuracy'] = max(self.training_stats['best_accuracy'], acc)
                    except:
                        pass
        
        elif "Epoch" in line and "completed" in line:
            self.training_stats['epochs_completed'] += 1
    
    def read_new_logs(self):
        """Read new log entries since last check"""
        if not Path(self.log_file).exists():
            return []
        
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
                return [line.strip() for line in new_lines if line.strip()]
        except:
            return []
    
    def get_model_files(self):
        """Get list of saved model files"""
        return list(self.models_dir.glob("wound_model_*.pth"))
    
    def display_progress(self):
        """Display current training progress"""
        print("\n" + "="*60)
        print(f"🧠 WOUND CNN TRAINING MONITOR")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Training status
        is_running = self.check_training_process()
        status = "🟢 RUNNING" if is_running else "🔴 STOPPED"
        print(f"Status: {status}")
        
        # Progress information
        if self.training_stats['current_epoch'] > 0:
            epoch_progress = f"{self.training_stats['current_epoch']}/{self.training_stats.get('total_epochs', '?')}"
            batch_progress = f"{self.training_stats['current_batch']}/{self.training_stats.get('total_batches', '?')}"
            
            print(f"Epoch Progress: {epoch_progress}")
            print(f"Batch Progress: {batch_progress}")
            print(f"Latest Loss: {self.training_stats['latest_loss']:.4f}")
            print(f"Best Accuracy: {self.training_stats['best_accuracy']:.2f}%")
        
        # Model files
        model_files = self.get_model_files()
        print(f"Saved Models: {len(model_files)}")
        
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            print(f"Latest Model: {latest_model.name}")
        
        return is_running
    
    def start_training_if_stopped(self):
        """Start training if it's not running"""
        if not self.check_training_process():
            print("🚀 Starting training...")
            try:
                # Start training in background
                subprocess.Popen([sys.executable, "minimal_wound_trainer.py"], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
                time.sleep(2)  # Give it time to start
                return True
            except Exception as e:
                print(f"❌ Failed to start training: {e}")
                return False
        return True
    
    def monitor_live(self, auto_restart=True):
        """Monitor training in real-time"""
        print("🔍 Starting real-time training monitor...")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                # Check if training should be restarted
                if auto_restart:
                    self.start_training_if_stopped()
                
                # Read new log entries
                new_lines = self.read_new_logs()
                
                # Parse new information
                for line in new_lines:
                    self.parse_log_line(line)
                    if any(keyword in line.lower() for keyword in ['epoch', 'loss', 'acc', 'error']):
                        print(f"📊 {line}")
                
                # Display current status
                is_running = self.display_progress()
                
                # Sleep before next check
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    
    def quick_status(self):
        """Get quick training status"""
        is_running = self.check_training_process()
        new_lines = self.read_new_logs()
        
        for line in new_lines:
            self.parse_log_line(line)
        
        model_files = self.get_model_files()
        
        return {
            'running': is_running,
            'epoch': self.training_stats['current_epoch'],
            'loss': self.training_stats['latest_loss'],
            'accuracy': self.training_stats['best_accuracy'],
            'models_saved': len(model_files),
            'latest_model': max(model_files, key=lambda x: x.stat().st_mtime).name if model_files else None
        }

def main():
    """Main monitoring function"""
    monitor = TrainingMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        monitor.monitor_live()
    else:
        # Quick status check
        status = monitor.quick_status()
        print(f"Training Status: {'Running' if status['running'] else 'Stopped'}")
        print(f"Current Epoch: {status['epoch']}")
        print(f"Latest Loss: {status['loss']:.4f}")
        print(f"Best Accuracy: {status['accuracy']:.2f}%")
        print(f"Models Saved: {status['models_saved']}")
        
        if status['latest_model']:
            print(f"Latest Model: {status['latest_model']}")

if __name__ == "__main__":
    main()