#!/usr/bin/env python3
"""
Real-time training dashboard for wound CNN
"""

import time
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def get_training_status():
    """Get current training status"""
    status = {
        'process_running': False,
        'current_epoch': 0,
        'current_batch': 0,
        'latest_loss': 0.0,
        'best_accuracy': 0.0,
        'models_saved': 0,
        'checkpoints_saved': 0,
        'last_update': 'Unknown'
    }
    
    # Check if training process is running
    try:
        result = subprocess.run(['pgrep', '-f', 'robust_wound_trainer'], 
                              capture_output=True, text=True)
        status['process_running'] = bool(result.stdout.strip())
    except:
        pass
    
    # Parse training log
    log_file = Path('robust_training_output.log')
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            for line in reversed(lines[-50:]):  # Check last 50 lines
                line = line.strip()
                
                if 'Epoch' in line and 'Batch' in line and 'Loss' in line:
                    # Extract epoch, batch, loss
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'Epoch' in part and i+1 < len(parts):
                            try:
                                epoch_str = parts[i+1].replace(',', '')
                                status['current_epoch'] = int(epoch_str)
                            except:
                                pass
                        
                        if 'Batch' in part and i+1 < len(parts):
                            try:
                                batch_info = parts[i+1].split('/')
                                if len(batch_info) >= 1:
                                    status['current_batch'] = int(batch_info[0])
                            except:
                                pass
                        
                        if 'Loss:' in part and i+1 < len(parts):
                            try:
                                status['latest_loss'] = float(parts[i+1])
                            except:
                                pass
                
                elif 'Val Acc:' in line:
                    # Extract validation accuracy
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'Acc:' in part and i+1 < len(parts):
                            try:
                                acc_str = parts[i+1].replace('%', '')
                                acc = float(acc_str)
                                status['best_accuracy'] = max(status['best_accuracy'], acc)
                            except:
                                pass
                
                elif 'Saved model:' in line:
                    status['models_saved'] += 1
                
                elif 'Saved checkpoint' in line:
                    status['checkpoints_saved'] += 1
                
                # Get timestamp of last update
                if any(keyword in line for keyword in ['Epoch', 'Loss', 'Acc']):
                    try:
                        timestamp = line.split(' - ')[0]
                        status['last_update'] = timestamp
                        break
                    except:
                        pass
        except:
            pass
    
    # Count saved model files
    model_files = list(Path('.').glob('*wound_model*.pth'))
    status['models_saved'] = len(model_files)
    
    # Count checkpoint files
    checkpoint_files = list(Path('.').glob('*checkpoint*.pth'))
    status['checkpoints_saved'] = len(checkpoint_files)
    
    return status

def display_dashboard():
    """Display training dashboard"""
    clear_screen()
    
    print("🧠 WOUND CNN TRAINING DASHBOARD")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    status = get_training_status()
    
    # Training Status
    status_indicator = "🟢 ACTIVE" if status['process_running'] else "🔴 STOPPED"
    print(f"Training Status: {status_indicator}")
    print()
    
    # Progress Information
    if status['current_epoch'] > 0:
        print("📊 TRAINING PROGRESS")
        print("-" * 30)
        print(f"Current Epoch: {status['current_epoch']}/20")
        print(f"Current Batch: {status['current_batch']}/61")
        print(f"Latest Loss: {status['latest_loss']:.4f}")
        print(f"Best Accuracy: {status['best_accuracy']:.2f}%")
        print()
        
        # Progress bar for epoch
        epoch_progress = (status['current_epoch'] / 20) * 100
        batch_progress = (status['current_batch'] / 61) * 100
        
        print(f"Epoch Progress: {epoch_progress:.1f}% [{'█' * int(epoch_progress/5):<20}]")
        print(f"Batch Progress: {batch_progress:.1f}% [{'█' * int(batch_progress/5):<20}]")
        print()
    
    # Model Information
    print("💾 MODEL STATUS")
    print("-" * 30)
    print(f"Models Saved: {status['models_saved']}")
    print(f"Checkpoints: {status['checkpoints_saved']}")
    print()
    
    # Recent Activity
    print("⏰ RECENT ACTIVITY")
    print("-" * 30)
    print(f"Last Update: {status['last_update']}")
    
    # Show recent log entries
    log_file = Path('robust_training_output.log')
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            recent_lines = []
            for line in reversed(lines[-10:]):
                line = line.strip()
                if any(keyword in line for keyword in ['Epoch', 'Loss', 'Acc', 'Saved']):
                    recent_lines.append(line)
                    if len(recent_lines) >= 3:
                        break
            
            for line in reversed(recent_lines):
                # Clean up the line for display
                if 'INFO' in line:
                    parts = line.split(' - INFO - ')
                    if len(parts) > 1:
                        print(f"  {parts[1]}")
                else:
                    print(f"  {line}")
        except:
            print("  No recent activity")
    
    print()
    print("=" * 60)
    print("Press Ctrl+C to stop monitoring")

def main():
    """Main dashboard loop"""
    try:
        while True:
            display_dashboard()
            time.sleep(10)  # Update every 10 seconds
    except KeyboardInterrupt:
        clear_screen()
        print("Training dashboard stopped.")

if __name__ == "__main__":
    main()