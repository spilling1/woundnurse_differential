#!/usr/bin/env python3
"""
YOLO Service Monitor with automatic restart
Keeps the YOLO service running in the background
"""

import subprocess
import time
import requests
import signal
import sys
import os
from datetime import datetime

class YoloMonitor:
    def __init__(self):
        self.process = None
        self.pid_file = "yolo.pid"
        self.log_file = "yolo_monitor.log"
        self.service_port = 8081
        self.running = True
        
    def log(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
            
    def is_service_healthy(self):
        """Check if YOLO service is responding"""
        try:
            response = requests.get(f"http://localhost:{self.service_port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_service(self):
        """Start the YOLO service"""
        if self.process and self.process.poll() is None:
            self.log("Service is already running")
            return True
            
        try:
            self.log("Starting YOLO service...")
            self.process = subprocess.Popen(
                ["python3", "yolo_service.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Write PID file
            with open(self.pid_file, "w") as f:
                f.write(str(self.process.pid))
                
            # Wait a bit and check if it started
            time.sleep(3)
            if self.process.poll() is None and self.is_service_healthy():
                self.log(f"Service started successfully with PID {self.process.pid}")
                return True
            else:
                self.log("Service failed to start properly")
                return False
                
        except Exception as e:
            self.log(f"Failed to start service: {e}")
            return False
    
    def stop_service(self):
        """Stop the YOLO service"""
        if self.process:
            try:
                self.log("Stopping YOLO service...")
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
                self.log("Service stopped")
            except:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.log("Service force-killed")
                except:
                    pass
            finally:
                self.process = None
                
        # Clean up PID file
        if os.path.exists(self.pid_file):
            os.remove(self.pid_file)
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.log("Starting YOLO service monitor...")
        
        while self.running:
            try:
                # Check if service is running and healthy
                if not self.process or self.process.poll() is not None:
                    self.log("Service process not running, starting...")
                    self.start_service()
                elif not self.is_service_healthy():
                    self.log("Service not responding, restarting...")
                    self.stop_service()
                    time.sleep(2)
                    self.start_service()
                else:
                    self.log("Service is healthy")
                
                # Wait before next check
                time.sleep(30)
                
            except KeyboardInterrupt:
                self.log("Received interrupt signal")
                break
            except Exception as e:
                self.log(f"Monitor error: {e}")
                time.sleep(10)
        
        self.log("Monitor shutting down...")
        self.stop_service()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.log(f"Received signal {signum}")
        self.running = False

def main():
    monitor = YoloMonitor()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, monitor.signal_handler)
    signal.signal(signal.SIGTERM, monitor.signal_handler)
    
    try:
        monitor.monitor_loop()
    except Exception as e:
        monitor.log(f"Fatal error: {e}")
        monitor.stop_service()
        sys.exit(1)

if __name__ == "__main__":
    main()