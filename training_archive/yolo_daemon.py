#!/usr/bin/env python3
"""
YOLO Service Daemon - Runs in background and keeps service alive
"""

import os
import sys
import time
import signal
import subprocess
import requests
from datetime import datetime
import threading
import atexit

class YoloDaemon:
    def __init__(self):
        self.pid_file = '/tmp/yolo_daemon.pid'
        self.service_pid_file = '/tmp/yolo_service.pid'
        self.log_file = 'yolo_daemon.log'
        self.service_log_file = 'yolo_service.log'
        self.service_process = None
        self.running = False
        self.service_port = 8081
        
    def daemonize(self):
        """Daemonize the process"""
        try:
            # First fork
            if os.fork() > 0:
                sys.exit(0)
        except OSError as e:
            sys.stderr.write(f"Fork failed: {e}\n")
            sys.exit(1)
            
        # Become session leader
        os.setsid()
        
        # Second fork
        try:
            if os.fork() > 0:
                sys.exit(0)
        except OSError as e:
            sys.stderr.write(f"Fork failed: {e}\n")
            sys.exit(1)
            
        # Change working directory and umask
        os.chdir('/')
        os.umask(0)
        
        # Redirect stdout, stderr
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Write PID file
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
            
        # Register cleanup
        atexit.register(self.cleanup)
        
    def cleanup(self):
        """Clean up on exit"""
        if self.service_process:
            self.service_process.terminate()
        if os.path.exists(self.pid_file):
            os.remove(self.pid_file)
        if os.path.exists(self.service_pid_file):
            os.remove(self.service_pid_file)
            
    def log(self, message):
        """Write to log file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
            
    def is_service_healthy(self):
        """Check if service is responding"""
        try:
            response = requests.get(f'http://localhost:{self.service_port}/health', timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def start_service(self):
        """Start the YOLO service"""
        if self.service_process and self.service_process.poll() is None:
            return True
            
        try:
            self.log("Starting YOLO service...")
            
            # Start service process
            self.service_process = subprocess.Popen(
                ['python3', 'yolo_service.py'],
                stdout=open(self.service_log_file, 'a'),
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                preexec_fn=os.setsid
            )
            
            # Save service PID
            with open(self.service_pid_file, 'w') as f:
                f.write(str(self.service_process.pid))
                
            # Wait for startup
            time.sleep(5)
            
            if self.service_process.poll() is None:
                self.log(f"Service started with PID {self.service_process.pid}")
                return True
            else:
                self.log("Service failed to start")
                return False
                
        except Exception as e:
            self.log(f"Error starting service: {e}")
            return False
            
    def stop_service(self):
        """Stop the YOLO service"""
        if self.service_process:
            try:
                self.log("Stopping YOLO service...")
                os.killpg(os.getpgid(self.service_process.pid), signal.SIGTERM)
                self.service_process.wait(timeout=10)
                self.log("Service stopped")
            except:
                try:
                    os.killpg(os.getpgid(self.service_process.pid), signal.SIGKILL)
                except:
                    pass
            finally:
                self.service_process = None
                
    def monitor_loop(self):
        """Main monitoring loop"""
        self.running = True
        self.log("YOLO daemon started")
        
        while self.running:
            try:
                # Check service health
                if (not self.service_process or 
                    self.service_process.poll() is not None or
                    not self.is_service_healthy()):
                    
                    self.log("Service needs restart")
                    self.stop_service()
                    time.sleep(2)
                    self.start_service()
                    
                # Wait before next check
                time.sleep(30)
                
            except Exception as e:
                self.log(f"Monitor error: {e}")
                time.sleep(10)
                
        self.log("YOLO daemon stopping")
        self.stop_service()
        
    def signal_handler(self, signum, frame):
        """Handle signals"""
        self.running = False
        
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 yolo_daemon.py [start|stop|restart|status]")
        sys.exit(1)
        
    daemon = YoloDaemon()
    command = sys.argv[1]
    
    if command == 'start':
        if os.path.exists(daemon.pid_file):
            print("Daemon already running")
            sys.exit(1)
        
        print("Starting YOLO daemon...")
        daemon.daemonize()
        
        # Set up signal handling
        signal.signal(signal.SIGTERM, daemon.signal_handler)
        signal.signal(signal.SIGINT, daemon.signal_handler)
        
        daemon.monitor_loop()
        
    elif command == 'stop':
        if not os.path.exists(daemon.pid_file):
            print("Daemon not running")
            sys.exit(1)
            
        with open(daemon.pid_file, 'r') as f:
            pid = int(f.read().strip())
            
        try:
            os.kill(pid, signal.SIGTERM)
            print("Daemon stopped")
        except OSError:
            print("Daemon not found")
            
    elif command == 'restart':
        # Stop first
        if os.path.exists(daemon.pid_file):
            with open(daemon.pid_file, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
            except OSError:
                pass
                
        # Start
        daemon.daemonize()
        signal.signal(signal.SIGTERM, daemon.signal_handler)
        signal.signal(signal.SIGINT, daemon.signal_handler)
        daemon.monitor_loop()
        
    elif command == 'status':
        if os.path.exists(daemon.pid_file):
            with open(daemon.pid_file, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)  # Check if process exists
                print(f"Daemon running (PID: {pid})")
                
                # Check service health
                if daemon.is_service_healthy():
                    print("Service is healthy")
                else:
                    print("Service not responding")
                    
            except OSError:
                print("Daemon not running")
        else:
            print("Daemon not running")
            
    else:
        print("Invalid command")
        sys.exit(1)

if __name__ == '__main__':
    main()