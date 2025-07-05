#!/bin/bash

# Simple YOLO service runner
echo "Starting YOLO service on port 8081..."

# Kill any existing processes
pkill -f "yolo_service.py" || true
pkill -f "uvicorn" || true

# Remove old PID file
rm -f yolo.pid

# Start the service with proper logging
python3 yolo_service.py > yolo_service.log 2>&1 &
PID=$!

# Save PID
echo $PID > yolo.pid
echo "Started YOLO service with PID: $PID"

# Wait for service to start up
sleep 5

# Check if service is running
if ps -p $PID > /dev/null; then
    echo "YOLO service is running"
    
    # Test health endpoint
    if curl -s http://localhost:8081/health > /dev/null; then
        echo "YOLO service is healthy"
        curl -s http://localhost:8081/health
    else
        echo "YOLO service is not responding"
        exit 1
    fi
else
    echo "YOLO service failed to start"
    exit 1
fi