#!/bin/bash
echo "Starting YOLO Wound Detection Service..."
python yolo_service.py &
YOLO_PID=$!
echo "YOLO service started with PID: $YOLO_PID"
echo $YOLO_PID > yolo.pid
sleep 5
curl -s http://localhost:8081/health && echo "YOLO service is healthy"