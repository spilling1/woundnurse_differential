#!/bin/bash

# YOLO Service Startup Script with automatic restart
SERVICE_NAME="yolo_service"
SERVICE_FILE="yolo_service.py"
PID_FILE="yolo.pid"
LOG_FILE="yolo_service.log"
PORT=8081

# Function to check if service is running
check_service() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to start service
start_service() {
    echo "Starting $SERVICE_NAME..."
    python3 "$SERVICE_FILE" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 3
    
    # Verify service started
    if check_service; then
        echo "$SERVICE_NAME started successfully (PID: $(cat $PID_FILE))"
        return 0
    else
        echo "Failed to start $SERVICE_NAME"
        return 1
    fi
}

# Function to stop service
stop_service() {
    if check_service; then
        PID=$(cat "$PID_FILE")
        echo "Stopping $SERVICE_NAME (PID: $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "$SERVICE_NAME stopped"
    else
        echo "$SERVICE_NAME is not running"
    fi
}

# Function to restart service
restart_service() {
    stop_service
    sleep 2
    start_service
}

# Function to check health
health_check() {
    if curl -s -f http://localhost:$PORT/health > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Main command handling
case "$1" in
    start)
        if check_service; then
            echo "$SERVICE_NAME is already running"
        else
            start_service
        fi
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        if check_service && health_check; then
            echo "$SERVICE_NAME is running and healthy"
        elif check_service; then
            echo "$SERVICE_NAME is running but not responding"
        else
            echo "$SERVICE_NAME is not running"
        fi
        ;;
    health)
        if health_check; then
            echo "Service is healthy"
            curl -s http://localhost:$PORT/health
        else
            echo "Service is not responding"
            exit 1
        fi
        ;;
    monitor)
        echo "Starting $SERVICE_NAME with monitoring..."
        while true; do
            if ! check_service || ! health_check; then
                echo "$(date): Service down, restarting..."
                restart_service
            else
                echo "$(date): Service is healthy"
            fi
            sleep 30
        done
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|health|monitor}"
        exit 1
        ;;
esac