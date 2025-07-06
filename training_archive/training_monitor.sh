#!/bin/bash
# Monitor CNN training progress

echo "🔄 Starting Training Monitor"
echo "=========================="

# Function to check training status
check_training() {
    # Check if training process is running
    if pgrep -f "minimal_wound_trainer" > /dev/null; then
        echo "✅ Training is ACTIVE"
        
        # Check for model files
        if ls wound_model_*.pth 1> /dev/null 2>&1; then
            echo "💾 Model files found:"
            ls -la wound_model_*.pth
        fi
        
        return 0
    else
        echo "⏹️  Training process finished"
        
        # Check final results
        if ls wound_model_*.pth 1> /dev/null 2>&1; then
            echo "🎉 TRAINING COMPLETED SUCCESSFULLY!"
            echo "📁 Model files created:"
            ls -la wound_model_*.pth
            return 1
        else
            echo "❌ Training failed - no model files found"
            return 2
        fi
    fi
}

# Monitor loop
while true; do
    clear
    echo "CNN Training Monitor - $(date)"
    echo "=============================="
    
    check_training
    status=$?
    
    if [ $status -eq 1 ]; then
        echo ""
        echo "🚀 Training completed successfully!"
        echo "Your custom wound detection model is ready!"
        break
    elif [ $status -eq 2 ]; then
        echo ""
        echo "❌ Training failed. Check logs for details."
        break
    fi
    
    echo ""
    echo "⏳ Monitoring... (Press Ctrl+C to stop)"
    sleep 30
done