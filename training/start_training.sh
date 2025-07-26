#!/bin/bash
# Asimov CLI Training Wrapper Script
# This script ensures environment variables are set and starts training with comprehensive logging

set -e  # Exit on any error

# Create training log with detailed startup information
echo "=================================" > /app/training.log
echo "ASIMOV CLI TRAINING STARTED" >> /app/training.log
echo "Timestamp: $(date)" >> /app/training.log
echo "=================================" >> /app/training.log

# Log system information
echo "" >> /app/training.log
echo "=== SYSTEM INFORMATION ===" >> /app/training.log
echo "Hostname: $(hostname)" >> /app/training.log
echo "Working directory: $(pwd)" >> /app/training.log
echo "User: $(whoami)" >> /app/training.log
echo "Python version: $(python --version 2>&1)" >> /app/training.log
echo "GPU info:" >> /app/training.log
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits >> /app/training.log 2>&1 || echo "GPU info not available" >> /app/training.log

# Log file structure
echo "" >> /app/training.log
echo "=== FILES IN /app ===" >> /app/training.log
ls -la /app >> /app/training.log

# Check and log environment variables
echo "" >> /app/training.log
echo "=== ENVIRONMENT VARIABLES ===" >> /app/training.log
echo "Checking required environment variables..." >> /app/training.log

REQUIRED_VARS=("HF_TOKEN" "BASE_MODEL_ID" "DATASET_ID" "LORA_MODEL_REPO" "LORA_TARGET_MODULES")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
        echo "❌ $var: NOT SET" >> /app/training.log
    else
        # Mask sensitive tokens in logs
        if [[ "$var" == *"TOKEN"* ]]; then
            masked_value="${!var:0:8}...${!var: -4}"
            echo "✅ $var: $masked_value" >> /app/training.log
        else
            echo "✅ $var: ${!var}" >> /app/training.log
        fi
    fi
done

# If any variables are missing, try to set defaults or exit
if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo "" >> /app/training.log
    echo "⚠️  Missing environment variables:" >> /app/training.log
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var" >> /app/training.log
    done
    
    # Try to set from backup methods
    echo "" >> /app/training.log
    echo "🔄 Attempting to load from backup sources..." >> /app/training.log
    
    # Try loading from .env file if it exists
    if [ -f "/app/.env" ]; then
        echo "📁 Loading from /app/.env file..." >> /app/training.log
        source /app/.env
    fi
    
    # Try loading from shell profile
    if [ -f "/root/.training_env" ]; then
        echo "📁 Loading from /root/.training_env file..." >> /app/training.log
        source /root/.training_env
    fi
    
    # Re-check after backup loading
    STILL_MISSING=()
    for var in "${MISSING_VARS[@]}"; do
        if [ -z "${!var}" ]; then
            STILL_MISSING+=("$var")
        fi
    done
    
    if [ ${#STILL_MISSING[@]} -ne 0 ]; then
        echo "" >> /app/training.log
        echo "❌ CRITICAL ERROR: Still missing required variables after all attempts:" >> /app/training.log
        for var in "${STILL_MISSING[@]}"; do
            echo "  - $var" >> /app/training.log
        done
        echo "" >> /app/training.log
        echo "🚨 Training cannot proceed. Please check your configuration." >> /app/training.log
        echo "💡 Available environment:" >> /app/training.log
        env | grep -E "(HF_|BASE_|DATASET|LORA)" >> /app/training.log 2>&1 || echo "No relevant environment variables found" >> /app/training.log
        
        # Exit with error
        exit 1
    else
        echo "✅ All variables recovered from backup sources!" >> /app/training.log
    fi
fi

# Change to app directory
cd /app

# Log final environment state
echo "" >> /app/training.log
echo "=== FINAL ENVIRONMENT STATE ===" >> /app/training.log
echo "All required environment variables are now set." >> /app/training.log

# Export all variables to ensure they're available to Python
export HF_TOKEN BASE_MODEL_ID DATASET_ID LORA_MODEL_REPO LORA_TARGET_MODULES
if [ ! -z "$DATASET_SUBSET" ]; then
    export DATASET_SUBSET
fi
if [ ! -z "$DATASET_CONFIG" ]; then
    export DATASET_CONFIG
fi

# Start Python training with comprehensive logging
echo "" >> /app/training.log
echo "=== STARTING PYTHON TRAINING ===" >> /app/training.log
echo "Command: python train.py" >> /app/training.log
echo "Starting at: $(date)" >> /app/training.log
echo "" >> /app/training.log

# Run training and capture all output
python train.py >> /app/training.log 2>&1

# Log completion
TRAINING_EXIT_CODE=$?
echo "" >> /app/training.log
echo "=== TRAINING COMPLETED ===" >> /app/training.log
echo "Completed at: $(date)" >> /app/training.log
echo "Exit code: $TRAINING_EXIT_CODE" >> /app/training.log

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!" >> /app/training.log
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE" >> /app/training.log
fi

echo "=================================" >> /app/training.log

exit $TRAINING_EXIT_CODE