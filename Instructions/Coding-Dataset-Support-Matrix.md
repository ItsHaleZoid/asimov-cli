# Coding Dataset Auto-Detection Support Matrix

## ✅ **Supported Dataset Patterns for Mistral Auto-Optimization**

The `training/train.py` script now supports comprehensive auto-detection for multiple coding datasets with various naming patterns:

### **🎯 Primary Datasets:**

#### **1. OpenCodeInstruct** (Recommended)
- ✅ `nvidia/OpenCodeInstruct`
- ✅ `sahilChiddarwar/OpenCodeInstruct`
- ✅ `m-a-p/OpenCodeInstruct`
- ✅ `opencodeinstruct`
- ✅ `open-code-instruct`
- ✅ `open_code_instruct`
- **Sample Size**: 10,000 examples
- **Optimization Level**: Maximum

#### **2. Microsoft rStar-Coder**
- ✅ `microsoft/rStar-Coder`
- ✅ `rstar-coder`
- ✅ `rStar-Coder`
- **Sample Size**: 8,000 examples
- **Optimization Level**: High (competitive programming focus)

#### **3. Code Alpaca Variants**
- ✅ `code_alpaca`
- ✅ `codealpaca`
- ✅ `evol-codealpaca`
- ✅ `theblackcat102/evol-codealpaca-v1`
- **Sample Size**: 6,000 examples
- **Optimization Level**: Medium-High

#### **4. WizardCoder**
- ✅ `wizardcoder`
- ✅ `wizard-coder`
- ✅ `WizardLM/WizardCoder`
- **Sample Size**: 6,000 examples
- **Optimization Level**: Medium-High

#### **5. Magicoder**
- ✅ `magicoder`
- ✅ `magic-coder`
- ✅ `ise-uiuc/Magicoder`
- **Sample Size**: 6,000 examples
- **Optimization Level**: Medium-High

### **🔧 Generic Coding Datasets:**
- ✅ `code-instruct`
- ✅ `coding-instruct`
- ✅ `programming-instruct`
- **Sample Size**: 5,000 examples
- **Optimization Level**: Standard

## 🚀 **Auto-Applied Optimizations**

When any supported coding dataset is detected with a Mistral model:

### **Training Parameters:**
- ✅ **Extended Sequences**: 4,096 tokens (vs. 2,048 standard)
- ✅ **Coding-Optimized Hyperparameters**: Lower LR (1e-4), 2 epochs
- ✅ **Memory Efficiency**: Gradient checkpointing enabled
- ✅ **BF16 Precision**: Mistral-optimized precision
- ✅ **Enhanced Prompting**: Coding-specific instruction formatting

### **Dataset-Specific Configurations:**

| Dataset Type | Sample Size | Context Length | Special Features |
|-------------|-------------|----------------|------------------|
| OpenCodeInstruct | 10,000 | 4,096 | Comprehensive coverage |
| rStar-Coder | 8,000 | 4,096 | Competitive programming |
| Code Alpaca | 6,000 | 4,096 | Evolution-based |
| WizardCoder | 6,000 | 4,096 | Instruction evolution |
| Magicoder | 6,000 | 4,096 | OSS-based synthesis |
| Generic Coding | 5,000 | 4,096 | Standard optimization |

## 📝 **Example Usage:**

### **Environment Variables:**
```bash
# OpenCodeInstruct (NVIDIA)
export DATASET_ID="nvidia/OpenCodeInstruct"
export BASE_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"

# rStar-Coder (Microsoft)
export DATASET_ID="microsoft/rStar-Coder"
export BASE_MODEL_ID="mistralai/Codestral-22B-v0.1"

# Code Alpaca
export DATASET_ID="theblackcat102/evol-codealpaca-v1"
export BASE_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"
```

### **Expected Output:**
```
🎯 DETECTED: OPENCODEINSTRUCT coding dataset with Mistral model
--> Applying optimized configuration for coding fine-tuning
✅ Loaded opencodeinstruct dataset: 5,000,000 examples
--> Using 10,000 examples for comprehensive coding training
--> Dataset optimization level: opencodeinstruct

🔧 APPLYING CODING-OPTIMIZED TRAINING PARAMETERS
--> Enhanced configuration for OpenCodeInstruct + Mistral

🎯 CODING OPTIMIZATION SUMMARY:
   Dataset: OPENCODEINSTRUCT (10,000 samples)
   Model: Mistral with BF16 precision
   Max sequence length: 4096 tokens
   Effective batch size: 8
   Total training steps: ~2500
   Optimized for: Code generation, debugging, and programming tasks
   Features: Extended sequences, enhanced prompting, coding-specific hyperparameters
```

## ✅ **Benefits:**

1. **Zero Configuration**: Automatic detection and optimization
2. **Multi-Dataset Support**: Works with all major coding datasets
3. **Flexible Naming**: Handles various repository naming patterns
4. **Scalable Optimization**: Different optimization levels per dataset type
5. **Mistral-Specific**: Optimized for Mistral architecture and instruction format
6. **Production Ready**: Comprehensive error handling and validation

The training script now provides **intelligent, zero-configuration optimization** for fine-tuning Mistral models on any supported coding dataset, automatically applying the most appropriate configuration for optimal coding performance.