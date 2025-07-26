# ==============================================================================
# Asimov CLI - Comprehensive LoRA Target Module Instruction Book
# ==============================================================================
#
# This file acts as a centralized "rulebook" or "registry" for LoRA fine-tuning.
# It maps a wide range of model architecture types from the Hugging Face `transformers`
# library to their recommended `target_modules`.
#
# A universal fine-tuning script can use this registry to automatically
# configure the correct LoRA setup for any supported model.
#
# The dictionary keys are the official architecture names found in a model's `config.json`.

import re
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
import torch
import numpy as np

# Base patterns for identifying module types through ML-based classification
MODULE_PATTERNS = {
    "attention": {
        "keywords": ["query", "key", "value", "qkv", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj", 
                    "q_lin", "k_lin", "v_lin", "attn", "self_attn", "cross_attn", "Wqkv", "c_attn",
                    "attention", "self_attention", "mha", "multihead", "query_key_value"],
        "weight": 1.0
    },
    "mlp": {
        "keywords": ["dense", "fc", "fc1", "fc2", "gate_proj", "up_proj", "down_proj", "wi", "wo", 
                    "ffn", "c_fc", "c_proj", "w1", "w2", "w3", "lin1", "lin2", "dense_h_to_4h", 
                    "dense_4h_to_h", "fc_in", "fc_out", "mlp", "feedforward", "feed_forward"],
        "weight": 1.0
    },
    "embedding": {
        "keywords": ["embed", "embedding", "token", "position", "pos_emb", "word_embeddings",
                    "positional_embeddings", "token_embeddings", "wte", "wpe"],
        "weight": 0.9  # Increased weight for embeddings
    },
    "normalization": {
        "keywords": ["norm", "layer_norm", "ln", "batch_norm", "bn", "group_norm", "rms_norm",
                    "input_layernorm", "post_attention_layernorm", "final_layer_norm"],
        "weight": 0.8  # Increased weight for normalization layers
    },
    "output": {
        "keywords": ["output", "head", "classifier", "lm_head", "prediction", "score", "logits"],
        "weight": 0.9  # Increased weight for output layers
    },
    "convolution": {
        "keywords": ["conv", "conv1d", "conv2d", "conv3d", "depthwise", "pointwise"],
        "weight": 0.9
    },
    "pooling": {
        "keywords": ["pool", "pooling", "avg_pool", "max_pool", "adaptive_pool"],
        "weight": 0.7  # Increased weight for pooling
    }
}

def classify_module_type(module_name: str) -> Tuple[str, float]:
    """
    Uses pattern matching and scoring to classify a module type.
    
    Args:
        module_name (str): Name of the module to classify
        
    Returns:
        Tuple[str, float]: (predicted_type, confidence_score)
    """
    scores = defaultdict(float)
    module_lower = module_name.lower()
    
    for module_type, config in MODULE_PATTERNS.items():
        for keyword in config["keywords"]:
            if keyword in module_lower:
                # Exact match gets full weight
                if keyword == module_lower:
                    scores[module_type] += config["weight"] * 2
                # Substring match gets partial weight
                elif keyword in module_lower:
                    scores[module_type] += config["weight"]
                    
                # Bonus for common patterns
                if re.search(rf'\b{re.escape(keyword)}\b', module_lower):
                    scores[module_type] += config["weight"] * 0.5
    
    if not scores:
        return "unknown", 0.0
    
    best_type = max(scores.items(), key=lambda x: x[1])
    return best_type[0], best_type[1]

def auto_categorize_modules(modules: List[str]) -> Dict[str, List[str]]:
    """
    Automatically categorizes a list of module names using ML-based classification.
    
    Args:
        modules (List[str]): List of module names to categorize
        
    Returns:
        Dict[str, List[str]]: Categorized modules by type
    """
    categorized = defaultdict(list)
    
    for module in modules:
        module_type, confidence = classify_module_type(module)
        if confidence > 0.3:  # Lower confidence threshold for more inclusive targeting
            categorized[module_type].append(module)
        else:
            categorized["unknown"].append(module)
    
    return dict(categorized)

def ai_infer_architecture_family(model_architecture_name: str, 
                                model_state_dict: Optional[Dict[str, Any]] = None) -> str:
    """
    Uses AI-based analysis to infer the architecture family for unknown models.
    
    Args:
        model_architecture_name (str): The architecture name to analyze
        model_state_dict (Optional[Dict[str, Any]]): Model state dict for deeper analysis
        
    Returns:
        str: Inferred architecture family or "unknown"
    """
    arch_lower = model_architecture_name.lower()
    
    # Pattern-based family detection with confidence scoring
    family_patterns = {
        "llama": {
            "patterns": ["llama", "vicuna", "alpaca", "wizard", "hermes", "solar", "zephyr", "yi"],
            "module_indicators": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "confidence": 0.9
        },
        "mistral": {
            "patterns": ["mistral", "mixtral", "zephyr", "neural"],
            "module_indicators": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "confidence": 0.9
        },
        "qwen": {
            "patterns": ["qwen", "qwen2", "tongyi"],
            "module_indicators": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_attn"],
            "confidence": 0.9
        },
        "gemma": {
            "patterns": ["gemma", "gemma2"],
            "module_indicators": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "confidence": 0.9
        },
        "bert": {
            "patterns": ["bert", "roberta", "electra", "deberta", "albert"],
            "module_indicators": ["query", "key", "value", "dense", "LayerNorm"],
            "confidence": 0.8
        },
        "gpt": {
            "patterns": ["gpt", "codegen", "starcoder", "santacoder"],
            "module_indicators": ["c_attn", "c_proj", "c_fc", "wte", "wpe", "ln_1", "ln_2"],
            "confidence": 0.8
        },
        "t5": {
            "patterns": ["t5", "flan", "ul2", "longt5"],
            "module_indicators": ["q", "k", "v", "o", "wi", "wo", "layer_norm"],
            "confidence": 0.8
        },
        "falcon": {
            "patterns": ["falcon", "rw"],
            "module_indicators": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "confidence": 0.8
        },
        "bloom": {
            "patterns": ["bloom", "bloomz"],
            "module_indicators": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "confidence": 0.8
        },
        "phi": {
            "patterns": ["phi", "phi3"],
            "module_indicators": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "confidence": 0.9
        }
    }
    
    family_scores = defaultdict(float)
    
    # Score based on architecture name patterns
    for family, config in family_patterns.items():
        for pattern in config["patterns"]:
            if pattern in arch_lower:
                # Exact match gets higher score
                if pattern == arch_lower or arch_lower.startswith(pattern):
                    family_scores[family] += config["confidence"] * 2
                else:
                    family_scores[family] += config["confidence"]
    
    # If we have state dict, analyze module patterns for additional confidence
    if model_state_dict:
        module_names = list(model_state_dict.keys())
        
        for family, config in family_patterns.items():
            module_match_count = 0
            for indicator in config["module_indicators"]:
                for module_name in module_names:
                    if indicator in module_name:
                        module_match_count += 1
                        break
            
            # Boost score based on module pattern matches
            match_ratio = module_match_count / len(config["module_indicators"])
            family_scores[family] += match_ratio * config["confidence"]
    
    # Return the family with highest score, or "unknown" if no good match
    if family_scores:
        best_family = max(family_scores.items(), key=lambda x: x[1])
        if best_family[1] > 0.5:  # Minimum confidence threshold
            return best_family[0]
    
    return "unknown"

def ai_generate_target_modules(model_architecture_name: str, 
                              model_state_dict: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Uses AI to generate target modules for completely unknown architectures.
    
    Args:
        model_architecture_name (str): The architecture name
        model_state_dict (Optional[Dict[str, Any]]): Model state dict for analysis
        
    Returns:
        List[str]: AI-generated target modules
    """
    if not model_state_dict:
        # Fallback to most common modern architecture pattern
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Extract unique module names from state dict
    module_names = set()
    for key in model_state_dict.keys():
        # Remove layer indices and weight/bias suffixes
        clean_name = re.sub(r'\.(\d+)\.', '.', key)  # Remove layer numbers
        clean_name = re.sub(r'\.(weight|bias)$', '', clean_name)  # Remove weight/bias
        clean_name = clean_name.split('.')[-1]  # Get just the module name
        if clean_name:
            module_names.add(clean_name)
    
    # Use ML classification to categorize modules
    categorized = auto_categorize_modules(list(module_names))
    
    # Build target list prioritizing attention and MLP modules
    target_modules = []
    
    # Add attention modules (highest priority)
    if "attention" in categorized:
        target_modules.extend(categorized["attention"])
    
    # Add MLP modules (high priority)
    if "mlp" in categorized:
        target_modules.extend(categorized["mlp"])
    
    # Add embedding modules (medium priority) if reasonable number
    if "embedding" in categorized and len(categorized["embedding"]) <= 3:
        target_modules.extend(categorized["embedding"])
    
    # Add normalization modules (lower priority) if small number
    if "normalization" in categorized and len(categorized["normalization"]) <= 2:
        target_modules.extend(categorized["normalization"])
    
    # If we don't have enough modules, add some unknowns with attention-like patterns
    if len(target_modules) < 4 and "unknown" in categorized:
        for module in categorized["unknown"]:
            if any(pattern in module.lower() for pattern in ["proj", "linear", "dense", "fc"]):
                target_modules.append(module)
                if len(target_modules) >= 6:
                    break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_targets = []
    for module in target_modules:
        if module not in seen:
            seen.add(module)
            unique_targets.append(module)
    
    # Ensure we have at least some targets
    if not unique_targets:
        # Ultimate fallback - try to find any projection-like modules
        for name in module_names:
            if any(pattern in name.lower() for pattern in ["proj", "linear", "dense", "fc", "attn"]):
                unique_targets.append(name)
                if len(unique_targets) >= 4:
                    break
    
    return unique_targets if unique_targets else ["q_proj", "k_proj", "v_proj", "o_proj"]

def get_lora_rank_recommendations(model_architecture: str, param_count: Optional[int] = None) -> Dict[str, int]:
    """
    Provides recommended LoRA ranks based on model architecture and parameter count.
    Higher ranks for finer and higher quality fine-tuning.
    
    Args:
        model_architecture (str): Model architecture name
        param_count (Optional[int]): Total parameter count of the model
        
    Returns:
        Dict[str, int]: Recommended ranks for different module types
    """
    base_ranks = {
        "small": {"attention": 16, "mlp": 32, "embedding": 16, "normalization": 8, "output": 16},
        "medium": {"attention": 32, "mlp": 64, "embedding": 32, "normalization": 16, "output": 32},
        "large": {"attention": 64, "mlp": 128, "embedding": 64, "normalization": 32, "output": 64},
        "xlarge": {"attention": 128, "mlp": 256, "embedding": 128, "normalization": 64, "output": 128}
    }
    
    # Determine size category
    if param_count:
        if param_count < 1e9:  # < 1B
            size_cat = "small"
        elif param_count < 7e9:  # < 7B
            size_cat = "medium"
        elif param_count < 30e9:  # < 30B
            size_cat = "large"
        else:
            size_cat = "xlarge"
    else:
        # Fallback based on architecture patterns
        if any(arch in model_architecture.lower() for arch in ["gpt2", "distilbert", "albert"]):
            size_cat = "small"
        elif any(arch in model_architecture.lower() for arch in ["llama", "mistral", "gemma"]):
            size_cat = "medium"
        else:
            size_cat = "medium"  # Default
    
    return base_ranks[size_cat]

def get_alpha_recommendations(rank: int) -> float:
    """
    Provides recommended alpha values based on LoRA rank.
    
    Args:
        rank (int): LoRA rank
        
    Returns:
        float: Recommended alpha value
    """
    # Common rule: alpha = rank * 2, but with practical bounds
    alpha = min(rank * 2, 128)
    return float(alpha)

MODEL_TARGETS = {
    # --- A ---
    "AlbertForMaskedLM": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings", "token_type_embeddings"],
        "normalization": ["LayerNorm"]
    },
    "AlbertForSequenceClassification": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings", "token_type_embeddings"],
        "normalization": ["LayerNorm"],
        "output": ["classifier"]
    },
    "AutoformerForPrediction": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "normalization": ["norm1", "norm2"]
    },
    "AquilaForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },

    # --- B ---
    "BartForConditionalGeneration": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["embed_tokens", "embed_positions"],
        "normalization": ["self_attn_layer_norm", "final_layer_norm"]
    },
    "BartForSequenceClassification": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["embed_tokens", "embed_positions"],
        "normalization": ["self_attn_layer_norm", "final_layer_norm"],
        "output": ["classification_head"]
    },
    "BertForMaskedLM": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings", "token_type_embeddings"],
        "normalization": ["LayerNorm"],
        "output": ["cls"]
    },
    "BertForSequenceClassification": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings", "token_type_embeddings"],
        "normalization": ["LayerNorm"],
        "output": ["classifier"]
    },
    "BertForQuestionAnswering": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings", "token_type_embeddings"],
        "normalization": ["LayerNorm"],
        "output": ["qa_outputs"]
    },
    "BigBirdForMaskedLM": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings"],
        "normalization": ["LayerNorm"]
    },
    "BlenderbotForConditionalGeneration": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["embed_tokens", "embed_positions"],
        "normalization": ["self_attn_layer_norm", "final_layer_norm"]
    },
    "BloomForCausalLM": {
        "attention": ["query_key_value", "dense"],
        "mlp": ["dense_h_to_4h", "dense_4h_to_h"],
        "embedding": ["word_embeddings"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "ln_f"]
    },
    "BloomForSequenceClassification": {
        "attention": ["query_key_value", "dense"],
        "mlp": ["dense_h_to_4h", "dense_4h_to_h"],
        "embedding": ["word_embeddings"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "ln_f"],
        "output": ["score"]
    },
    "BaiChuanForCausalLM": {
        "attention": ["W_pack", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },

    # --- C ---
    "CodeGenForCausalLM": {
        "attention": ["qkv_proj", "out_proj"],
        "mlp": ["fc_in", "fc_out"],
        "embedding": ["wte", "wpe"],
        "normalization": ["ln_1", "ln_2", "ln_f"]
    },
    "CodeLlamaForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "CLIPVisionModel": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["class_embedding", "position_embedding"],
        "normalization": ["layer_norm1", "layer_norm2", "pre_layrnorm", "post_layernorm"]
    },
    "CLIPTextModel": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["token_embedding", "position_embedding"],
        "normalization": ["layer_norm1", "layer_norm2", "final_layer_norm"]
    },
    "ChatGLMForConditionalGeneration": {
        "attention": ["query_key_value", "dense"],
        "mlp": ["dense_h_to_4h", "dense_4h_to_h"],
        "embedding": ["word_embeddings"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "final_layernorm"]
    },
    "CohereForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },

    # --- D ---
    "Data2VecVisionForImageClassification": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["embeddings"],
        "normalization": ["layernorm_before", "layernorm_after"],
        "output": ["classifier"]
    },
    "DebertaV2ForMaskedLM": {
        "attention": ["query_proj", "key_proj", "value_proj", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings"],
        "normalization": ["LayerNorm"]
    },
    "DebertaV2ForSequenceClassification": {
        "attention": ["query_proj", "key_proj", "value_proj", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings"],
        "normalization": ["LayerNorm"],
        "output": ["classifier"]
    },
    "DeepseekV3ForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "DistilBertForMaskedLM": {
        "attention": ["q_lin", "k_lin", "v_lin", "out_lin"],
        "mlp": ["ffn", "lin1", "lin2"],
        "embedding": ["word_embeddings", "position_embeddings"],
        "normalization": ["sa_layer_norm", "output_layer_norm"]
    },
    "DistilBertForSequenceClassification": {
        "attention": ["q_lin", "k_lin", "v_lin", "out_lin"],
        "mlp": ["ffn", "lin1", "lin2"],
        "embedding": ["word_embeddings", "position_embeddings"],
        "normalization": ["sa_layer_norm", "output_layer_norm"],
        "output": ["classifier"]
    },
    "DptForDepthEstimation": {
        "attention": ["query", "key", "value"],
        "mlp": ["dense"],
        "embedding": ["embeddings"],
        "normalization": ["layernorm_before", "layernorm_after"]
    },

    # --- E ---
    "ElectraForMaskedLM": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings", "token_type_embeddings"],
        "normalization": ["LayerNorm"]
    },
    "ElectraForSequenceClassification": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings", "token_type_embeddings"],
        "normalization": ["LayerNorm"],
        "output": ["classifier"]
    },
    
    # --- F ---
    "FalconForCausalLM": {
        "attention": ["query_key_value", "dense"],
        "mlp": ["dense_h_to_4h", "dense_4h_to_h"],
        "embedding": ["word_embeddings"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "ln_f"]
    },
    "FlanT5ForConditionalGeneration": {
        "attention": ["q", "k", "v", "o"],
        "mlp": ["wi", "wo", "wi_0", "wi_1"],
        "embedding": ["embed_tokens"],
        "normalization": ["layer_norm", "final_layer_norm"]
    },
    "FuyuForCausalLM": {
        "attention": ["query_key_value", "dense"],
        "mlp": ["dense_h_to_4h", "dense_4h_to_h"],
        "embedding": ["word_embeddings"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },

    # --- G ---
    "GemmaForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "Gemma2ForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "GPT2LMHeadModel": {
        "attention": ["c_attn", "c_proj"],
        "mlp": ["c_fc", "c_proj"],
        "embedding": ["wte", "wpe"],
        "normalization": ["ln_1", "ln_2", "ln_f"]
    },
    "GPT2ForSequenceClassification": {
        "attention": ["c_attn", "c_proj"],
        "mlp": ["c_fc", "c_proj"],
        "embedding": ["wte", "wpe"],
        "normalization": ["ln_1", "ln_2", "ln_f"],
        "output": ["score"]
    },
    "GPTJForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc_in", "fc_out"],
        "embedding": ["wte"],
        "normalization": ["ln_1", "ln_f"]
    },
    "GPTNeoForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["c_fc", "c_proj"],
        "embedding": ["wte", "wpe"],
        "normalization": ["ln_1", "ln_2", "ln_f"]
    },
    "GPTNeoXForCausalLM": {
        "attention": ["query_key_value", "dense"],
        "mlp": ["dense_h_to_4h", "dense_4h_to_h"],
        "embedding": ["embed_in"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "final_layer_norm"]
    },
    "GLMForCausalLM": {
        "attention": ["query_key_value", "dense"],
        "mlp": ["dense_h_to_4h", "dense_4h_to_h"],
        "embedding": ["word_embeddings"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },

    # --- H ---
    "HermesChatModel": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },

    # --- I ---
    "IdeficsForVisionText2Text": {
        "attention": ["q_proj", "k_proj", "v_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },
    "InternLMForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "InternLM2ForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },

    # --- J ---
    "Jamba": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },

    # --- K ---
    "KimiForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },

    # --- L ---
    "LlamaForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "Llama2ForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "Llama3ForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "LongT5ForConditionalGeneration": {
        "attention": ["q", "k", "v", "o"],
        "mlp": ["wi", "wo", "wi_0", "wi_1"],
        "embedding": ["embed_tokens"],
        "normalization": ["layer_norm", "final_layer_norm"]
    },
    "LlaVAForConditionalGeneration": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },
    
    # --- M ---
    "M2M100ForConditionalGeneration": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["embed_tokens", "embed_positions"],
        "normalization": ["self_attn_layer_norm", "final_layer_norm"]
    },
    "MBartForConditionalGeneration": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["embed_tokens", "embed_positions"],
        "normalization": ["self_attn_layer_norm", "final_layer_norm"]
    },
    "MistralForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "Mistral7BForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "MixtralForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["w1", "w2", "w3"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "MptForCausalLM": {
        "attention": ["Wqkv", "out_proj"],
        "mlp": ["up_proj", "down_proj"],
        "embedding": ["wte", "wpe"],
        "normalization": ["norm_1", "norm_2", "norm_f"]
    },
    "MPTForCausalLM": {
        "attention": ["Wqkv", "out_proj"],
        "mlp": ["up_proj", "down_proj"],
        "embedding": ["wte", "wpe"],
        "normalization": ["norm_1", "norm_2", "norm_f"]
    },

    # --- N ---
    "NemotronForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },

    # --- O ---
    "OPTForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["embed_tokens", "embed_positions"],
        "normalization": ["self_attn_layer_norm", "final_layer_norm"]
    },
    "OlmoForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },

    # --- P ---
    "PegasusForConditionalGeneration": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["embed_tokens", "embed_positions"],
        "normalization": ["self_attn_layer_norm", "final_layer_norm"]
    },
    "PhiForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "dense"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "final_layernorm"]
    },
    "Phi3ForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "PersimmonForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },
    
    # --- Q ---
    "Qwen2ForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "QwenForCausalLM": {
        "attention": ["c_attn", "c_proj"],
        "mlp": ["w1", "w2"],
        "embedding": ["wte"],
        "normalization": ["ln_1", "ln_2", "ln_f"]
    },
    "Qwen2VLForConditionalGeneration": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },

    # --- R ---
    "RobertaForMaskedLM": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings", "token_type_embeddings"],
        "normalization": ["LayerNorm"]
    },
    "RobertaForSequenceClassification": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings", "token_type_embeddings"],
        "normalization": ["LayerNorm"],
        "output": ["classifier"]
    },
    "RobertaForQuestionAnswering": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings", "token_type_embeddings"],
        "normalization": ["LayerNorm"],
        "output": ["qa_outputs"]
    },

    # --- S ---
    "StableLmForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },
    "SwinForImageClassification": {
        "attention": ["query", "key", "value"],
        "mlp": ["dense"],
        "embedding": ["embeddings"],
        "normalization": ["norm"],
        "output": ["classifier"]
    },
    "SolarForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },
    "StarcoderForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["c_fc", "c_proj"],
        "embedding": ["wte", "wpe"],
        "normalization": ["ln_1", "ln_2", "ln_f"]
    },

    # --- T ---
    "T5ForConditionalGeneration": {
        "attention": ["q", "k", "v", "o"],
        "mlp": ["wi", "wo", "wi_0", "wi_1"],
        "embedding": ["embed_tokens"],
        "normalization": ["layer_norm", "final_layer_norm"]
    },
    "T5ForSequenceClassification": {
        "attention": ["q", "k", "v", "o"],
        "mlp": ["wi", "wo", "wi_0", "wi_1"],
        "embedding": ["embed_tokens"],
        "normalization": ["layer_norm", "final_layer_norm"],
        "output": ["classifier"]
    },
    
    # --- U ---
    "UL2ForConditionalGeneration": {
        "attention": ["q", "k", "v", "o"],
        "mlp": ["wi", "wo", "wi_0", "wi_1"],
        "embedding": ["embed_tokens"],
        "normalization": ["layer_norm", "final_layer_norm"]
    },

    # --- V ---
    "ViTForImageClassification": {
        "attention": ["query", "key", "value"],
        "mlp": ["dense"],
        "embedding": ["embeddings"],
        "normalization": ["layernorm_before", "layernorm_after"],
        "output": ["classifier"]
    },
    "VicunaForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },
    
    # --- W ---
    "Wav2Vec2ForCTC": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "embedding": ["feature_projection"],
        "normalization": ["layer_norm"],
        "output": ["lm_head"]
    },
    "WhisperForConditionalGeneration": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["embed_tokens", "embed_positions"],
        "normalization": ["self_attn_layer_norm", "final_layer_norm"]
    },
    "WizardLMForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },

    # --- X ---
    "XGLMForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["fc1", "fc2"],
        "embedding": ["embed_tokens", "embed_positions"],
        "normalization": ["self_attn_layer_norm", "final_layer_norm"]
    },
    "XLMForMaskedLM": {
        "attention": ["q_lin", "k_lin", "v_lin"],
        "mlp": ["lin1", "lin2"],
        "embedding": ["embeddings"],
        "normalization": ["layer_norm1", "layer_norm2"]
    },
    "XLMRobertaForMaskedLM": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings"],
        "normalization": ["LayerNorm"]
    },
    "XLMRobertaForSequenceClassification": {
        "attention": ["query", "key", "value", "dense"],
        "mlp": ["dense", "LayerNorm"],
        "embedding": ["word_embeddings", "position_embeddings"],
        "normalization": ["LayerNorm"],
        "output": ["classifier"]
    },

    # --- Y ---
    "YiForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm", "norm"]
    },

    # --- Z ---
    "ZephyrForCausalLM": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embedding": ["embed_tokens"],
        "normalization": ["input_layernorm", "post_attention_layernorm"]
    },
}

# Architecture family mappings for better fallback handling
ARCHITECTURE_FAMILIES = {
    "llama": ["LlamaForCausalLM", "Llama2ForCausalLM", "Llama3ForCausalLM", "CodeLlamaForCausalLM", "VicunaForCausalLM"],
    "mistral": ["MistralForCausalLM", "Mistral7BForCausalLM", "MixtralForCausalLM"],
    "qwen": ["QwenForCausalLM", "Qwen2ForCausalLM", "Qwen2VLForConditionalGeneration"],
    "gemma": ["GemmaForCausalLM", "Gemma2ForCausalLM"],
    "bert": ["BertForMaskedLM", "BertForSequenceClassification", "BertForQuestionAnswering"],
    "roberta": ["RobertaForMaskedLM", "RobertaForSequenceClassification", "RobertaForQuestionAnswering"],
    "gpt": ["GPT2LMHeadModel", "GPT2ForSequenceClassification", "GPTJForCausalLM", "GPTNeoForCausalLM", "GPTNeoXForCausalLM"],
    "t5": ["T5ForConditionalGeneration", "T5ForSequenceClassification", "FlanT5ForConditionalGeneration"],
}

def get_target_modules(model_architecture_name: str, target_type: str = "all", 
                      model_state_dict: Optional[Dict[str, Any]] = None) -> list:
    """
    Looks up the recommended LoRA target modules for a given model architecture.
    Now uses AI-powered inference for unknown architectures.

    Args:
        model_architecture_name (str): The official architecture name from the model's config.json.
        target_type (str): One of "all", "attention", "mlp", "embedding", "normalization", or "output".
        model_state_dict (Optional[Dict[str, Any]]): Model state dict for AI inference if needed.

    Returns:
        list: A list of string names for the target modules, or AI-generated list if not found.
    """
    config = MODEL_TARGETS.get(model_architecture_name)

    if not config:
        # Try to find a similar architecture in the same family first
        for family, architectures in ARCHITECTURE_FAMILIES.items():
            if any(arch.lower() in model_architecture_name.lower() for arch in architectures):
                # Use the first architecture in the family as a template
                config = MODEL_TARGETS.get(architectures[0])
                if config:
                    print(f"Info: Using {architectures[0]} target modules for {model_architecture_name} (family: {family})")
                    break
        
        if not config:
            # Use AI to infer the architecture family and generate targets
            print(f"Info: Architecture '{model_architecture_name}' not found in registry. Using AI inference...")
            
            # First, try to infer the family
            inferred_family = ai_infer_architecture_family(model_architecture_name, model_state_dict)
            
            if inferred_family != "unknown" and inferred_family in ARCHITECTURE_FAMILIES:
                # Use the inferred family's template
                template_arch = ARCHITECTURE_FAMILIES[inferred_family][0]
                config = MODEL_TARGETS.get(template_arch)
                if config:
                    print(f"Info: AI inferred family '{inferred_family}', using {template_arch} as template")
            
            if not config:
                # Generate modules directly using AI
                ai_targets = ai_generate_target_modules(model_architecture_name, model_state_dict)
                print(f"Info: AI generated target modules: {ai_targets}")
                return ai_targets

    if target_type == "all":
        # Return all module types for comprehensive fine-tuning
        all_modules = []
        for module_list in config.values():
            all_modules.extend(module_list)
        return all_modules
    elif target_type in config:
        return config[target_type]
    else:
        print(f"Warning: Unknown target_type '{target_type}'. Returning all modules.")
        all_modules = []
        for module_list in config.values():
            all_modules.extend(module_list)
        return all_modules

def infer_target_modules_from_model(model_state_dict: Dict[str, Any], 
                                   filter_types: List[str] = ["attention", "mlp", "embedding", "normalization"]) -> Dict[str, List[str]]:
    """
    Uses ML-based classification to automatically infer target modules from a model's state dict.
    
    Args:
        model_state_dict (Dict[str, Any]): The model's state dictionary
        filter_types (List[str]): Types of modules to include in results
        
    Returns:
        Dict[str, List[str]]: Categorized modules suitable for LoRA targeting
    """
    module_names = list(model_state_dict.keys())
    
    # Extract layer names (remove .weight, .bias suffixes)
    cleaned_names = []
    for name in module_names:
        clean_name = re.sub(r'\.(weight|bias)$', '', name)
        if clean_name not in cleaned_names:
            cleaned_names.append(clean_name)
    
    # Auto-categorize using ML classification
    categorized = auto_categorize_modules(cleaned_names)
    
    # Filter to only include requested types
    filtered_result = {}
    for module_type in filter_types:
        if module_type in categorized:
            filtered_result[module_type] = categorized[module_type]
    
    return filtered_result

def analyze_model_complexity(model_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes model complexity metrics for LoRA configuration optimization.
    
    Args:
        model_state_dict (Dict[str, Any]): The model's state dictionary
        
    Returns:
        Dict[str, Any]: Analysis results including parameter counts and recommendations
    """
    total_params = 0
    layer_counts = defaultdict(int)
    param_distribution = defaultdict(int)
    
    for name, param in model_state_dict.items():
        if hasattr(param, 'numel'):
            param_count = param.numel()
            total_params += param_count
            
            # Categorize by layer type
            module_type, _ = classify_module_type(name)
            layer_counts[module_type] += 1
            param_distribution[module_type] += param_count
    
    # Calculate recommendations
    complexity_score = total_params / 1e6  # In millions
    recommended_ranks = get_lora_rank_recommendations("auto", total_params)
    
    return {
        "total_parameters": total_params,
        "complexity_score": complexity_score,
        "layer_counts": dict(layer_counts),
        "parameter_distribution": dict(param_distribution),
        "recommended_ranks": recommended_ranks,
        "size_category": "small" if total_params < 1e9 else "medium" if total_params < 7e9 else "large"
    }

def get_optimal_lora_config(model_architecture: str, 
                           model_state_dict: Optional[Dict[str, Any]] = None,
                           target_type: str = "all",
                           training_data_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Provides an optimal LoRA configuration based on model characteristics.
    Enhanced for finer and higher quality fine-tuning.
    
    Args:
        model_architecture (str): Model architecture name
        model_state_dict (Optional[Dict[str, Any]]): Model state dict for analysis
        target_type (str): Type of modules to target ("all" for comprehensive fine-tuning)
        training_data_size (Optional[int]): Size of training dataset
        
    Returns:
        Dict[str, Any]: Complete LoRA configuration
    """
    target_modules = get_target_modules(model_architecture, target_type, model_state_dict)
    
    # Analyze model if state dict provided
    if model_state_dict:
        analysis = analyze_model_complexity(model_state_dict)
        param_count = analysis["total_parameters"]
        recommended_ranks = analysis["recommended_ranks"]
    else:
        param_count = None
        recommended_ranks = get_lora_rank_recommendations(model_architecture)
    
    # Determine rank based on target type - higher ranks for better quality
    if target_type == "attention":
        rank = recommended_ranks["attention"]
    elif target_type == "mlp":
        rank = recommended_ranks["mlp"]
    elif target_type == "embedding":
        rank = recommended_ranks["embedding"]
    elif target_type == "normalization":
        rank = recommended_ranks["normalization"]
    else:
        # For "all" or mixed targeting, use the highest rank for best quality
        rank = max(recommended_ranks.values())
    
    # Adjust rank based on training data size - higher ranks for more data
    if training_data_size:
        if training_data_size < 1000:
            rank = max(rank // 2, 8)  # Minimum rank 8 for quality
        elif training_data_size > 50000:
            rank = min(rank * 2, 256)  # Higher cap for large datasets
        elif training_data_size > 10000:
            rank = min(int(rank * 1.5), 128)  # Moderate increase
    
    alpha = get_alpha_recommendations(rank)
    
    return {
        "target_modules": target_modules,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.05,  # Lower dropout for better learning
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "analysis": analysis if model_state_dict else None
    }

def validate_target_modules(model, target_modules: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validates target modules against actual model structure.
    
    Args:
        model: The actual model object
        target_modules (List[str]): Proposed target modules
        
    Returns:
        Tuple[List[str], List[str]]: (valid_modules, invalid_modules)
    """
    valid_modules = []
    invalid_modules = []
    
    # Get all module names from the model
    model_modules = set()
    for name, _ in model.named_modules():
        model_modules.add(name.split('.')[-1])  # Get just the module name
    
    for target in target_modules:
        if target in model_modules:
            valid_modules.append(target)
        else:
            invalid_modules.append(target)
    
    return valid_modules, invalid_modules

if __name__ == '__main__':
    # --- Example Usage ---
    print("--- LoRA Target Module Instruction Book with AI-Powered Architecture Inference ---")

    # Simulate getting the architecture name from a model's config.json
    test_architectures = ["MistralForCausalLM", "GPT2LMHeadModel", "Qwen2ForCausalLM", "T5ForConditionalGeneration", "Wav2Vec2ForCTC", "Llama3ForCausalLM", "Gemma2ForCausalLM"]
    
    for arch in test_architectures:
        print(f"\nLooking up modules for: {arch}")

        # Get all recommended target modules for comprehensive fine-tuning
        all_modules = get_target_modules(arch, target_type="all")
        print(f"  - All Targets (Comprehensive): {all_modules}")

        # Get only the attention layer modules
        attention_modules = get_target_modules(arch, target_type="attention")
        print(f"  - Attention Targets Only: {attention_modules}")
        
        # Get optimal LoRA configuration for high-quality fine-tuning
        optimal_config = get_optimal_lora_config(arch, target_type="all")
        print(f"  - Optimal Config (High Quality): rank={optimal_config['r']}, alpha={optimal_config['lora_alpha']}")
        
        # Demonstrate ML classification
        print(f"  - ML Classification Results:")
        all_module_names = all_modules
        categorized = auto_categorize_modules(all_module_names)
        for category, modules in categorized.items():
            print(f"    {category}: {modules}")
    
    # Test AI-powered inference for unknown architectures
    print(f"\n--- Testing AI-Powered Architecture Inference ---")
    unknown_architectures = ["CustomLlamaModel", "MyNovelTransformer", "InnovativeAttentionModel"]
    
    # Simulate a comprehensive state dict for demonstration
    fake_state_dict = {
        "model.embed_tokens.weight": torch.randn(32000, 4096),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4096, 4096),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(4096, 4096),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4096, 4096),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(11008, 4096),
        "model.layers.0.mlp.up_proj.weight": torch.randn(11008, 4096),
        "model.layers.0.mlp.down_proj.weight": torch.randn(4096, 11008),
        "model.layers.0.input_layernorm.weight": torch.randn(4096),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(4096),
        "model.norm.weight": torch.randn(4096),
        "lm_head.weight": torch.randn(32000, 4096),
    }
    
    for unknown_arch in unknown_architectures:
        print(f"\nTesting AI inference for: {unknown_arch}")
        
        # Test family inference
        inferred_family = ai_infer_architecture_family(unknown_arch, fake_state_dict)
        print(f"  - AI inferred family: {inferred_family}")
        
        # Test target module generation
        ai_targets = get_target_modules(unknown_arch, target_type="all", model_state_dict=fake_state_dict)
        print(f"  - AI generated targets: {ai_targets}")
        
        # Test AI module generation directly
        direct_ai_targets = ai_generate_target_modules(unknown_arch, fake_state_dict)
        print(f"  - Direct AI generation: {direct_ai_targets}")
    
    print(f"\n--- Model Analysis Example ---")
    analysis = analyze_model_complexity(fake_state_dict)
    print(f"Analysis: {analysis}")
    
    # Test comprehensive targeting
    inferred = infer_target_modules_from_model(fake_state_dict, 
                                             filter_types=["attention", "mlp", "embedding", "normalization", "output"])
    print(f"Inferred comprehensive targets: {inferred}")
