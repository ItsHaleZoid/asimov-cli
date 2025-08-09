#!/usr/bin/env python3
"""
Debug script to check what HF_TOKEN value is actually being passed
"""
import os
import sys

def main():
    print("=== HF TOKEN DEBUG SCRIPT ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check specific token provided in instructions
    test_token = "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"
    print(f"\n=== TESTING SPECIFIC TOKEN: {test_token} ===")
    print(f"Token length: {len(test_token)}")
    print(f"Token starts with 'hf_': {test_token.startswith('hf_')}")
    print(f"Token format appears valid: {len(test_token) == 37 and test_token.startswith('hf_')}")
    
    # Check all environment variables
    print("\n=== ALL ENVIRONMENT VARIABLES ===")
    for key, value in sorted(os.environ.items()):
        if 'HF' in key.upper() or 'TOKEN' in key.upper():
            print(f"{key}: {value}")
    
    # Specifically check HF_TOKEN
    print("\n=== HF_TOKEN SPECIFIC CHECK ===")
    hf_token = os.getenv("HF_TOKEN")
    print(f"HF_TOKEN exists: {hf_token is not None}")
    print(f"HF_TOKEN value: '{hf_token}'")
    print(f"HF_TOKEN type: {type(hf_token)}")
    print(f"HF_TOKEN length: {len(hf_token) if hf_token else 0}")
    
    if hf_token:
        print(f"HF_TOKEN first 10 chars: '{hf_token[:10]}'")
        print(f"HF_TOKEN last 10 chars: '{hf_token[-10:]}'")
        print(f"HF_TOKEN contains whitespace: {any(c.isspace() for c in hf_token)}")
        print(f"HF_TOKEN contains special chars: {any(not c.isalnum() and c not in '_-' for c in hf_token)}")
        print(f"HF_TOKEN matches test token: {hf_token == test_token}")
    
    # Test if the specific token can access HF Hub
    print(f"\n=== TESTING SPECIFIC TOKEN HF HUB ACCESS ===")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=test_token)
        user_info = api.whoami()
        print(f"✅ Specific token is valid! User: {user_info['name']}")
        
        # Test specific model access with the provided token
        print("\n=== TESTING GEMMA MODEL ACCESS WITH SPECIFIC TOKEN ===")
        try:
            model_info = api.model_info("google/gemma-3-1b-it", token=test_token)
            print(f"✅ Can access google/gemma-3-1b-it with specific token! Model ID: {model_info.id}")
        except Exception as e:
            print(f"❌ Cannot access google/gemma-3-1b-it with specific token: {e}")
            
    except ImportError:
        print("⚠️  huggingface_hub not available, cannot test token")
    except Exception as e:
        print(f"❌ Specific token validation failed: {e}")
    
    # Test if environment token can access HF Hub
    print("\n=== TESTING ENVIRONMENT TOKEN HF HUB ACCESS ===")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        print(f"✅ Environment token is valid! User: {user_info['name']}")
        
        # Test specific model access
        print("\n=== TESTING GEMMA MODEL ACCESS WITH ENVIRONMENT TOKEN ===")
        try:
            model_info = api.model_info("google/gemma-3-1b-it", token=hf_token)
            print(f"✅ Can access google/gemma-3-1b-it with environment token! Model ID: {model_info.id}")
        except Exception as e:
            print(f"❌ Cannot access google/gemma-3-1b-it with environment token: {e}")
            
    except ImportError:
        print("⚠️  huggingface_hub not available, cannot test token")
    except Exception as e:
        print(f"❌ Environment token validation failed: {e}")
    
    # Check if transformers can use the specific token
    print("\n=== TESTING TRANSFORMERS ACCESS WITH SPECIFIC TOKEN ===")
    try:
        from transformers import AutoTokenizer
        print("Testing tokenizer access with specific token...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", token=test_token)
        print(f"✅ Tokenizer loaded successfully with specific token! Vocab size: {len(tokenizer)}")
    except ImportError:
        print("⚠️  transformers not available, cannot test")
    except Exception as e:
        print(f"❌ Tokenizer loading failed with specific token: {e}")
    
    # Check if transformers can use the environment token
    print("\n=== TESTING TRANSFORMERS ACCESS WITH ENVIRONMENT TOKEN ===")
    try:
        from transformers import AutoTokenizer
        print("Testing tokenizer access with environment token...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", token=hf_token)
        print(f"✅ Tokenizer loaded successfully with environment token! Vocab size: {len(tokenizer)}")
    except ImportError:
        print("⚠️  transformers not available, cannot test")
    except Exception as e:
        print(f"❌ Tokenizer loading failed with environment token: {e}")
    
    print("\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    main()