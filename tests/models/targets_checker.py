from transformers import AutoConfig

model_id = "openai-community/gpt2"
config = AutoConfig.from_pretrained(model_id)
print(config.to_dict()) # This will print all available settings
