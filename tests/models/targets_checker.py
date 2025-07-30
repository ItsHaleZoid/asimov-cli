from transformers import AutoConfig

model_id = "google/gemma-3-4b-it"
config = AutoConfig.from_pretrained(model_id)
print(config.to_dict()) # This will print all available settings
