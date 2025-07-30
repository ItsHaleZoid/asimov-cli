# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("deepseek-ai/deepseek-vl2-small", torch_dtype="auto"),