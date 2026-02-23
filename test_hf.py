from transformers import AutoModel, AutoTokenizer

model_id = "zai-org/GLM-OCR"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("Tokenizer loaded!")
