from transformers import GPT2Tokenizer

# Tải tokenizer của GPT-2 (hoặc mô hình mà bạn đang sử dụng)
tokenizer = WhisperTokenizer.from_pretrained(f'openai/whisper-base.en', language='en', task='transcribe')

print(tokenizer.get_vocab())

# Chuyển đổi token ID 50257 thành văn bản
# token_id = 50257
# decoded_token = tokenizer.decode([token_id])

# print(f"Token ID {token_id} corresponds to: {decoded_token}")
