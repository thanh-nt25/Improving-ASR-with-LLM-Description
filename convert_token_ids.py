from transformers import GPT2Tokenizer

from transformers_prompt import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperPromptForConditionalGeneration, GenerationConfig, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
# Tải tokenizer của GPT-2 (hoặc mô hình mà bạn đang sử dụng)
tokenizer = WhisperTokenizer.from_pretrained(f'openai/whisper-base.en', language='en', task='transcribe')

# print(tokenizer.get_vocab())

token_id = 50360
decoded_token = tokenizer.decode([token_id])

print(f"Token ID {token_id} corresponds to: {decoded_token}")
