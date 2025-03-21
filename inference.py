import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio
import librosa

def load_audio(audio_path, target_sr=16000):
    """
    Load audio file and resample if necessary
    
    Args:
        audio_path (str): Path to the audio file
        target_sr (int): Target sampling rate (default 16000 for Whisper)
    
    Returns:
        numpy.ndarray: Loaded and resampled audio array
    """
    waveform, original_sr = librosa.load(audio_path, sr=None)
    if original_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=target_sr)
    return waveform

def transcribe_audio(model_path, audio_path, device='cuda'):
    """
    Transcribe an audio file using a fine-tuned Whisper model
    
    Args:
        model_path (str): Path to the fine-tuned model
        audio_path (str): Path to the audio file to transcribe
        device (str): Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        dict: Contains original and transcribed text
    """
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    
    # Load and process audio
    audio = load_audio(audio_path)
    input_features = processor.feature_extractor(audio, return_tensors="pt").input_features.to(device)
    
    # Load ground truth transcript if exists
    ground_truth = "Ground truth not available"
    try:
        # Adjust this path according to your dataset structure
        transcript_path = audio_path.replace('.mp3', '.txt')
        with open(transcript_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
    except FileNotFoundError:
        print(f"No transcript found for {audio_path}")
    
    # Generate transcription
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    generated_ids = model.generate(
        input_features, 
        forced_decoder_ids=forced_decoder_ids,
        max_length=225
    )
    
    # Decode transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return {
        "original_transcript": ground_truth,
        "model_transcript": transcription,
        "audio_path": audio_path
    }

def main():
    # Example usage
    model_path = "/content/drive/MyDrive/Thesis/Improving-ASR-with-LLM-Description/results/test/checkpoint-5700"
    audio_path = "/path/to/your/specific/audio/file.mp3"  # Replace with your specific audio file
    
    result = transcribe_audio(model_path, audio_path)
    
    print("=" * 50)
    print("Audio File:", result['audio_path'])
    print("=" * 50)
    print("Original Transcript:\n", result['original_transcript'])
    print("\n" + "=" * 50 + "\n")
    print("Model Transcript:\n", result['model_transcript'])

if __name__ == "__main__":
    main()