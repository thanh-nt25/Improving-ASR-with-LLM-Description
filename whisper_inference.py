import os
import argparse
import torch
import torchaudio
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from IPython.display import display, HTML

def main():
    parser = argparse.ArgumentParser(description="Run inference with Whisper model and compare with ground truth")
    parser.add_argument("--model_id", type=str, default="thanh-nt25/whisper-earning", 
                        help="Hugging Face model ID or local path")
    parser.add_argument("--audio_dir", type=str, required=True, 
                        help="Directory containing audio files and corresponding JSON files")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to run on (cuda:0, cpu, etc.)")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Determine torch dtype
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    
    print(f"Loading model from {args.model_id}...")
    print(f"Using device: {device}, dtype: {torch_dtype}")
    
    # Load model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        chunk_length_s=30,
        batch_size=args.batch_size,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # Find all MP3 files in the directory and subfolders
    audio_files = []
    for root, _, files in os.walk(args.audio_dir):
        for file in files:
            if file.endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each audio file
    for audio_file in audio_files:
        print(f"\n{'='*100}")
        print(f"Processing: {audio_file}")
        print(f"{'='*100}")
        
        base_name = os.path.splitext(audio_file)[0]
        json_file = f"{base_name}.json"
        
        # Check if corresponding JSON file exists
        if not os.path.exists(json_file):
            print(f"Warning: No corresponding JSON file found for {audio_file}")
            continue
        
        try:
            # Load ground truth transcript
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                original_transcript = json_data.get("text", "No transcript found in JSON")
            
            # Load audio
            speech_array, sampling_rate = torchaudio.load(audio_file)
            speech = speech_array[0].numpy()
            
            # Run inference
            result = pipe(speech, sampling_rate=sampling_rate)
            model_transcript = result["text"]
            
            # Print comparison to console
            print(f"\n{'-'*40}")
            print("ORIGINAL TRANSCRIPT:")
            print(f"{'-'*40}")
            print(f"{original_transcript}")
            
            print(f"\n{'-'*40}")
            print("MODEL TRANSCRIPT:")
            print(f"{'-'*40}")
            print(f"{model_transcript}")
            
            # If running in Colab or Jupyter, display more nicely formatted HTML
            try:
                audio_filename = os.path.basename(audio_file)
                html_output = f"""
                <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                    <h3 style="color: #2c3e50;">{audio_filename}</h3>
                    <div style="display: flex; flex-direction: row;">
                        <div style="flex: 1; padding: 10px; background-color: #f8f9fa; margin-right: 5px; border-radius: 3px;">
                            <h4 style="color: #3498db;">Original Transcript</h4>
                            <p style="white-space: pre-wrap;">{original_transcript}</p>
                        </div>
                        <div style="flex: 1; padding: 10px; background-color: #f8f9fa; margin-left: 5px; border-radius: 3px;">
                            <h4 style="color: #e74c3c;">Model Transcript</h4>
                            <p style="white-space: pre-wrap;">{model_transcript}</p>
                        </div>
                    </div>
                </div>
                """
                display(HTML(html_output))
            except:
                # Not running in an environment that supports HTML display, that's okay
                pass
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    print("\nInference and comparison completed!")

if __name__ == "__main__":
    main()