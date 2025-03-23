import os
import argparse
import re
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Process audio files by groups and show transcripts")
    parser.add_argument("--test_folder", type=str, required=True, 
                        help="Main test folder containing audio and JSON files")
    parser.add_argument("--model_id", type=str, default="thanh-nt25/whisper-earning", 
                        help="Hugging Face model ID or local path")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to run on (cuda:0, cpu, etc.)")
    
    args = parser.parse_args()
    test_folder = args.test_folder
    
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
        batch_size=8,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # First, collect all files grouped by directory and prefix
    grouped_files = {}
    
    for root, dirs, files in os.walk(test_folder):
        # Skip if no MP3 files
        mp3_files = [f for f in files if f.endswith('.mp3')]
        if not mp3_files:
            continue
        
        folder_name = os.path.basename(root)
        grouped_files[root] = defaultdict(list)
        
        # Group files by prefix
        for file in files:
            if file.endswith('.mp3') or file.endswith('.json'):
                # Extract prefix from filenames like "Michael J. Hartshorn_1_1.mp3"
                match = re.match(r'(.+?)_(\d+)_(\d+)\.(mp3|json)$', file)
                if match:
                    person_name = match.group(1)
                    prefix = match.group(2)
                    number = int(match.group(3))
                    extension = match.group(4)
                    
                    key = f"{person_name}_{prefix}"
                    grouped_files[root][key].append({
                        'file': file,
                        'path': os.path.join(root, file),
                        'number': number,
                        'extension': extension
                    })
    
    # Now process each group
    for folder_path, prefix_groups in grouped_files.items():
        folder_name = os.path.basename(folder_path)
        print(f"\n{'='*100}")
        print(f"Processing folder: {folder_name}")
        print(f"{'='*100}")
        
        for prefix, files in prefix_groups.items():
            # Sort files by number
            sorted_files = sorted(files, key=lambda x: x['number'])
            
            # Organize into pairs of MP3 and JSON
            file_pairs = []
            mp3_dict = {}
            json_dict = {}
            
            for file_info in sorted_files:
                if file_info['extension'] == 'mp3':
                    mp3_dict[file_info['number']] = file_info
                elif file_info['extension'] == 'json':
                    json_dict[file_info['number']] = file_info
            
            numbers = sorted(set(mp3_dict.keys()) | set(json_dict.keys()))
            
            print(f"\n{'-'*100}")
            print(f"Group: {prefix}")
            print(f"{'-'*100}")
            
            for number in numbers:
                mp3_info = mp3_dict.get(number)
                json_info = json_dict.get(number)
                
                if mp3_info:
                    print(f"\nFile: {mp3_info['file']}")
                    print(f"{'-'*40}")
                    
                    # Get original transcript
                    original_transcript = "No transcript found"
                    if json_info:
                        try:
                            with open(json_info['path'], 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                                original_transcript = json_data.get("text", "No transcript found in JSON")
                        except Exception as e:
                            print(f"Error reading JSON: {str(e)}")
                    
                    # Run model inference
                    try:
                        result = pipe(mp3_info['path'])
                        model_transcript = result["text"]
                        
                        print(f"ORIGINAL TRANSCRIPT:")
                        print(f"{'-'*20}")
                        print(f"{original_transcript}")
                        
                        print(f"\nMODEL TRANSCRIPT:")
                        print(f"{'-'*20}")
                        print(f"{model_transcript}")
                        
                    except Exception as e:
                        print(f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    main()