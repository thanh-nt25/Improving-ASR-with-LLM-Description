# Add this at the beginning of your whisper_fine.py file
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch.serialization
torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])

from datasets import Audio
import torch
from transformers_prompt import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperPromptForConditionalGeneration, GenerationConfig, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers.trainer_callback import TrainerCallback
from utils_prompt import compute_wer, compute_wer_ocw, DataCollatorSpeechS2SWhitPadding
from data.dataloader import PromptWhisperDataset
import os
import json
from huggingface_hub import login, HfApi
import argparse
torch.manual_seed(1004)
torch.cuda.manual_seed_all(1004)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # try:
    #     torch.multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     # Context đã được thiết lập, bỏ qua
    #     pass
    parser = argparse.ArgumentParser(description='whisper prompt tuning')

    parser.add_argument('--exp-name', type=str, default="test", help="path to save result")
    parser.add_argument('--model', type=str, default="base.en", help="path to save result")
    parser.add_argument('--batch', type=int, default=2, help="batch size")
    parser.add_argument('--epoch', type=int, default=10, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--prompt', action='store_true', help="whether to use prompt to decoder")
    parser.add_argument('--dataset', type=str, default="ocw", help="path to save result")
    parser.add_argument('--freeze', action='store_true', help="whether to freeze whisper")
    parser.add_argument('--eval', action='store_true', help="only evaluation")
    
    parser.add_argument('--random', action='store_true', help="context perturbation")
    parser.add_argument('--basic', action='store_true', help="collected description")
    
    # Add arguments for Hugging Face integration and checkpointing
    parser.add_argument('--save-hf', action='store_true', help="Save model to Hugging Face Hub")
    parser.add_argument('--hf-repo', type=str, default=None, help="Hugging Face repository name (e.g., username/repo-name)")
    parser.add_argument('--hf-token', type=str, default=None, help="Hugging Face API token")
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")
    parser.add_argument('--checkpoint-path', type=str, default=None, help="Path or HF repo to resume training from")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Device:", device)
    args.prompt = True
    
    # Login to Hugging Face if saving to Hub
    if args.save_hf and (args.hf_token or args.hf_repo):
        if args.hf_token:
            login(token=args.hf_token)
        else:
            print("Please set HUGGING_FACE_HUB_TOKEN environment variable or use --hf-token")
            login()
        
        if not args.hf_repo:
            args.hf_repo = f"{os.environ.get('HUGGINGFACE_USERNAME', 'user')}/whisper-{args.model}-{args.dataset}"
            print(f"No repository name specified, using: {args.hf_repo}")
    print(f"save_hf: {args.save_hf}, hf_repo: {args.hf_repo}")
    
    # prepare feature extractor, tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f'openai/whisper-{args.model}')
    tokenizer = WhisperTokenizer.from_pretrained(f'openai/whisper-{args.model}', language='en', task='transcribe')
    processor = WhisperProcessor.from_pretrained(f'openai/whisper-{args.model}', language='en', task='transcribe')

    # data collator  
    data_collator = DataCollatorSpeechS2SWhitPadding(processor=processor)
    
    # Đường dẫn đến dữ liệu trên Kaggle
    data_root = "/kaggle/input/ocw-biasing"
    
    if args.dataset == 'earning':
        data_train = PromptWhisperDataset(base_path=os.path.join(data_root,"Earnings_Call/"), phase='train', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, random=args.random)
        data_eval = PromptWhisperDataset(base_path=os.path.join(data_root,"Earnings_Call/"), phase='dev', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
        data_test = PromptWhisperDataset(base_path=os.path.join(data_root,"Earnings_Call/"), phase='test', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
    elif args.dataset == 'ocw':
        data_train = PromptWhisperDataset(base_path=os.path.join(data_root,"OCW/"), phase='train', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic, random=args.random)
        data_eval = PromptWhisperDataset(base_path=os.path.join(data_root,"OCW/"), phase='dev', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
        data_test = PromptWhisperDataset(base_path=os.path.join(data_root,"OCW/"), phase='test', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
    else:
        raise ValueError("Wrong dataset")

    # load model - Check if resuming from checkpoint
    checkpoint_path = None
    # Thay đổi phần load checkpoint
    if args.resume:
        if args.checkpoint_path:
            try:
                # Thử load từ đường dẫn cụ thể 
                model = WhisperPromptForConditionalGeneration.from_pretrained(
                    args.checkpoint_path, 
                    local_files_only=False  # Cho phép tải từ internet
                )
                print(f"Successfully loaded model from {args.checkpoint_path}")
            except Exception as e:
                print(f"Error loading specific checkpoint: {e}")
                print("Falling back to the main repository")
                
                # Nếu load checkpoint riêng lẻ thất bại, thử load từ repository chính
                try:
                    model = WhisperPromptForConditionalGeneration.from_pretrained(
                        f"thanh-nt25/whisper-earning", 
                        local_files_only=False
                    )
                except Exception as e:
                    print(f"Error loading main repository: {e}")
                    print("Falling back to original model")
                    model = WhisperPromptForConditionalGeneration.from_pretrained(f'openai/whisper-{args.model}')
    
    if args.prompt:
        if args.eval and args.checkpoint_path:
            # For evaluation with specified checkpoint
            model = WhisperPromptForConditionalGeneration.from_pretrained(args.checkpoint_path)
            print(f"Model loaded from {args.checkpoint_path} for evaluation!")
        elif checkpoint_path and args.resume:
            # For resuming training
            try:
                model = WhisperPromptForConditionalGeneration.from_pretrained(checkpoint_path)
                print(f"Successfully loaded model from checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Falling back to original model")
                model = WhisperPromptForConditionalGeneration.from_pretrained(f'openai/whisper-{args.model}')
        else:
            # For initial training
            model = WhisperPromptForConditionalGeneration.from_pretrained(f'openai/whisper-{args.model}')
            
        # Freeze all parameters
        for name, param in model._named_members(lambda module: module._parameters.items()):
            if args.freeze: 
                param.requires_grad = False
            else:
                param.requires_grad = True

        for name, module in model.named_modules():
            if 'decoder' in name:
                for param in module.parameters():
                    param.requires_grad = True
    else:
        print("Prompt must be used.")
        raise(ValueError)
    
    model.to(device)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # Thư mục đầu ra trên Kaggle
    # root_path = "/kaggle/working"
    # if args.dataset == 'earning':
    #     output_dir = os.path.join(root_path, "results", args.exp_name)
    # else:
    #     output_dir = os.path.join(root_path, "results_ocw", args.exp_name)
    
    # os.makedirs(output_dir, exist_ok=True)

    iteration_steps = int(len(data_train) * args.epoch // args.batch)

    eval_step = int((len(data_train) // 2) // args.batch)
    log_step = int((len(data_train) // 50) // args.batch)

    print("Train data len:", len(data_train))
    print("Eval data len:", len(data_eval))
    print("Test data len:", len(data_test))

    print("Max steps:", iteration_steps)
    print("eval step:", eval_step)
    print("log step:", log_step)
    
    generation_config = GenerationConfig(
        pos_token_id=50360
    )
    
    # Configure HF Hub settings based on arguments
    hub_strategy = "every_save" if args.save_hf else None
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.hf_repo}/checkpoint" if args.save_hf else None,  
        hub_model_id=args.hf_repo if args.save_hf else None,
        hub_strategy=hub_strategy,
        push_to_hub=args.save_hf,
        save_strategy="steps", # eval steps
        weight_decay=0.01,
        dataloader_num_workers=1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1,  
        learning_rate=args.lr,
        warmup_steps=100,
        max_steps=iteration_steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=256, # Điều chỉnh batch size nếu cần
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=eval_step,
        eval_steps=eval_step,
        logging_steps=log_step,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        dataloader_pin_memory=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        model=args.model,
        remove_unused_columns=False,
        pos_token_id=tokenizer.convert_tokens_to_ids("<|startofprev|>")
    )
    print(f"hub_model_id: {training_args.hub_model_id}")
    print(f"hub_strategy: {training_args.hub_strategy}")
    print(f"push_to_hub: {training_args.push_to_hub}")
    print(f"output_dir: {training_args.output_dir}")

    if args.dataset == 'earning':
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=data_train,
            eval_dataset=data_eval,
            data_collator=data_collator,
            compute_metrics=compute_wer,
            tokenizer=processor.feature_extractor,
        )
    else:
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=data_train,
            eval_dataset=data_eval,
            data_collator=data_collator,
            compute_metrics=compute_wer_ocw,
            tokenizer=processor.feature_extractor,
        )

    if not args.eval:
        print("Start Training!")
        
        # Resume from checkpoint if specified
        resume_from_checkpoint = checkpoint_path if args.resume else None
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        
        # Push to hub if requested
        if args.save_hf and args.hf_repo:
            print(f"Pushing final model to Hugging Face Hub: {args.hf_repo}")
            trainer.push_to_hub()

    print("Start Evaluation!!")
    if args.prompt:
        print("Using prompt")
    
    # Run evaluation on test set
    test_results = trainer.evaluate(data_test)
    print(test_results)
    
    # Save test results locally
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as t:
        t.write(str(test_results))
    
    # Save test results in JSON format for easier parsing
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # If saving to HuggingFace Hub, add the test metrics
    if args.save_hf and args.hf_repo:
        try:
            # Create a model card with test metrics or update existing one
            model_card_content = f"""---
language: en
license: apache-2.0
datasets:
- {args.dataset}
metrics:
- wer
model-index:
- name: whisper-{args.model}-{args.dataset}
  results:
  - task: 
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: {args.dataset} test
      type: {args.dataset}
    metrics:
    - name: WER
      type: wer
      value: {test_results.get('wer', 'N/A')}
tags:
- whisper
- asr
- speech
- audio
- {args.dataset}
---

# Whisper Fine-tuned Model

This model is a fine-tuned version of [`openai/whisper-{args.model}`] on the {args.dataset} dataset.

## Test Results

- eval_loss: {test_results.get('eval_loss', 'N/A')}
- eval_wer: {test_results.get('wer', 'N/A')}
- eval_runtime: {test_results.get('eval_runtime', 'N/A')}
- eval_samples_per_second: {test_results.get('eval_samples_per_second', 'N/A')}

## Training Parameters

- Model: whisper-{args.model}
- Dataset: {args.dataset}
- Prompt: {args.prompt}
- Learning Rate: {args.lr}
- Batch Size: {args.batch}
- Epochs: {args.epoch}
- Frozen: {args.freeze}
"""

            # Path to save the README.md file locally
            readme_path = os.path.join(output_dir, "README.md")
            with open(readme_path, "w") as f:
                f.write(model_card_content)
            
            # Create a metadata file with the test metrics
            metadata = {
                "test_metrics": {
                    "eval_loss": float(test_results.get('eval_loss', 0)),
                    "eval_wer": float(test_results.get('wer', 0)),
                    "eval_runtime": float(test_results.get('eval_runtime', 0)),
                    "eval_samples_per_second": float(test_results.get('eval_samples_per_second', 0))
                },
                "training_params": {
                    "model": args.model,
                    "dataset": args.dataset,
                    "prompt": args.prompt,
                    "batch_size": args.batch,
                    "learning_rate": args.lr,
                    "epochs": args.epoch,
                    "frozen": args.freeze
                }
            }
            
            metadata_path = os.path.join(output_dir, "test_metrics.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
                
            # Push these files to the Hub
            if args.save_hf and args.hf_repo:
                print("Pushing test metrics and README to Hugging Face Hub...")
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=args.hf_repo,
                    repo_type="model"
                )
                api.upload_file(
                    path_or_fileobj=metadata_path,
                    path_in_repo="test_metrics.json",
                    repo_id=args.hf_repo,
                    repo_type="model"
                )
                print("Test metrics successfully uploaded to Hugging Face Hub!")
        except Exception as e:
            print(f"Error uploading test metrics to Hugging Face Hub: {e}")