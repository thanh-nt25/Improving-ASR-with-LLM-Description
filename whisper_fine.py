# Add this at the beginning of your whisper_fine.py file
import warnings
warnings.filterwarnings("ignore")

import os
import re
import traceback
import numpy as np
import torch.serialization
torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])

from datasets import Audio
import torch
from transformers import TrainerCallback
from huggingface_hub import HfApi, hf_hub_download
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

    parser.add_argument('--exp-name', type=str, default="", help="path to save result")
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

    # Thư mục đầu ra trên Kaggle
    root_path = "/kaggle/working"
    
    output_dir = os.path.join(root_path, "results", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    def download_latest_checkpoint(repo_id):
        """
        Tải checkpoint mới nhất từ Hugging Face về local
        """
        try:
            api = HfApi()

            # Lấy toàn bộ file trong repository
            all_files = api.list_repo_files(repo_id)

            # Lọc ra các thư mục checkpoint chứa file quan trọng
            checkpoints = [
                f for f in all_files 
                if re.search(r'checkpoints/checkpoint-\d+/model\.safetensors$', f)
            ]
            
            if not checkpoints:
                raise ValueError(f"Không tìm thấy checkpoint nào trong repository {repo_id}")
            
            # Sắp xếp checkpoint theo số thứ tự giảm dần
            sorted_checkpoints = sorted(
                checkpoints, 
                key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)), 
                reverse=True
            )
            
            # Lấy checkpoint mới nhất
            latest_checkpoint = sorted_checkpoints[0]
            print(f"Latest checkpoint: {latest_checkpoint}")
            
            # Lấy thư mục checkpoint
            checkpoint_folder = '/'.join(latest_checkpoint.split('/')[:-1])
            print(f"Checkpoint folder: {checkpoint_folder}")
          
            # Tạo thư mục local để lưu checkpoint - loại bỏ hoàn toàn phần checkpoints/
            local_checkpoint_dir = os.path.join(root_path, "results/")
            print(f"Local checkpoint dir: {local_checkpoint_dir}")
            os.makedirs(local_checkpoint_dir, exist_ok=True)
            
            # Các file cần tải
            files_to_download = [
                'config.json', 
                'generation_config.json', 
                'model.safetensors',
                'training_args.bin',
                'optimizer.pt',
                'preprocessor_config.json',
                'rng_state.pth',
                'scheduler.pt',
                'trainer_state.json'
            ]
            
            # Tải từng file
            for file in files_to_download:
                try:
                    full_file_path = os.path.join(checkpoint_folder, file)
                    hf_hub_download(
                        repo_id=repo_id, 
                        filename=full_file_path,
                        local_dir=local_checkpoint_dir,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    print(f"Không thể tải file {file}: {e}")
            
            print(f"Đã tải checkpoint mới nhất: {checkpoint_folder}")
            local_path_downloaded = os.path.join(local_checkpoint_dir, checkpoint_folder)
            print(f"Local path downloaded: {local_path_downloaded}")
            return local_path_downloaded
        
        except Exception as e:
            print(f"Lỗi khi tải checkpoint: {e}")
            traceback.print_exc()
            raise

    def download_specific_checkpoint(checkpoint_name):
        """
        Tải checkpoint cụ thể từ Hugging Face về local
        """
        try:
            repo_id = "thanh-nt25/whisper-earning"
            api = HfApi()
            
            all_files = api.list_repo_files(repo_id)
            
            checkpoint_path = f"checkpoints/{checkpoint_name}"
            if not any(checkpoint_path in f for f in all_files):
                raise ValueError(f"Không tìm thấy checkpoint {checkpoint_name}")
            
            # local_checkpoint_dir = os.path.join(root_path, "results", "checkpoints", checkpoint_name)
            local_checkpoint_dir = os.path.join(root_path, "results")
            os.makedirs(local_checkpoint_dir, exist_ok=True)
            
            files_to_download = [
                'config.json', 
                'generation_config.json', 
                'model.safetensors',
                'training_args.bin',
                'optimizer.pt',
                'preprocessor_config.json',
                'rng_state.pth',
                'scheduler.pt',
                'trainer_state.json'
            ]
            
            for file in files_to_download:
                try:
                    full_file_path = os.path.join(f"checkpoints/{checkpoint_name}", file)
                    hf_hub_download(
                        repo_id=repo_id, 
                        filename=full_file_path,
                        local_dir=local_checkpoint_dir,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    print(f"Không thể tải file {file}: {e}")
            
            print(f"Đã tải checkpoint cụ thể: {checkpoint_name}")
            # print(f"Local checkpoint dir: {local_checkpoint_dir}")
            local_checkpoint_dir = os.path.join(local_checkpoint_dir, f"checkpoints/{checkpoint_name}")
            return local_checkpoint_dir # full path
        
        except Exception as e:
            print(f"Lỗi khi tải checkpoint cụ thể: {e}")
            traceback.print_exc()
            raise
        
    checkpoint_dir = None     

    if args.resume:
        try:
            if not args.checkpoint_path:
                checkpoint_dir = download_latest_checkpoint("thanh-nt25/whisper-earning")
            else:
                checkpoint_dir = download_specific_checkpoint(args.checkpoint_path)
                print(f"checkpoint_dir of case have checkpoint_path {checkpoint_dir}")
            
            model = WhisperPromptForConditionalGeneration.from_pretrained(
                checkpoint_dir, 
                local_files_only=True
            )
            print(f"Loaded checkpoint from {checkpoint_dir}")
        
        except Exception as e:
            print(f"Checkpoint loading failed: {e}")
            traceback.print_exc()
            raise
    
    if args.prompt:
        if args.eval and args.checkpoint_path:
            # For evaluation with specified checkpoint
            model = WhisperPromptForConditionalGeneration.from_pretrained(args.checkpoint_path)
            print(f"Model loaded from {args.checkpoint_path} for evaluation!")
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
    
    model.to(device)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    

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
        output_dir = output_dir,  
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
        per_device_eval_batch_size=32, # Điều chỉnh batch size nếu cần
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

    class HuggingFaceHubCallback(TrainerCallback):
        def __init__(self, hub_repo):
            self.hub_repo = hub_repo
            self.api = HfApi()
            self.uploaded_checkpoints = set()  # Theo dõi các checkpoint đã upload

        def on_save(self, args, state, control, **kwargs):
            # Tìm tất cả các checkpoint trong thư mục output
            checkpoints = [
                d for d in os.listdir(args.output_dir) 
                if d.startswith('checkpoint-')
            ]
            
            if not checkpoints:
                print("No checkpoints found to push")
                return

            try:
                # Upload các checkpoint vào thư mục checkpoints/
                for checkpoint in checkpoints:
                    # Chỉ upload checkpoint nếu chưa được upload trước đó
                    if checkpoint not in self.uploaded_checkpoints:
                        checkpoint_path = os.path.join(args.output_dir, checkpoint)
                        self.api.upload_folder(
                            folder_path=checkpoint_path,
                            path_in_repo=f"checkpoints/{checkpoint}",
                            repo_id=self.hub_repo,
                            repo_type="model"
                        )
                        self.uploaded_checkpoints.add(checkpoint)
                        print(f"Pushed {checkpoint} to {self.hub_repo}/checkpoints/")
                
                # Các file thường được push_to_hub=True tự động xử lý
                standard_files = [
                    "config.json", "pytorch_model.bin", "training_args.bin", 
                    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                    "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json",
                    "model.safetensors", "optimizer.pt", "scheduler.pt", "scaler.pt",
                    "trainer_state.json", "README.md"
                ]
                
                # Upload các file không phải là file tiêu chuẩn hoặc checkpoint
                for item in os.listdir(args.output_dir):
                    item_path = os.path.join(args.output_dir, item)
                    # Chỉ upload file không thuộc danh sách tiêu chuẩn và không phải thư mục checkpoint
                    if os.path.isfile(item_path) and not item.startswith('.') and item not in standard_files:
                        try:
                            self.api.upload_file(
                                path_or_fileobj=item_path,
                                path_in_repo=item,
                                repo_id=self.hub_repo,
                                repo_type="model"
                            )
                            print(f"Pushed additional file {item} to root")
                        except Exception as e:
                            print(f"Error uploading file {item}: {e}")
                
                print(f"Successfully pushed checkpoint folders and non-standard files")
            except Exception as e:
                print(f"Error in main push operation: {e}")

    # Trong phần khởi tạo Trainer
    hf_hub_callback = HuggingFaceHubCallback(hub_repo=args.hf_repo) if args.save_hf else None


    if args.dataset == 'earning':
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=data_train,
            eval_dataset=data_eval,
            data_collator=data_collator,
            compute_metrics=compute_wer,
            tokenizer=processor.feature_extractor,
            callbacks=[hf_hub_callback] if hf_hub_callback else None
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
            callbacks=[hf_hub_callback] if hf_hub_callback else None
        )

    if not args.eval:
        print("Start Training!")
        
        if args.checkpoint_path and args.resume:
            print(f"Resuming training from specificy checkpoint: {args.checkpoint_path}")
            print(f"Checkpont dir: {checkpoint_dir}")
        elif args.resume:
            print(f"Resuming training from checkpoint: {checkpoint_dir}")
      
        trainer.train(resume_from_checkpoint=checkpoint_dir)        
        
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
    result = trainer.evaluate(data_test)
    print(result)

    with open(os.path.join(root_path, "results", args.exp_name, 'result.txt'), 'w') as t:
        t.write(str(result))
    
    