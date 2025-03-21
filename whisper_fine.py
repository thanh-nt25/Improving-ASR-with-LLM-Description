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
from huggingface_hub import login
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
    
    # data_root = "/data/jwsuh/whisper-datasets/main"
    data_root = "/content/drive/MyDrive/Thesis/Improving-ASR-with-LLM-Description"
    
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
    if args.resume:
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
            print(f"Resuming from specified checkpoint: {checkpoint_path}")
        elif args.save_hf and args.hf_repo:
            checkpoint_path = args.hf_repo
            print(f"Resuming from HF Hub checkpoint: {checkpoint_path}")
        else:
            # Look for local checkpoint
            root_path = "/content/drive/MyDrive/Thesis/Improving-ASR-with-LLM-Description/"
            if args.dataset == 'earning':
              checkpoint_dir = os.path.join(root_path, "results", args.exp_name)
            else:
              checkpoint_dir = os.path.join(root_path, "results_ocw", args.exp_name)

            if os.path.exists(checkpoint_dir):
                checkpoint_path = checkpoint_dir
                print(f"Resuming from local checkpoint: {checkpoint_path}")
            else:
                print("No checkpoint found. Starting from scratch.")
    
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
    
    #root_path = "results/"
    root_path = "/content/drive/MyDrive/Thesis/Improving-ASR-with-LLM-Description/"
    if args.dataset == 'earning':
      os.makedirs(os.path.join(root_path, "results", args.exp_name), exist_ok=True)
    else:
      os.makedirs(os.path.join(root_path, "results_ocw", args.exp_name), exist_ok=True)

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
        weight_decay=0.01,
        output_dir=os.path.join(root_path, "results", args.exp_name),
        hub_model_id=args.hf_repo if args.save_hf else None,
        hub_strategy=hub_strategy,
        push_to_hub=True,
        dataloader_num_workers=1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=args.lr,
        warmup_steps=100,
        max_steps=iteration_steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=256,
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

    # class CustomProgressCallback(TrainerCallback):
    #   def __init__(self):
    #       self.prediction_bar = None
    #       self.last_updated_step = 0
          
    #   def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
    #       if not hasattr(eval_dataloader, "__len__"):
    #           return
              
    #       total_steps = len(eval_dataloader)
    #       # Chỉ hiển thị 10 cập nhật trong toàn bộ quá trình
    #       display_interval = max(1, total_steps // 10)
          
    #       current_step = self.last_updated_step + 1
    #       self.last_updated_step = current_step
          
    #       # Chỉ hiển thị tại một số điểm nhất định
    #       if current_step == 1 or current_step == total_steps or current_step % display_interval == 0:
    #           print(f"\rEvaluation: {current_step}/{total_steps} ({(current_step/total_steps*100):.1f}%)", end="")
            
    #   def on_evaluate(self, args, state, control, **kwargs):
    #       print("\nEvaluation completed.")
    #       self.last_updated_step = 0


    if args.dataset == 'earning':
      trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data_train,
        eval_dataset=data_eval,
        data_collator=data_collator,
        compute_metrics=compute_wer,
        tokenizer=processor.feature_extractor,
        # callbacks=[CustomProgressCallback()]
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
          # callbacks=[CustomProgressCallback()]
      )

    if not args.eval:
        print("Start Training!")
        
        # Resume from checkpoint if specified
        resume_from_checkpoint = checkpoint_path if args.resume else None
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        
        # Push to hub if requested
        if args.save_hf and args.hf_repo:
            print(f"Pushing final model to Hugging Face Hub: {args.hf_repo}")
            trainer.push_to_hub()

    print("Start Evaluation!!")
    if args.prompt:
        print("Using prompt")
    result = trainer.evaluate(data_test)
    print(result)
    
    # print results
    with open(os.path.join(root_path, "results", args.exp_name, 'result.txt'), 'w') as t:
        t.write(str(result))