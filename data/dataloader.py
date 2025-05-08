import numpy as np
import os

import torch
import torchaudio.transforms as at
import torchaudio
import editdistance
import av

import json
import random


class calc_metrics:
    def __init__(self):
        pass

    def __call__(self, refs, preds):
        """
        refs are output from dataloader, so uses the collate fn, that already contains the normalization
        preds are the output of whisper tokenizer, which doesn't have dataset specific normalization

        they should both in list (list of list)
        """
        distance = 0
        tokens = 0
        wer_list = []
        processed_preds = []
        processed_refs = []
        exclude = [",", "?", ".", "!", ";"]
        for ref, pred in zip(refs, preds):
            pred = pred.lower()
            pred = "".join(ch for ch in pred if ch not in exclude)
            processed_preds.append(pred)
            processed_refs.append(ref)  # do not process ref
            cur_dist = editdistance.distance(pred.split(" "), ref.split(" "))
            cur_tokens = len(ref.split(" "))
            wer_list.append(cur_dist / cur_tokens)
            distance += cur_dist
            tokens += cur_tokens

        return {"wer": distance / tokens}, (wer_list, processed_preds, processed_refs)


def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    with av.open(wave_path, metadata_errors="ignore") as container:
        decode = container.decode(audio=0)
        aframes_list = [frame.to_ndarray() for frame in decode]
        aframes = np.concatenate(aframes_list, 1)
        # Convert to float32 for processing. We normalize by dividing by 32768.0 (2^15) to get range [-1, 1]
        wav = torch.from_numpy(aframes).float() / 32768.0
        wav = wav.mean(dim=0)  # Taking the mean to convert from stereo to mono
        cur_sample_rate = container.streams.audio[0].rate
        if cur_sample_rate != sample_rate:
            resampler = at.Resample(orig_freq=cur_sample_rate, new_freq=sample_rate)
            wav = resampler(wav)
        if wav.mean() == 0:
            print(wave_path, "empty!")
    return wav


class PromptWhisperDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, phase, feature_extractor, tokenizer, prompt=False, audio_type=".wav", sample_rate=16000, random=False, basic=False):
        super().__init__()
        self.phase = phase
        self.base_path = base_path
        self.sample_rate = sample_rate
        self.prompt = prompt
        self.random_prompt = random
        self.data = []
        self.prompt_pool = []
        self.audio_type = audio_type
        self.basic = basic
        self._load_data()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def _initialize_prompt_pool(self):
        # Walk through the directory structure to build the prompt pool
        for root, dirs, files in os.walk(os.path.join(self.base_path, self.phase)):
            json_files = [f for f in files if f.endswith('.json')]
            for json_file_name in json_files:
                json_file_path = os.path.join(root, json_file_name)
                with open(json_file_path, 'r', encoding='utf-8') as json_file:
                    json_data = json.load(json_file)
                    prompt = json_data.get("prompt", "")
                    if prompt:
                        self.prompt_pool.append(prompt)


    def _load_data(self):
        idx = 1
        # Walk through the directory structure
        for root, dirs, files in os.walk(os.path.join(self.base_path, self.phase)):
            print("Idx sample data: ", idx)
            idx = idx + 1
            wav_files = [f for f in files if f.endswith(f'{self.audio_type}')]
            json_files = [f for f in files if f.endswith('.json')]
            for wav_file in wav_files:
                base_name = os.path.splitext(wav_file)[0]
                json_file_name = f"{base_name}.json"
                if json_file_name in json_files:
                    json_file_path = os.path.join(root, json_file_name)
                    # Open the json file and extract "text" and "prompt"
                    with open(json_file_path, 'r', encoding='utf-8') as json_file:
                        json_data = json.load(json_file)
                        text = json_data.get("text", "")
                        prompt = json_data.get("prompt", "")
                        random_prompt = random.choice(self.prompt_pool) if self.prompt_pool else ""
                        basic = json_data.get("basic", "")
                    self.data.append([os.path.join(root, wav_file),
                        prompt,
                        random_prompt,
                        basic,
                        text
                    ])

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, id):
    #     audio_path, prompt, random_prompt, basic_prompt, raw_text = self.data[id]
    #     # Load and process audio
    #     audio, _ = torchaudio.load(audio_path)
    #     audio = audio.squeeze().numpy()  # Converting to NumPy array if not already
    #     processed_audio = self.feature_extractor(audio, sampling_rate=self.sample_rate).input_features
    #     processed_audio = torch.tensor(processed_audio[0])  # Ensure processed_audio is a tensor
    #     # Encode text
    #     encoded_labels = torch.tensor(self.tokenizer.encode(raw_text.lower()))  # Convert to tensor

    #     if self.prompt:
    #         if self.random_prompt and 'train' in self.phase:
    #             if torch.rand([]) < 0.05 and 'train' in self.phase:
    #                 encoded_prompt = self.tokenizer.encode(random_prompt.lower(), add_special_tokens=False)
    #             else:
    #                 encoded_prompt = self.tokenizer.encode(prompt.lower(), add_special_tokens=False)
    #         elif self.basic:
    #             encoded_prompt = self.tokenizer.encode(basic_prompt.lower(), add_special_tokens=False)
    #         else:
    #             encoded_prompt = self.tokenizer.encode(prompt.lower(), add_special_tokens=False)

    #         if len(encoded_prompt) > 190:
    #             encoded_prompt = encoded_prompt[:190]

    #         encoded_prompt = torch.tensor(encoded_prompt)  # Ensure encoded_prompt is a tensor
    #         return {
    #             "input_features": processed_audio,
    #             "prompt": encoded_prompt,  # Including the prompt in the output
        #         "labels": encoded_labels
        #     }
        # else:
        #     print("prompt must be used.")
        #     raise(ValueError)
    def __getitem__(self, id):
      audio_path, prompt, random_prompt, basic_prompt, raw_text = self.data[id]
      try:
          # Load and process audio
          audio, _ = torchaudio.load(audio_path)
          audio = audio.squeeze().numpy()  # Converting to NumPy array if not already
          processed_audio = self.feature_extractor(audio, sampling_rate=self.sample_rate).input_features
          processed_audio = torch.tensor(processed_audio[0])  # Ensure processed_audio is a tensor

          # Encode text
          encoded_labels = torch.tensor(self.tokenizer.encode(raw_text.lower()))  # Convert to tensor

          if self.prompt:
              if self.random_prompt and 'train' in self.phase:
                  if torch.rand([]) < 0.05 and 'train' in self.phase:
                      encoded_prompt = self.tokenizer.encode(random_prompt.lower(), add_special_tokens=False)
                  else:
                      encoded_prompt = self.tokenizer.encode(prompt.lower(), add_special_tokens=False)
              elif self.basic:
                  encoded_prompt = self.tokenizer.encode(basic_prompt.lower(), add_special_tokens=False)
              else:
                  encoded_prompt = self.tokenizer.encode(prompt.lower(), add_special_tokens=False)

              if len(encoded_prompt) > 190:
                  encoded_prompt = encoded_prompt[:190]

              encoded_prompt = torch.tensor(encoded_prompt)  # Ensure encoded_prompt is a tensor

              # Check that all values are valid tensors
              if processed_audio is None or encoded_prompt is None or encoded_labels is None:
                  raise ValueError("One of the required tensors is None")

              return {
                  "input_features": processed_audio,
                  "prompt": encoded_prompt,  # Including the prompt in the output
                  "labels": encoded_labels
              }
          else:
              print("prompt must be used.")
              raise(ValueError)
      except Exception as e:
          print(f"Error processing sample {id}, file: {audio_path}, error: {str(e)}")
          # Return a default sample or skip this sample
          # For debugging, it's better to raise the exception first to see what's happening
          raise e


# new PromptWhisperDataset for bold kind of dataset
# class PromptWhisperDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         base_path,
#         phase,
#         feature_extractor,
#         tokenizer,
#         prompt=False,
#         audio_type=".wav",
#         sample_rate=16000,
#         random=False,
#         basic=False,
#         hf_data=None,
#     ):
#         super().__init__()
#         self.phase = phase
#         self.base_path = base_path
#         self.sample_rate = sample_rate
#         self.prompt = prompt
#         self.random_prompt = random
#         self.audio_type = audio_type
#         self.basic = basic
#         self.feature_extractor = feature_extractor
#         self.tokenizer = tokenizer
#         self.prompt_pool = []

#         # NEW: support both HF-style data and local file-based
#         if hf_data is not None:
#             self.data = hf_data  # list of dicts with 'audio', 'text', etc.
#             self.use_hf_format = True
#         else:
#             self.data = []
#             self.use_hf_format = False
#             self._load_data()

#     def _load_data(self):
#         idx = 1
#         for root, dirs, files in os.walk(os.path.join(self.base_path, self.phase)):
#             print("Idx sample data: ", idx)
#             idx += 1
#             wav_files = [f for f in files if f.endswith(self.audio_type)]
#             json_files = [f for f in files if f.endswith(".json")]
#             for wav_file in wav_files:
#                 base_name = os.path.splitext(wav_file)[0]
#                 json_file_name = f"{base_name}.json"
#                 if json_file_name in json_files:
#                     json_path = os.path.join(root, json_file_name)
#                     with open(json_path, "r", encoding="utf-8") as jf:
#                         j = json.load(jf)
#                     self.data.append(
#                         {
#                             "audio_path": os.path.join(root, wav_file),
#                             "prompt": j.get("prompt", ""),
#                             "random_prompt": "",  # can be set later
#                             "basic": j.get("basic", ""),
#                             "text": j.get("text", ""),
#                         }
#                     )

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         try:
#             item = self.data[idx]

#             if self.use_hf_format:
#                 # Use in-memory audio from HuggingFace
#                 audio_array = item["audio"]["array"]
#                 sampling_rate = item["audio"]["sampling_rate"]
#                 text = item["text"]
#                 prompt = item.get("prompt", "")
#                 basic_prompt = item.get("basic", "")
#                 random_prompt = ""
#             else:
#                 # Use file-based audio
#                 audio_array, _ = torchaudio.load(item["audio_path"])
#                 audio_array = audio_array.squeeze().numpy()
#                 sampling_rate = self.sample_rate
#                 text = item["text"]
#                 prompt = item.get("prompt", "")
#                 basic_prompt = item.get("basic", "")
#                 random_prompt = item.get("random_prompt", "")

#             # Feature extraction
#             input_features = self.feature_extractor(
#                 audio_array, sampling_rate=sampling_rate
#             ).input_features
#             input_features = torch.tensor(input_features[0])

#             # Label encoding
#             labels = torch.tensor(self.tokenizer.encode(text.lower()))

#             if self.prompt:
#                 if self.random_prompt and "train" in self.phase:
#                     if torch.rand([]) < 0.05:
#                         encoded_prompt = self.tokenizer.encode(
#                             random_prompt.lower(), add_special_tokens=False
#                         )
#                     else:
#                         encoded_prompt = self.tokenizer.encode(
#                             prompt.lower(), add_special_tokens=False
#                         )
#                 elif self.basic:
#                     encoded_prompt = self.tokenizer.encode(
#                         basic_prompt.lower(), add_special_tokens=False
#                     )
#                 else:
#                     encoded_prompt = self.tokenizer.encode(
#                         prompt.lower(), add_special_tokens=False
#                     )

#                 encoded_prompt = encoded_prompt[:190]
#                 encoded_prompt = torch.tensor(encoded_prompt)

#                 return {
#                     "input_features": input_features,
#                     "prompt": encoded_prompt,
#                     "labels": labels,
#                 }
#             else:
#                 return {"input_features": input_features, "labels": labels}

#         except Exception as e:
#             print(f"Error at idx {idx}: {e}")
#             raise
