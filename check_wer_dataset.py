# Lưu file này dưới tên check_empty_references.py
import os
import sys
import argparse
import traceback
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from tqdm import tqdm

# Thêm thư mục gốc vào đường dẫn để import các module
sys.path.append("/kaggle/working")

# Import các module từ codebase
from transformers_prompt import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor
from utils_prompt import BasicTextNormalizer
from data.dataloader import PromptWhisperDataset

class WERChecker:
    def __init__(self, args):
        """
        Khởi tạo WERChecker để kiểm tra lỗi empty references
        
        Args:
            args: Tham số từ argparse giống như trong whisper_fine.py
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Khởi tạo tokenizer, feature extractor và processor
        self.tokenizer = WhisperTokenizer.from_pretrained(
            f'openai/whisper-{args.model}', 
            language='en', 
            task='transcribe'
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            f'openai/whisper-{args.model}'
        )
        self.processor = WhisperProcessor.from_pretrained(
            f'openai/whisper-{args.model}', 
            language='en', 
            task='transcribe'
        )
        
        self.normalizer = BasicTextNormalizer()
        
        # Đường dẫn dữ liệu
        self.data_root = "/kaggle/input/ocw-biasing"
        
    def load_datasets(self):
        """
        Tải datasets từ cấu hình args giống như trong pipeline chính
        """
        try:
            # Tải dataset theo cùng logic với whisper_fine.py
            if self.args.dataset == 'earning':
                self.eval_dataset = PromptWhisperDataset(
                    base_path=os.path.join(self.data_root, "Earnings_Call/"), 
                    phase='dev', 
                    feature_extractor=self.feature_extractor, 
                    audio_type=".mp3", 
                    tokenizer=self.tokenizer, 
                    prompt=self.args.prompt, 
                    basic=self.args.basic
                )
                self.test_dataset = PromptWhisperDataset(
                    base_path=os.path.join(self.data_root, "Earnings_Call/"), 
                    phase='test', 
                    feature_extractor=self.feature_extractor, 
                    audio_type=".mp3", 
                    tokenizer=self.tokenizer, 
                    prompt=self.args.prompt, 
                    basic=self.args.basic
                )
            elif self.args.dataset == 'ocw':
                self.eval_dataset = PromptWhisperDataset(
                    base_path=os.path.join(self.data_root, "OCW/"), 
                    phase='dev', 
                    feature_extractor=self.feature_extractor, 
                    audio_type=".mp3", 
                    tokenizer=self.tokenizer, 
                    prompt=self.args.prompt, 
                    basic=self.args.basic
                )
                self.test_dataset = PromptWhisperDataset(
                    base_path=os.path.join(self.data_root, "OCW/"), 
                    phase='test', 
                    feature_extractor=self.feature_extractor, 
                    audio_type=".mp3", 
                    tokenizer=self.tokenizer, 
                    prompt=self.args.prompt, 
                    basic=self.args.basic
                )
            else:
                raise ValueError(f"Dataset không hợp lệ: {self.args.dataset}")
                
            print(f"Đã tải dataset đánh giá: {len(self.eval_dataset)} mẫu")
            print(f"Đã tải dataset kiểm thử: {len(self.test_dataset)} mẫu")
            
            return True
        except Exception as e:
            print(f"Lỗi khi tải datasets: {e}")
            traceback.print_exc()
            return False
    
    def check_dataset(self, dataset, name="Evaluation"):
        """
        Kiểm tra dataset để tìm các chuỗi tham chiếu rỗng
        
        Args:
            dataset: Dataset cần kiểm tra
            name: Tên của dataset (để hiển thị)
        """
        print(f"\n===== KIỂM TRA DATASET {name} =====")
        
        # Thống kê
        empty_references = 0
        empty_after_norm = 0
        short_references = 0
        valid_samples = 0
        problematic_indices = []
        
        batch_size = 16  # Batch size cho kiểm tra
        
        # Kiểm tra từng mẫu trong dataset
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_indices = list(range(i, min(i + batch_size, len(dataset))))
            
            for idx in batch_indices:
                try:
                    # Lấy mẫu từ dataset
                    sample = dataset[idx]
                    
                    # Lấy labels
                    if 'labels' in sample:
                        labels = sample['labels']
                        
                        # Chuyển đổi -100 thành pad_token_id
                        processed_labels = [
                            token_id if token_id != -100 else self.tokenizer.pad_token_id 
                            for token_id in labels
                        ]
                        
                        # Giải mã thành text
                        label_text = self.tokenizer.decode(processed_labels, skip_special_tokens=True)
                        
                        # Kiểm tra chuỗi rỗng
                        if not label_text or label_text.strip() == "":
                            empty_references += 1
                            problematic_indices.append((idx, "empty_reference", label_text))
                            continue
                        
                        # Kiểm tra chuỗi đặc biệt
                        if label_text == 'ignore_time_segment_in_scoring':
                            problematic_indices.append((idx, "ignore_segment", label_text))
                            continue
                        
                        # Kiểm tra sau khi chuẩn hóa
                        normalized = self.normalizer(label_text)
                        if not normalized or normalized.strip() == "":
                            empty_after_norm += 1
                            problematic_indices.append((
                                idx, 
                                "empty_after_norm", 
                                f"Original: '{label_text}', Normalized: '{normalized}'"
                            ))
                            continue
                        
                        # Kiểm tra chuỗi quá ngắn
                        if len(normalized.split()) < 2:
                            short_references += 1
                            problematic_indices.append((
                                idx, 
                                "short_reference", 
                                f"Normalized: '{normalized}', Word count: {len(normalized.split())}"
                            ))
                            continue
                            
                        valid_samples += 1
                    else:
                        print(f"Mẫu {idx} không có trường 'labels'")
                
                except Exception as e:
                    print(f"Lỗi khi xử lý mẫu {idx}: {e}")
        
        # Báo cáo kết quả
        print(f"\n===== KẾT QUẢ KIỂM TRA DATASET {name} =====")
        print(f"Tổng số mẫu: {len(dataset)}")
        print(f"Số mẫu hợp lệ: {valid_samples}")
        print(f"Số mẫu có chuỗi tham chiếu rỗng: {empty_references}")
        print(f"Số mẫu có chuỗi rỗng sau chuẩn hóa: {empty_after_norm}")
        print(f"Số mẫu có chuỗi quá ngắn (dưới 2 từ): {short_references}")
        
        # Hiển thị các mẫu có vấn đề
        if problematic_indices:
            print("\n===== MẪU CÓ VẤN ĐỀ =====")
            for idx, issue_type, content in problematic_indices[:10]:  # Hiển thị 10 mẫu đầu tiên
                print(f"- Sample {idx}: {issue_type}")
                print(f"  Content: {content}")
                print()
            
            if len(problematic_indices) > 10:
                print(f"... và {len(problematic_indices) - 10} mẫu khác có vấn đề")
        
        return {
            "total": len(dataset),
            "valid": valid_samples,
            "empty_references": empty_references,
            "empty_after_norm": empty_after_norm,
            "short_references": short_references,
            "problematic_indices": problematic_indices
        }
    
    def check_all_datasets(self):
        """
        Kiểm tra tất cả datasets và đưa ra phân tích tổng hợp
        """
        print("\n===== BẮT ĐẦU KIỂM TRA DATASETS =====")
        print(f"Dataset: {self.args.dataset}")
        print(f"Model: whisper-{self.args.model}")
        
        # Tải datasets
        if not self.load_datasets():
            return
        
        # Kiểm tra datasets
        eval_results = self.check_dataset(self.eval_dataset, "EVALUATION")
        test_results = self.check_dataset(self.test_dataset, "TEST")
        
        # Phân tích tổng hợp
        print("\n===== PHÂN TÍCH TỔNG HỢP =====")
        
        total_samples = eval_results["total"] + test_results["total"]
        total_problematic = (
            eval_results["empty_references"] + test_results["empty_references"] +
            eval_results["empty_after_norm"] + test_results["empty_after_norm"]
        )
        
        print(f"Tổng số mẫu đã kiểm tra: {total_samples}")
        print(f"Tổng số mẫu có vấn đề: {total_problematic} ({total_problematic/total_samples*100:.2f}%)")
        
        # Đề xuất giải pháp
        if total_problematic > 0:
            print("\n===== ĐỀ XUẤT GIẢI PHÁP =====")
            print("1. SỬA TRONG HÀM compute_wer:")
            print("   Thêm đoạn code sau vào hàm compute_wer trong utils_prompt.py trước dòng tính toán WER:")
            print("   ```python")
            print("   # Lọc bỏ các chuỗi rỗng trước khi tính WER")
            print("   pre_strs = [pred for pred in pre_strs if pred.strip()]")
            print("   label_strs = [ref for ref in label_strs if ref.strip()]")
            print("   ")
            print("   # Kiểm tra nếu không còn dữ liệu sau khi lọc")
            print("   if not pre_strs or not label_strs:")
            print("       print(\"Cảnh báo: Không có dữ liệu hợp lệ để tính WER\")")
            print("       return {'wer': float('nan')}")
            print("   ```")
            print("\n2. TIỀN XỬ LÝ DATASET:")
            print("   Xem xét lọc bỏ các mẫu có vấn đề trước khi huấn luyện")
            
            # Ghi danh sách mẫu có vấn đề vào file để xử lý sau
            problematic_file = os.path.join("/kaggle/working", "problematic_samples.txt")
            with open(problematic_file, "w", encoding="utf-8") as f:
                f.write("===== MẪU CÓ VẤN ĐỀ =====\n")
                f.write(f"Dataset: {self.args.dataset}\n\n")
                
                f.write("EVALUATION DATASET:\n")
                for idx, issue_type, content in eval_results["problematic_indices"]:
                    f.write(f"- Sample {idx}: {issue_type}\n")
                    f.write(f"  Content: {content}\n\n")
                
                f.write("\nTEST DATASET:\n")
                for idx, issue_type, content in test_results["problematic_indices"]:
                    f.write(f"- Sample {idx}: {issue_type}\n")
                    f.write(f"  Content: {content}\n\n")
            
            print(f"\nDanh sách chi tiết các mẫu có vấn đề đã được lưu vào: {problematic_file}")
        else:
            print("\n✅ Không phát hiện vấn đề nào trong datasets. Có thể tiến hành huấn luyện.")

def parse_args():
    parser = argparse.ArgumentParser(description='Check for empty references in datasets')
    
    # Các tham số giống với whisper_fine.py
    parser.add_argument('--model', type=str, default="base.en", help="Whisper model size")
    parser.add_argument('--dataset', type=str, default="ocw", choices=["ocw", "earning"], help="Dataset to check")
    parser.add_argument('--basic', action='store_true', help="Use basic description")
    parser.add_argument('--prompt', action='store_true', help="Use prompt", default=True)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Run the checker
    checker = WERChecker(args)
    checker.check_all_datasets()