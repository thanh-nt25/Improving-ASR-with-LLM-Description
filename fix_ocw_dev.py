# Lưu file này dưới tên fix_ocw_dev.py
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import tqdm

# Thêm thư mục code của bạn vào path
sys.path.append("/kaggle/working")

# Import các module cần thiết từ code của bạn
from transformers_prompt import WhisperTokenizer
from utils_prompt import BasicTextNormalizer

def fix_ocw_dev_set():
    """
    Kiểm tra và loại bỏ các mẫu có chuỗi tham chiếu rỗng trong OCW dev set
    để tránh lỗi "one or more references are empty strings"
    """
    print("===== KIỂM TRA CHUỖI RỖNG TRONG OCW DEV SET =====")
    
    # Khởi tạo tokenizer và normalizer
    tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-base.en', language='en', task='transcribe')
    normalizer = BasicTextNormalizer()
    
    # Đường dẫn đến OCW dev set
    ocw_dev_path = "/kaggle/input/ocw-biasing/OCW/dev"
    
    if not os.path.exists(ocw_dev_path):
        print(f"Không tìm thấy thư mục OCW dev: {ocw_dev_path}")
        return
    
    # Danh sách chứa thông tin về các file có vấn đề
    problematic_files = []
    
    # Đếm số lượng JSON file
    json_files = []
    for root, dirs, files in os.walk(ocw_dev_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Đang kiểm tra {len(json_files)} file JSON trong OCW dev set...")
    
    # Kiểm tra từng file
    for json_path in tqdm.tqdm(json_files):
        try:
            # Đọc file JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Lấy trường text
            text = data.get('text', '')
            
            # Kiểm tra xem text có rỗng không
            if not text or text.strip() == '':
                problematic_files.append((json_path, "Empty text"))
                continue
            
            # Chuẩn hóa text và kiểm tra
            normalized_text = normalizer(text)
            if not normalized_text or normalized_text.strip() == '':
                problematic_files.append((json_path, "Empty after normalization"))
                continue
            
            # Kiểm tra độ dài sau khi chuẩn hóa
            if len(normalized_text.split()) < 2:
                problematic_files.append((json_path, f"Too short: '{normalized_text}'"))
                continue
            
        except Exception as e:
            problematic_files.append((json_path, f"Error: {str(e)}"))
    
    # Hiển thị thông tin về các file có vấn đề
    if problematic_files:
        print(f"\nTìm thấy {len(problematic_files)} file có vấn đề:")
        for path, reason in problematic_files:
            print(f"- {os.path.basename(path)}: {reason}")
        
        # Loại bỏ các file có vấn đề
        print("\nĐang loại bỏ các file có vấn đề để tránh lỗi 'one or more references are empty strings'...")
        
        for path, reason in problematic_files:
            try:
                # Đổi tên file json
                backup_path = path + ".empty_ref"
                os.rename(path, backup_path)
                
                # Nếu có file audio tương ứng, cũng đổi tên nó
                audio_path = path.replace('.json', '.mp3')
                if os.path.exists(audio_path):
                    os.rename(audio_path, audio_path + ".empty_ref")
                
                print(f"Đã loại bỏ: {os.path.basename(path)}")
            except Exception as e:
                print(f"Lỗi khi loại bỏ {os.path.basename(path)}: {e}")
        
        print(f"\n✅ Đã loại bỏ {len(problematic_files)} file có vấn đề.")
        print("Bạn có thể chạy whisper_fine.py mà không lo lắng về lỗi 'one or more references are empty strings'.")
    else:
        print("\n✅ Không tìm thấy file nào có vấn đề. Dataset sạch.")

if __name__ == "__main__":
    fix_ocw_dev_set()