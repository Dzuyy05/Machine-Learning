import pandas as pd
import numpy as np
from pathlib import Path
import re
from scipy.signal import butter, filtfilt

# ==========================================
# 1. BỘ LỌC TÍN HIỆU (Filter)
# ==========================================
def butter_bandpass_filter(data, fs=200, lowcut=10, highcut=99, order=4):
    """Bộ lọc Butterworth bậc 4, dải tần 10-99 Hz theo bài báo."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# ==========================================
# 2. TRÍCH XUẤT ĐẶC TRƯNG (Feature Engineering)
# ==========================================
def extract_paper_features(window):
    """
    Tính toán chính xác 4 đặc trưng theo Công thức (2) - (5) trong bài báo.
    Đầu vào: window (200 samples, 8 channels)
    Đầu ra : Mảng 1D (32 phần tử) gồm 4 đặc trưng x 8 kênh.
    """
    # (2) Mean Absolute Value (MAV)
    mav = np.mean(np.abs(window), axis=0)
    
    # (3) Root Mean Square (RMS)
    rms = np.sqrt(np.mean(window**2, axis=0))
    
    # (4) Simple Square Integral (SSI)
    ssi = np.sum(window**2, axis=0)
    
    # (5) Variance (VAR)
    var = np.var(window, axis=0)
    
    # Nối tất cả các đặc trưng lại với nhau
    return np.hstack([mav, rms, ssi, var])

# ==========================================
# 3. CẮT CỬA SỔ & RÚT TRÍCH (Dynamic Segmentation)
# ==========================================
def process_signal(data, window_size=200, step_size=10):
    """Trượt cửa sổ 200 mẫu, bước nhảy 10 mẫu."""
    features_list = []
    num_samples = len(data)
    
    for start_idx in range(0, num_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window = data[start_idx:end_idx, :]
        
        # Trích xuất đặc trưng ngay lập tức để tiết kiệm RAM
        window_features = extract_paper_features(window)
        features_list.append(window_features)
        
    return np.array(features_list)

# ==========================================
# 4. LUỒNG XỬ LÝ CHÍNH (Subject-specific Pipeline)
# ==========================================
DATA_DIR = Path('/home/duy/Documents/Machine-Learning/EMG-Data/RSE-emg-data')

GESTURE_MAP = {
    'rest': 0,
    'indexfingerextension': 1,
    'middlefingerextension': 2,
    'cylindricalgrip': 3,
    'closedgrip': 4,
    'closedfist': 4 
}

def run_pipeline():
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    csv_files = list(DATA_DIR.rglob('*.csv'))
    print(f"Bắt đầu pipeline mô phỏng bài báo với {len(csv_files)} files CSV...")
    
    for file_path in csv_files:
        # Xác định nhãn cử chỉ
        parent_folder = file_path.parent.name
        normalized_folder = re.sub(r'[\s\-_]', '', parent_folder.lower())
        
        label_id = None
        for key, val in GESTURE_MAP.items():
            if key in normalized_folder:
                label_id = val
                break
        if label_id is None:
            continue
            
        # Xác định tập Train hay Test (Subject 10 là Test, phần còn lại là Train)
        # Dùng path.parts để bắt chính xác tên thư mục 'User 10'
        is_test_subject = any('user 10' in part.lower() for part in file_path.parts)
        
        try:
            # Đọc dữ liệu
            df = pd.read_csv(file_path)
            signal_cols = ['emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']
            
            if not all(col in df.columns for col in signal_cols):
                signal_cols = [col for col in df.columns if col.lower() in signal_cols]
                if len(signal_cols) != 8:
                    continue
                    
            raw_signals = df[signal_cols].values
            
            # Bước 1: Lọc Band-pass
            filtered_signals = butter_bandpass_filter(
                raw_signals, fs=200, lowcut=10, highcut=99, order=4
            )
            
            # Bước 2 & 3: Cắt cửa sổ và Tính đặc trưng (MAV, RMS, SSI, VAR)
            features = process_signal(filtered_signals, window_size=200, step_size=10)
            
            # Bước 4: Lưu vào danh sách tương ứng
            if len(features) > 0:
                y_array = np.full(shape=(len(features),), fill_value=label_id)
                
                if is_test_subject:
                    X_test.append(features)
                    y_test.append(y_array)
                else:
                    X_train.append(features)
                    y_train.append(y_array)
                    
        except Exception as e:
            print(f"Lỗi file {file_path.name}: {e}")

    # Gộp thành các ma trận chuẩn NumPy
    X_train_final = np.concatenate(X_train, axis=0).astype(np.float32)
    y_train_final = np.concatenate(y_train, axis=0).astype(np.int8)
    
    X_test_final = np.concatenate(X_test, axis=0).astype(np.float32)
    y_test_final = np.concatenate(y_test, axis=0).astype(np.int8)
    
    return X_train_final, y_train_final, X_test_final, y_test_final

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = run_pipeline()
    
    print("\n--- KẾT QUẢ TIỀN XỬ LÝ THEO BÀI BÁO ---")
    print(f"Tập Huấn luyện (Train - User 1 đến 9):")
    print(f" - Số mẫu: {X_train.shape[0]}")
    print(f" - Kích thước X: {X_train.shape} (32 cột = 8 kênh x 4 đặc trưng)")
    
    print(f"\nTập Kiểm thử (Test - Độc quyền User 10):")
    print(f" - Số mẫu: {X_test.shape[0]}")
    print(f" - Kích thước X: {X_test.shape}")
    
    # Lưu kết quả
    np.save('RSE_X_train_paper.npy', X_train)
    np.save('RSE_y_train_paper.npy', y_train)
    np.save('RSE_X_test_paper.npy', X_test)
    np.save('RSE_y_test_paper.npy', y_test)
    print("\nĐã đóng gói dữ liệu xong! Bạn có thể đưa thẳng X_train và y_train vào mô hình học máy.")