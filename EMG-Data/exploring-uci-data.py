import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

# ==========================================
# 1. BỘ LỌC TÍN HIỆU (Filter)
# ==========================================
def butter_bandpass_filter(data, fs=1000, lowcut=10, highcut=99, order=4):
    """
    Bộ lọc Butterworth bậc 4.
    Lưu ý: fs=1000 Hz do file UCI đo timestamp bằng mili-giây.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# ==========================================
# 2. TRÍCH XUẤT 4 ĐẶC TRƯNG THEO BÀI BÁO
# ==========================================
def extract_paper_features(window):
    """Tính toán MAV, RMS, SSI, VAR cho cửa sổ 200 mẫu."""
    mav = np.mean(np.abs(window), axis=0)
    rms = np.sqrt(np.mean(window**2, axis=0))
    ssi = np.sum(window**2, axis=0)
    var = np.var(window, axis=0)
    return np.hstack([mav, rms, ssi, var])

# ==========================================
# 3. CẮT CỬA SỔ & LỌC NHÃN
# ==========================================
def process_signal_uci(data, labels, window_size=200, step_size=10):
    features_list = []
    labels_list = []
    num_samples = len(data)
    
    for start_idx in range(0, num_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Biểu quyết nhãn (Majority Voting) cho cửa sổ hiện tại
        current_labels = labels[start_idx:end_idx]
        most_frequent_label = np.bincount(current_labels.astype(int)).argmax()
        
        # QUAN TRỌNG: Chỉ trích xuất đặc trưng nếu cử chỉ KHÁC 0 (Unmarked data)
        if most_frequent_label != 0:
            window = data[start_idx:end_idx, :]
            window_features = extract_paper_features(window)
            
            features_list.append(window_features)
            labels_list.append(most_frequent_label)
            
    if len(features_list) > 0:
        return np.array(features_list), np.array(labels_list)
    return np.array([]), np.array([])

# ==========================================
# 4. LUỒNG XỬ LÝ CHÍNH
# ==========================================
DATA_DIR = Path('/home/duy/Documents/Machine-Learning/EMG-Data/UCI-emg-data/EMG_data_for_gestures-master')

def run_uci_pipeline():
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    txt_files = list(DATA_DIR.rglob('*.txt'))
    print(f"Bắt đầu pipeline mô phỏng bài báo với {len(txt_files)} files TXT từ UCI...")
    
    for file_path in txt_files:
        # Tập UCI tổ chức thư mục theo số (01, 02... 36)
        parent_folder = file_path.parent.name
        try:
            subject_id = int(parent_folder)
        except ValueError:
            continue
            
        # ÁP DỤNG QUY TẮC CỦA BÀI BÁO: Phân lập Subject 9 và 18 cho tập Test
        is_test_subject = (subject_id == 9 or subject_id == 18)
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            df.columns = [col.lower().strip() for col in df.columns]
            
            signal_cols = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8']
            
            if not all(col in df.columns for col in signal_cols) or 'class' not in df.columns:
                continue
                
            raw_signals = df[signal_cols].values
            labels = df['class'].values
            
            # Bước 1: Lọc
            filtered_signals = butter_bandpass_filter(
                raw_signals, fs=1000, lowcut=10, highcut=99, order=4
            )
            
            # Bước 2 & 3: Cắt cửa sổ và Rút trích
            features, window_labels = process_signal_uci(
                filtered_signals, labels, window_size=200, step_size=10
            )
            
            if len(features) > 0:
                if is_test_subject:
                    X_test.append(features)
                    y_test.append(window_labels)
                else:
                    X_train.append(features)
                    y_train.append(window_labels)
                    
        except Exception as e:
            print(f"Lỗi file {file_path.name}: {e}")

    # Ép kiểu dữ liệu để tối ưu RAM
    X_train_final = np.concatenate(X_train, axis=0).astype(np.float32)
    y_train_final = np.concatenate(y_train, axis=0).astype(np.int8)
    
    X_test_final = np.concatenate(X_test, axis=0).astype(np.float32)
    y_test_final = np.concatenate(y_test, axis=0).astype(np.int8)
    
    return X_train_final, y_train_final, X_test_final, y_test_final

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = run_uci_pipeline()
    
    print("\n--- KẾT QUẢ TIỀN XỬ LÝ THEO BÀI BÁO (UCI DATASET) ---")
    print(f"Tập Huấn luyện (Train - 34 Subjects):")
    print(f" - Số mẫu: {X_train.shape[0]}")
    print(f" - Kích thước X: {X_train.shape} (32 cột = 8 kênh x 4 đặc trưng)")
    
    print(f"\nTập Kiểm thử (Test - Độc quyền Subject 9 và 18):")
    print(f" - Số mẫu: {X_test.shape[0]}")
    print(f" - Kích thước X: {X_test.shape}")
    
    # Lưu kết quả
    np.save('UCI_X_train_paper.npy', X_train)
    np.save('UCI_y_train_paper.npy', y_train)
    np.save('UCI_X_test_paper.npy', X_test)
    np.save('UCI_y_test_paper.npy', y_test)
    print("\nĐã đóng gói dữ liệu UCI xong! Hoàn toàn không có hiện tượng Data Leakage.")