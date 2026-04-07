import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ==========================================
# HÀM HỖ TRỢ: VẼ ĐỒ THỊ TIME DOMAIN VÀ FFT
# ==========================================
def plot_signal_and_fft(df, dataset_name, time_col=None, signal_col='channel1', fs=1000):
    """
    Vẽ tín hiệu gốc và phổ FFT của một kênh EMG.
    fs: Tần số lấy mẫu (Sampling rate), mặc định giả định là 1000Hz (phổ biến với EMG).
    """
    plt.figure(figsize=(15, 8))
    
    # 1. Đồ thị miền thời gian (Time Domain)
    plt.subplot(2, 1, 1)
    if time_col and time_col in df.columns:
        plt.plot(df[time_col], df[signal_col], color='b', alpha=0.7)
        plt.xlabel('Thời gian (ms hoặc s)')
    else:
        plt.plot(df.index, df[signal_col], color='b', alpha=0.7)
        plt.xlabel('Số mẫu (Samples)')
        
    plt.ylabel('Biên độ')
    plt.title(f'[{dataset_name}] Tín hiệu thô miền thời gian - Kênh: {signal_col}')
    plt.grid(True)

    # 2. Đồ thị miền tần số (Frequency Domain - FFT)
    plt.subplot(2, 1, 2)
    # Lấy dữ liệu tín hiệu, trừ đi giá trị trung bình để loại bỏ thành phần DC (0Hz)
    signal = df[signal_col].values
    signal = signal - np.mean(signal) 
    
    n = len(signal)
    # Tính toán FFT
    yf = rfft(signal)
    xf = rfftfreq(n, 1 / fs)
    
    plt.plot(xf, np.abs(yf), color='r')
    plt.xlabel('Tần số (Hz)')
    plt.ylabel('Biên độ (Năng lượng)')
    plt.title(f'[{dataset_name}] Phổ tần số (FFT) - Kiểm tra nhiễu điện lưới')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# HÀM MỚI: VẼ TOÀN BỘ CÁC KÊNH (MULTI-CHANNEL TRACE)
# ==========================================
def plot_all_channels(df, dataset_name, time_col=None):
    """
    Vẽ toàn bộ các kênh tín hiệu (thường là 8 kênh) theo dạng xếp tầng để quan sát tổng thể.
    """
    # Tìm các cột là kênh tín hiệu (bỏ qua cột thời gian hoặc class nếu có)
    # RSE thường có tên 'CH1'..'CH8', UCI thường là 'channel1'..'channel8'
    signal_cols = [col for col in df.columns if 'ch' in col.lower() or col.isdigit() or type(col) == int]
    
    # Nếu không tìm thấy theo tên, lấy 8 cột đầu tiên (bỏ qua time/class)
    if not signal_cols:
        signal_cols = [col for col in df.columns if col != time_col and col != 'class'][:8]

    num_channels = len(signal_cols)
    if num_channels == 0:
        print("Không tìm thấy cột tín hiệu nào để vẽ.")
        return

    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 2 * num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes] # Đảm bảo axes luôn có thể lặp qua (iterable)

    fig.suptitle(f'[{dataset_name}] Tín hiệu đa kênh (Multi-channel Trace)', fontsize=16)

    time_data = df[time_col] if time_col and time_col in df.columns else df.index

    for i, col in enumerate(signal_cols):
        axes[i].plot(time_data, df[col], color='b', alpha=0.8, linewidth=0.5)
        axes[i].set_ylabel(col, rotation=0, labelpad=40, ha='right')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
        # Loại bỏ viền trên/dưới để nhìn liền mạch hơn
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)

    axes[-1].set_xlabel('Thời gian (ms hoặc mẫu)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Để chừa không gian cho suptitle
    plt.show()

# ==========================================
# PHẦN 1: KHÁM PHÁ DỮ LIỆU UCI (File .txt)
# ==========================================
print("--- KHÁM PHÁ TẬP DỮ LIỆU UCI ---")
# Thay đường dẫn này bằng đường dẫn thực tế trên máy cậu
uci_path = 'EMG-Data/UCI-emg-data/EMG_data_for_gestures-master/01/1_raw_data_13-12_22.03.16.txt'

try:
    # Tập UCI thường phân tách bằng tab ('\t')
    df_uci = pd.read_csv(uci_path, sep='\t')
    print("5 dòng đầu tiên của UCI dataset:")
    print(df_uci.head())
    print("\nThông tin tập dữ liệu UCI:")
    print(df_uci.info())
    
    # Giả định cột đầu tiên là time, cột thứ 2 là channel 1
    cols = df_uci.columns
    time_column = cols[0] if 'time' in cols[0].lower() else None
    signal_column = cols[1] # Thường là 'channel1'
    
    # Tần số lấy mẫu của UCI EMG thường là 1000Hz (Mỗi mẫu cách nhau 1ms)
    # Chúng ta plot thử Kênh 1
    plot_signal_and_fft(df_uci, dataset_name="UCI Dataset (User 01)", time_col=time_column, signal_col=signal_column, fs=1000)

    # TRỰC QUAN HÓA TOÀN BỘ 8 KÊNH CỦA UCI
    plot_all_channels(df_uci, dataset_name="UCI Dataset (User 01)", time_col=time_column)

except Exception as e:
    print(f"Lỗi khi đọc file UCI: {e}")


# ==========================================
# PHẦN 2: KHÁM PHÁ DỮ LIỆU RSE (File .csv)
# ==========================================
print("\n--- KHÁM PHÁ TẬP DỮ LIỆU RSE ---")
# Thay đường dẫn này bằng đường dẫn thực tế trên máy cậu
rse_path = 'EMG-Data/RSE-emg-data/User 1/Closed Grip/u1-closed-fist-set-1.csv'

try:
    # Tập RSE chuẩn CSV
    df_rse = pd.read_csv(rse_path)
    print("5 dòng đầu tiên của RSE dataset:")
    print(df_rse.head())
    print("\nThông tin tập dữ liệu RSE:")
    print(df_rse.info())
    
    # Giả định lấy cột tín hiệu đầu tiên (thường là CH1)
    cols = df_rse.columns
    time_column = cols[0] if 'time' in cols[0].lower() else None
    signal_column = cols[0] if time_column is None else cols[1]
    
    # Giả định Sampling Rate của thiết bị RSE cũng là 1000Hz (Cần điều chỉnh nếu đồ thị FFT nhìn bất thường)
    plot_signal_and_fft(df_rse, dataset_name="RSE Dataset (User 1 - Closed Grip)", time_col=time_column, signal_col=signal_column, fs=1000)

    # TRỰC QUAN HÓA TOÀN BỘ 8 KÊNH CỦA RSE
    plot_all_channels(df_rse, dataset_name="RSE Dataset (User 1 - Closed Grip)", time_col=time_column)

except Exception as e:
    print(f"Lỗi khi đọc file RSE: {e}")