import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================
# HÀM BỘ LỌC (Band-pass 10-99 Hz)
# ==========================================
def butter_bandpass_filter(data, fs, lowcut=10, highcut=99, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    # Lọc 1D cho 1 kênh tín hiệu
    return filtfilt(b, a, data)

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN FILE MẪU
# ==========================================
# Lấy 1 file bất kỳ của UCI (Tần số 1000Hz)
uci_path = '/home/duy/Documents/Machine-Learning/EMG-Data/UCI-emg-data/EMG_data_for_gestures-master/01/1_raw_data_13-12_22.03.16.txt'

# Lấy 1 file bất kỳ của RSE (Tần số 200Hz)
rse_path = '/home/duy/Documents/Machine-Learning/EMG-Data/RSE-emg-data/User 1/Closed Grip/u1-closed-fist-set-1.csv'

# ==========================================
# ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ==========================================
# 1. Xử lý UCI Dataset
df_uci = pd.read_csv(uci_path, sep='\t')
df_uci.columns = [col.lower().strip() for col in df_uci.columns]
uci_raw = df_uci['channel1'].values
uci_filtered = butter_bandpass_filter(uci_raw, fs=1000)

# 2. Xử lý RSE Dataset
df_rse = pd.read_csv(rse_path)
rse_raw = df_rse['emg1'].values
rse_filtered = butter_bandpass_filter(rse_raw, fs=200)

# ==========================================
# VẼ ĐỒ THỊ (TÁI TẠO FIGURE 4)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
fig.suptitle('Comparison between raw and filtered data of the UCI sEMG and RSE sEMG dataset (Figure 4)', fontsize=16, fontweight='bold')

# --- HÀNG 1: UCI DATASET ---
# UCI Raw
axes[0, 0].plot(uci_raw, color='steelblue', linewidth=0.8)
axes[0, 0].set_title('UCI Dataset: Raw sEMG Signal (Channel 1)')
axes[0, 0].set_ylabel('Amplitude (Normalized V)')
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# UCI Filtered
axes[0, 1].plot(uci_filtered, color='seagreen', linewidth=0.8)
axes[0, 1].set_title('UCI Dataset: Filtered sEMG Signal (10-99 Hz)')
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# --- HÀNG 2: RSE DATASET ---
# RSE Raw
axes[1, 0].plot(rse_raw, color='darkorange', linewidth=0.8)
axes[1, 0].set_title('RSE Dataset: Raw sEMG Signal (EMG 1)')
axes[1, 0].set_ylabel('Amplitude (ADC Counts)')
axes[1, 0].set_xlabel('Samples')
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# RSE Filtered
axes[1, 1].plot(rse_filtered, color='firebrick', linewidth=0.8)
axes[1, 1].set_title('RSE Dataset: Filtered sEMG Signal (10-99 Hz)')
axes[1, 1].set_xlabel('Samples')
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

# Căn chỉnh layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()