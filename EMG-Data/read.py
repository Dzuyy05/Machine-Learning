import numpy as np

# 1. Tải dữ liệu từ ổ cứng lên RAM
print("Đang nạp dữ liệu...")
X_data = np.load('UCI_X_processed.npy')
y_data = np.load('UCI_y_processed.npy')
print("Nạp dữ liệu thành công!\n")

# 2. Kiểm tra thông tin tổng quan
print("--- THÔNG TIN TẬP ĐẶC TRƯNG (X) ---")
print(f"Kích thước (Shape) : {X_data.shape}")
print(f"Kiểu dữ liệu (Type): {X_data.dtype}")
# Kì vọng: (Số lượng mẫu, 200, 8) và kiểu float32 hoặc float64

print("\n--- THÔNG TIN TẬP NHÃN (y) ---")
print(f"Kích thước (Shape) : {y_data.shape}")
print(f"Kiểu dữ liệu (Type): {y_data.dtype}")
# Kì vọng: (Số lượng mẫu,) và kiểu int 

# 3. Xem thử một mẫu (Sample) bất kỳ
sample_idx = 0  # Chọn mẫu đầu tiên
print(f"\n--- MẪU DỮ LIỆU SỐ {sample_idx} ---")
print(f"Nhãn cử chỉ: {y_data[sample_idx]}")
print(f"Kích thước ma trận của 1 cửa sổ: {X_data[sample_idx].shape}")
# In thử 5 giá trị đầu tiên của kênh số 1 (emg1) trong cửa sổ này
print(f"5 giá trị tín hiệu đầu tiên của Kênh 1: {X_data[sample_idx, :5, 0]}")