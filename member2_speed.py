import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sir_superspreader_simulation import EpidemicSimulator

# ==========================================
# THIẾT LẬP THÔNG SỐ CHUNG
# ==========================================
r0 = 1.0
L = 10.0 * r0
N = 636 
NUM_RUNS = 1000  # Chạy 1000 lần như bạn đã setup
MAX_STEPS = 40
DATA_FILE = "member2_data_cache_v2.pkl"

def get_velocity_from_avg_curve(rf_avg):
    """
    Tính vận tốc từ ĐƯỜNG CONG TRUNG BÌNH (Average Curve).
    Tự động dò tìm vùng tuyến tính đang phát triển để tính hệ số góc,
    bỏ qua vùng đi ngang (plateau) khi chạm biên hoặc khi dịch tắt.
    """
    max_rf = np.max(rf_avg)
    
    # Định nghĩa mốc bão hòa: Cắt bỏ 15% phần chóp đuôi bão hòa
    # Giới hạn tuyệt đối là 8.5 (vì biên L-r0 = 9.0)
    threshold = min(8.5, 0.85 * max_rf) 
    
    # Lấy các điểm dữ liệu nằm trong giai đoạn đang tăng tốc
    valid_indices = np.where((rf_avg >= 0.5) & (rf_avg <= threshold))[0]
    
    # Backup: Nếu đường cong quá dốc (chạm biên quá nhanh trong 1-2 bước)
    if len(valid_indices) < 3:
        valid_indices = np.where(rf_avg <= 0.9 * max_rf)[0]
         
    if len(valid_indices) >= 2:
        x = valid_indices
        y = rf_avg[valid_indices]
        slope, _ = np.polyfit(x, y, 1) # Tính hệ số vi phân bậc nhất
        return slope if slope > 0 else 0.0
    return 0.0

# ==========================================
# PHA 1: CHẠY MÔ PHỎNG VÀ LƯU DỮ LIỆU 
# ==========================================
lambda_list_fig6 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
lambda_list_fig7 = np.arange(0.0, 1.05, 0.05) 

if not os.path.exists(DATA_FILE):
    print(f"Đang chạy mô phỏng mới với {NUM_RUNS} runs...")
    
    rf_results_fig6 = {}
    vel_strong = []
    vel_hub = []

    # --- Hình 6: Quỹ đạo thời gian (Chỉ chạy Strong model) ---
    print("\n--- Đang xử lý Hình 6 ---")
    for lam in lambda_list_fig6:
        rf_avg = np.zeros(MAX_STEPS)
        for _ in range(NUM_RUNS):
            sim = EpidemicSimulator(N, L, r0, lam, model_type='strong')
            rf = sim.run()["rf_history"]
            length = min(len(rf), MAX_STEPS)
            rf_avg[:length] += rf[:length]
            if length < MAX_STEPS:
                rf_avg[length:] += rf[-1] # Kéo dài giá trị bão hòa cho các bước còn lại
        
        rf_avg /= NUM_RUNS
        rf_results_fig6[lam] = rf_avg
        print(f" Xong lambda = {lam}")

    # --- Hình 7: Vận tốc (Chạy cả Strong và Hub) ---
    print("\n--- Đang xử lý Hình 7 ---")
    for lam in lambda_list_fig7:
        rf_s_avg = np.zeros(MAX_STEPS)
        rf_h_avg = np.zeros(MAX_STEPS)
        
        for _ in range(NUM_RUNS):
            # Tích lũy quỹ đạo Strong
            sim_s = EpidemicSimulator(N, L, r0, lam, model_type='strong')
            rf_s = sim_s.run()["rf_history"]
            len_s = min(len(rf_s), MAX_STEPS)
            rf_s_avg[:len_s] += rf_s[:len_s]
            if len_s < MAX_STEPS: rf_s_avg[len_s:] += rf_s[-1]
            
            # Tích lũy quỹ đạo Hub
            sim_h = EpidemicSimulator(N, L, r0, lam, model_type='hub')
            rf_h = sim_h.run()["rf_history"]
            len_h = min(len(rf_h), MAX_STEPS)
            rf_h_avg[:len_h] += rf_h[:len_h]
            if len_h < MAX_STEPS: rf_h_avg[len_h:] += rf_h[-1]
            
        rf_s_avg /= NUM_RUNS
        rf_h_avg /= NUM_RUNS
        
        # TÍNH VẬN TỐC TỪ ĐƯỜNG TRUNG BÌNH
        vel_strong.append(get_velocity_from_avg_curve(rf_s_avg))
        vel_hub.append(get_velocity_from_avg_curve(rf_h_avg))
        print(f" Xong lambda = {lam:.2f}")

    cached_data = {"fig6": rf_results_fig6, "fig7_strong": vel_strong, "fig7_hub": vel_hub}
    with open(DATA_FILE, "wb") as f:
        pickle.dump(cached_data, f)
else:
    print(f"Đang load dữ liệu từ {DATA_FILE}...")
    with open(DATA_FILE, "rb") as f:
        cached_data = pickle.load(f)
    rf_results_fig6 = cached_data["fig6"]
    vel_strong = cached_data["fig7_strong"]
    vel_hub = cached_data["fig7_hub"]

# ==========================================
# PHA 2: TRỰC QUAN HÓA (VẼ BIỂU ĐỒ)
# ==========================================
print("\nĐang vẽ biểu đồ...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Vẽ Hình 6
colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']
markers = ['o', '*', 's', 'p', '^', 'v']
for i, lam in enumerate(lambda_list_fig6):
    x_plot = np.arange(0, MAX_STEPS, 1) # Giữ nguyên bước 1 để thấy rõ độ mượt
    y_plot = (rf_results_fig6[lam] / r0)[x_plot]
    ax1.plot(x_plot, y_plot, marker=markers[i], color=colors[i], linestyle='', fillstyle='none', label=rf'$\lambda={lam}$')

ax1.set_title("Figure 6: Propagation Distance over Time")
ax1.set_xlabel("time step")
ax1.set_ylabel(r"$r_f / r_0$")
ax1.set_xlim(0, 40)
ax1.set_ylim(0, 12)
ax1.legend(loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.5)

# Vẽ Hình 7
ax2.plot(lambda_list_fig7, vel_strong, 'ro', label='Strong infectiousness model')
ax2.plot(lambda_list_fig7, vel_hub, 'bs', fillstyle='none', label='Hub model')
ax2.set_title("Figure 7: Velocity of propagation")
ax2.set_xlabel(r"$\lambda$")
ax2.set_ylabel(r"velocity (/$r_0 \cdot s$)")
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0, 1.6)
ax2.legend(loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()