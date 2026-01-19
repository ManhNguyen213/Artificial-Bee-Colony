import matplotlib.pyplot as plt
from abc_algorithm import ABC as OriginalABC, objective_function
from ae_abc import AEABC

# ========================
# Chạy ABC gốc
# ========================
bounds = [(-5, 5), (-5, 5)]
abc = OriginalABC(objective_function, bounds, pop_size=20, max_iter=500, limit=50)
best_pos_abc, best_val_abc = abc.run()
history_abc = abc.history[:50]  # Lấy 50 vòng đầu để so sánh

# ========================
# Chạy AEABC
# ========================
aeabc = AEABC(objective_function, bounds, pop_size=20, max_iter=500, limit=50)
best_pos_aeabc, best_val_aeabc = aeabc.run()
history_aeabc = aeabc.history[:50]

# ========================
# Vẽ đồ thị so sánh
# ========================
plt.figure(figsize=(10,6))
plt.plot(history_abc, color='blue', marker='o', markersize=3, linewidth=1, label='ABC')
plt.plot(history_aeabc, color='green', marker='o', markersize=3, linewidth=1, label='AEABC')

plt.title("So sánh đồ thị hội tụ ABC vs AEABC (50 vòng đầu)")
plt.xlabel("Iteration")
plt.ylabel("Best objective value")
plt.legend()
plt.grid(True)
plt.show()
