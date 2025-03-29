import numpy as np 
from scipy.interpolate import KroghInterpolator
from scipy.optimize import minimize_scalar, root_scalar
import matplotlib.pyplot as plt

# 時間節點
T = np.array([0, 3, 5, 8, 13])
# 對應的位置與速度
D = np.array([0, 200, 375, 620, 990])
V = np.array([75, 77, 80, 74, 72])

# 分別建立位置與速度的插值器
position_interp = KroghInterpolator(T, D)
velocity_interp = KroghInterpolator(T, V)

# 驗證：插值器在節點處是否準確
def verify_interpolation():
    for t, d, v in zip(T, D, V):
        calc_d = position_interp(t)
        calc_v = velocity_interp(t)
        assert np.isclose(calc_d, d, rtol=1e-5), f"在 t={t} 時，位置計算錯誤"
        assert np.isclose(calc_v, v, rtol=1e-5), f"在 t={t} 時，速度計算錯誤"
    print("✅ 驗證通過：所有節點處的位置和速度均符合。")

verify_interpolation()

# 問題 a：預測 t=10 秒的位置和速度
t_eval = 10
position = position_interp(t_eval)
speed = velocity_interp(t_eval)
print(f"\n問題 a:")
print(f"t={t_eval} 秒時，位置 = {position:.2f} 英尺")
print(f"t={t_eval} 秒時，速度 = {speed:.2f} 英尺/秒")

# 問題 b：是否超過 55 mph（= 80.67 ft/s）
speed_limit_fts = 55 * 5280 / 3600
print(f"\n問題 b:")
print(f"速度限制：55 mph = {speed_limit_fts:.2f} ft/s")

# 找出何時第一次超速
exceed_time = None
for i in range(len(T) - 1):
    t_start, t_end = T[i], T[i+1]
    v_start = velocity_interp(t_start)
    v_end = velocity_interp(t_end)

    if (v_start - speed_limit_fts) * (v_end - speed_limit_fts) < 0:
        sol = root_scalar(lambda t: velocity_interp(t) - speed_limit_fts,
                          bracket=[t_start, t_end], method='brentq')
        if sol.converged:
            exceed_time = sol.root
            break
    elif v_start >= speed_limit_fts:
        exceed_time = t_start
        break
    elif v_end >= speed_limit_fts:
        exceed_time = t_end
        break

if exceed_time is not None:
    print(f"汽車在 t={exceed_time:.5f} 秒時首次超過 55 mph。")
else:
    print("汽車未超過 55 mph 限制。")

# 問題 c：最大速度與其時間
result = minimize_scalar(lambda t: -velocity_interp(t), bounds=(0, 13), method='bounded')
max_speed = -result.fun
max_time = result.x
print(f"\n問題 c:")
print(f"最大速度為 {max_speed:.2f} ft/s，發生在 t={max_time:.2f} 秒。")

# 畫速度圖
t_plot = np.linspace(0, 13, 1000)
v_plot = velocity_interp(t_plot)
plt.plot(t_plot, v_plot, label='速度 (ft/s)')
plt.scatter(T, V, color='red', label='已知速度')
plt.axhline(speed_limit_fts, color='green', linestyle='--', label='55 mph')
plt.xlabel('時間 (秒)')
plt.ylabel('速度 (ft/s)')
plt.title('汽車速度隨時間變化')
plt.legend()
plt.grid(True)
plt.show()
