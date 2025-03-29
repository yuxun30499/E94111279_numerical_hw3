import numpy as np

# 題目給的 x 與 對應的 e^{-x} (y)
x_data = np.array([0.3, 0.4, 0.5, 0.6])
y_data = np.array([0.740818, 0.670320, 0.606531, 0.548812])

# 對應的 f_i = x_i - y_i
f_data = x_data - y_data

max_iter = 10
print("Iter |     x_new      |     f_new        ")
print("----------------------------------------")

for i in range(max_iter):
    # (1) 對 (f_data, x_data) 做多項式插值 => 反插值 (f -> x)
    poly_inv = np.polyfit(f_data, x_data, deg=3)
    # f=0 時的 x 值
    x_new = np.polyval(poly_inv, 0)
    
    # (2) 對 (x_data, y_data) 做多項式插值 => 正插值 (x -> y)
    poly_dir = np.polyfit(x_data, y_data, deg=3)
    y_new = np.polyval(poly_dir, x_new)
    
    # (3) 計算新的 f_new
    f_new = x_new - y_new
    
    print(f"{i:4d} | {x_new:14.8f} | {f_new:14.8e}")
    
    # 判斷是否收斂
    if abs(f_new) < 1e-12:
        break
    
    # (4) 更新資料：捨棄最舊的一點，加入 (x_new, y_new, f_new)
    x_data = np.append(x_data[1:], x_new)
    y_data = np.append(y_data[1:], y_new)
    f_data = np.append(f_data[1:], f_new)

print("\nApprox. root =", x_new)
