import numpy as np
from scipy.interpolate import lagrange
import pandas as pd
import math

# 題目提供的點
x_data = np.array([0.698, 0.733, 0.768, 0.803])
y_data = np.array([0.7661, 0.7432, 0.7193, 0.6946])

# 目標插值點與真實值
x_target = 0.750
true_value = 0.7317  # 題目提供的 cos(0.750)

# 假設最大導數值 M = 1
M = 1

# 存放結果
results = []

# 建立不同階數的插值
for n in range(2, 6):
    x_subset = x_data[:n]
    y_subset = y_data[:n]

    # 建立 Lagrange 多項式
    poly = lagrange(x_subset, y_subset)
    estimate = poly(x_target)

    # 計算實際誤差（用真實值）
    true_error = abs(true_value - estimate)

    # 計算理論誤差界線（Error Bound）
    product_term = 1
    for xi in x_subset:
        product_term *= abs(x_target - xi)
    
    error_bound = (M / math.factorial(n)) * product_term

    results.append({
        "Degree": n - 1,
        "Estimate": estimate,
        "True Error": true_error,
        "Error Bound": error_bound
    })

# 輸出結果
df = pd.DataFrame(results)
print(df)
