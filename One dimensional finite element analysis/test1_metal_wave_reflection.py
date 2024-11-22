import numpy as np
import laplace_fem as fem
import matplotlib.pyplot as plt


def analytical_solution(k0, m, x, theta, epsilon, miu):
    """金属衬底介质片平面波反射的解析解

    Args:
        x:要求的场点坐标
        theta:入射角
        m:将介质片划分的薄层数
        k0:自由空间的k
        epsilon:相对介电常常数
        miu:相对磁导率

    Returns:
        计算得到的反射功率
    """
    k_xm = k0 * np.sqrt(miu * epsilon - (np.sin(theta.reshape(-1, 1)) ** 2))
    k_xm = k_xm.T
    # 计算电场的反射功率
    R_me = -1 * np.ones(len(theta))
    R_mm = np.ones(len(theta))
    for i in range(m):
        # 电场反射系数计算
        lambda_me = (miu[i] * k_xm[i + 1] - miu[i + 1] * k_xm[i]) / (
            miu[i] * k_xm[i + 1] + miu[i + 1] * k_xm[i]
        )
        R_me = (
            (lambda_me + R_me * np.exp(-2.0j * k_xm[i] * x[i + 1]))
            / (1 + lambda_me * R_me * np.exp(-2.0j * k_xm[i] * x[i + 1]))
        ) * np.exp(2.0j * k_xm[i + 1] * x[i + 1])

        # 磁场反射系数计算
        lambda_mm = (epsilon[i] * k_xm[i + 1] - epsilon[i + 1] * k_xm[i]) / (
            epsilon[i] * k_xm[i + 1] + epsilon[i + 1] * k_xm[i]
        )
        R_mm = (
            (lambda_mm + R_mm * np.exp(-2.0j * k_xm[i] * x[i + 1]))
            / (1 + lambda_mm * R_mm * np.exp(-2.0j * k_xm[i] * x[i + 1]))
        ) * np.exp(2.0j * k_xm[i + 1] * x[i + 1])
    return R_me, R_mm


# 在此例子中,平面波的波速是光速,设入射波的波长为1mm
lamda_wave = 1
# 真空中波数k_0为2π/λ
k0 = 2 * np.pi / lamda_wave
# 介质片厚度为5λ
thickness = 5 * lamda_wave
# 划分薄层数
m = 100
# 场点坐标
x = np.arange(0, thickness + thickness / m, thickness / m)
# 入射角度,90个角度
max_theta = np.pi / 2
theta = np.arange(0, max_theta + max_theta / 90, max_theta / 90)
# 相对介电常数
epsilon = 4 + (2 - 0.1j) * ((1 - np.arange(0, 1, 1 / m)) ** 2)
epsilon = np.append(epsilon, 1)
# 相对磁导率
miu = np.ones(m) * (2 - 0.1j)
miu = np.append(miu, 1)

R_me, R_mm = analytical_solution(k0, m, x, theta, epsilon, miu)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(theta / max_theta * 90, abs(R_me))
plt.title("Electric field reflection coefficient")
plt.xlim(xmin=0, xmax=90)
plt.yticks(np.arange(0, 1.01, 0.1))  # 设置 y 轴刻度
plt.ylim(ymin=0, ymax=1)

plt.subplot(1, 2, 2)
plt.plot(theta / max_theta * 90, abs(R_mm))
plt.title("Magnetic field reflection coefficient")
plt.xlim(xmin=0, xmax=90)
plt.yticks(np.arange(0, 1.01, 0.1))  # 设置 y 轴刻度
plt.ylim(ymin=0, ymax=1)
plt.show()
