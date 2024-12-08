import numpy as np
from laplace_fem import *
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
    for i in range(m - 1):
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
x = np.arange(thickness / m, thickness + thickness / m, thickness / m)  # 场点坐标
max_theta = np.pi / 2
# theta = np.arange(0, max_theta + max_theta / 90, max_theta / 90)  # 入射角度,90个角度
theta = np.array([0])

epsilon = 4 + (2 - 0.1j) * ((1 - np.arange(1 / m, 1, 1 / m)) ** 2)
miu = np.ones_like(epsilon) * (2 - 0.1j)
epsilon = np.append(epsilon, 1)  # 相对介电常数
miu = np.append(miu, 1)  # 相对磁导率

# 解析解求解
analytical_Rme, analytical_Rmm = analytical_solution(k0, m, x, theta, epsilon, miu)
print("解析解的反射系数: ", abs(analytical_Rme))


# 有限元方法求解电场
m = 150
x = np.arange(0, thickness + thickness / m, thickness / m)  # 场点坐标
epsilon = 4 + (2 - 0.1j) * ((1 - np.arange(1 / m, 1 + 1 / m, 1 / m)) ** 2)
miu = np.ones_like(epsilon) * (2 - 0.1j)

E0 = 1  # 初始电场值,随便设一个
H0 = 1

l = thickness / m * np.ones(m)  # 每个分段区域长度
alpha_e = 1 / miu
f_e = np.zeros_like(alpha_e)

fem_e = None
E_l = np.ones((theta.shape[0], m + 1)) * 1j
numerical_Rme = np.ones(theta.shape[0]) * 1j
for i in range(theta.shape[0]):
    # print("求解第", i, "个角度")
    beta_e = -(k0**2) * (epsilon - 1 / miu * (np.sin(theta[i]) ** 2))
    gama_e = k0 * np.cos(theta[i]) * 1j
    q_e = (
        np.cos(theta[i]) * 2j * k0 * E0 * np.exp(1j * k0 * np.cos(theta[i]) * thickness)
    )
    if fem_e == None:
        fem_e = LaplaceFEM(m, alpha_e, beta_e, f_e, l)
    else:
        fem_e.reset_params(alpha_e, beta_e, f_e, l)
    # 输入边界条件类型
    fem_e.boundary_condition([gama_e, q_e, 0])
    # 求解
    fem_e.solve()
    # 计算电场以及反射系数
    E_l[i] = fem_e.forward(x)
    numerical_Rme[i] = E_l[i][-1] - E0 * np.exp(1j * k0 * thickness * np.cos(theta[i]))
    print("电场数值解的反射系数: ", abs(numerical_Rme[i]))


# # 有限元方法求解磁场
# alpha_m = 1 / epsilon
# f_m = np.zeros_like(alpha_m)

# fem_m = None
# H_l = np.ones((theta.shape[0], m + 1)) * 1j
# numerical_Rmm = np.ones(theta.shape[0]) * 1j
# for i in range(theta.shape[0]):
#     # print("求解第", i, "个角度")
#     beta_m = -(k0**2) * (miu - 1 / epsilon * (np.sin(theta[i]) ** 2))
#     gama_m = k0 * np.cos(theta[i]) * 1j
#     q_m = (
#         2j * k0 * np.cos(theta[i]) * H0 * np.exp(1j * k0 * np.cos(theta[i]) * thickness)
#     )
#     if fem_m == None:
#         fem_m = LaplaceFEM(m, alpha_m, beta_m, f_m, l)
#     else:
#         fem_m.reset_params(alpha_m, beta_m, f_m, l)
#     # 输入边界条件类型
#     fem_m.boundary_condition([gama_m, q_m, 0])
#     # 求解
#     fem_m.solve()
#     # 计算磁场以及反射系数
#     H_l[i] = fem_m.forward(x)
#     numerical_Rmm[i] = (
#         H_l[i][-1] - H0 * np.exp(1j * k0 * thickness * np.cos(theta[i]))
#     ) / (H0 * np.exp(-1j * k0 * thickness * np.cos(theta[i])))
#     # print("磁场数值解的反射系数: ", abs(numerical_Rmm[i]))


# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(
#     theta / max_theta * 90,
#     abs(analytical_Rme),
#     label="Analytical Solution",
#     color="blue",
#     linestyle="-",
# )
# plt.plot(
#     theta / max_theta * 90,
#     abs(numerical_Rme),
#     label="Numerical Solution",
#     color="red",
#     linestyle="--",
# )
# plt.legend(loc="upper right")
# plt.title("Electric field reflection coefficient")
# plt.xlim(xmin=0, xmax=90)
# plt.yticks(np.arange(0, 1.01, 0.1))  # 设置 y 轴刻度
# plt.ylim(ymin=0, ymax=1)

# plt.subplot(1, 2, 2)
# plt.plot(
#     theta / max_theta * 90,
#     abs(analytical_Rmm),
#     label="Analytical Solution",
#     color="blue",
#     linestyle="-",
# )
# plt.plot(
#     theta / max_theta * 90,
#     abs(numerical_Rmm),
#     label="Numerical Solution",
#     color="red",
#     linestyle="--",
# )
# plt.legend(loc="upper right")
# plt.title("Magnetic field reflection coefficient")
# plt.xlim(xmin=0, xmax=90)
# plt.yticks(np.arange(0, 1.01, 0.1))  # 设置 y 轴刻度
# plt.ylim(ymin=0, ymax=1)
# plt.show()
