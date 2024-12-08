import numpy as np


class LaplaceFEM:
    """
    一维边值问题数值解通用有限元程序
    """

    def __init__(self, M, alpha, beta, f, l):
        """读入M,然后读入相关参数

        Args:
            M:将解域分割成M个小子域
            alpha:每个子域单元内的物理参数
            beta:每个子域单元内的物理参数
            f:每个子域中的已知源或者激励函数
            l:每个子域单元长度

        returns:
            无
        """
        self.M = M
        self.reset_params(alpha, beta, f, l)

    def reset_params(self, alpha, beta, f, l):
        """重置参数
        Args:
            alpha:每个子域单元内的物理参数
            beta:每个子域单元内的物理参数
            f:每个子域中的已知源或者激励函数
            l:每个子域单元长度

        returns:
            无
        """
        if (
            alpha.shape[0] != self.M
            or beta.shape[0] != self.M
            or f.shape[0] != self.M
            or l.shape[0] != self.M
        ):
            raise ValueError("输入长度不匹配")
        self.alpha = alpha
        self.beta = beta
        self.f = f
        self.l = l
        # 需要计算的矩阵和向量
        self.K = np.zeros((self.M + 1, self.M + 1), dtype=complex)
        self.b = np.zeros(self.M + 1, dtype=complex)

    def boundary_condition(self, params):
        """读入边界条件,判断是否为狄利克雷型

        Args:
            params:边界条件的参数值,如果是狄利克雷型则为p,否则为gama和q
            is_dirichlet:是否为狄利克雷型,默认为是

        Returns:
            无
        """
        self.p = 0
        self.gama = 0
        self.q = 0
        [self.gama, self.q, self.p] = params

    def print_input(self):
        """打印出输入数据

        Args:
            无

        Returns:
            无
        """
        print("分割成%d" % self.M, "个子域单元")
        for i in range(self.M):
            print("第", i, "个单元")
            print("alpha_", i, "=", self.alpha[i])
            print("beta_", i, "=", self.beta[i])
            print("f_", i, "=", self.f[i])
            print("l_", i, "=", self.l[i])
            print("输入狄利克雷边界条件: p = ", self.p)
            print("输入诺曼型边界条件: gama = ", self.gama, "q = ", self.q)

    def __compute_K_and_b__(self):
        """计算K矩阵和b向量

        Args:
            无

        Returns:
            无
        """
        # print("计算K矩阵和b向量")
        K_ii_vector = self.alpha / self.l + self.beta * self.l / 3
        K_ij_vector = -self.alpha / self.l + self.beta * self.l / 6

        # 加入诺曼边界条件修正
        np.fill_diagonal(
            self.K, np.append(K_ii_vector, 0) + np.insert(K_ii_vector, 0, 0)
        )
        self.K[-1, -1] = self.K[-1, -1] + self.gama
        # 主对角线
        self.a = np.append(K_ii_vector, 0) + np.insert(K_ii_vector, 0, 0)
        self.a[-1] = self.a[-1] + self.gama
        # 副对角线
        self.c = K_ij_vector
        # 填充主对角线下一条对角线
        np.fill_diagonal(self.K[1:], K_ij_vector)
        # 填充主对角线上一条对角线
        np.fill_diagonal(self.K[:, 1:], K_ij_vector)

        b_i = self.f * self.l / 2
        self.b = np.append(b_i, 0) + np.insert(b_i, 0, 0)
        self.b[-1] = self.b[-1] + self.q
        self.b.reshape(-1, 1)
        self.b = self.b - self.p * self.K[:, 0]

        self.b[0] = self.p
        # 狄利克雷边界条件添加
        self.K[0, :] = 0  # 重置第一行
        self.K[:, 0] = 0  # 重置第一列
        self.K[0, 0] = 1  # 狄利克雷条件,x=0处为0

        self.fai_e = np.zeros(self.M + 1, dtype=complex)
        self.a[0] = 1
        self.c[0] = 0

    def solve(self):
        """先计算矩阵,然后使用高斯消元法计算方程组的解,得到偏微分方程计算结果

        Args:
            无

        Returns:
            多项式系数
        """
        # print("使用有限元方法计算偏微分方程")
        # 计算K矩阵和向量
        self.__compute_K_and_b__()
        # 高斯消元法计算拟合多项式系数

        try:
            self.fai_e[0] = self.p
            a = self.a
            b = self.b
            for i in range(self.M):
                a[i + 1] = a[i + 1] - (self.c[i] ** 2) / a[i]
                b[i + 1] = b[i + 1] - self.c[i] * b[i] / a[i]

            self.fai_e[-1] = b[-1] / a[-1]
            for i in range(self.M):
                self.fai_e[-1 - 1 - i] = (
                    b[-1 - 1 - i] - self.c[-1 - i] * self.fai_e[-1 - i]
                ) / a[-1 - 1 - i]
        except np.linalg.LinAlgError:
            print("该矩阵不可逆")

    def forward(self, x):
        """结合计算得到的拟合多项式系数,得到偏微分方程的解

        Args:
            x: 输入要求解的场点坐标,行向量

        Returns:
            坐标对应场点的值
        """
        fai = np.zeros_like(x, dtype=complex)
        x_e = np.zeros((self.M + 1, 1))
        for i in range(self.M + 1):
            x_e[i][0] = np.sum(self.l[0:i])
        l_e = self.l.reshape(-1, 1)
        N_e1 = np.zeros_like(x_e)
        N_e2 = np.zeros_like(x_e)
        fai_array = self.fai_e.reshape(1, -1)
        for i in range(x.shape[0]):
            # N_e1是一个列向量
            if x[i] < x_e.max():
                N_e1[:-1, :] = (
                    (x_e[1:, :] - x[i])
                    / l_e
                    * (x[i] < x_e[1:, :])
                    * (x[i] >= x_e[:-1, :])
                )
                N_e2[1:, :] = (
                    (x_e[1:, :] - x[i])
                    / l_e
                    * (x[i] < x_e[1:, :])
                    * (x[i] >= x_e[:-1, :])
                )
            fai[i] = np.dot(fai_array, N_e1) + np.dot(fai_array, N_e2)
        else:
            fai[i] = self.fai_e[-1]
        return fai
