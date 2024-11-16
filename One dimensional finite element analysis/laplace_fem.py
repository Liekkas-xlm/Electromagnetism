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
        if alpha.length != M or beta.length != M or f.length != M or l.length != M:
            raise ValueError("输入长度不匹配")
        self.alpha = alpha
        self.beta = beta
        self.f = f
        self.l = l
        # 需要计算的矩阵和向量
        self.K = np.zeros((self.M + 1, self.M + 1))
        self.b = np.zeros(self.M + 1)

    def boundary_condition(self, params, is_dirichlet=True):
        """读入边界条件,判断是否为狄利克雷型

        Args:
            params:边界条件的参数值,如果是狄利克雷型则为p,否则为gama和q
            is_dirichlet:是否为狄利克雷型,默认为是

        Returns:
            无
        """
        self.is_dirichlet = is_dirichlet
        self.p = 0
        self.gama = 0
        self.q = 0
        if is_dirichlet:
            self.p = params
        else:
            [self.gama, self.q] = params

    def print_input(self):
        """打印出输入数据

        Args:
            无

        Returns:
            无
        """
        print("分割成%d" % self.M + "个子域单元")
        for i in range(self.M):
            print("第" + i + "个单元")
            print("alpha_" + i + "=" + self.alpha[i])
            print("beta_" + i + "=" + self.beta[i])
            print("f_" + i + "=" + self.f[i])
            print("l_" + i + "=" + self.l[i])
        if self.is_dirichlet:
            print("输入狄利克雷边界条件: p=" + self.p)
        else:
            print("输入诺曼型边界条件: gama=" + self.gama + " q=" + self.q)

    def compute_K_and_b(self):
        """计算K矩阵和b向量

        Args:
            无

        Returns:
            无
        """
        K_ii_vector = self.alpha / self.l + self.beta * self.l / 3
        K_ij_vector = -self.alpha / self.l + self.beta * self.l / 6

        # 加入诺曼边界条件修正
        np.fill_diagonal(
            self.K, np.append(K_ii_vector, 0) + np.insert(K_ii_vector, 0, 0) + self.gama
        )
        # 填充主对角线下一条对角线
        np.fill_diagonal(self.K[1:], K_ij_vector)
        # 填充主对角线上一条对角线
        np.fill_diagonal(self.K[:, 1:], K_ij_vector)

        b_i = self.f * self.l / 2 + self.q
        self.b = np.append(b_i, 0) + np.insert(b_i, 0, 0)

        # 狄利克雷边界条件添加
        self.K[0, :] = 0  # 重置第一行
        self.K[:, 0] = 0  # 重置第一列
        self.K[0, 0] = 1  # 狄利克雷条件,x=0处为0

        self.b.reshape(-1, 1)
        self.b = self.b - self.p * self.K[:, 1]
        self.b[0] = self.p

    def gaussian_elimination(self):
        """高斯消元法计算方程组的解,返回偏微分方程计算结果

        Args:
            无

        Returns:
            多项式系数
        """
        self.fai_e = np.zeros(self.M + 1)
        try:
            self.fai_e = np.dot(self.b, np.linalg.inv(self.K))
        except np.linalg.LinAlgError:
            print("该矩阵不可逆")

    def forward(self, x):
        """结合计算得到的拟合多项式系数,得到偏微分方程的解

        Args:
            x: 输入要求解的场点坐标,行向量

        Returns:
            坐标对应场点的值
        """
        x_e = np.zeros(self.M + 1)
        for i in range(self.M):
            x_e[i + 1] = np.sum(self.l[0:i])
        x_colum = x.reshape(-1, 1)

        N_e1 = (x_e[1:] - x_colum) / self.l
        N_e2 = (x_e[:-1] - x_colum) / self.l

        # φ是一个列向量
        fai = np.dot(N_e1, self.fai_e[0:-1]) + np.dot(N_e2, self.fai_e[1:])
        return fai
