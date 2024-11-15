import numpy as np


class LaplaceFEM:
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

    def compute_K(self):
        """计算K矩阵

        Args:
            无

        Returns:
            K矩阵
        """
        self.K = np.zeros((self.M, self.M))
        K_ii_vector = self.alpha / self.l + self.beta * self.l / 3
        K_ij_vector = -self.alpha / self.l + self.beta * self.l / 6
        np.fill_diagonal(
            self.K, np.append(K_ii_vector, 0) + np.insert(K_ii_vector, 0, 0)
        )
        # 填充主对角线下一条对角线
        np.fill_diagonal(self.K[1:], K_ij_vector)
        # 填充主对角线上一条对角线
        np.fill_diagonal(self.K[:, 1:], K_ij_vector)

        pass
