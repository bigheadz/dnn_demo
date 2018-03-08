# from DNA import TAnt
import abc
import pandas as pd
import numpy as np

from DNA import IDNA, match
from matplotlib import pyplot as plt

def DenseDNAs(model_maker):

    class _DenseDNAs(IDNA):
        """
        一个Dense的DNA表达， 能够翻译成RNA， 转换成对应的Dense结构
        layer.n         是个连续的变量，有layer.n.r决定， 有几率突变到0， 就是这一层被隐藏， 不起作用
        layer.n.r       随机+-一个随机值， 并且可能突变到0
        layer.act       激活函数, 取值为0, 1, 2, 3, 4， 对应None, relu, sigmoid, tanh, softmax
        layer.act.r     激活函数变化不具有连续性和惯性， 所以突变方式应该为突变， 并且突变后， 有一定几率保持一段时间, 所以突变后， 一旦某个值生存下来， 这一层暂时应该保持不变
        layer.norm      是否更一个norm
        layer.norm.r    让norm改变后被锁定一段时间
        """

        # TODO 如何让机器自己学习到， 有些突变经常是无效， 并且有害的， 比如norm会导致计算能力的浪费， sigmoid会导致整个网络反馈变慢
        # TODO 首先对于网络的各种突变可能性， 我们可以通过一些参数来手动调节

        def __init__(self, act_init_table=None, act_mutate_table=None, act_lock_rate=0.1, act_lock_turn=5,
                     norm_init_table=None, norm_lock_turn=4, norm_lock_rate=0.1, rna_gap=16, layer_mutate_to_0=0.01):

            # dna 到 rna的进度， 比如16的话， 就是以 1/16为精度来调整rna的值， 这样可以减少rna映射出来的数量,
            # 避免小的变化而重新训练网络, 这个值进度越高， rna映射出来的数量越多, None的时候， 不进行变化
            self.layer_mutate_to_0 = layer_mutate_to_0
            self.rna_gap = rna_gap
            self.norm_lock_rate = norm_lock_rate
            self.norm_lock_turn = norm_lock_turn
            self.act_lock_turn = act_lock_turn

            self.act_lock_rate = act_lock_rate

            self.norm_init_table = None
            self.set_norm_init_table(norm_init_table)

            self.act_mutate_table = None
            self.set_act_mutate_table(act_mutate_table)

            self.act_init_table = None
            self.set_act_table(act_init_table)

        def set_rna_gap(self, rna_gap=16):
            self.rna_gap = rna_gap

        def set_norm_init_table(self, norm_init_table=None):
            self.norm_init_table = np.array([0.7, 0.3]) if norm_init_table is None else np.array(norm_init_table).astype(
                np.float)
            _sum = self.norm_init_table.sum()
            if _sum == 0:
                self.norm_init_table = 0.5
            else:
                self.norm_init_table /= _sum

            return self.norm_init_table.cumsum()

        def set_act_mutate_table(self, act_mutate_table=None):
            # None, relu, sigmoid, tanh, softmax
            self.act_mutate_table = np.array([[0.0, 0.7, 0.1, 0.1, 0.1],
                                              [0.7, 0.0, 0.1, 0.1, 0.1],
                                              [0.1, 0.1, 0.0, 0.7, 0.1],
                                              [0.1, 0.1, 0.7, 0.0, 0.1],
                                              [0.3, 0.3, 0.2, 0.2, 0.0]]
                                             ) if act_mutate_table is None else act_mutate_table
            _sum = self.act_mutate_table.sum(axis=1)
            self.act_mutate_table /= _sum.reshape(-1, 1)
            self.act_mutate_table = pd.DataFrame(self.act_mutate_table.T)
            # print("act_mutate_table\n", self.act_mutate_table)

        def set_act_table(self, act_table=None):
            act_table = np.array([0.4, 0.3, 0.1, 0.1, 0.1]) if act_table is None else act_table
            # 归一化
            tsum = act_table.sum()
            if tsum == 0:
                act_table = 1 / len(act_table)
            else:
                act_table = act_table / act_table.sum()
            # 转换成可查表的比例
            self.act_init_table = act_table.cumsum()

        def create(self, data=None, num=None, n_mean_std=(5, 2)):
            """
            :param n_mean_std: 确认初始化的时候， 这一层dense的分布曲线
            :param act_table:
            :param data:
            :param num:
            :param n_min_max: 估计的layer_n的最小值和最大值
            :return: 返回一个Dense对应的DNA的pandas列表
            """
            assert not (data is None and num is None)

            if data is None:
                n_mean, n_std = n_mean_std
                dna = pd.DataFrame(np.random.randn(num, 6) if data is None else data,
                                   columns=["layer", "layer_r", "act", "act_r", "norm", "norm_r"])
                dna.layer = np.random.randn(num, 1) * n_std + n_mean
                dna.act = self.seed_to_act(num=num)
                dna.act_r = self.act_lock_turn
                dna.norm = np.where(np.random.rand(num, 1) < self.norm_init_table[0], 0, 1)
                dna.norm_r = self.norm_lock_turn
            else:
                dna = pd.DataFrame(np.random.randn(num, 6) if data is None else data,
                                   columns=["layer", "layer_r", "act", "act_r", "norm", "norm_r"])

            return dna

        def seed_to_act(self, num=None, seed=None):
            def get_r(i):
                if i == 4:
                    return 4
                return np.where(seed < self.act_init_table[i], i, get_r(i + 1))

            seed = np.random.rand(num, 1) if seed is None else seed
            r = get_r(0)
            # print(r)
            return r

        def mutate(self, dna):
            # layer
            # 有一定的几率， dnar重新回到N(0, 1)区间, 作用就是踩刹车
            dna.layer_r = np.where(np.random.rand(*dna.layer_r.shape) > 0.1,
                                   dna.layer_r + np.random.randn(*dna.layer_r.shape),
                                   np.random.randn(*dna.layer_r.shape))
            dna.layer = dna.layer + dna.layer_r
            # 有一定的几率， layer会突变到-1， 就是说一段时间, 或者一层都是被屏蔽的状态
            dna.layer = np.where(np.random.rand(*dna.layer.shape) < self.layer_mutate_to_0, -0.5, dna.layer)

            # act
            seed = np.random.rand(*dna.act_r.shape)
            change_acts_indies = (dna.act_r <= 0) & (seed < self.act_lock_rate)

            def mutate_act(sub_act):
                return sub_act.map(
                    lambda act: self.act_mutate_table.sample(weights=self.act_mutate_table.iloc[:, act]).index.tolist()[0])

            dna.loc[change_acts_indies, "act"] = mutate_act(dna.loc[change_acts_indies, "act"])  # 目前还是有一定的几率， 自己变成给自己的
            dna.loc[change_acts_indies, "act_r"] = self.act_lock_turn + 1
            dna.loc[dna.act_r > 0, "act_r"] -= 1  # 因为之前被锁定的已经+1, >0的条件避免act_r没有改变的情况

            # norm
            seed = np.random.rand(*dna.norm.shape)
            change_norm_indies = (dna.norm_r <= 0) & (seed < self.norm_lock_rate)
            dna.loc[change_norm_indies, "norm"] = ~dna.loc[change_norm_indies, "norm"] + 2  # 取反操作
            dna.loc[change_norm_indies, "norm_r"] = self.norm_lock_turn + 1
            dna.loc[dna.norm_r > 0, "norm_r"] -= 1
            return dna

        def sex_propagation(self, father, mother):
            """让father和mother来随机交换DNA"""
            # 整个DNA组作为一个整体， 不再拆分了
            seed = np.random.rand(len(father))
            sf = np.where(seed > 0.5, 0, 1)
            sm = np.where(seed > 0.5, 1, 0)
            father = father.apply(lambda s: s * sf)
            mother = mother.apply(lambda s: s * sm)
            return father.reset_index(drop=True) + mother.reset_index(drop=True)

        def to_RNA(self, dna):
            rna = pd.DataFrame(dna.iloc[:, [0, 2, 4]], columns=["layer", "act", "norm"])
            # 这个是普通的算法， 其实，应该dna较小的时候， 翻译出来rna精确到个位数， 但是dna较大的时候， 精确到10位数就可以了
            rna.layer = (rna.layer * 10 + 0.5).astype(np.int)
            if self.rna_gap:  # 减小样本数量, 通过较大数据跳值的方法， 比如rna_gap=16的时候， 34->32
                gap = np.ceil(rna.layer / self.rna_gap)
                gap[gap == 0] = 1
                rna.layer = (np.floor(rna.layer / gap) * gap).astype(np.int)
            # 为了后续压缩方便， 一旦出现layer<=0的情况， 就直接
            rna.loc[rna.layer <= 0, :] = np.nan
            return rna

        def to_dense(self, rna_row, inputs):
            """明显， 就一行一行的来建model就可以了"""
            if np.isnan(rna_row.layer):
                return inputs
            return model_maker(inputs, rna_row.layer, rna_row.act, rna_row.norm)

    return _DenseDNAs

