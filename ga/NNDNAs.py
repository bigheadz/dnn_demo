from DNA import IDNA, match
import numpy as np
import pandas as pd

from DenseDNAs import DenseDNAs


def NNDNAs(model_maker):
    class _NNDNAs(IDNA):
        """
        构成普通的NN， 或者DNN的网络结构
        其核心提供了对DNA进行操作的方法， 和一些基因算法中的超参数的设置和修改
        """

        def __init__(self, max_layer_n, n_mean_stds=None, denseDNAs=None):
            self.denseDNAs = denseDNAs if denseDNAs else [DenseDNAs(model_maker)() for _ in range(max_layer_n)]
            self.max_layer_n = max_layer_n
            self.n_mean_stds = n_mean_stds if n_mean_stds else [(10, 2) for _ in range(self.max_layer_n)]

        def create(self, max_pop_n=20):
            """
            :param n_mean_stds: 每一层的初始化的（均值， 方差)
            :return:返回所有的DNA的pandas列表， 每一行一个DNA， 对应一个层
            """
            dnas_list = [self.denseDNAs[i].create(num=max_pop_n, n_mean_std=self.n_mean_stds[i]) for i in
                         range(self.max_layer_n)]

            return pd.concat(dnas_list, axis=1, keys=self.get_dna_columns())

        def evolve(self, dnas, fitness):
            """
            优胜劣汰， 变异， 包括了有性生殖"
            :param dnas:  重要的是， dnas是一个下层dnas的列表， 列表的元素是某个结构的dnas
            :param fitness:
            :return:
            """
            # fitness转换成排名计分
            fitness = fitness.rank()
            fitness += 1  # 避免排位很低的， 一点机会都没有
            dnas['fitness'] = fitness

            # 优胜劣汰
            surviors = dnas.sample(frac=0.2, weights=dnas.fitness)

            # 有性生殖
            father = dnas.sample(frac=0.3, weights=dnas.fitness)
            mother = dnas.sample(frac=0.4, weights=dnas.fitness)
            father, mother = match(father, mother, childsize=int(len(fitness) * 0.8), father_weight=father.fitness,
                                   mother_weight=mother.fitness)

            children = self.sex_propagation(father, mother)
            dnas = surviors.append(children).reset_index(drop=True)
            return self.mutate(dnas)

            # TODO: 交换DNA
            # 变异
            # return self.mutate()

        def mutate(self, dnas):
            """
            变异
            :param dnas:
            :return: 变异后的dnas列表
            """
            dnas = [self.denseDNAs[i].mutate(dnas.loc[:, "layer{}".format(i)].copy()) for i in range(self.max_layer_n)]
            return pd.concat(dnas, axis=1, keys=self.get_dna_columns())

        def to_RNA(self, dna):
            def nans(shape, dtype=float):
                a = np.empty(shape, dtype)
                a.fill(np.nan)
                return a

            """翻译成RNA， RNA是可以表达， 并且也可以被保存的"""
            rna = [self.denseDNAs[i].to_RNA(dna["layer{}".format(i)]) for i in range(self.max_layer_n)]
            rna = pd.concat(rna, axis=1, keys=self.get_dna_columns())

            # 压缩rna, 避免因为0 10 10和10 0 10 而出现的两次重复的计算, 实际压缩就是把类似 [nan, 1, nan, 2, 3] 变成 [1, 2, 3, nan, nan]
            _nans = nans(rna.shape[1])
            rna = rna.apply(lambda row: np.hstack([row.dropna().values, _nans])[:rna.shape[1]], axis=1)
            return rna

        def to_model(self, rna_row, inputs):
            for i in range(self.max_layer_n):
                inputs = self.denseDNAs[i].to_dense(rna_row["layer{}".format(i)], inputs)

            return inputs

        def get_fitness(self, rna, inputs):
            for i in range(len(rna)):
                # print("get_fitness #{}".format(i))
                model = self.to_model(rna.iloc[i, :], inputs)
                print(model)
                # TODO:
            return pd.Series(np.random.randn(len(rna)))

        def sex_propagation(self, father, mother):
            # 用denseDNA对每个Dense进行sex_propagation
            children = [
                self.denseDNAs[i].sex_propagation(father.loc[:, "layer{}".format(i)],
                                                  mother.loc[:, "layer{}".format(i)])
                for i in range(self.max_layer_n)]
            return pd.concat(children, axis=1, keys=self.get_dna_columns())

        def get_dna_columns(self):
            return ["layer{}".format(i) for i in range(self.max_layer_n)]

    return _NNDNAs


if __name__ == '__main__':
    nnDNAs = NNDNAs(print)(5)
    dnas = nnDNAs.create()
    print(dnas)
    # dnas = nnDNAs.evolve(dnas, fitness)
    for i in range(5):
        dnas = nnDNAs.evolve(dnas, nnDNAs.get_fitness(nnDNAs.to_RNA(dnas)))
