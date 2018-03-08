import abc
import pandas as pd
import numpy as np


class IDNA:
    def create(self):
        pass

    def mutate(self, dna):
        """变异， 可能包括了有性生殖"""
        pass

    def to_RNA(self, dna):
        """翻译成RNA， RNA是可以表达， 并且也可以被保存的"""
        return None

    def sex_propagation(self, father, mother):
        return None


def match(father, mother, childsize=None, father_weight=None, mother_weight=None):
    """让father和mother配对，无weight"""
    if childsize is None:
        childsize = max(len(father), len(mother))

    mother = mother.sample(childsize, replace=True, weights=mother_weight)
    father = father.sample(childsize, replace=True, weights=father_weight)

    return father, mother
