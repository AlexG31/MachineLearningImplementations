#encoding:utf8
import os, sys, pdb
import numpy as np

class CellBase(object):
    '''Base class for Neral Cells, with function and its derivitaves.'''
    def __init__(self):
        pass
    def f(self, x):
        return x
    def df(self, x):
        return 1.0
    def safe_f(self, x):
        if isinstance(x, list):
            if len(x) == 0:
                raise ValueError('Empty list cannot pass neural cell!')
        elif isinstance(x, tuple):
            if len(x) == 0:
                raise ValueError('Empty list cannot pass neural cell!')
        elif isinstance(x, np.ndarray):
            if x.size == 0:
                raise ValueError('Empty array cannot pass neural cell!')
        elif isinstance(x, int):
            pass
        elif isinstance(x, float):
            pass
        else:
            raise ValueError('Invalid x of type {}:{}'.format(type(x), x))
        return self.f(x)

    def safe_df(self, x):
        if isinstance(x, list):
            if len(x) == 0:
                raise ValueError('Empty list cannot pass neural cell!')
        elif isinstance(x, tuple):
            if len(x) == 0:
                raise ValueError('Empty list cannot pass neural cell!')
        elif isinstance(x, np.ndarray):
            if x.size == 0:
                raise ValueError('Empty array cannot pass neural cell!')
        elif isinstance(x, int):
            pass
        elif isinstance(x, float):
            pass
        else:
            raise ValueError('Invalid x of type {}:{}'.format(type(x), x))
        return self.df(x)

                

class SigmodCell(CellBase):
    def __init__(self, c):
        self.c = c

    def f(self, x):
        return 1.0 / (1 + np.exp(-self.c * np.array(x)))
    def df(self, x):
        fx = self.f(x)
        return self.c * fx * (1.0 - fx)

class ReluCell(CellBase):
    def f(self, x):
        x = np.array(x)
        x[x <= 0] = 0
        return x
    def df(self, x):
        x = np.array(x)
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
        

def Test():
    c = SigmodCell(1.0)
    # print c.safe_f(None)
    # print c.safe_df(None)


if __name__ == '__main__':
    Test()
