from numba import vectorize, jit, cuda
import numpy as np
from timeit import default_timer as timer


def cpu_func(a):
    for i in range(10000000):
        a[i] += 1

# This method gives ValueError
# @cuda.jit
# def gpu_func(a):
#     for i in range(10000000):
#         a[i] += 1


@vectorize(['float64(float64)'], target='cuda')
def gpu_func_1(a):
    return a+1


if __name__ == '__main__':
    n = 10000000
    a = np.ones(n, dtype=np.float64)

    #start = timer()
    #cpu_func(a)
    #print('Processing time Without GPU: ', timer()-start)

    start = timer()
    gpu_func_1(a)
    print('Processing time With GPU: ', timer()-start)

