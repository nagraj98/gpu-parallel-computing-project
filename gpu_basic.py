from numba import cuda, vectorize, jit
import numpy as np

#for measuring exact time
from timeit import default_timer as timer

#Normal function to run on CPU
def cpufunc(a):
    for i in range(a.size):
        a[i] += 1

#function optimised for GPU

#Way 1 --- not working
@cuda.jit(target = "cuda")
def gpufunc1(a):
    for i in range(a.size + 1):
        a[i] += 1

#Way 2 -- using vectorize
@vectorize(['float64(float64)'],target="cuda")
def gpufunc2(a):
    return a+1

if __name__=="__main__":
    n = 10000000
    a = np.ones(n,dtype = np.float64)
    print(a.size)

    start = timer()
    cpufunc(a)
    print("processing time without GPU:",timer()-start)
    # print(a)

    start = timer()
    gpufunc2(a)
    print("processing time with GPU:", timer()-start)
    # print(a)