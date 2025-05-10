import ctypes
cpp = ctypes.CDLL('./hello.so')
from time import time 
import numpy as np 
if __name__ == '__main__': 

    aa = cpp.hehe1(); print('hehe1 void:', aa)
    aa = cpp.hehe2(); print('hehe2 int:', aa)
 
    t1  = time() 
    cm1 = 0
    for ii in range(10000): cm1 += ii + 1
    t2  = time() 
    print('sum via python loops takes', t2 - t1, 's; result = ', cm1)
    t1  = time() 
    cm2 = cpp.cm(10000)  
    t2  = time() 
    print('sum via c      loops takes', t2 - t1, 's; result = ', cm2)

    shape = (3,3)
    data  = np.empty(shape, dtype=float,order='C')
    print(data)
    #shape表示维度
    #dtype为数据类型
    #order为保存数据是按行还是按列
