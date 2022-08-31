import ctypes
from time import time 
cpp = ctypes.CDLL('./hello.so')

if __name__ == '__main__': 

    t1  = time() 
    cm1 = 0
    for ii in range(10000): cm1 += ii + 1
    t2  = time() 
    print(t2 - t1)  

    t1  = time() 
    cm2 = cpp.cm(10000)  
    t2  = time() 
    print(t2 - t1)  

    print( cm1 )
    print( cm2 )
