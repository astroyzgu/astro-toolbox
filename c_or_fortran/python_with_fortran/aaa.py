from my_plugin import ffi
import numpy as np
import my_module 

@ffi.def_extern()
def hello_world():
    my_module.hello_world()

@ffi.def_extern()
def add_one(a, n, w):
    n = my_module.asarray(ffi, n, 1)[0]
    T = ffi.getctype(ffi.typeof(a).item)
    print(T, a, n) #, ffi.unpack(n,1))
    print(T.count, ffi.sizeof(a))
    a = np.frombuffer(ffi.buffer(a, n*ffi.sizeof(a)), np.dtype('float64'))
    a[:] += 1
    w = np.frombuffer(ffi.buffer(w, n*ffi.sizeof(w)), np.dtype('float64'))
    w[:] = a[:]
    #print(arr)
    #arr = my_module.asarray(ffi, a, n)
    #print(arr)

@ffi.def_extern()
def plot_one(x, nx, y, ny, string):
    nx = my_module.asarray(ffi, nx, 1)[0]
    x  = my_module.asarray(ffi, x, nx)
    ny = my_module.asarray(ffi, ny, 1)[0]
    y  = my_module.asarray(ffi, y, ny)
    print( ffi.getctype(ffi.typeof(string).item ), ffi.sizeof(string) )
    string_bytes = ffi.string(string[0:32]) #.encode('UTF-8') )
    print( string_bytes, type(string_bytes) )  
    string_ = str(string_bytes, encoding='UTF-8') 
    string_ = bytes.decode(string_bytes ) 
    string_ = string_.rstrip()
    print('input x:', x)
    print('input y:', y)
    print('input string:', string_ )
    my_module.plot_demo(x, y, string_)
