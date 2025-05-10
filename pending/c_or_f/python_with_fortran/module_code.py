from my_plugin import ffi
import numpy as np
from ctype2dtype import asarray, asstring
import userdefined  

# user defined function 
#@ffi.def_extern() 
#def testdata(x, y): 

# extern defition for test 
@ffi.def_extern() 
def sharedata32(x, nx):
    nx = asarray(ffi, nx, 1)[0]
    x  = asarray(ffi, x, nx)
    x  = x + 1
    return x

@ffi.def_extern() 
def f2py_test32(x, nx, char0):
    nx = asarray(ffi, nx, 1)[0]
    x  = asarray(ffi, x, nx)
    print('in python:', x[:2], nx, x.dtype)
    char0   = asstring(ffi, char0, length = 256)
    print('in python:', char0, type(char0), len(char0)  )
    print('fortran has transit data into python') 
    print('In this way, the memorey is shared in python and fortran')
    print('--------------------------------------------')
    print('In python,  if assign [0.987654321, 987654321.0] to array') 
    print('In fortran, the value is also changed since it occupies the same memory') 
    x[0] = 0.987654321
    x[1] = 987654321.0
    print('in python:', x[:2], nx, x.dtype)

@ffi.def_extern() 
def f2py_test64(x, nx, char0):
    nx = asarray(ffi, nx, 1)[0]
    x  = asarray(ffi, x, nx)
    print('in python:', x[:2], nx, x.dtype)
    char0   = asstring(ffi, char0, length = 256)
    print('in python:', char0, type(char0), len(char0)  )
    print('fortran has transit data into python') 
    print('In this way, the memorey is shared in python and fortran')
    print('--------------------------------------------')
    print('In python,  if assign [0.987654321, 987654321.0] to array') 
    print('In fortran, the value is also changed since it occupies the same memory') 
    x[0] = 0.987654321
    x[1] = 987654321.0
    print('in python:', x[:2], nx, x.dtype)

