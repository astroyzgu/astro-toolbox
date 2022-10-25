#Create the dictionary mapping ctypes to np dtypes.
import numpy as np
ctype2dtype = {}
# Integer types
for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = '%s%d_t' % (prefix, 8 * (2**log_bytes))
        dtype = '%s%d' % (prefix[0], 2**log_bytes)
        ctype2dtype[ctype] = np.dtype(dtype)
# Floating point types
ctype2dtype['int']    = np.dtype('int32')
ctype2dtype['float']  = np.dtype('f4')
ctype2dtype['double'] = np.dtype('f8')
#print(ctype2dtype)
dtype2ctype = {} 
for k, v in ctype2dtype.items(): 
    dtype2ctype[v] = k
#print(dtype2ctype)

def cast2ctype(ffi, arr): 
    ctype = dtype2ctype[arr.dtype]
    arr_ctype = ffi.cast('%s *'%ctype, arr.ctypes.data)
    return arr_ctype

def asstring(ffi, ptr, length = 256): 
    string_bytes = ffi.string( ptr[0:length])
    string_ = str( string_bytes, encoding='UTF-8') 
    string_ = string_.rstrip()
    # print(type(string_) )
    return string_ 

def asarray(ffi, ptr, shape, **kwargs):
    length = np.prod(shape)
    # Get the canonical C type of the elements of ptr as a string.
    T = ffi.getctype(ffi.typeof(ptr).item) 
    # print(T)
    # print( T, shape, length, ffi.sizeof(T) )
    if T not in ctype2dtype:
        raise RuntimeError("Cannot create an array for element type: %s" % T) 
    a = np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(T)), ctype2dtype[T])\
          .reshape(shape, **kwargs)
    print(type(a))
    return a
