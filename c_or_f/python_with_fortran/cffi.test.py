import cffi  
import sys
import numpy as np
ffi = cffi.FFI()

###---- NULL
print(ffi.NULL) # <cdata 'void *' NULL>
### C语言代码的整数和浮点值映射到Python的常规
# => init, long, float, double 和 char

#Create the dictionary mapping ctypes to np dtypes.                                                                                                  
import numpy as np
ctype2dtype = {}
# Floating point types
ctype2dtype['float'] =  np.dtype('f4')
ctype2dtype['double'] = np.dtype('f8')

# Integer types
for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = '%s%d_t' % (prefix, 8 * (2**log_bytes))
        dtype = '%s%d' % (prefix[0], 2**log_bytes)
        # print( ctype )
        # print( dtype )
        ctype2dtype[ctype] = np.dtype(dtype)
for k, v in ctype2dtype.items():
        print(k, v)
print('#########################')

# testing string C语言类型必须是指针或数组 
string= ffi.new("char *", b'1') # 单个字符
string= ffi.new("char[]", b'C string, ending')   # 定义字符串 
                           #1234567890123456
string = b'\xc5C\xe2\xfe\x7f'
print(string) # <cdata 'char[]' owning 17 bytes>
print(ffi.string(string) ) #b'C string, ending'
print(ffi.string(string).decode("UTF-8") ) #b'C string, ending'
print(str(string))
print('#########################')
# 单个整形和浮点型 



# 一维数组
a = ffi.new("int[]", 3)  # 三元素整型数组
# 也可以这样定义 a   = ffi.new("int[3]", [1,2,3])
print(len(a), a, type(a)) # 返回值a为<cdata>指针
a[0]=-1
a[2]=10        # 不能使用负索引，不能使用步长
print(list(a))          # 转为list（适用小型数组）
print(ffi.unpack(a, 3)) # 转为list（适合大数据量）
# 推荐直接转为numpy.array
# 如下： 

# CFFI数组和Numpy相互转换
print('##### CFFI数组和Numpy相互转换')
a = np.arange(5, dtype=np.int64) # np.int16 == c_int, 
                                 # np.int64 == c_long_long
print('创建ndarray数组a:', a, '  dtype = %s'%a.dtype)
print('#### ndarray to CFFI数组')
b   = ffi.from_buffer("long*", a) 
b   = ffi.cast("long*", a.ctypes.data) 
b   = ffi.new( "long[5]", [1,2,3,4,5]) # <cdata 'double[3]' owning 24 bytes>
arr = ffi.unpack(b, 5)
c = np.frombuffer(ffi.buffer(b, ffi.sizeof(b)), np.dtype('int64'))
print(b, c)
# 虽然a和b的地址不一样 
print(id(a), id(b)) 
print(ffi.sizeof(b))
# 但a和b在内存上有联系，改变b会直接影响a,改变a却不会改变b
print(a, b, ffi.unpack(b, 5), arr)
b[3] = 5 
print(a, b, ffi.unpack(b, 5), arr)
a[3] = 6 
print(a, b, ffi.unpack(b, 5), arr)

#### CFFI数组 to ndarray 
print('#### CFFI数组 to ndarray') 

arr   = ffi.new( "double[3]", [1.0,2.0,3.0]) # <cdata 'double[3]' owning 24 bytes>
T     = ffi.getctype( ffi.typeof(arr).item)
print(arr, 'ctype = %s'%T, len(arr) )
print('直接转为list：', list(arr))
shape = ffi.typeof(arr).length
arr[1] = -1
print(ffi.unpack(arr, 3), ', ', arr, ', length = %s'%shape)  
#print(ffi.typeof(arr), ffi.sizeof(arr)) #, sys.getsizeof(arr[1])) 
arr = np.frombuffer(ffi.buffer(arr, ffi.sizeof(arr)),  np.dtype('float64'))
print(arr)
print(arr.dtype, arr.astype(np.int64) )

##### 二维数组 

