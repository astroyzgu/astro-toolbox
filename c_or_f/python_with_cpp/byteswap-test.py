import numpy as np 

#------------------------------------------------------------------------
filename = './test.b'
datain = np.array([8.0, 256.0, 21.415]); print(datain.dtype)  
datain.tofile(filename) 
datain = np.array([8, 256, 124, 125, 120]); print(datain.dtype) 
datain.tofile(filename) 

f = open(filename,'rb')   
dataout = np.fromfile(f, dtype = np.int64, count = 1) 
print(dataout)
dataout = np.fromfile(f, dtype = np.int64, count = 2) 
print(dataout)
f.close() 




print('#---------------------------------------------------------------------')
import os 
f = open('test', 'w+')
f.write('0123456789') 
f.close() 
f = open('test', 'rb+')
f.seek( 3, 0); print( f.read(1), f.tell()  )
f.seek( 2, 2); print( f.read(1), f.tell() )

print('#---------------------------------------------------------------------')

print(os.SEEK_END) # 2 代表从文件末尾算起
print(os.SEEK_SET) # 0 代表从文件开头开始算起

# -32768~+32767
# b1111111
print( bin(255), int('0b11111111', 2) )
A = np.array([1, 100, 300, 8755, 255 ], dtype=np.int16) # 1个字节（byte）<==> 8个比特(bit), -2^7-1 ~ + 2^7  
A = np.array([int('0x1122', 16), int('0x2211', 16) ], dtype=np.int16)  
#---------------------------------------
# 大端 hi
print(A)
print(list(map(bin, A)))
print(A.byteswap(inplace=True))
print(list(map(bin, A)))
