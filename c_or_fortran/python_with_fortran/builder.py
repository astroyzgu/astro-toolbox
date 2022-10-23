import cffi  
ffibuilder = cffi.FFI() # 声明了外部函数接口(FFI)对象

with open('header_code.c') as file_obj:
     content = file_obj.read() 
header = content

with open('module_code.py') as file_obj:
     content = file_obj.read()
module = content


with open("plugin.h", "w") as f:
    f.write(header)

ffibuilder.embedding_api(header)
ffibuilder.set_source("my_plugin", r'''
    #include "plugin.h"
''')

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target= "libplugin.dylib", verbose=True)
ffibuilder.compile(target= "libplugin.so", verbose=True)

#import os 
#path = './libplugin/'
#if not os.path.exists(path): os.system('mkdir %s'%path) 
#os.system('mv plugin.c %s'%path ) 
#os.system('mv plugin.o %s'%path ) 
#os.system('mv plugin.h %s'%path ) 
#os.system('mv libplugin.so %s'%path ) 
#os.system('mv libplugin.dylib %s'%path ) 
