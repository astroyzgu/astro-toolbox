import numpy as np 

def wrap(func): #函数包装器
    def wrapper(*args, **kwargs):  # 指定宇宙无敌参数
        return func(*args, **kwargs)
    return wrapper  

def readgrp(filename): 
    hid, ns  = np.loadtxt(filename, dtype = 'int', unpack = True)  
    return hid, ns
 
def readsub(filename): 
    shid, sid = np.loadtxt(filename,dtype = 'int', unpack = True)  
    return shid, sid  

def haloinfo(): 
     hid,  ns  = readgrp('./database/grp.txt')
     shid, sid = readgrp('./database/sub.txt')
     datah = np.random.uniform( size = ( np.shape(hid)[0],  6) ) 
     datas = np.random.uniform( size = ( np.shape(sid)[0],  6) ) 
     print('The number of    hole %s'%np.shape(hid)[0])
     print('The number of subhole %s'%np.shape(sid)[0])
     return hid, datah, shid, sid, datas 
if __name__ == '__main__':  
    haloinfo() 
