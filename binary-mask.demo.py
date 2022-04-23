import numpy as np            
targetmask = np.random.randint(0, 2**4, size = 10) # [low, high) 
print( targetmask )
print( targetmask & 0b0001 ) # LRG
print( targetmask & 0b0010 ) # ELG
print( targetmask & 0b0100 ) # QSO
print( targetmask & 0b1000 ) # BGS # 也可以 targetmask & 2**2
