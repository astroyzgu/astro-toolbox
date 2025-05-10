from astropy.table import Table 
import numpy as np 

a = [1.0, np.ma.masked]
b = [np.ma.masked, 'val']
t = Table([a, b], names=('a', 'b'))
# t        t.mask 
# a   b    a     b  
#--- ---  ----- -----
#1.0  --  False  True
# -- val   True False
#t[t['a'].mask] = 0
#t[t['a'].mask].mask = False

t = t.filled(-99) 
print(t)
print(t.mask)
print( np.mean( t['a'])  )

