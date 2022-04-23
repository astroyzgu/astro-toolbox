import numpy as np            
targetmask = np.random.randint(0, 2**4, size = 10) # [low, high) 
print( targetmask )
print( targetmask & 0b0001 ) # LRG
print( targetmask & 0b0010 ) # ELG
print( targetmask & 0b0100 ) # QSO
print( targetmask & 0b1000 ) # BGS # 也可以 targetmask & 2**2


print('#---------------------------------')
import numpy as np
from desitarget import targetmask
from desiutil.bitmask import BitMask
_bitdefs  = targetmask.load_mask_bits() 
desi_mask = BitMask('desi_mask', _bitdefs) 
bgs_mask  = BitMask('bgs_mask',  _bitdefs) 
mws_mask  = BitMask('mws_mask',  _bitdefs) 

types = ['LRG', 'ELG', 'QSO', 'ELG_LOP', 'ELG_HIP', 'ELG|LRG|QSO', 'BGS_ANY', 'MWS_ANY']
#bitnum = 2**0=1  2**1=2 2**2=4  2**5=32    2*6=64   111=1+2+4=7    2**61,     2**62
#         1*2**0 + 1*2**1 + 1*2**2 = 7 <==> 'ELG|LRG|QSO'
from astropy.table import Table
tab = Table()
tab['ID'] = [0,1,2,3,4,5,6,7]
tab['DESI_TARGET'] = [0,1,2,3,4,5,6,7]
mask = desi_mask.mask('LRG')  
selt = (tab['DESI_TARGET'] & mask) > 0
tab['isLRG'] = 0; tab['isLRG'][selt] = 1

mask = desi_mask.mask('ELG')  
selt = (tab['DESI_TARGET'] & mask) > 0
tab['isELG'] = 0; tab['isELG'][selt] = 1

tab['targetmask'] =  2**0*tab['isLRG'] + 2**1*tab['isELG']
print(tab)
print('#---------------------------------') 
for tp in types:
    mask = desi_mask.mask(tp) # -> 返回一个整型 
    selt = ( mask & tab['DESI_TARGET'])>0
    print(tp,mask,np.log2(mask)) #'bit%s'%(bin(mask) ) )

#--------------------------------------
# ID DESI_TARGET isLRG isELG targetmask
#--- ----------- ----- ----- ----------
#  0           0     0     0          0
#  1           1     1     0          1
#  2           2     0     1          2
#  3           3     1     1          3
#  4           4     0     0          0
#  5           5     1     0          1
#  6           6     0     1          2
#  7           7     1     1          3
#--------------------------------------
# LRG 1 0.0
# ELG 2 1.0
# QSO 4 2.0
# ELG_LOP 32 5.0
# ELG_HIP 64 6.0
# ELG|LRG|QSO 7 2.807354922057604
# BGS_ANY 1152921504606846976 60.0
# MWS_ANY 2305843009213693952 61.0
#
#
# end 
