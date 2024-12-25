import glob 
import numpy as np
from astropy.table import Table, vstack

max_rows = 10 
# max_rows = None
idir     = '/home/yzgu/work/work-desi/gfinder/' 
tagnames  = ['DESIDR9.y3.v1', 'DESIDR9.y1.v1', 'DESIDR9.edr.v1'] 
tagnames  = ['DESIDR9.y3.v2'] 
odir     = '/home/yzgu/work/work-desi/gfinder/'
for tagname in tagnames: 
    print(tagname) 
    t1 = Table.read(idir + tagname + '_NGC_galaxyID.fits', memmap = True)
    t2 = Table.read(idir + tagname + '_SGC_galaxyID.fits', memmap = True)
    t  = vstack([t1, t2]) 
    t.write(odir + tagname + '_galaxyID.fits', overwrite = True )

    t1 = Table.read(idir + tagname + '_NGC_group.fits', memmap = True)
    t2 = Table.read(idir + tagname + '_SGC_group.fits', memmap = True)
    t2['groupID'] = t2['groupID'] + np.max(t1['groupID']) 
    t  = vstack([t1, t2]) 
    t.write(odir + tagname + '_group.fits', overwrite = True )

    t1 = Table.read(idir + 'i' + tagname + '_NGC_1.fits', memmap = True)
    t2 = Table.read(idir + 'i' + tagname + '_SGC_1.fits', memmap = True)
    t2['groupID'] = t2['groupID'] + np.max(t1['groupID']) 
    t  = vstack([t1, t2]) 
    t.write(odir + 'i' + tagname + '_1.fits', overwrite = True )

    t1 = Table.read(idir + 'i' + tagname + '_NGC_2.fits', memmap = True)
    t2 = Table.read(idir + 'i' + tagname + '_SGC_2.fits', memmap = True)
    t2['groupID'] = t2['groupID'] + np.max(t1['groupID']) 
    t  = vstack([t1, t2]) 
    t.write(odir + 'i' + tagname + '_2.fits', overwrite = True )