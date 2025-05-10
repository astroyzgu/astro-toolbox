# gfinder tofits -i -o 
# 
import glob 
import numpy as np
from astropy.table import Table
# max_rows = 10 
max_rows = None
idir     = '/home/xhyang/work/Gfinder/DESIDR9Y3/data/' 
tagname  = 'DESIDR9.y3.v1_NGC'
odir     = '/home/yzgu/data/desi/yzgu/seedcat/gfinder/'


idir     = '/home/xhyang/work/Gfinder/DESIDR9Y3/datamore/'
# tagname  = 'DESIDR9.y3.v2_NGC'
odir     = '/home/yzgu/work/work-desi/gfinder/'

grpfiles = glob.glob(idir + 'DESIDR9.*_group' ) 
print(grpfiles)
tagnames = []
for grpfile in grpfiles: 
    tagname = grpfile.split('/')[-1].replace('_group', '')
    print(tagname)
    tagnames.append(tagname)

for tagname in tagnames: 
    #--------------------------------------------------------------------
    grpfile  = tagname + '_group'
    print()
    print(grpfile) 
    groupID,  richness, ra_group,  dec_group, z_grp,   logM_group, logL_group = np.genfromtxt(
        idir + grpfile, max_rows=max_rows, unpack=True) 
    for ii in range(3): 
        print(groupID[ii].astype('int'),  richness[ii].astype('int'), ra_group[ii],  dec_group[ii], z_grp[ii],   logM_group[ii], logL_group[ii]) 
    t = Table()
    t['groupID']   = groupID.astype('int')
    t['richness']  = richness.astype('int')
    t['ra_group']  = ra_group
    t['dec_group'] = dec_group
    t['z_grp']     = z_grp
    t['logM_group']= logM_group
    t['logL_group']= logL_group
    print(t[:3])
    t.write(odir + grpfile + '.fits', overwrite = True) 

    #--------------------------------------------------------------------
    galfile  = tagname + '_galaxyID'
    print()
    print(galfile) 

    galaxyID = np.genfromtxt(
        idir + galfile, max_rows=max_rows, unpack=True) 
    for ii in range(3): 
        print(galaxyID[ii].astype('int') ) 
    t = Table()
    t['galaxyID']   = galaxyID.astype('int')
    print(t[:3])
    t.write(odir + galfile + '.fits', overwrite = True) 

    igalfile1  =  'i' + tagname + '_1'
    print()
    print(igalfile1) 

    galaxyID, groupID, rank, auxiliaryID = np.genfromtxt(
        idir + igalfile1, max_rows=max_rows, unpack=True) 
    for ii in range(3): 
        print(galaxyID[ii].astype('int'), groupID[ii].astype('int'), rank[ii].astype('int'), auxiliaryID[ii].astype('int') ) 
    t = Table()
    t['galaxyID']   = galaxyID.astype('int')
    t['groupID']    = groupID.astype('int')
    t['rank']       = rank.astype('int')
    print(t[:3])
    t.write(odir + igalfile1 + '.fits', overwrite = True) 



    igalfile2  = 'i' + tagname + '_2'
    print()
    print(igalfile2) 

    groupID, galaxyID = np.genfromtxt(
        idir + igalfile2, max_rows=max_rows, unpack=True) 
    for ii in range(3): 
        print(groupID[ii].astype('int'), galaxyID[ii].astype('int')) 
    t = Table()
    t['groupID']    = groupID.astype('int')
    t['galaxyID']   = galaxyID.astype('int')
    print(t[:3])
    t.write(odir + igalfile2+ '.fits', overwrite = True) 


