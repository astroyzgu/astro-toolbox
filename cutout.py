from astropy.io import fits
from astropy.table import Table
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy import units as u
import numpy as np

import argparse
def getargs(): 
    '''
    Returns
    -------
    k: dict
    '''
    parser = argparse.ArgumentParser(
    prog = 'cutout', 
    formatter_class = argparse.ArgumentDefaultsHelpFormatter) 
    parser.spilog = ''
    parser.description = '''
    Cutout image from raw image (sci or wht)                                    
    '''

#    [USAGE 1] cutout one image from raw image: 
#           python cutout.py -i input.fits -o -output.fits -c 150.0 30.0 -s 0.0028 0.0028 --size_in_degree True 
#    [USAGE 2] cutout images from raw image:
#           python cutout.py -i input.fits -f opt.list --size_in_degree True 
          
    parser.add_argument('-i', '--inputname',  help = 'name of input image',  type = str)
    parser.add_argument('-o', '--outputname', help = 'name of output image', type = str)
    parser.add_argument('-c', '--center',  nargs = '+',  help = 'center coordinate of the cutout in degree', type = float)
    parser.add_argument('-s', '--size', nargs = '+',  help = 'size of the cutout in degree', type = str)
    parser.add_argument('-f', '--filename',  help = 'filename', type = str)
    parser.add_argument('--size_in_degree', default = True,  help = 'is the unit of size in pixel?', type = bool)
    #nargs='+', 1 or more
    #      '*', 0 or more 
    #      '?', 0 or 1
    args   = parser.parse_args()
    kwargs = args._get_kwargs() # list: [{arg1: val1}, {arg2:val2}] 
    k = {} 
    for arg in kwargs: 
        if arg[1] is not None: 
            k[arg[0]] = arg[1]  
    # print(k)
    return k

def readconf(filename): 
    f = open(filenmae, 'r')
    lines = f.readlines() 
    f.close() 
    for ii in range(len(lines)): 
        l = line.split() 
        
#        else: 
#            print('Skip line %s: %s'%(ii, lines[ii]) )

if __name__ == '__main__':
    args       = getargs() 
    if 'filename' not in args.keys() :  
       inputnames  = np.atleast_1d( args['inputname'] ) 
       outputnames = np.atleast_1d( args['outputname']) 
       sizes       = np.atleast_1d( args['size'] )
       sizes = np.array([eval(s) for s in sizes]).astype('float')  
       if args['size_in_degree'] is True: sizes = sizes*u.degree 
       centers     = [(args['center'][0], args['center'][1])] 
    else: 
       filename    = args['filename']
       t           = Table.read(filename, format = 'ascii')
       if 'inputname' not in t.columns: inputnames  = np.atleast_1d( args['inputname'] ) 
       if 'inputname'     in t.columns: inputnames  = np.atleast_1d( t['inputname'] ) 

       outputnames  = np.atleast_1d( t['outputname'] ) 
       if 'size' not in t.columns: 
           sizes = np.atleast_1d( args['size'] )
           sizes = np.array([eval(s) for s in sizes]).astype('float')  
       if 'size'     in t.columns: sizes  = np.atleast_1d( t['size'] ) 
       if args['size_in_degree'] is True: sizes = sizes*u.degree 
       if 'ra'  not in t.columns:  print('ra is not in the list'); exit()
       if 'dec' not in t.columns: print('dec is not in the list'); exit() 
       centers = np.vstack( [ np.atleast_1d( t['ra'] ), np.atleast_1d( t['dec'] ) ] ).T

    for ii in range(len(outputnames)):
        if len(inputnames) != 1: inputname  = inputnames[ii]
        if len(inputnames) == 1: inputname  = inputnames[0]
        outputname = outputnames[ii]
        if len(sizes) == 1: size = [sizes, sizes]
        if len(sizes) == 2: size =  sizes
        center     = centers[ii]
        #----------------------------------------------------------------------
        hdu = fits.open( inputname ); 
        data = hdu[0].data; header = hdu[0].header
        hdu_new  = hdu;
        hdr_wcs2d= wcs.WCS(header);
        center   = SkyCoord(center[0], center[1], frame = 'icrs', unit = 'deg')
        cutout   = Cutout2D(data, center, size = (size[0], size[1]), wcs = hdr_wcs2d)
        hdu_new[0].data = cutout.data
        hdu_new[0].header.update(cutout.wcs.to_header())
        hdu_new.writeto(outputname, overwrite = True)
