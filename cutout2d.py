from astropy.io import fits
from astropy.table import Table
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy import units as u
import numpy as np

import argparse
def getargs(): 
    parser = argparse.ArgumentParser(
    prog = 'cutout2d', 
    formatter_class = argparse.ArgumentDefaultsHelpFormatter) 
    parser.spilog = ''
    parser.description = '''Cutout image from raw image (sci or wht)'''
    parser.add_argument('-f', '--filename', help = 'filename', type = str)
    parser.add_argument('-o', '--outputname', default = 'cutout.fits', help = 'outputname', type = str)
    parser.add_argument('-c', '--center',  nargs = '+',  help = 'center coordinate of the cutout in degree', type = float)
    parser.add_argument('-s', '--size',    nargs = '+',  help = 'size of the cutout in degree', type = float)
    #nargs='+', 1 or more
    #      '*', 0 or more 
    #      '?', 0 or 1
    args = parser.parse_args()
    #print(args.size)
    return args

if __name__ == '__main__':
    args       = getargs() 
    img_file   = args.filename 
    outputname = args.outputname 
    hdu = fits.open(img_file); 
    data = hdu[0].data; header = hdu[0].header
    hdu_new  = hdu;
    hdr_wcs2d= wcs.WCS(header);
    center   = SkyCoord(args.center[0], args.center[1], frame = 'icrs', unit = 'deg')
    cutout   = Cutout2D(data, center, size = (args.size[0]*u.degree, args.size[1]*u.degree), wcs = hdr_wcs2d)
    hdu_new[0].data = cutout.data
    hdu_new[0].header.update(cutout.wcs.to_header())
    # print np.shape(cutout.data)
    hdu_new.writeto(outputname, overwrite = True)
