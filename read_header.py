from astropy.io import fits
import numpy as np
import argparse

def getargs(): 
    parser = argparse.ArgumentParser(
    prog = 'fits2zpt', 
    formatter_class = argparse.ArgumentDefaultsHelpFormatter) 
    parser.spilog = 'ST_ZPT = -2.5*np.log10(PHOTFLAM) - 21.10'
    parser.description = '''Calculate the magnitude zeropoints of 'fits' file'''
    parser.add_argument('-f', '--filename', help = 'filename', type = str)
    args = parser.parse_args() 
    return args 

def fits2zpt(fitsname, return_AB = True):  

    hdu = fits.open(fitsname)
    hdr = hdu[0]
    PHOTFLAM = hdr.header['PHOTFLAM']
    PHOTPLAM = hdr.header['PHOTPLAM']
    hdu.close() 
    #PHOTZPT  = hdr.header['PHOTZPT']
    #PHOTBW   = hdr.header['PHOTBW']
    #CCDGAIN  = hdr.header['CCDGAIN']
    ST_ZPT = -2.5*np.log10(PHOTFLAM) - 21.10
    AB_ZPT = -2.5*np.log10(PHOTFLAM) - 5*np.log10(PHOTPLAM) - 2.408 
    print(('#  FILENAME: %s' % fitsname ))
    print(('#  PHOTFLAM  PHOTPLAM    AB_ZPT    ST_ZPT'))
    # e.g., #     0.000 35682.278    26.479    30.549
    print(('#' + '%10.3f'*4)%(PHOTFLAM, PHOTPLAM, AB_ZPT, ST_ZPT))
    if return_AB: 
        return AB_ZPT
    else:  
        return ST_ZPT

if __name__ == '__main__': 
    args = getargs() 
    AB_ZPT = fits2zpt(args.filename)
