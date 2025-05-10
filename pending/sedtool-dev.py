#USAGE: 
#python sedtool.py -phot -spec  -filter  --code FAST -xaxis um 1E3 1E5 -yaxis mJy 1E-21 1E-19
#
from astropy.table import Table 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    plot the results of sed fitting (mJy for cigale or AB_ZEROPOINT == 25 for FAST).  
    '''
    parser.add_argument('-phot', help = 'name of input photometrics',type = str)
    parser.add_argument('-spec', help = 'name of input spectroscopics',type = str)
    parser.add_argument('-translate', help = 'File of translator of filter names',type = str)
    parser.add_argument('-code', help = 'cigale',type = str)
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



def readfilters(filterpath, filternames, translate = None, code = 'cigale'):  
    if os.path.exits(filterpath):  
        print(filterpath)
    #return wavelength 



# read_phot - read_filter -> plot 
# read_spec -> plot 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_fast_filter(filename): 
    '''
    Read the filter file for FAST
    parameter
    ---------
    filename: the filter file for FAST (e.g., FILTER.RES.v7.R300)
    return
    --------
    clamb: array_like 
          the central wavelength of filter
    lamb: list 
          array_like, the wavelengths of filter
    tran: list
          array_like, the transitions of filter
    '''
    f = open(filename, 'r')
    lines = f.readlines()
    infos = []; lambs = []; trans = []; clambs = []
    ifilter = 0; flag = 1 
    for ii, line in enumerate(lines):
        if(flag == 1): 
            ifilter = ifilter + 1
            nfilter = eval(line.split()[0]) 
            info    = line.split()[1]
            flag = 0
            lamb = []; tran = []
            counts = 0
            infos.append( info ) 
            continue 
        if(flag == 0): 
            lamb.append( eval(line.split()[1])  )
            tran.append( eval(line.split()[2])  )
            counts = counts + 1
            if( nfilter == counts):
                flag = 1
        if(flag == 1):   
            lamb = np.array(lamb); tran = np.array(tran)
            lambs.append( lamb )
            trans.append( tran ) 
            clambs.append( np.sum(tran*lamb)/np.sum(tran) )
            continue 
    f.close()
    clambs = np.array(clambs)
    return clambs, trans, lambs

def load_fast_translate(filename): 
    '''
    read FAST translate
    parameter
    ---------
    filename: the translate file for FAST (e.g., UVISTA_final_v4.1.translate)
    return 
    --------- 
    filternames: array_like
              the filter names need to translate () 
    ifilters: array_like
              the filter number in the filter file for FAST (see load_fast_filter)              
    '''
    filternames, translators =  np.loadtxt(filename, dtype = 'str',  unpack = True)  
    ival = np.char.find(translators, 'F'); 
    ierr = np.char.find(translators, 'E');
    filternames = filternames[ival>=0]
    ifilters    = np.char.replace(translators[ival>=0], 'F', '').astype('int') - 1
    return filternames, ifilters

def filter2wave(filterfile, translate, inputnames = None, code = 'FAST'):  
    '''
    parameter
    ---------
    filternames: 
    
    return
    ------
    clambs: the central wavelength, corresponding to the input filternames
    ''' 
    filternames, ifilters = load_fast_translate(filterfile) 
    clambs, _, _          = load_fast_filter(translate) 
    clambs = clambs[ifilters]
    if inputnames is not None: 
        indx   = np.isin(inputnames, filternames)
        filternames = filternames[indx]; 
        clambs      = clambs[indx]
    return filternames, clambs

class readphots(): 
    '''
    retrun: 
    '''
    def __init__(self, photfile): 
        self.f = open(photfile, 'r')
        line = self.f.readline() 
        head = np.char.array( line.split() )
        indx = np.arange(len(head)) 
        ierr = indx[ np.char.find(head, 'e', end = 1) == 0]; 
        herr = head[ierr]; 
        hval = np.char.replace(herr, 'e', '', count = 1)
        ival = indx[ np.isin(head, hval) ] 
        #print(herr, ierr)
        #print(hval, ival)
        self.ival = ival
        self.ierr = ierr
        self.head = head 
    def __next__(self): 
        line = self.f.readline() 
        if len(line) == 0: 
            print('Finish reading.') 
            return None  
        data = np.array( line.split() ) #.astype('float')
        name= data[0]; 
        redshift = data[1].astype('float') 
        val = data[self.ival].astype('float'); 
        err = data[self.ierr].astype('float');         
        return name, redshift, val, err 

from astropy.constants import iau2015 as const
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)

def fab2fmu(F, zp = -48.6):
    ''' 
    MagAB  = -2.5log10(F)  + zp
    1). zp = -48.6; unit =           erg s^(−1)Hz^(−1)cm^(−2)
    2). zp =  25.0; unit = 3.631E-30 erg s^(−1)Hz^(−1)cm^(−2)
    m_ν    = -2.5log10(Fν) - 48.6 = -2.5log10(F) + zp 
    Fν == F*10**[-(zp + 48.6)/2.5], unit: erg s^(−1)Hz^(−1)cm^(−2) 
    paramter
    --------
    F:  array_like 
        Flux 
    zp: float
        the zeropoint of AB magnitdue, deflaut: 25
    return
    --------
    f:  array_like 
        the flux density per frequency, unit: erg s^(−1)Hz^(−1)cm^(−2) 
    '''
    return F*10**(-0.4*( zp + 48.6))

def convertflux(fla = None, fmu = None, mu = None, la = None): 
    '''
    Given wavelentgh (or frequency), convert flux density per wavelentgh to flux density per frequency (or in reverse).  
    parameter
    ---------
    fla: array_like
         the flux density per wavelentgh in the unit of erg/cm**2/s/Angstrom
    fmu: array_like 
         the flux density per frequency in the unit of erg/cm**2/s/Hz
    la:  array_like 
         the wavelentgh in the unit of Angstrom (1E-10 m)
    mu:  array_like 
         the frequency  in the unit of Hz (1 s^-1)
    ------
    output: 
    fla (or fmu): array_like
    mu (or la): array_like
    '''
    args = {} 
    if fla is not None: args['fla'] = fla
    if fmu is not None: args['fmu'] = fmu
    if  la is not None: args['la']  =  la
    if  mu is not None: args['mu']  =  mu 
    unit_la = u.erg/u.s/u.cm/u.cm/u.AA # ==> erg/cm**2/s/Angstrom
    unit_mu = u.erg/u.s/u.cm/u.cm/u.Hz # ==> erg/cm**2/s/Hz 
    speed_of_light= 2.9979246*1E08*u.m/u.s # ==> 3*10^8 m/s    
    #print(args, list(args.keys()) )
    #print(  np.isin( ['la', 'mu'], list( args.keys() )  ) )
    if np.isin([ 'la',  'mu'], list(args.keys()) ).sum() != 1: raise ValueError('either la or mu')
    if np.isin(['fla', 'fmu'], list(args.keys()) ).sum() != 1: raise ValueError('either fla or fmu')
    if  'la' in args.keys(): la  = la*u.AA; mu = speed_of_light/la; mu = mu.to('Hz')
    if  'mu' in args.keys(): mu  = mu*u.Hz; la = speed_of_light/(mu).to('s^-1'); la = la.to('AA')
    if 'fla' in args.keys(): fla = fla*unit_la; fmu = (fla*la/mu).to(unit_mu)
    if 'fmu' in args.keys(): fmu = fmu*unit_mu; fla = (fmu*mu/la).to(unit_la)
        
    if np.isin(['fla', 'mu'], list(args.keys()) ).sum() == 2: return fmu, la
    if np.isin(['fla', 'la'], list(args.keys()) ).sum() == 2: return fmu, mu
    if np.isin(['fmu', 'mu'], list(args.keys()) ).sum() == 2: return fla, la
    if np.isin(['fmu', 'la'], list(args.keys()) ).sum() == 2: return fla, mu

#def main(args): 
#    args = getargs()
#    print(args) 
    # read the photometics
    #if 'phot' in args.keys(): 
    #    translate = readtranslate(args['translate']) 
    #    phots     = readphots(args['phot']); 
    #    filternames, ival, ierr = phots.filtername()
    # plot

if __name__ == '__main__':  
    args = {};
    filterfile = 'FILTERS/UVISTA_final_v4.1.translate' 
    translate  = 'FILTERS/FILTER.RES.v7.R300'
    filternames, clambs = filter2wave(filterfile, translate) 
    photfile   = 'sed_candidate_for_plot.tbl'
    phots      = readphots( photfile ) 
    unit_la = u.erg/u.s/u.cm/u.cm/u.AA # ==> erg/cm**2/s/Angstrom
    unit_mu = u.erg/u.s/u.cm/u.cm/u.Hz # ==> erg/cm**2/s/Hz 
    flag  = True; ii = 0  
    while flag:  
        data = phots.__next__() 
        if data is None: break 
        name, redshift, val, err  = data 
        print( '%s: running %s object at z = %.2f'%(ii, name, redshift) )
        fig, ax = plt.subplots(1,1)
        print( len(val), len(clambs) ) 
        fmu = fab2fmu(val, zp = 25) 
        flamb, cmu = convertflux(fmu = fmu, la = clambs)
        ax.scatter(clambs/(1+redshift), fmu, label = 'SED' ) 
        
        tab = Table.read('out/%s_best_model.fits'%name)
        lamb= np.array( tab['wavelength'])*u.nm; lamb = lamb.to(u.AA)
        flux= np.array( tab['Fnu'])*u.mJy; flux = flux.to(unit_mu)  
        ax.plot(lamb/(1+redshift), flux, label = 'best model')
        ax.set_xlim(1E3, 1E5)
        ax.set_xlabel(r'$\rm Angstrom$') 
        ax.set_ylabel(r'$\rm erg/cm^2/s/Hz$') 
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        plt.show() 
        plt.close()
        ii = ii + 1 
