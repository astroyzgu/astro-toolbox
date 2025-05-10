#!/usr/bin/env python

import sys 
import argparse
import numpy as np
import shutil
import os 
import healpy as hp
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
class weightcp(object): 
    '''
    weight_cp.upweight_fofgroup(ra, dec, indx, angle) return  wht, indx_cp 
            upweight_nearest(ra, dec, indx, angle)  return  wht, indx_cp
            assignz_nearest(ra, dec, indx, angle, z=None)   return newz, indx_cp
            kdsearch(ra1, dec1, ra2, dec2, angle)
    '''
    def upweight_nearest(ra, dec, indx, angle): 
        '''
        data: xyz 
        indx: array-like, bool
            galaxies are assigned, if True
        w: weight, if None, all galaxies are weighted by 1. 
        '''
        w   = np.ones_like(ra)
        xyz = hp.ang2vec(ra, dec, lonlat = True)

        max_sep = 2*np.sin(0.5*angle*np.pi/180); # 角度转为弧度转为对应笛卡尔坐标下的直线距离
        #------------------------------------------------------------
        ngal  = np.shape(data)[0]
        
        index = np.zeros(ngal).astype(bool) 
        igal  = np.arange(ngal) 
        index[indx]  = True
        igal1 = igal[ index]; #   有光谱的样本编号, igal1
        igal2 = igal[~index]; # 没有光谱的样本编号, igal2
    
        kd   = KDTree(xyz[ index])
        num1 = kd.count_neighbors(xyz[ index], max_sep) 
        num2 = kd.count_neighbors(xyz[~index], max_sep) 
        w[ index] = 1.0*num2/num1
        d, i = kd.query( xyz[~index], k = 1, workers = -1)

        select = d < max_sep;
        igal2_ = igal2[   select  ] # 符合条件的，没有有光谱的样本编号, indx[igal2_] === False
        igal1_ = igal1[ i[select] ] # 符合条件的，有光谱的样本编号,    indx[igal1_] === True 

        # upweighting by nearby untargetwed objects (igal1_, igal2_)
        #w      = np.array([1,1,0.5,1])*1.0
        #igal1_ = np.array([0,0,1,3])#*1.0
        #igal2_ = np.array([1,1,2,1])#*1.0
        
        newwht    = w.copy()
        for i1, i2 in zip(igal1_,igal2_):
            newwht[i1] = newwht[i1] + w[i2]
        return newwht, index


# def foregroundmask(target_ra, target_dec, maskfile, outputfile): 
#     t = Table.read(maskfile)
#     indx = ellipse_masking(target_ra, target_dec, t['ra'], t['dec'], t['sma'], t['pa'], t['ba'] )
#     t = Table()
#     t['masked'] = indx
#     t.write(outputfile, overwrite = True)


class PipeUI(object):
    def __init__(self):    
        parser = argparse.ArgumentParser(
            description="DESI pipeline control",
            usage="""lsscat_pipe <command> [options]

selfunc run \
--ra   lsdr9_prop.bgsbright.fits RA   \
--dec  lsdr9_prop.bgsbright.fits DEC  \
--z    lsdr9_prop.bgsbright.fits z    \
--zsrc lsdr9_prop.bgsbright.fits zsrc \
--nside 128 \
--Xname mag_r \
--X lsdr9_prop.bgsbright.fits mag_r \
--Xbin 16.  16.5 17.  17.5 18.  18.5 19.  19.5  \
--Xname z \
--X lsdr9_prop.bgsbright.fits z \
--Xbin 0.00 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0  \
--prod_dir './output-y3/'

Where supported commands are (use lsscat_pipe <command> --help for details):
   (------- High-Level -------)
   run      Run a full production to the Production directory. 
   status   Overview of production (not yet).
   (------- Low-Level --------)
   elligeom return the weighted counting located in the elli(cire) shapes. 
   """)
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            sys.exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)() # 获取一个对象的属性值或方法 
        #                               ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#--------------------------------------------------------------------
    def elligeom(self): 
        parser = argparse.ArgumentParser(
            description="Run the total pipelines to the Production directory",
            usage="""pipeUI.py elligeom [options] (use --help for details)

pipeUI.py elligeom [source catalog] --radec ra dec --mskfile ellimask.fits --output maskcounts.fits
pipeUI.py elligeom [source catalog] --radec ra dec --mskfile ellimask.fits --output maskcounts.fits --remove 
pipeUI.py elligeom [source catalog] --radec ra dec --mskfile ellimask.fits --output maskcounts.fits --remove --newcolname ''

            """)
        parser.add_argument("inputs", metavar='datafile', type=str, nargs = '*', 
                            help="Input catalog")
        parser.add_argument("--radec", type=str, default=None, nargs=2,
                            help="Provide RA by filename and colname, e.g., ra.fits 1")
        parser.add_argument("--mskfiles", nargs = '*', 
                            help="Provide ellipse masks: id ra dec sma pa ba w")
        parser.add_argument('--output', default=None, type=str, 
                            help='Path to the output file.')
        parser.add_argument('--remove', action='store_true', 
                            help='whether removing the masked rows') 
        parser.add_argument('--newcolname', default='ncover', type = str, 
                            help='the additional colname of the counting results')
        args = parser.parse_args(sys.argv[2:]) 

        for inputname in args.inputs: 
            target_ra   = readcol_from_fits(inputname,  args.radec[0])
            target_dec  = readcol_from_fits(inputname,  args.radec[1])

            msk  = vstack( [read_mskfile(mskfile) for mskfile in args.mskfiles] )
            counting = ellipse_counting(target_ra, target_dec, 
                                        msk['ra'], msk['dec'], msk['sma'], msk['pa'], msk['ba'], msk['w']) 
            t = Table.read(inputname)
            if args.newcolname!='': t[args.newcolname] = counting
            if args.remove:  t = t[counting > 0] 
            if args.output is None: 
                outputname = inputname.replace('.fits', '.elligemo.fits')
            else: 
                outputname = args.output
            print('writing into %s'%outputname)
            t.write(outputname, overwrite = True)    

    def run(self): 

        parser = argparse.ArgumentParser(
            description="Run the total pipelines to the Production directory",
            usage="""desi_pipe run [options] (use --help for details)
            
      1. data.fits        # fmt: id, ra, dec, z, zsrc, X1, X2, X3, ...
      2. hppix-data.fits  # healpix pixel number 
      3. rand.fits        # fmt: id, ra, dec, z, zsrc, X1, X2, X3, ...
      4. hppix-rand.fits  # healpix pixel number of random catalog
      5. hpmap.fits       # healpix map of all the data 
         hpmap_X1.fits    # cumulative healpix map as a function of X1
         hpmap_X2.fits    # cumulative healpix map as a function of X2
         ...
      6. clustering-data.fits # fmt: wht wht_sys wht_cp wht_noz nz wht_fkp ...
      7. clustering-rand.fits # fmt: wht wht_sys wht_cp wht_noz nz wht_fkp ...
      8. fig/hpmap.png    # plot of the healpix map
             compl_X1.png # plot of the completeness as a function of X1
             hpmap_X1_{lower}_{upper}.png # plot of the map as a function of X1

            """) 
        parser.add_argument("--ra", type=str, default=None, nargs=2,
                            help="Provide RA by filename and colname, e.g., ra.fits 1")
        parser.add_argument("--dec", default=None, nargs=2,
                            help="Provide DEC by filename and colname, e.g., dec.fits 2")
        parser.add_argument("--z", default=None, nargs=2,
                            help="Provide the redshift by filename and colname, e.g., z.fits 3. \
                            if z > 0, good-z.")
        parser.add_argument("--zsrc", default=None, nargs=2,
                            help="Provide the redshift source. if zsrc > 0, have spec observation;")
        # parser.add_argument("--zcls", default=None, nargs=2,
        #                     help="Provide the redshift source. if zcls > 0, have spec observation and right class;")
        parser.add_argument("--nside", default=128, type=int,
                            help="Set the nside of the healpix")
        parser.add_argument("--Xname",  default=None, nargs='+', type=str, action='append',
                            help="Provide the name of the parameter X")
        parser.add_argument("--X",  default=None, nargs='+', type=str, action='append', 
                            help="Provide the set of parameter X related to probility of owning spec-z.")
        parser.add_argument("--Xbin", default=None, nargs='+', type=float, action='append',
                            help="Provide the bin edges of the parameter X")
        parser.add_argument("--prod_dir", default='./output/',  type=str, 
                            help="Production directory to store the output")
        args = parser.parse_args(sys.argv[2:]) 
        self.args = vars(args)

        prod_dir     = args.prod_dir 
        prod_fig_dir = os.path.join(args.prod_dir, 'fig')

        os.makedirs(prod_dir,     exist_ok=True) 
        os.makedirs(prod_fig_dir, exist_ok=True)
        
        terminal_width = shutil.get_terminal_size().columns
        print('---  configure  ---'.ljust(int(terminal_width), '-') )
        print_args(args, show_shape = False)

        print('---  reading  ---'.ljust(int(terminal_width), '-') )
        self.data = self._from_args2dict(args) 

        print('---  data  ---'.ljust(int(terminal_width), '-') )
        print_args(args, show_shape =  True)

        print('---  nside == {:}  ---'.format(self.data['nside']).ljust(int(terminal_width), '-') )
        import healpy as hp
        self.data['hppix'] = hp.ang2pix(self.data['nside'], self.data['ra'], self.data['dec'], lonlat=True) 
        print("{:<10}".format('hppix'), np.shape(self.data['hppix']), self.data['hppix'][:3]) 
        unq, cnt = np.unique(self.data['hppix'], return_counts=True)
        pixarea = hp.nside2pixarea(self.data['nside'], degrees=True)
        print("    Pixel   area: %.2f sq.deg."%pixarea)
        print("    Unique pixel: %s"%len(unq)) 
        print("    Total   area: %.2f sq.deg."%(len(unq)*pixarea))
        print
        print('---  modeling data ---'.ljust(int(terminal_width), '-') ) 
        # plot the fracmap dependence on one of the parameters X1, X2, ..., and C
        
        compl1 = BinMap(Xbins = None, Cbins = None, interp = None)
        compl1.fit(None, self.data['zsrc'] > 0, C=self.data['hppix']) 
        compl2 = BinMap(Xbins=self.data['Xbin'], interp='nearest') 
        compl2.fit(self.data['X'], self.data['zsrc'] > 0, C=self.data['hppix'])

        print('saveing fracmap')
        print(self.data['nside'], np.shape(np.squeeze(compl1.valmap)), )
        print(self.data['nside'], np.shape(np.squeeze(compl2.valmap)), )

        np.save(os.path.join(prod_dir, 'complmap1.npy'), np.squeeze(compl1.valmap).T ) 
        np.save(os.path.join(prod_dir, 'complmap2.npy'), np.squeeze(compl2.valmap).T ) 

        self.data['compl'] = compl1.predict_proba(None, C=self.data['hppix']) 
        self.data['w']     = 1.0/compl2.predict_proba(self.data['X'], C=self.data['hppix']) 
        self.data['w'][np.isinf(self.data['w'])] = 0
        self.data['w'][np.isnan(self.data['w'])] = 0
        cartview(compl1.valmap, fname=os.path.join(prod_fig_dir, 'hpmap.png'), nside = self.data['nside'], ctitle='Fraction of Spec') 

        for ii in range(len(self.data['Xbin'])): 
            compl = BinMap(Xbins=[self.data['Xbin'][ii]], interp=None) 
            compl.fit(self.data['X'][:, ii].reshape(-1,1), self.data['zsrc']>0, C=self.data['hppix'])

            prod_X_dir = os.path.join(prod_fig_dir, 'X%s'%ii)
            os.makedirs(prod_X_dir, exist_ok=True)            
            compl.fracmap1d(0, fname=os.path.join(prod_X_dir, 'compl_X%s.png'%(ii)), label = self.data['Xname'][ii]) 

            for jj in range(len(self.data['Xbin'][ii])-1):
                lo = self.data['Xbin'][ii][jj]; up = self.data['Xbin'][ii][jj+1]
                cartview(compl.valmap[jj,:], 
                        fname=os.path.join(prod_X_dir, 'compl_X%s_%.3f_%.3f.png'%(ii,lo,up)), 
                        title='%s: %.3f < %s < %.3f'%(self.data['Xname'][ii],lo,self.data['Xname'][ii],up), ctitle='Fraction of Spec', 
                        nside = self.data['nside'], vmin=0, vmax=1)

        # 
        # modeling the fracmap dependence on parameters X and C
        # 


        print
        print('---  rand  ---'.ljust(int(terminal_width), '-') )
        # 
        idata = np.arange(len(self.data['ra'])) 
        self.rand = self.data.copy() 
        irand = np.random.choice(idata, 2*len(idata), replace = True) 
        self.rand['ra'], self.rand['dec'], _ = sphrand_healpy(len(irand), self.rand['nside'], np.unique(self.data['hppix']) )  
        self.rand['z']     = self.data['z'][irand]
        self.rand['zsrc']  = self.rand['zsrc'][irand]
        self.rand['X']  = self.rand['X'][irand]
        self.rand['hppix'] = hp.ang2pix(self.rand['nside'], self.rand['ra'], self.rand['dec'], lonlat=True)
        self.rand['compl'] = compl1.predict_proba(None, C=self.rand['hppix']) 
        self.rand['w']     = 1.0/compl2.predict_proba(self.rand['X'], C=self.rand['hppix']) 

        self.rand['w'][np.isinf(self.rand['w'])] = 0
        self.rand['w'][np.isnan(self.rand['w'])] = 0

        print("write the galaxy catalog to %s"%os.path.join(prod_dir, 'data.fits'))
        from astropy.table import Table, hstack
        t1 = Table( {k: self.data[k] for k in ['ra', 'dec', 'hppix', 'z', 'zsrc', 'compl', 'w']} )
        t2 = Table( {self.data['Xname'][k]: self.data['X'][:,k] for k in range(self.data['nX'])})
        tdata =  hstack([t1, t2]); 
        tdata.add_column(idata, 0, name = 'idata')
        tdata = tdata[tdata['zsrc'] > 0]; # del tdata['zsrc']
        print('%s lines'%len(tdata))
        tdata.write(os.path.join(prod_dir, 'data.fits'), overwrite=True) 

        print("write the random catalog to %s"%os.path.join(prod_dir, 'rand.fits'))
        t1 = Table( {k: self.rand[k] for k in ['ra', 'dec', 'hppix', 'z', 'zsrc', 'compl', 'w']} )
        t2 = Table( {self.rand['Xname'][k]: self.rand['X'][:,k] for k in range(self.rand['nX'])})
        trand =  hstack([t1, t2]);
        trand.add_column(irand, 0, name = 'idata')
        trand = trand[trand['zsrc'] > 0]; # del trand['zsrc']
        print('%s lines'%len(trand))
        trand.write(os.path.join(prod_dir, 'rand.fits'), overwrite=True) 
        print('---  end  ---'.ljust(int(terminal_width), '-') )

    def _data(self, dict):
        pass
    def _data2hpmap(self, dict):
        pass
    def _hpmap2fig(self, hpmap, X, Xbin):
        pass
    def _hpmap2data(self, hpmap, X, Xbin):
        pass
    def _data2clustering(self, data):
        pass

    def _from_args2dict(self, args):
        '''
        Convert the input arguments to a dictionary with data. 
        '''
        dict = vars(args) # Convert the input arguments to a dictionary. 
        keynames = ['ra', 'dec', 'nside', 'output_dir'] 
        if not args.ra   is None: dict['ra']   = readcol_from_fits(args.ra[0],   args.ra[1])
        if not args.dec  is None: dict['dec']  = readcol_from_fits(args.dec[0],  args.dec[1])
        if not args.nside is None: dict['nside'] = int(args.nside)
        if not args.Xname is None: 
            dict['Xname'] = np.array(args.Xname).flatten()
        else: 
            dict['Xname'] = [ 'X%i'%ii for ii in range(len(Xbin))]
        if not args.z    is None: dict['z'] = readcol_from_fits(args.z[0],  args.z[1])
        if not args.zsrc is None: dict['zsrc'] = readcol_from_fits(args.zsrc[0],  args.zsrc[1])
        if not args.prod_dir is None: dict['prod_dir'] = args.prod_dir 

        X = []; Xbin = [];
        if args.X is not None:
            if len(args.X)!=len(args.Xbin): 
                ValueError("The length of X and Xbin should be the same.")
            else: 
                dict['nX'] = len(args.Xbin)
            for argX, argXbin in zip(args.X, args.Xbin): 
                X_    = readcol_from_fits(argX[0],  argX[1])
                Xbin_ = np.histogram_bin_edges(X_, argXbin)
                X.append(np.reshape(X_, (-1,1)) );  
                Xbin.append(Xbin_); 
            X = np.concatenate(X, axis=1)
        dict['X'] = X; 
        dict['Xbin'] = Xbin
        return dict

def print_args(args, show_shape = False):
    try: 
        args = vars(args).copy()
    except:
        pass

    for key, value in args.items():
        if show_shape: 
            try: 
                if key == 'Xbin': 
                    print("{:<10}".format(key), [np.shape(v) for v in value]) 
                    for ii, v in enumerate(value): 
                        print("    {:<8}".format('%s%s'%(key, ii) ), len(v), v)
                elif key == 'zsrc':
                    print("{:<10}".format(key), np.shape(value), value[:3]) 
                    unq, cnt = np.unique(value > 0, return_counts=True)
                    for unq_, cnt_ in zip(unq, cnt):
                        print("    {:<8}".format('spec-z is %5s'%unq_), '%7.4f'%(cnt_/np.sum(cnt)*100), '%') 
                elif key == 'X': 
                    print("{:<10}".format(key), np.shape(value))
                    value = np.transpose(value)
                    for ii, v in enumerate(value): 
                        print("    {:<8}".format('%s%s'%(key, ii) ), len(v), v[:3])           
                elif key == 'nside': 
                    print("{:<10}".format(key), value)
                elif key == 'ra' or key == 'dec' or key == 'z': 
                    print("{:<10}".format(key), np.shape(value), value[:3])
                else:
                    print("{:<10}".format(key), value)
            except:
                print("{:<10}".format(key), value)
        else: 
            print("{:<10}".format(key), value)


# class lsstools():
#     def __init__(self):
#         pass
#     @staticmethod 



def readcol_from_fits(filename, colname, funit = 1, verbose=False, range=None): #  [0,1000000]): # 
    from astropy.io import fits
    print('reading %s colunm from %s'%(colname, filename))  
    hdul = fits.open(filename, memmap=True)
    try: 
        colnames=hdul[funit].columns.names
        colname = colnames[int(colname)]
    except:
        pass
    if range is not None: 
        colvalue =  hdul[funit].data[colname][range[0]:range[1]]
    else:
        colvalue =  hdul[funit].data[colname]
    if verbose: print(colname, len(colvalue), colvalue[:3]) 
    return np.array(colvalue)

def sphrand_uniform(nrand, ramin, ramax, decmin, decmax): 
    '''
	Draw a random sample with uniform distribution on a sphere

    Parameters
    ----------
    nrand : int
        the number of the random points to return
    ramin, ramax: float
        Right Ascension between ramin and ramax [degrees] 
    decmin, decmax: float 
        Declination between decmin and decmax [degrees]
    Returns
    -------
    ra, dec : ndarray
        the random sample on the sphere within the given limits.
        arrays have shape equal to nrand.
    skycov: float 
        sky coverage [deg^2].
    '''
    zmax = np.sin( np.asarray(decmax) * np.pi / 180.)
    zmin = np.sin( np.asarray(decmin) * np.pi / 180.)

    z   = np.random.uniform(zmin, zmax,  nrand)
    DEC = (180. / np.pi) * np.arcsin(z) 
    RA  = np.random.uniform(ramin, ramax, nrand)

    skycov = (zmax - zmin)*180/np.pi *(ramax - ramin)
    return RA, DEC, skycov

def sphrand_healpy(nrand, nside, pix):  #, ramin = None, ramax = None, decmin = None, decmax = None): 
    '''
	Draw a random sample with uniform distribution on the given region of a sphere defined by healpix. 

    Parameters
    ----------
    nrand : int
        the number of the random points to return
    nside: int 
        nside of the healpy pixelization 
    pix: ndarray, int 
        pixel number(s)
    Returns
    -------
    ra, dec : ndarray
        the random sample on the sphere within the given region defined by healpix.
        arrays have shape equal to nrand.
    skycov: float 
        sky coverage [deg^2].
    ''' 
    pix      = np.asarray(pix).astype(int)
    pixarea  = hp.nside2pixarea(nside, degrees = True)
    skycov_healpy = pixarea*len(pix) 
    
    lon, lat = hp.pix2ang(nside, pix, lonlat = True)
    indx_box = np.hstack([np.argmax(lon), np.argmin(lon), np.argmax(lat), np.argmin(lat)])
    vec      = hp.boundaries(nside, pix[indx_box], step = 1)
    lon, lat = hp.vec2ang(vec, lonlat = True) 
    ramax  = np.max(lon); ramin  = np.min(lon) 
    decmax = np.max(lat); decmin = np.min(lat)
    nrand_ = 0
    RA = []; DEC = []
    while nrand_ < nrand:
        arand, drand, __skycov__ = sphrand_uniform( int(nrand*1.2), ramin, ramax, decmin, decmax)
        pix_rand = hp.ang2pix(nside, arand, drand, lonlat = True) 
        indx     = np.isin(pix_rand, pix)
        nrand_ = nrand_ + np.sum(indx)
        RA.append( arand[indx]) 
        DEC.append(drand[indx]) 
        # print('Generating %s random points. Targeting --> %s.'%(nrand_, nrand) )
    RA   = np.hstack(RA)
    DEC  = np.hstack(DEC)
    indx = np.arange(nrand).astype('int') # , replace = False)
    return RA[indx], DEC[indx], skycov_healpy
    
class BinMap:
    def __init__(self, Xbins=None, Cbins=None, interp=None):
        '''
        Class for creating a bin map based on input data.

        Parameters:
        -----------
        Xbins: list, optional
            List of bin edges for the X variables.
        Cbins: list, optional
            List of bin edges for the C variables.
        '''
        self.Xbins = Xbins
        self.Cbins = Cbins
        self.interp = interp

    def interpdd(self, data, method='nearest'):
        from scipy import interpolate
        z = data.copy()
        valid_index  = np.where(~np.isnan(data))
        valid_value  = data[valid_index]
        indices      = np.where( np.isnan(data))
        z[indices]   = interpolate.griddata(valid_index, valid_value, indices, method=method, fill_value=0, rescale = True)
        return z

    def assure2d(self, val):
        '''
        Helper function to ensure that the input array is 2-dimensional.

        Parameters:
        -----------
        val: array-like
            Input array.

        Returns:
        --------
        val_2d: array-like
            2-dimensional version of the input array.
        '''
        val = np.array(val)
        ndim = np.ndim(val)
        if ndim == 0:
            return np.empty((0, 2))
        if ndim == 1:
            return val[:, np.newaxis]
        if ndim > 2:
            return val.reshape(-1, val.shape[-1])
        else:
            return val

    def autobins(self, X=None, C=None):
        '''
        Automatically determine the bin edges based on the input data.

        Parameters:
        -----------
        X: array-like, optional
            Input X variables.
        C: array-like, optional
            Input C variables.

        Returns:
        --------
        bins: list
            List of bin edges for X and C variables.
        '''
        X = self.assure2d(X)
        if self.Cbins is None: 
            C  = self.assure2d(C)
            if np.prod(np.shape(C)) != 0:
                Cmaxs = np.max(self.assure2d(C), axis=0)
                self.Cbins = [np.arange(Cmax + 2) for Cmax in Cmaxs]
            else: 
                self.Cbins = []

        if self.Xbins is None: 
            X = self.assure2d(X) 
            if np.prod(np.shape(X)) != 0:
                self.Xbins = [np.histogram_bin_edges(X[:, ii], bins=10, range=None, weights=None) for ii in range(X.shape[1])]
            else:
                self.Xbins = []

        if len(self.Xbins) == 0: 
           self.bins = self.Cbins
        elif len(self.Cbins) == 0:
           self.bins = self.Xbins
        else:
           self.bins = self.Xbins + self.Cbins
        return self.bins

    def fit(self, X, y, C=None):
        '''
        Fit the bin map using the input data.

        Parameters:
        -----------
        X: array-like
            Input X variables.
        y: array-like
            Target variable.
        C: array-like, optional
            Input C variables.
        '''
        if C is None:
            self.C = None
            self.X = X
            data = self.assure2d(X)
        elif X is None:
            self.C = C
            self.X = None
            data = self.assure2d(C)
        else:
            self.C = C
            self.X = X
            data = np.hstack([self.assure2d(X), self.assure2d(C)])
        self.y = y
        bins = self.autobins(X, C)
        
        histdd1, _ = np.histogramdd(data, bins=bins, weights=y)
        histdd2, _ = np.histogramdd(data, bins=bins, weights=None)
        self.histdd1 = histdd1
        self.histdd2 = histdd2 
        self.valmap  = np.zeros_like(histdd2) + np.nan        
        # self.shape   = histdd2.shape

        Cbincout = np.bincount(C)
        exec_code = 'self.valmap['+':,'*len(self.Xbins)+'Cbincout == 0] = -np.inf'
        # print(exec_code)
        exec(exec_code)
        
        self.valmap[histdd2 != 0] = histdd1[histdd2 != 0] / histdd2[histdd2 != 0]
        if self.interp is not None: 
            print('maxmin', np.nanmax(self.valmap), np.nanmin(self.valmap)  )
            self.valmap = self.interpdd(self.valmap, method=self.interp) 
            print('maxmin', np.nanmax(self.valmap), np.nanmin(self.valmap)  )
            self.valmap_pad = np.pad(self.valmap, 1, mode='edge')
        else: 
            self.valmap_pad = np.pad(self.valmap, 1, mode='constant', constant_values=np.nan)
        exec('self.valmap_pad['+':,'*len(self.Xbins)+' 0] = 0')
        exec('self.valmap_pad['+':,'*len(self.Xbins)+'-1] = 0')

    def predict_proba(self, X, C=None):
        '''
        Predict the probabilities based on the input data.

        Parameters:
        -----------
        X: array-like
            Input X variables.
        C: array-like, optional
            Input C variables.

        Returns:
        --------
        proba: array-like
            Predicted probabilities.
        '''
        if C is None:
            data = self.assure2d(X)
        elif X is None:
            data = self.assure2d(C)
        else:
            data = np.hstack([self.assure2d(X), self.assure2d(C)])
        # print(np.shape(data),)
        queryslice = tuple(np.digitize(data[:, ii], bins=self.bins[ii], right=False) for ii in range(data.shape[1]))
        proba      = self.valmap_pad[queryslice]
        self.index_outer  = np.isnan(proba) 
        num_of_outer = np.sum(self.index_outer)  
        if num_of_outer > 0: 
            if self.interp is None: 
                print('Warning: %s values are out of range, returning to nan.'%num_of_outer ) 
            else: 
                print('Warning: %s values are out of range, returning to use interped values.'%num_of_outer)
        return proba
    
    def fracmap1d(self, ii, fname = None, verbose = True, label = None): 
        nsample = self.X.shape[0]
        fig, ax = plt.subplots(1, 1, figsize = (5, 5)) 
        edges   = np.histogram_bin_edges(self.X[:,ii], bins = self.Xbins[ii]) 
        x = (edges[1:] + edges[:-1])*0.5 
        h,  _ = np.histogram(self.X[:,ii], bins = edges)
        h1, _ = np.histogram(self.X[:,ii][self.y > 0], bins = edges) 
        ax.plot(x,  h/np.sum(h), drawstyle = 'steps-mid', color = 'b', label = 'Total  distribution')
        ax.plot(x, h1/np.sum(h), drawstyle = 'steps-mid', color = 'r', label = 'Spec-z distribution')
        ax.plot(x, h1/h, drawstyle = 'steps-mid', marker = '*', color = 'k', label = 'Completeness')
        ax.set_xticks(edges)
        if label is None: ax.set_xlabel('X%s'%ii)
        if label is not None: ax.set_xlabel(label)
        ax.set_ylabel('Fraction of Spec') 
        ax.legend()
        if fname is not None: 
            if verbose: print('%s'%fname)
            plt.savefig(fname)
        plt.close()

    # def hpcartview(self, fname=None, nside=None, ii = None): 
    #     import math
    #     # vmap  = self.valmap[]
    #     npix  = np.shape(self.valmap)[-1]
    #     vmap  = np.zeros(12*nside**2) - np.inf
    #     vmap[:npix] = self.valmap 
    #     vmap[np.isnan(vmap)|np.isinf(vmap)] = -np.inf
    #     cartview(vmap, fname=fname)
    
def cartview(vmap, fname=None, title=None, nside=None, vmin=None, vmax=None, ctitle=None, verbose=True):
    from healpy.newvisufunc import projview, newprojplot
    if nside is None: 
       nside = hp.npix2nside(len(vmap)) 
    else: 
       vmap_ = vmap.copy()
       vmap  = np.zeros(12*nside**2) - np.inf
       vmap[:len(vmap_)] = vmap_
    if title is None: title = 'nside=%d'%nside
    if ctitle is None: ctitle = 'surface number density [deg^-2]'
    projview(
        vmap,
        rot = [0, 0, 0], 
        coord=["G"],
        graticule=True,
        graticule_labels=True,
        unit=ctitle,
        xlabel="longitude", 
        ylabel="latitude",
        title=title, 
        cb_orientation="vertical",
        projection_type="cart",
        nest=False,
        min=vmin,
        max=vmax
    ); 
    from astropy.coordinates import SkyCoord 
    theta = np.linspace(0, 360, 101)
    gp  = SkyCoord(theta, theta*0, frame='galactic', unit='deg')
   #  print(gp.icrs.ra.degree, gp.icrs.dec.degree)
   #  isort = np.argsort(gp.icrs.ra.degree)
   #  newprojplot(gp.icrs.ra.degree[isort] , gp.icrs.dec.degree[isort], color = 'k') # , lonlat = True)
    hp.graticule(dpar=30, dmer=30) 
    plt.tight_layout()
    if fname is not None: 
        if verbose: print('%s'%fname)
        plt.savefig(fname, bbox_inches='tight')
    plt.close()


def __circle_counting(ra,  dec, u, v, a, w = None): 
    '''
    Count the number of points within a circle of radius a centered at (u, v) in the (ra, dec) coordinates.

    Parameters:
    - ra (array-like): The right ascension coordinates of the points.
    - dec (array-like): The declination coordinates of the points.
    - u (float or array-like): The right ascension of the center of the circle.
    - v (float or array-like): The declination of the center of the circle.
    - a (float): The radius of the circle in degrees.
    - w (array-like, optional): The weights of the points. Default is None.

    Returns:
    - indx (array): The number of points within the circle for each center (u, v). 
    '''
    a = np.atleast_1d(a)
    xyzstar = hp.ang2vec(u,  v,   lonlat = True).reshape(-1, 3)
    xyzgala = hp.ang2vec(ra, dec, lonlat = True).reshape(-1, 3)
    indx =  np.zeros( len(ra) )*0.0
    from scipy.spatial import KDTree
    radius  = 2*np.sin(a/2*np.pi/180)
    kd_tree = KDTree(xyzgala); 
    indice  = kd_tree.query_ball_point(xyzstar, r = radius)
    if w is None: w = np.ones(len(u))
    counts  = np.array([len(ind) for ind in indice]) 
    wht     = np.repeat(w, counts)
    indice  = np.hstack(indice) 
    unq, inv = np.unique( indice,return_inverse=True)
    wcount   = np.bincount(inv, wht.reshape(-1))
    if len(unq) != 0:  indx[unq] = wcount
    return indx

def __ellipse_counting(ra,  dec, u, v, a, pa, ba, w = None): 
    '''
    Count the number of points within an ellipse centered at (u, v) in the (ra, dec) coordinates.

    Parameters:
    - ra (array-like): The right ascension coordinates of the points.
    - dec (array-like): The declination coordinates of the points.
    - u (float or array-like): The right ascension of the center of the ellipse.
    - v (float or array-like): The declination of the center of the ellipse.
    - a (float): The semi-major axis of the ellipse in degrees.
    - pa (float): The position angle of the ellipse in degrees.
    - ba (float): The axis ratio of the ellipse.
    - w (array-like, optional): The weights of the points. Default is None.

    Returns:
    - indx (array): The number of points within the ellipse for each center (u, v). 
    '''

    xyzstar = hp.ang2vec(u,  v,   lonlat = True).reshape(-1, 3)
    xyzgala = hp.ang2vec(ra, dec, lonlat = True).reshape(-1, 3) 
    a = np.atleast_1d(a)
    pa= np.atleast_1d(pa)*np.pi/180.0
    ba= np.atleast_1d(ba)
    from scipy.spatial import KDTree
    radius  = 2*np.sin(a/2*np.pi/180) 
    kd_tree = KDTree(xyzgala); 
    indice  = kd_tree.query_ball_point(xyzstar, r = radius)
    # indice  = np.unique(np.hstack(indice) ).astype(int)
    # indx[indice] = True 
    istar =  np.arange(len(u) ) 
    indx  =  np.zeros( len(ra) )*0.0
    if w is None: w = np.ones(len(u))
    for istar_, x_, y_, a_, pa_, ba_, w_ in zip(istar, u, v, a, pa, ba, w): 
        igala_ = indice[istar_]; 
        if len(igala_) == 0: continue 
        igala_ = np.array(igala_)
        # Transform the point to the ellipse's coordinates
        ra_  = ra[igala_]
        dec_ = dec[igala_]
        ra_prime  =  (ra_  - x_) * np.cos(pa_) * np.cos(dec_*np.pi/180) - (dec_ - y_) * np.sin(pa_)
        dec_prime =  (ra_  - x_) * np.sin(pa_) * np.cos(dec_*np.pi/180) + (dec_ - y_) * np.cos(pa_)
        a_prime  =  a_*ba_ # *np.cos(dec*np.pi/180)
        b_prime  =  a_
        ang      = ra_prime**2/a_prime**2 + dec_prime**2/b_prime**2
        igala_   = igala_[ang < 1]
        if len(igala_) == 0: continue
        indx[igala_] = indx[igala_] + w_
    return indx

def ellipse_counting(ra, dec, u, v, a, pa, ba, w = None):  
    ''' 
    Count the number of points within an ellipse centered at (u, v) in the (ra, dec) coordinates.

    parameters
    ----------
    ra, dec:  float, scalars or array-like
	Angular coordinates of input targets on the sphere
    u, v: float, scalars or array-like
	Angular coordinates of central point of ellipse on the sphere
    a: float, scalars or array-like
	The length of Semi-major axis of ellipse [degree] 
    pa:  float, scalars or array-like
	Position angle [degree]. PA == 0 is North (ΔDEC>0), PA = 90 is WEST (ΔRA > 0). 
    ba:  float, scalars or array-like
	b/a [0,1]. if ba == 1, the shape is circle, namely the cap on the sphere. 
    w: array-like, optional
    Weights of the points. Default is None. 

    Returns
    -------
    indx: array
        Number of points within the ellipse for each center (u, v).
    ''' 
    u   = np.atleast_1d(u) 
    v   = np.atleast_1d(v) 
    if w is None: w = 1
    if isinstance(a,  (int, float)): a   = u*0.0 + a 
    if isinstance(pa, (int, float)): pa  = u*0.0 + pa 
    if isinstance(ba, (int, float)): ba  = u*0.0 + ba
    if isinstance(w,  (int, float)): w   = u*0.0 + w 
    a   = np.atleast_1d(a) 
    pa  = np.atleast_1d(pa) 
    ba  = np.atleast_1d(ba) 

    indx     = np.zeros(len(ra))#.astype(bool) 
    iscircle = (ba==1)
    if np.sum( iscircle) != 0: 
        # print('Number of circle masking:', np.sum(iscircle))
        indx1     = __circle_counting(ra, dec, 
                                   u[iscircle], v[iscircle], a[iscircle], w[iscircle])
        indx      = indx + indx1
    if np.sum(~iscircle) != 0: 
        # print('Number of ellipse masking:', np.sum(~iscircle))
        indx2    = __ellipse_counting(ra, dec, 
                                   u[~iscircle],  v[~iscircle], a[~iscircle], 
                                   pa[~iscircle], ba[~iscircle],w[~iscircle]) 
        indx      = indx + indx2
    return indx

def __circle_masking(ra,  dec, u, v, a): 
    a = np.atleast_1d(a)
    xyzstar = hp.ang2vec(u,  v,   lonlat = True).reshape(-1, 3)
    xyzgala = hp.ang2vec(ra, dec, lonlat = True).reshape(-1, 3)
    indx =  np.zeros( len(ra) ).astype(bool)
    from scipy.spatial import KDTree
    radius  = 2*np.sin(a/2*np.pi/180)
    kd_tree = KDTree(xyzgala); 
    indice  = kd_tree.query_ball_point(xyzstar, r = radius)
    indice  = np.unique(np.hstack(indice) ).astype(int)
    indx[indice] = True
    return indx

def __ellipse_masking(ra,  dec, u, v, a, pa, ba): 
    xyzstar = hp.ang2vec(u,  v,   lonlat = True).reshape(-1, 3)
    xyzgala = hp.ang2vec(ra, dec, lonlat = True).reshape(-1, 3) 
    a = np.atleast_1d(a)
    pa= np.atleast_1d(pa)*np.pi/180.0
    ba= np.atleast_1d(ba)
    from scipy.spatial import KDTree
    radius  = 2*np.sin(a/2*np.pi/180) 
    kd_tree = KDTree(xyzgala); 
    indice  = kd_tree.query_ball_point(xyzstar, r = radius)
    # indice  = np.unique(np.hstack(indice) ).astype(int)
    # indx[indice] = True
    istar = np.arange(len(u) ) 
    indx =  np.zeros( len(ra) ).astype(bool)
    for istar_, x_, y_, a_, pa_, ba_ in zip(istar, u, v, a, pa, ba): 
        igala_ = indice[istar_]; 
        # print(istar_, igala_)
        if len(igala_) == 0: continue 
        igala_ = np.array(igala_)
        # Transform the point to the ellipse's coordinates
        ra_  = ra[igala_]
        dec_ = dec[igala_]
        ra_prime  =  (ra_  - x_) * np.cos(pa_) * np.cos(dec_*np.pi/180) - (dec_ - y_) * np.sin(pa_)
        dec_prime =  (ra_  - x_) * np.sin(pa_) * np.cos(dec_*np.pi/180) + (dec_ - y_) * np.cos(pa_)
        a_prime  =  a_*ba_ # *np.cos(dec*np.pi/180)
        b_prime  =  a_
        ang      = ra_prime**2/a_prime**2 + dec_prime**2/b_prime**2
        igala_   = igala_[ang < 1]
        if len(igala_) == 0: continue
        indx[igala_] = True
    return indx 

def ellipse_plot(x, y, a, pa, ba, lonlat = True): 
    theta = np.arange(0, 2*np.pi, 0.01)
    x_prime= a*ba * np.cos(theta)#*np.cos(y*np.pi/180)
    y_prime= a    * np.sin(theta)
    pa     = -pa*np.pi/180.0
    rotation_matrix = np.array([[np.cos(pa), -np.sin(pa)],
                                [np.sin(pa),  np.cos(pa)]])
    rotated_ellipse = np.dot(rotation_matrix, np.array([x_prime, y_prime])) 
    dec  = rotated_ellipse[1, :] + y 
    if lonlat: 
        ra   = rotated_ellipse[0, :]/np.cos(dec*np.pi/180) + x
    else: 
        ra   = rotated_ellipse[0, :] + x
    return ra, dec

def ellipse_masking(ra, dec, u, v, a, pa, ba):  
    ''' 
    Returns a boolean array of the same shape as ra that is True where the position 
(ra, dec) is in the ellipse shape. 

    parameters
    ----------
    ra, dec:  float, scalars or array-like
	Angular coordinates of input targets on the sphere
    u, v: float, scalars or array-like
	Angular coordinates of central point of ellipse on the sphere
    a: float, scalars or array-like
	The length of Semi-major axis of ellipse [degree] 
    pa:  float, scalars or array-like
	Position angle [degree]. PA == 0 is North (ΔDEC>0), PA = 90 is WEST (ΔRA > 0). 
    ba:  float, scalars or array-like
	b/a [0,1]. if ba == 1, the shape is circle, namely the cap on the sphere. 
    Returns
    -------
    vetomap: ndarray
	 An array with size = 12*nside*nside. 1.0 means w/i survey; 0.0 means w/o survey.  
    nside: int 
         nside of the healpy pixelization
    ''' 

    u   = np.atleast_1d(u) 
    v   = np.atleast_1d(v) 
    if isinstance(a,  (int, float)): a   = u*0.0 + a 
    if isinstance(pa, (int, float)): pa  = u*0.0 + pa 
    if isinstance(ba, (int, float)): ba  = u*0.0 + ba 
    a   = np.atleast_1d(a) 
    pa  = np.atleast_1d(pa) 
    ba  = np.atleast_1d(ba) 

    indx     = np.zeros(len(ra)).astype(bool) 
    iscircle = (ba==1)
    if np.sum( iscircle) != 0: 
        # print('Number of circle masking:', np.sum(iscircle))
        indx1     = __circle_masking(ra, dec, 
                                   u[iscircle], v[iscircle], a[iscircle])
        indx[indx1] = True
    if np.sum(~iscircle) != 0: 
        # print('Number of ellipse masking:', np.sum(~iscircle))
        indx2    = __ellipse_masking(ra, dec, 
                                   u[~iscircle],  v[~iscircle], a[~iscircle], 
                                   pa[~iscircle], ba[~iscircle]) 
        indx[indx2] = True
    return indx

def read_mskfile(mskfile): 
    tab  = Table.read(mskfile, format = 'fits', memmap = True)
    col0 = tab.colnames[0]
    if tab[col0].dtype.kind != 'f': 
        pass; # the 0th col is the mask name
    else: 
        tab.add_column(np.arange(len(tab)), 0, name = 'imask')
    ntab = len(tab.colnames)
    expect_colnames = ['imask', 'ra', 'dec', 'sma', 'pa', 'ba', 'w']
    if ntab <  4: ValueError('ValueError: masking table less than 3 columns') 
    if ntab == 4: colnames = ['imask', 'ra', 'dec', 'sma']
    if ntab == 5: colnames = ['imask', 'ra', 'dec', 'sma', 'w']
    if ntab == 6: colnames = ['imask', 'ra', 'dec', 'sma', 'pa', 'ba']
    if ntab == 7: colnames = ['imask', 'ra', 'dec', 'sma', 'pa', 'ba', 'w']
    tab.rename_columns(tab.colnames, colnames)
    if not 'pa' in colnames: tab['pa'] = 0.0
    if not 'ba' in colnames: tab['ba'] = 1.0
    if not 'w'  in colnames: tab['w']  = 1.0
    return tab[expect_colnames]

if __name__ == '__main__': 
      p = PipeUI()
