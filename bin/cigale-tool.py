#!/usr/bin/env python

import os 
import sys 
import argparse
import subprocess 
import math 
import shutil
import numpy as np
import glob
import datetime
from astropy.table import Table, vstack, unique, join

terminal_width = shutil.get_terminal_size().columns
current_dir    = os.getcwd() 

# sed -e "/data_file/s/^data_file*/da/" pcigale.ini | head -n 20
# tt     = tt[(tt['redshift'] > 0)&(tt['redshift'] < 9999)&(~np.isnan(tt['redshift']))&(~np.isinf(tt['redshift']))]

# sed -i -e "s/cores.*/cores = 10/g" */run*/pcigale.ini
# sed -i -e "s/redshift_decimals.*/redshift_decimals = 2/g" */run*/pcigale.ini

# import numpy as np
# import shutil
# import healpy as hp
# import matplotlib.pyplot as plt
# from astropy.table import Table

class PipeUI(object):
    def __init__(self):    
        parser = argparse.ArgumentParser(
            description="cigale pipeline control",
            usage="""cigale <command> [options]

Where supported commands are (use lsscat_pipe <command> --help for details):
   (------- High-Level -------)
   build    build the production directory by divided input.cat into serveral subdirs. 
   fitting  fitting the catalogs in the serveral subdirs
   updatepbs 
   summary  summary the catalogs into one files. 
   status   Overview of fitting. 
   tractor  generate input catalog from lsdr9 tractor catalogs
   rmnull      
        t = t.filled(-99)
        try: 
            t1['igal'] = t1['igal'].filled(999999)
            t2['igal'] = t2['igal'].filled(999999)
        except:
            pass    
USAGE: 
pipeUI.py build lsdr9_z0.sed_for_cigale.fits \
          -c pcigale_north_ngc.ini \
          -o ./ext_north_ngc0/ \
          --number_in_blocks 100000 \
          --resort
pipeUI.py fitting --proddir ./ext_north_ngc0/ --ntask_torque 5 --submit 
pipeUI.py status ./ext_north_ngc0/ # check finished or not
pipeUI.py summary --proddir ./ext_north_ngc0/
pipeUI.py tractor sweep-240p055-250p060.fits --newzfile specfuji.fits --output lsdr9_for_cigale.fits --id 'id' --redshift 'Z' --method 'targetid'
   """)
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            sys.exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)() # 获取一个对象的属性值或方法 

    def rmnull(self): 
        parser = argparse.ArgumentParser(
            description="Run the total pipelines to the Production directory",
            usage="desi_pipe run [options] (use --help for details)") 
        parser.add_argument('inputfile',  type=str, help='The tractor like catalog')
        parser.add_argument('outputfile', type=str, help='Output file path')
        parser.add_argument('--filledcols', nargs='+', default=None)
        parser.add_argument('--filledvals', nargs='+')

        args = parser.parse_args(sys.argv[2:]) 
        # Use the arguments
        inputfile    = args.inputfile
        outputfile   = args.outputfile
        filledcols   = args.filledcols 
        filledvals   = args.filledvals
        t = Table.read(inputfile) 
        print(t)
        print(filledcols)
        print(filledvals)
        nloop = np.min([len(filledcols), len(filledvals)]) 
        for ii in range(nloop): 
            filledcol = filledcols[ii]
            filledval = filledvals[ii]
            try:
                filledval = float(filledval) if '.' in filledval else int(filledval) 
            except: 
                pass
            print('Filling col%s'%ii, 'colname = %s'%filledcol, 'filledval = %s'%filledval)  
            t[filledcol] = t[filledcol].filled(filledval)
        if nloop < len(filledvals): 
            filledval = filledvals[-1]
            try:
                filledval = float(filledval) if '.' in filledval else int(filledval) 
                t = t.filled(filledval) 
                print('The reamining colnames are filled by %s'%filledval)  
            except: 
                pass
        print(t)
        print() 
        print('Writing into %s'% outputfile )
        print() 
        t.write(outputfile, overwrite = True) 
        print('rmnull done!') 
        print() 

    def fitsvstack(self): 
        # Create the parser
        parser = argparse.ArgumentParser(description='Load a file path.')
        parser.add_argument('filepath', type=str, nargs = '+', help='The path to the file.')
        parser.add_argument('output', type=str, help='The path to output file.')
        parser.add_argument('--uniquekey', type=str, default = '', required=False,help='The path to output file.')

        args   = parser.parse_args(sys.argv[2:])
        filepath = args.filepath
        output   = args.output
        uniquekey= args.uniquekey

        # Use glob to get all the file paths
        if len(filepath)==1: 
            filepaths = glob.glob(filepath[0])
        else: 
            filepaths = filepath
        # Initialize an empty list to store arrays
        t = []; uniqueval = np.array([])
        # Loop through the file paths and read the data
        for filepath in filepaths:
            t_ = Table.read(filepath, memmap = True)  # Assuming the files are text files with numerical data
            print(filepath, len(t_), end = ';  ')
            if uniquekey != '':  
                indx = np.isin(t_[uniquekey], uniqueval) 
                uniqueval = np.append(uniqueval, t_[uniquekey][~indx]) 
                t_   = t_[~indx]
            print(len(t_), 'are appended.') 
            t.append(t_)
        # Stack the arrays vertically
        t  = vstack(t)
        print(t)
        t.write(output, overwrite = True)



    def tractor(self): 
        parser = argparse.ArgumentParser(
            description="Run the total pipelines to the Production directory",
            usage="desi_pipe run [options] (use --help for details)") 
        parser.add_argument('tractorcat',  type=str, help='The tractor like catalog')
        parser.add_argument('--newzfile', type=str, required=True, help='Path to the new redshift file')
        parser.add_argument('--output', type=str,  default = 'lsdr9_for_cigale.fits', help='Output file path')
        parser.add_argument('--id', type=str, default='id', help='ID parameter')
        parser.add_argument('--redshift', type=str, default='Z', help='Redshift parameter')
        parser.add_argument('--method', type=str, default='targetid', help='Method parameter')
        parser.add_argument('--region', type=str, default=None, help='Method parameter')
        args = parser.parse_args(sys.argv[2:]) 

        # Use the arguments
        tractorcat = args.tractorcat
        newzfile   = args.newzfile
        output     = args.output
        id         = args.id
        redshift   = args.redshift
        method     = args.method
        region     = args.region
        print(args) 
        tsed       = tractorcat2cigalesed(tractorcat, newzfile, method = method, id = id, region=region, 
                                          redshift = redshift)
        print(tsed)
        dirname = os.path.dirname(os.path.abspath(output)) 
        os.makedirs(dirname, exist_ok=True)
        tsed.write(output, overwrite = True)
#------------------------------------------------------------------
    def build(self): 
        parser = argparse.ArgumentParser(
            description="Run the total pipelines to the Production directory",
            usage="desi_pipe run [options] (use --help for details)") 
        parser.add_argument('input', metavar='datafile', type=str, 
                            help="data file")
        parser.add_argument("-c", "--config", default="pcigale.ini", type=str, 
                            help="config file")
        parser.add_argument("-o", "--proddir", default='./output/', type=str, # required = False,  
                            help="Production directory to store the output")
        parser.add_argument("--nblock", default=1, type=int, # required = False,  
                            help="number of blocks") 
        parser.add_argument("-n", "--number_in_blocks", default=1, type=int, # required = False,  
                            help="number_in_blocks") 
        parser.add_argument("-nhdr", "--nheader", default=1, type=int, # required = False,  
                            help="line number of header")
        parser.add_argument("--resort", action='store_true', default=False,
                            help='resort the catalog by redshifts') 
        args = parser.parse_args(sys.argv[2:]) 
        self.args = vars(args)
        print(self.args) 
        self.datafile   = args.input
        self.config     = args.config
        self.output_dir = args.proddir 
        self.nblock     = args.nblock
        self.number_in_blocks     = args.number_in_blocks
        self.nheader    = args.nheader
        self.issort      = args.resort
        #-------------------------------------------
        # check if the output directory exists 
        os.makedirs(self.output_dir, exist_ok = True)  

        # cmd = 'sed -e "s/data_file.*/data_file = %s/g" pcigale.ini  %s | head -n 20 '%(self.datafile.split('/')[-1], self.config) 
        current_config = os.path.join(self.output_dir, 'pcigale.ini') 
        cmd    = 'sed -e "s/data_file.*/data_file = %s/g" %s > %s'%(self.datafile.split('/')[-1], self.config, current_config) 
        output = subprocess.run(cmd, shell = True, capture_output=True, text=True)
        cmd = 'cp %s %s'%(self.config.replace('.ini', '.ini.spec'), current_config.replace('.ini', '.ini.spec')); print(cmd)
        output = subprocess.run(cmd, shell = True, capture_output=True, text=True)

        # print(output.stdout) 

        if self.datafile.split('.')[-1] == 'fits': 
            nlines_in_blocks, newdirs = split_fits( self.datafile, self.output_dir, resort = self.issort, number_in_blocks = self.number_in_blocks)
        else: 
            nlines_in_blocks, newdirs = split_text( self.datafile, self.output_dir, nblock = self.nblock, resort = self.issort, number_in_blocks = self.number_in_blocks, nheaders = self.nheader, )

        for ii in range(len(newdirs)): 
            newdir    = newdirs[ii]
            newconfig = os.path.join(newdir, 'pcigale.ini') 
            cmd = 'cp %s %s'%(current_config, newconfig); print(cmd)
            output = subprocess.run(cmd, shell = True, capture_output=True, text=True)   
            # cmd   = 'sed -e "s/redshift_decimals.*/redshift_decimals = 2/g; s/cores.*/cores = 4/g" %s > %s'%(current_config, newconfig) 
            # output = subprocess.run(cmd, shell = True, capture_output=True, text=True)    
            cmd = 'cp %s %s'%(current_config.replace('.ini', '.ini.spec'), newconfig.replace('.ini', '.ini.spec')); print(cmd)
            output = subprocess.run(cmd, shell = True, capture_output=True, text=True)    

    def fitting(self): 
        parser = argparse.ArgumentParser(
            description="Run the total pipelines to the Production directory",
            usage="desi_pipe run [options] (use --help for details)") 
        parser.add_argument("--proddir", default='./output/', type=str, # required = False,  
                            help="Production directory to store the output")
        parser.add_argument("--iblocks", default=None, type=int, nargs = '*', # required = False,  
                            help="number id of blocks to fit") 
        parser.add_argument("--ntask_torque", default = 0, type=int, 
                            help='submit ntask through torque') 
        parser.add_argument("--submit", action='store_true', default = False, 
                            help='submit directly or not') 
        args = parser.parse_args(sys.argv[2:]) 
        self.args = vars(args)
        self.output_dir = args.proddir 
        self.iblocks    = args.iblocks
        if args.iblocks is None: 
            subdirs = glob.glob(os.path.join(self.output_dir, 'run*') )
            self.iblocks = np.array([subdir.split('run')[-1] for subdir in subdirs]).astype('i')
            self.iblocks = np.sort(self.iblocks)
        print()
        print('# running cigale '.ljust(terminal_width, '#') )
        print('# nblock = ', len(self.iblocks) )

        if args.ntask_torque > 0: 
            systimestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            scriptname = os.path.join(self.output_dir, 'job-submit-all.qsub')
            PBStasks = ''
            for iblocks in self.iblocks: PBStasks = PBStasks + '%s,'%iblocks
            PBStasks = PBStasks[:-1]
            PBStasks = PBStasks + '%' + '%s'% args.ntask_torque
            torque_script(scriptname, PBStasks) 

            if args.submit: 
                os.chdir(current_dir) 
                os.chdir(self.output_dir); 
                cmd = 'qsub ' + 'job-submit-all.qsub'; print(cmd)
                output = subprocess.run(cmd, shell = True, capture_output=True, text=True)
                os.chdir(current_dir) 
        else: 
            for iblock in self.iblocks: 
                newdir      = os.path.join(self.output_dir, 'run%04i'%iblock) 
                print('working on %s'%newdir)
                script_cigale_in_fold(newdir) 

    def updatepbs(self): 
        parser = argparse.ArgumentParser(
            description="Run the total pipelines to the Production directory",
            usage="desi_pipe run [options] (use --help for details)") 
        parser.add_argument("proddirs", default='./output/', type=str, nargs = '+', # required = False,  
                            help="Production directory to store the output")
        args = parser.parse_args(sys.argv[2:]) 
        self.args = vars(args)
        for output_dir in args.proddirs: 
            print(output_dir)
            update_pbs(output_dir)

    def status(self): 
        parser = argparse.ArgumentParser(
            description="Run the total pipelines to the Production directory",
            usage="desi_pipe run [options] (use --help for details)") 
        parser.add_argument("proddirs", default='./output/', type=str, nargs = '+', # required = False,  
                            help="Production directory to store the output")

        args = parser.parse_args(sys.argv[2:]) 
        self.args = vars(args)
        for output_dir in args.proddirs: 
            check_status(output_dir) 
            
    def summary(self): 
        parser = argparse.ArgumentParser(
            description="Run the total pipelines to the Production directory",
            usage="desi_pipe run [options] (use --help for details)") 
        parser.add_argument("--proddir", default='./output/', type=str, # required = False,  
                            help="Production directory to store the output")
        parser.add_argument("--iblocks", default=None, type=int, nargs = '*', # required = False,  
                            help="number id of blocks to summary") 
        parser.add_argument("--remain_colnames", 
                            default = ['id', 'best.universe.redshift', 'best.stellar.m_star', 
                                       'bayes.stellar.m_star',
                                       'best.sfh.sfr',  'best.sfh.sfr10Myrs', 'best.sfh.sfr100Myrs',   # 'best.dust.luminosity', 
                                       'bayes.sfh.sfr', 'bayes.sfh.sfr10Myrs', 'bayes.sfh.sfr100Myrs', # 'bayes.dust.luminosity', 
                                       'best.chi_square', 'best.reduced_chi_square'], 
                            type=str, nargs = '*', 
                            help="the names of remaining colunms")
        parser.add_argument("--sort_by_id", action='store_true', default=False,
                            help='sort by the id') 
        
        args = parser.parse_args(sys.argv[2:]) 
        self.args = vars(args)
        self.output_dir = args.proddir 
        self.iblocks    = args.iblocks
        self.remain_colnames = args.remain_colnames
        sort_by_id = args.sort_by_id
        
    
        if args.iblocks is None: 
            subdirs = glob.glob( os.path.join(self.output_dir, 'run*') )
            # self.iblocks = np.array([subdir.split('run')[-1] for subdir in subdirs]).astype('i')
            # self.iblocks = np.sort(self.iblocks)
        print()
        print('# summary cigale '.ljust(terminal_width, '#') )
        print('  remain_colnames:', self.remain_colnames) 
        print('  sort_by_id:', sort_by_id) 
        print()
        print(f'summarying {self.output_dir}')
        print()

        print()

        tab = []
        for subdir in subdirs: 
            # newdir      = os.path.join(self.output_dir, 'run%04i'%iblock) 
            newresult   = os.path.join(subdir, 'out', 'results.fits')
            if os.path.exists(newresult) is False: 
                print('%s is not found.'% newresult); continue
            # print('reading: ', newresult, end = '  ')
            t  = Table.read(newresult, memmap = True)
            # print(len(t))
            if len(self.remain_colnames) == 0: self.remain_colnames = t.colnames 
            tab.append(t[self.remain_colnames]) 
        print()
        print('stacking...', end = '  ')
        if len(tab) == 1: 
            tab = tab[0]
        else: 
            tab = vstack(tab)
        print(len(tab))
        print()
        newdir      = os.path.join(self.output_dir, 'summary.fits')
        print(tab)

        if sort_by_id: 
            print('# ---- sort by id')
            tab.sort('id')
            print(tab)
        print('saving into:', newdir)
        tab.write(newdir, overwrite = True)
        print()
        print('end')
# '''
# id           bayes.sfh.sfr       bayes.sfh.sfr_err    bayes.sfh.sfr100Myrs  bayes.sfh.sfr100Myrs_err     bayes.sfh.sfr10Myrs  bayes.sfh.sfr10Myrs_err           bayes.BASS_g       bayes.BASS_g_err          bayes.BASS_r        bayes.BASS_r_err         bayes.MzLS_z        bayes.MzLS_z_err          bayes.WISE1        bayes.WISE1_err          bayes.WISE2        bayes.WISE2_err     best.chi_square  best.reduced_chi_square  best.attenuation.B_B90  best.attenuation.E_BVs.stellar.old  best.attenuation.E_BVs.stellar.young  best.attenuation.FUV  best.attenuation.V_B90  best.attenuation.ebvs_old_factor  best.attenuation.powerlaw_slope  best.attenuation.uv_bump_amplitude  best.attenuation.uv_bump_wavelength  best.attenuation.uv_bump_width  best.sfh.age  best.sfh.burst_age  best.sfh.f_burst  best.sfh.tau_burst  best.sfh.tau_main  best.stellar.age_m_star  best.stellar.imf  best.stellar.metallicity  best.stellar.old_young_separation_age   best.universe.age  best.universe.luminosity_distance  best.universe.redshift  best.attenuation.stellar.old  best.attenuation.stellar.young    best.dust.luminosity  best.sfh.integrated            best.sfh.sfr     best.sfh.sfr100Myrs      best.sfh.sfr10Myrs        best.stellar.lum     best.stellar.lum_ly  best.stellar.lum_ly_old  best.stellar.lum_ly_young    best.stellar.lum_old  best.stellar.lum_young  best.stellar.m_gas  best.stellar.m_gas_old  best.stellar.m_gas_young  best.stellar.m_star  best.stellar.m_star_old  best.stellar.m_star_young       best.stellar.n_ly   best.stellar.n_ly_old  best.stellar.n_ly_young           best.BASS_g           best.BASS_r          best.MzLS_z           best.WISE1            best.WISE2
# '''
def script_cigale_in_fold(datadir): 
    os.chdir(datadir) 
    cmd = 'pcigale run'; # print(cmd)
    os.system(cmd) 
    # output = subprocess.run(cmd, shell = True, capture_output=True, text=True)
    # print(output.stdout)
    os.chdir(current_dir) 

def split_fits(datafile, output_dir, 
                 number_in_blocks=100000, 
                 lolimz=0, 
                 uplimz=9999, 
                 resort=False, 
                 zstart_by_step=0.0):
    print() 
    print('Running split_fits...')
    print() 
    resort_by_redshift = resort
    print(f"Datafile: {datafile}")
    print(f"Number in blocks: {number_in_blocks}")
    print(f"Lower limit of redshift: {lolimz}")
    print(f"Upper limit of redshift: {uplimz}")
    print(f"Resort by redshift: {resort_by_redshift}")
    print(f"Start by step: {zstart_by_step}")
    t = Table.read(datafile, memmap = True)
    ndata  =len(t)
    zmax   = np.max(t['redshift'])
    zmin   = np.min(t['redshift'])
    print('datafile:', datafile) 
    print('ndata:',    ndata) 
    print('redshift range:', np.min(zmin), np.max(zmax) )

    if not lolimz is None:
        indx   = (t['redshift'] <  lolimz)|(np.isnan(t['redshift']))|(t['redshift'] <= lolimz)
        print(f'Found {np.sum(indx)} redshifts are < {lolimz}, nan, or inf. --> Set to z = {lolimz}; ') 
        t['redshift'][indx] = lolimz
        try: 
            t['redshift'] = t['redshift'].filled(lolimz)
        except: 
            pass 
    if not uplimz is None:
        indx   = t['redshift'] >  uplimz
        print(f'Found {np.sum(indx)} redshifts are > {uplimz}. --> Set to z = {uplimz}; ') 
        t['redshift'][indx] = uplimz
    print()
    print('Do the data partition; calculate the number in each block')
    print()
    zstart_by_step = int(zstart_by_step*100)/100 
    num_in_blocks = []
    if resort_by_redshift: 
        print('--> resort by redshift')
        t.sort('redshift')
        indx1  = t['redshift'] <= lolimz
        num_in_blocks.append(np.sum(indx1)) 
        indx2  =(t['redshift'] <  zstart_by_step)&(t['redshift'] > lolimz) 
        indx3  = t['redshift'] >= zstart_by_step
        if np.sum(indx2) > 0: 
            ntotal = np.sum(indx2)
            nblock = int(ntotal/number_in_blocks) 
            # num_in_blocks.append()
            num_in_blocks.extend([number_in_blocks] * nblock )
            num_in_blocks.extend([ntotal - number_in_blocks*nblock]) 
        if np.sum(indx3) > 0: 
            redshift = t['redshift'][t['redshift'] >= zstart_by_step]
            bins = np.arange(zstart_by_step, np.max(redshift)+0.01, 0.01) 
            hist, edge = np.histogram(redshift, bins)
            num_in_blocks.extend(list(hist)) 
    else: 
        ntotal = np.sum(len(t))
        nblock = int(ntotal/number_in_blocks) 
        num_in_blocks.extend([number_in_blocks] * nblock )
        num_in_blocks.extend([ntotal - number_in_blocks*nblock]) 
    num_in_blocks = np.array(num_in_blocks) 
    num_in_blocks = num_in_blocks[ num_in_blocks != 0 ]
    accumulatedid = np.cumsum( np.insert(num_in_blocks, 0, 0) ); 
    newdirs = []
    for iblock in range(len(num_in_blocks)):
        newdir = os.path.join(output_dir, 'run%04i'%iblock)
        os.makedirs(newdir, exist_ok = True)
        newdirs.append(newdir)
        newfile= os.path.join(newdir, datafile.split('/')[-1])
        print(f'Writing {newfile}')
        leftid  = accumulatedid[iblock]
        rightid = accumulatedid[iblock+1]
        t_      = t[leftid:rightid] 
        t_.write(newfile, overwrite = True) 
    return num_in_blocks, newdirs

def split_fits_old(datafile, output_dir, nblock = 1, resort = False, number_in_blocks = None): 
    t = Table.read(datafile)
    ndata=len(t)
    print('datafile:', datafile) 
    print('ndata:',  ndata) 
    if resort: 
        abnormal = (t['redshift'] < 0)|(np.isinf(t['redshift']))|(np.isnan(t['redshift']))
        if np.sum(abnormal) > 0: t['redshift'][abnormal] = 0.0 
        try: 
            t['redshift'] = t['redshift'].filled(0)
        except: 
            pass
        t.sort('redshift')
    nlines_in_blocks = []; newdirs = []
    if np.sum(t['redshift']==0) > 0: 
        t_=t[t['redshift']==0]
        print(t_[:3])
        t =t[t['redshift']!=0]
        iblock = 0
        newdir = os.path.join(output_dir, 'run%04i'%iblock)
        newdirs.append(newdir)
        newfile= os.path.join(newdir, datafile.split('/')[-1])
        os.makedirs(newdir, exist_ok = True)
        t_.write(newfile, overwrite = True) 
        nlines_in_blocks.append(len(t_) )  
        print(newdir)
        print('iblock:', 0, '(block of redshift == 0)') 
        print('number in blocks:', len(t_) )  
        ndata=len(t)
        print('Remaining records %s'%ndata)

    if number_in_blocks is not None: 
        nblock = math.ceil(ndata/number_in_blocks) 
    else: 
        number_in_blocks = math.ceil(ndata/nblock) 
    print('nblock:', nblock) 
    print('number in blocks:', number_in_blocks) 
    for iblock in range(nblock): 
        t_=t[iblock*number_in_blocks:(iblock+1)*number_in_blocks]
        iblock = iblock + 1
        newdir = os.path.join(output_dir, 'run%04i'%iblock)
        print(newdir)
        newdirs.append(newdir)
        newfile= os.path.join(newdir, datafile.split('/')[-1])
        os.makedirs(newdir, exist_ok = True)
        print('block %s in %s'%(iblock, nblock) )
        print(t_[:3])
        t_.write(newfile, overwrite = True) 
        nlines_in_blocks.append(len(t_) )  
    if number_in_blocks*nblock >= ndata: # 总数不够
        print('numbers in each block are', nlines_in_blocks) 
    else: 
        Warning('number_in_blocks*nblock < ndata') 
    return nlines_in_blocks, newdirs

def split_text(datafile, output_dir, nblock = 1, resort = False, nheaders = 1, number_in_blocks = None): 
    # splitcat into 
    f = open(datafile, 'r') 
    lines = f.readlines(); 
    f.close()

    headers = lines[:nheaders]
    lines   = np.char.array(lines[nheaders:])
    ndata   = len(lines)
    print('datafile:', datafile) 
    print('nheader:', nheaders) 
    print('ndata:',  ndata) 
    print('nblock:', nblock) 

    if resort: 
        redshifts = np.array([line.split()[1] for line in lines]).astype('f8')
        lines     = lines[np.argsort(redshifts)]
        redshifts = redshifts[np.argsort(redshifts)] 
        abnormal  = (redshifts < 0)|(np.isinf(redshifts))|(np.isnan(redshifts))
        if np.sum(abnormal) > 0: redshifts[abnormal] = 0.0

    # fsh=open(os.path.join(output_dir, 'run.sh'), 'w')
    # for ipart in range(0, nblock):
    #     cmd = 'cd run%04i;cp ../pcigale.ini.spec .; cp ../pcigale.ini .; pcigale run; cd ../ \n'%ipart
    #     print(cmd)
    #     fsh.writelines(cmd)
    # fsh.close()
    if number_in_blocks is not None: 
        nblock = math.ceil(ndata/number_in_blocks) 
    else: 
        number_in_blocks = math.ceil(ndata/nblock) 
    nlines_in_blocks = []; newdirs = []
    for iblock in range(nblock): 
        newdir = os.path.join(output_dir, 'run%04i'%iblock)
        newdirs.append(newdir) 
        newfile= os.path.join(newdir, datafile)
        os.makedirs(newdir, exist_ok = True)  
        f=open(newfile, 'w')
        f.writelines(headers) 
        newlines = lines[iblock*number_in_blocks:(iblock+1)*number_in_blocks]
        nlines_in_blocks.append(len(newlines) ) 
        f.writelines(newlines)
        f.close()
    if number_in_blocks*nblock >= ndata: # 总数不够
        print('numbers in each block are', nlines_in_blocks) 
    else: 
        Warning('number_in_blocks*nblock < ndata') 
    return nlines_in_blocks, newdirs

def torque_script(filename, PBStasks='0', debug = True): 
    '''
    72cores = 376 G
    1 core -- 5G
    '''
    f = open(filename, 'w') 
    print('#!/bin/sh', file=f)
    print('#PBS -N cigale', file=f)
    print('#PBS -l nodes=1:ppn=1', file=f)
    print('#PBS -l mem=20gb', file=f)
    print('#PBS -l walltime=3:00:00', file=f)
    print('#PBS -q debug', file=f)
    print('#PBS -o job.out', file=f)
    print('#PBS -e job.err', file=f)
    print('#PBS -t %s'%PBStasks, file=f)
    print() 

    print('cd $PBS_O_WORKDIR', file=f)
    print('module load anaconda/anaconda3', file=f)
    print('source activate', file=f)
    print('conda deactivate', file=f)
    print('conda activate cigale2022.1', file=f)
    print('export MKL_NUM_THREADS="1" ', file=f)
    print('export NUMEXPR_NUM_THREADS="1" ', file=f)
    print('export OMP_NUM_THREADS="1" ', file=f)

    print('arrayid=$(printf "%04d" $PBS_ARRAYID)', file=f)
    print('echo "START" run$arrayid', file=f)
    print('cd run$arrayid;', file=f)
    print('pcigale run;', file=f)
    print('cd ../', file=f)
    print('echo "END" run$arrayid', file=f)
    f.close()


# def summray(): 
#     import numpy as np
#     from astropy.table import Table

#     outputnames = ['sweepnorth_ngc', 'sweepsouth_ngc', 'sweepsouth_sgc']
#     for outputname in outputnames: 
#         iseq, redshift = np.genfromtxt('%s/%s_for_cigale.cat'%(outputname, outputname), dtype = 'str', usecols=[0,1], unpack = True, skip_header=1) # ,max_rows = 4)
#         iseq = np.char.array(iseq, unicode = 'utf8')
#         iseq = np.char.split(iseq, sep ='-')
#         brickid = np.array([iseq_[0] for iseq_ in iseq]).astype(int)
#         objid   = np.array([iseq_[1] for iseq_ in iseq]).astype(int)
#         redshift= redshift.astype('float')
#         print(outputname, brickid)
#         print(outputname, objid)
#         print(outputname, redshift)
#         lmass = np.genfromtxt('cat/%s.txt'%(outputname), dtype = 'float', usecols=2, unpack = True, skip_header=0) #, max_rows = 4)
#         lmass = np.log10(lmass)
#         lmass = lmass  + np.log10(0.7) 
#         print(lmass, np.log10(0.7) )
#         t = Table() 
#         t['brickid'] = brickid
#         t['objid']   = objid 
#         t['redshift']= redshift
#         t['lmass']   = lmass
#         t.write('%s.fits'%outputname, overwrite = True)

    # print( iseq.char.split('-') )
def update_pbs(output_dir): 
    iblocks_failed = check_status(output_dir)
    if len(iblocks_failed) == 0: 
        exit() 
    # update PBStasks 
    PBStasks = ''
    for iblocks in iblocks_failed: 
        PBStasks = PBStasks + '%s,'%iblocks
    PBStasks = PBStasks[:-1]
    scriptname = os.path.join(output_dir, 'job-complement.qsub')
    print('update PBStasks %s'%PBStasks, ' into ', scriptname)
    torque_script(scriptname, PBStasks) 

def check_status(output_dir): 
    subdirs = glob.glob( os.path.join(output_dir, 'run*') )
    iblocks = np.array([subdir.split('run')[-1] for subdir in subdirs]).astype('i')
    iblocks = np.sort(iblocks)
    print()
    print('# status of cigale '.ljust(terminal_width, '#') )
    print(output_dir)
    print()
    iblocks_failed = [] 
    for iblock in iblocks: 
        newdir      = os.path.join(output_dir, 'run%04i'%iblock) 
        newresult   = os.path.join(newdir, 'out', 'results.fits')
        if os.path.exists(newresult): 
            pass; # print('exist.')
        else: 
            print('check:', newresult, end = ' ')
            print('No exits.') 
            iblocks_failed.append(iblock)
    return iblocks_failed


class desi: 
    def checklsdr9_dec32375(ra, dec, release): 
        '''
        32.375度规则
        (l < 0): release = 9010/9012
        (l > 0)&(δ < 32.375): release = 9010/9012
        (l > 0)&(δ > 32.375): release = 9011 
        ''' 
        from astropy.coordinates import SkyCoord
        c_icrs = SkyCoord(ra=ra, dec=dec, unit = 'degree', frame='icrs') 
        l = c_icrs.galactic.l.degree
        b = c_icrs.galactic.b.degree
        indxa  =  release == 9011
        indxb  = (release == 9010)|(release == 9012)

        indx1  = (b >= 0)&(dec >= 32.375)
        indx2  = (b >= 0)&(dec <  32.375)
        indx3  = (b <  0)
        indx_n  = (indx1&indxa)
        indx_s  = (indx2&indxb)|(indx3&indxb) 
        return indx_n, indx_s
 
    def decode_tragetid(targetid):  
        # print(bin(targetid)[2:].rjust(63, '0')  ) 
        # print(bin(((1 << 22) - 1) <<  0)[2:].rjust(63, '0')  ) 
        # print(bin(((1 << 20) - 1) << 22)[2:].rjust(63, '0')  ) 
        # print(bin(((1 << 16) - 1) << 42)[2:].rjust(63, '0')  ) 
        # print(bin(((1 <<  1) - 1) << 58)[2:].rjust(63, '0')  ) 
        # print(bin(((1 <<  1) - 1) << 59)[2:].rjust(63, '0')  ) 
        # print(bin(((1 <<  2) - 1) << 60)[2:].rjust(63, '0')  ) 
        targetid = np.atleast_1d(targetid)
        negative_tragetid = targetid <  0 
        objid   = (targetid & ( ((1 << 22) - 1) <<  0) ) >>  0
        brickid = (targetid & ( ((1 << 20) - 1) << 22) ) >> 22
        release = (targetid & ( ((1 << 16) - 1) << 42) ) >> 42
        ismock  = (targetid & ( ((1 <<  1) - 1) << 58) ) >> 58
        issky   = (targetid & ( ((1 <<  1) - 1) << 59) ) >> 59
        gaiadr  = (targetid & ( ((1 <<  2) - 1) << 60) ) >> 60
        if np.sum(negative_tragetid) > 0: 
            objid[negative_tragetid]   = -1
            brickid[negative_tragetid] = 0
            release[negative_tragetid] = 0
            ismock[negative_tragetid]  = 0
            issky[negative_tragetid]   = -1
            gaiadr[negative_tragetid]  = -1
        return release, brickid, objid, ismock, issky, gaiadr

def tractorcat2cigalesed(tractorcat, desispecfile, method = 'targetid', id = None, redshift = None, region = None, filled = 999999): 
    t2 = Table.read(tractorcat, memmap = True); # left catalog 
    print('left')
    print(t2)
    print(t2[999998:1000003])
    if not  id  in t2.colnames: 
        print('No "%s" columns is found'% id )
        print('Add id columns starting from 0')
        t2.add_column(np.arange(len(t2)), 0, 'id')
    else: 
        try: 
            t2['id'] = t2[id].filled(filled)
        except:
            t2['id'] = t2[id]
    t2['__leftorderid__'] = np.arange(len(t2))
    if 'redshift' in t2.colnames: del t2['redshift'] 

    
    t1 = Table.read(desispecfile, memmap = True)
    if not redshift in t1.colnames: 
        print('No "%s" columns is found'% redshift )
        if 'Z' in t1.colnames: redshift = 'Z'
        print('Using "%s" columns as "redshift" '% redshift )
    t1['redshift'] =  t1[redshift]
    t1['__rightorderid__'] = np.arange(len(t1))

    if 'id' in t1.colnames: del t1['id']
    print('right')
    print(t1)
    if isinstance(method, str):
        if method == 'targetid': 
            release, brickid, objid, ismock, issky, gaiadr = desi.decode_tragetid(t1['TARGETID']) 
            t1['RELEASE']     = release
            t1['BRICKID']     = brickid
            t1['OBJID'] = objid
            t1 = t1['RELEASE', 'BRICKID', 'OBJID', 'redshift', '__rightorderid__']
            t2 = join(t2, t1, join_type = 'inner') # , keep_order=True)
        elif method == 'releasebrickidobjid': 
            t2 = join(t2, t1, join_type = 'inner') # , keep_order=True)
        else: 
            t2 = join(t2, t1[method, 'redshift', '__rightorderid__'], join_type = 'inner')
    t2.sort('__leftorderid__')  
    flux_colnames      = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4']
    fluxivar_colnames  = ['FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FLUX_IVAR_W3', 'FLUX_IVAR_W4']
    fluxerr_colnames   = ['FLUX_ERR_G', 'FLUX_ERR_R', 'FLUX_ERR_Z', 'FLUX_ERR_W1', 'FLUX_ERR_W2', 'FLUX_ERR_W3', 'FLUX_ERR_W4']
    trans_colnames     = ['MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2', 'MW_TRANSMISSION_W3', 'MW_TRANSMISSION_W4']
    for flux_colname, trans_colname in  zip(flux_colnames, trans_colnames): 
        if not flux_colname in t2.colnames: continue
        if not trans_colname in t2.colnames: continue
        t2[flux_colname] =  t2[flux_colname]/t2[trans_colname]*3.631E-3
    for fluxivar_colname, fluxerr_colname, trans_colname  in zip(fluxivar_colnames, fluxerr_colnames, trans_colnames) : 
        if not fluxivar_colname in t2.colnames: continue
        if not trans_colname in t2.colnames: continue
        t2[fluxerr_colname] = 1.0/np.sqrt( t2[fluxivar_colname] ) # ivar to err: 
        t2[fluxerr_colname] = t2[fluxerr_colname]/t2[trans_colname]*3.631E-3 # maggy to mJy 
    if region == 'N': t2['REGION'] = 'N'
    if region == 'S': t2['REGION'] = 'S'
    keepcolnames = np.array( ['id', 'RELEASE', 'BRICKID', 'OBJID', 'REGION', 'RA', 'DEC', 'TARGET_RA', 'TARGET_DEC',  'redshift', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4', 
              'FLUX_ERR_G', 'FLUX_ERR_R', 'FLUX_ERR_Z', 'FLUX_ERR_W1', 'FLUX_ERR_W2', 'FLUX_ERR_W3', 'FLUX_ERR_W4', '__leftorderid__', '__rightorderid__'] )
    keepcolnames = list( keepcolnames[ np.isin(keepcolnames, np.array(t2.colnames) )]  ) 
    print('keepcolnames:', keepcolnames) 
    t2   = t2[keepcolnames] 
    
    filternames_in_sweep  = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4', 'FLUX_ERR_G', 'FLUX_ERR_R', 'FLUX_ERR_Z', 'FLUX_ERR_W1', 'FLUX_ERR_W2', 'FLUX_ERR_W3', 'FLUX_ERR_W4'] 
    filternames_in_cigale = filternames_in_sweep 
    if 'north' in tractorcat: filternames_in_cigale = ['BASS_g',  'BASS_r',  'MzLS_z',   'WISE1', 'WISE2', 'WISE3', 'WISE4', 'BASS_g_err',   'BASS_r_err',   'MzLS_z_err',    'WISE1_err',  'WISE2_err',  'WISE3_err',  'WISE4_err'] 
    if 'south' in tractorcat: filternames_in_cigale = ['DECaLS_g','DECaLS_r','DECaLS_z', 'WISE1', 'WISE2', 'WISE3', 'WISE4', 'DECaLS_g_err', 'DECaLS_r_err', 'DECaLS_z_err',  'WISE1_err',  'WISE2_err',  'WISE3_err',  'WISE4_err'] 
    if 'north_ngc' in tractorcat: filternames_in_cigale = ['BASS_g',  'BASS_r',  'MzLS_z',   'WISE1', 'WISE2', 'WISE3', 'WISE4', 'BASS_g_err',   'BASS_r_err',   'MzLS_z_err',    'WISE1_err',  'WISE2_err',  'WISE3_err',  'WISE4_err'] 
    if 'south_ngc' in tractorcat: filternames_in_cigale = ['DECaLS_g','DECaLS_r','DECaLS_z', 'WISE1', 'WISE2', 'WISE3', 'WISE4', 'DECaLS_g_err', 'DECaLS_r_err', 'DECaLS_z_err',  'WISE1_err',  'WISE2_err',  'WISE3_err',  'WISE4_err'] 
    if 'south_sgc' in tractorcat: filternames_in_cigale = ['DECaLS_g','DECaLS_r','DECaLS_z', 'WISE1', 'WISE2', 'WISE3', 'WISE4', 'DECaLS_g_err', 'DECaLS_r_err', 'DECaLS_z_err',  'WISE1_err',  'WISE2_err',  'WISE3_err',  'WISE4_err'] 
    if region == 'N': filternames_in_cigale = ['BASS_g',  'BASS_r',  'MzLS_z',   'WISE1', 'WISE2', 'WISE3', 'WISE4', 'BASS_g_err',   'BASS_r_err',   'MzLS_z_err',    'WISE1_err',  'WISE2_err',  'WISE3_err',  'WISE4_err'] 
    if region == 'S': filternames_in_cigale = ['DECaLS_g','DECaLS_r','DECaLS_z', 'WISE1', 'WISE2', 'WISE3', 'WISE4', 'DECaLS_g_err', 'DECaLS_r_err', 'DECaLS_z_err',  'WISE1_err',  'WISE2_err',  'WISE3_err',  'WISE4_err'] 
    for name1, name2 in zip(filternames_in_sweep, filternames_in_cigale): 
        if name1 in t2.colnames: t2.rename_column(name1, name2) 
    return t2 


if __name__ == '__main__': 
      p = PipeUI() 
