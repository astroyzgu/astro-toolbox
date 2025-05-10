#!/usr/bin/env python 
import os 
import sys
import argparse 
import re 
import numpy as np 
import logging
from astropy.table import Table, vstack

# *_group:  groupID richness ra_group dec_group  z_grp      logM_group   logL_group  
# *_galaxyID:  galaxyID, redshift
# i*_1 (4 columns): galaxyID, groupID, rank, auxiliary ID (not used)
# i*_2 (2 columns): groupID,  galaxyID
# rank: rank == 1, central (brightest); rank == 2, satallite; 

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class PipeUI(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="tool for yang's group catalog",
            usage="""gfindertool <command> [options]
            gfindertool txt2fits /path/to/group/catalog/* --output ./r19.5/ # --maxrow  3
            gfindertool merge   ./r19.5/* --newtag allsky --output ./r19.5/ # --maxrow  3
""")
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            sys.exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)() # 获取一个对象的属性值或方法 
    def merge(self): 
        parser = argparse.ArgumentParser(
            description="Run the total pipelines to the Production directory",
            usage="desi_pipe run [options] (use --help for details)")
        parser.add_argument('inlist', type=str, nargs   = '*', help='The tractor like catalog') 
        parser.add_argument('--newtag', type=str, default = 'mergesky')
        parser.add_argument('--maxrow', default=None, type = int) 
        parser.add_argument('--output', default='./', type = str) 
        args     = parser.parse_args(sys.argv[2:])
        max_rows = args.maxrow
        odir     = args.output
        newtag   = args.newtag 
        os.makedirs(odir, exist_ok=True)
        infilenames = [infilename.split('/')[-1] for infilename in args.inlist]
        tagnames = extract_tagnames(infilenames)  
        print('tagname: ', tagnames)
        grpfiles = []; galfiles = []; igalfile1s = []; igalfile2s = []; 
        for tagname in tagnames: 
            grpfile, galfile, igalfile1, igalfile2 = subfiles_of_tagname(tagname, args.inlist)  
            print(tagname, ':', grpfile, galfile, igalfile1, igalfile2) 
            galfiles.append(galfile)
            grpfiles.append(grpfile)
            igalfile1s.append(igalfile1)
            igalfile2s.append(igalfile2)

        t = []
        for ii in range(len(tagnames)): 
            t_ = Table.read(galfiles[ii], memmap = True)[:max_rows]
            t.append(t_)
        t = vstack(t)
        t.write(os.path.join(odir, newtag + '_galaxyID.fits'), overwrite = True)

        t = []
        for ii in range(len(tagnames)): 
            t_ = Table.read(grpfiles[ii], memmap = True)[:max_rows]
            t_['groupID'] = t_['groupID'] + len(t) 
            t.append(t_)
        t = vstack(t)
        t.write(os.path.join(odir, newtag + '_group.fits'), overwrite = True)

        t = []
        for ii in range(len(tagnames)): 
            t_ = Table.read(igalfile1s[ii], memmap = True)[:max_rows]
            t_['groupID'] = t_['groupID'] + len(t) 
            t.append(t_)
        t = vstack(t)
        t.write(os.path.join(odir, 'i'+newtag + '_1.fits'), overwrite = True)

        t = []
        for ii in range(len(tagnames)): 
            t_ = Table.read(igalfile1s[ii], memmap = True)[:max_rows]
            t_['groupID'] = t_['groupID'] + len(t) 
            t.append(t_)
        t = vstack(t)
        t.write(os.path.join(odir, 'i'+newtag + '_2.fits'), overwrite = True)



    def txt2fits(self): 
        parser = argparse.ArgumentParser(
            description="Run the total pipelines to the Production directory",
            usage="desi_pipe run [options] (use --help for details)")
        parser.add_argument('inlist', type=str, nargs = '*', help='The tractor like catalog')
        parser.add_argument('--maxrow', default=None, type = int) 
        parser.add_argument('--output', default='./', type = str) 
        args     = parser.parse_args(sys.argv[2:])
        max_rows = args.maxrow 
        odir     = args.output 
        os.makedirs(odir, exist_ok=True)
        print('odir: ', odir)
        infilenames = [infilename.split('/')[-1] for infilename in args.inlist]

        tagnames = extract_tagnames(infilenames)  
        print('tagname: ', tagnames)
        for tagname in tagnames: 
            grpfile, galfile, igalfile1, igalfile2 = subfiles_of_tagname(tagname, args.inlist) 
            
            ### ---

            idir    = os.path.dirname(grpfile)
            grpfile = os.path.basename(grpfile) 
            groupID,  richness, ra_group,  dec_group, z_grp, logM_group, logL_group = np.genfromtxt(
                os.path.join(idir, grpfile), max_rows=max_rows, unpack=True)
            for ii in range(3): 
                logging.debug(f"{groupID[ii].astype('int')} {richness[ii].astype('int')} {ra_group[ii]} {dec_group[ii]} {z_grp[ii]} {logM_group[ii]} {logL_group[ii]}") 
            t = Table() 
            t['groupID']   = groupID.astype('int')
            t['richness']  = richness.astype('int')
            t['ra_group']  = ra_group
            t['dec_group'] = dec_group
            t['z_grp']     = z_grp
            t['logM_group']= logM_group
            t['logL_group']= logL_group
            logging.debug(t[:3])
            print(os.path.join(idir, grpfile)) 
            print(os.path.join(odir, grpfile + '.fits')) 
            t.write( os.path.join(odir, grpfile + '.fits'), overwrite=True) 

            ### ---

            idir    = os.path.dirname(galfile)
            galfile = os.path.basename(galfile) 
            print(idir, odir, galfile) 
            galaxyID = np.genfromtxt(
                os.path.join(idir, galfile), max_rows=max_rows, unpack=True) 
            for ii in range(3): 
                logging.debug(galaxyID[ii].astype('int')) 
                
            t = Table()
            t['galaxyID']   = galaxyID.astype('int')
            logging.debug(t[:3])
            t.write( os.path.join(odir, galfile+'.fits'), overwrite=True) 

            ### ---

            idir      = os.path.dirname(igalfile1)
            igalfile1 = os.path.basename(igalfile1) 
            print(idir, odir, igalfile1) 
            galaxyID, groupID, rank, auxiliaryID = np.genfromtxt(
                os.path.join(idir, igalfile1), max_rows=max_rows, unpack=True) 
            for ii in range(3): 
                logging.debug(f"{galaxyID[ii].astype('int')} {groupID[ii].astype('int')} {rank[ii].astype('int')} {auxiliaryID[ii].astype('int')}") 
            
            t = Table()
            t['galaxyID']   = galaxyID.astype('int')
            t['groupID']    = groupID.astype('int')
            t['rank']       = rank.astype('int')
            logging.debug(t[:3])
            t.write( os.path.join(odir, igalfile1+'.fits'), overwrite=True) 
            
            ### ---

            idir      = os.path.dirname(igalfile2)
            igalfile2 = os.path.basename(igalfile2) 
            print(idir, odir, igalfile2) 

            groupID, galaxyID = np.genfromtxt(
                os.path.join(idir, igalfile2), max_rows=max_rows, usecols = (0,1), unpack=True) 
            for ii in range(3): 
                logging.debug(f"{groupID[ii].astype('int')} {galaxyID[ii].astype('int')}") 
                
            t = Table()
            t['groupID']    = groupID.astype('int')
            t['galaxyID']   = galaxyID.astype('int')
            logging.debug(t[:3])
            t.write( os.path.join(odir, igalfile2+ '.fits'), overwrite=True)

def subfiles_of_tagname(tagname, infilenames): 
    grpfile  = tagname + '_group'
    galfile  = tagname + '_galaxyID'
    igalfile1  = 'i' + tagname + '_1'
    igalfile2  = 'i' + tagname + '_2' 
    grpfile = [word for word in infilenames if grpfile in word][0]
    galfile = [word for word in infilenames if galfile in word][0]
    igalfile1 = [word for word in infilenames if igalfile1 in word][0]
    igalfile2 = [word for word in infilenames if igalfile2 in word][0]
    return grpfile, galfile, igalfile1, igalfile2

def extract_tagnames(infilenames): 
    extract1 = np.unique([re.search(r'i(.*?)_1(.*?)', x).group(1) for x in infilenames if re.search(r'i(.*?)_1', x)])
    extract2 = np.unique([re.search(r'i(.*?)_2(.*?)', x).group(1) for x in infilenames if re.search(r'i(.*?)_2', x)])
    extract3 = np.unique([re.search(r'(.*?)_galaxyID(.*?)', x).group(1) for x in infilenames if re.search(r'(.*?)_galaxyID', x)])
    extract4 = np.unique([re.search(r'(.*?)_group(.*?)', x).group(1) for x in infilenames if re.search(r'(.*?)_group', x)]) 
    uniq, count = np.unique( np.hstack([extract1, extract2, extract3, extract4]), return_counts=True ) 
    return uniq[count == 4]

if __name__ == '__main__': 
    p = PipeUI() 