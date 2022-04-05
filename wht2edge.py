#------------------------------------------------------------------------------
# Convert weight maps to the rough profile of survey 
#
# Writen by Gu Yizhou
# Refer: 
#------------------------------------------------------------------------------

from astropy.io import fits
import numpy as np
import pandas as pd
import cv2
from astropy.table import Table
from astropy.io import ascii
from descartes import PolygonPatch
from shapely.geometry import Polygon,MultiPoint  #多边形
from sklearn.cluster import DBSCAN
from astropy.nddata import block_reduce
from astropy.wcs import WCS
#import aplpy

def wht2poly_from_fits(fitsname, block_size = 20): 
    # read weight maps
    hdul = fits.open(fitsname)
    hdu  = hdul[0]
    w    = WCS(hdu.header)
    img  = np.array(hdu.data)
    hdul.close()
    # if image is too large, downsample it. 
    # If data is not perfectly divisible by block_size along a given axis
    # then the data will be trimmed (from the end) along that axis. 
    #block_size = 20
    img  = block_reduce(img, block_size) 
    # convert weight to binary map 
    indx = np.where( img > 0  ) 
    img[:, :] = 0
    img[indx] = 255
    img = img.astype(np.uint8)
    # open-cv提取轮廓
    contours, hera_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours -> Nx1x2, 保存了轮廓坐标的信息
    # hera_[0] -> Nx4 的数据, col0, col1, col2, col3 
    polygons = []
    areas    = []
    for icontour, contour in enumerate(contours): # 逐一建立轮廓  
        ipixel_x = contour[:,0,0] # 轮廓像素横坐标
        ipixel_y = contour[:,0,1] # 轮廓像素纵坐标
        ialpha   = (ipixel_x+0.5)*block_size  # 还原横坐标（下采样前的像素坐标）
        idelta   = (ipixel_y+0.5)*block_size  # 还原纵坐标（下采样前的像素坐标）
        # 利用WCS将像素坐标转回赤道坐标
        coord    = w.pixel_to_world( ialpha, idelta )
        ra       = coord.ra.degree
        dec      = coord.dec.degree
        poly     = Polygon([(ra_, dec_) for ra_, dec_ in zip(ra, dec)])
        polygons.append(poly)
        areas.append(poly.area)
    areas     = np.array(areas)
    icontours = np.argsort(-areas)
    polygons  = [polygons[icontour] for icontour in icontours]
    return polygons, areas
        
if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.description='''Convert weight maps to the rough profile of survey '''
    parser.add_argument("whtsname",  help="The filename of weight maps ")
    parser.add_argument("blocksize", help="The integer block size along each axis. Downsample a data array by applying a sum function to local blocks", default = 20,  type = int )

    args = parser.parse_args()
    whtsname  = args.whtsname
    blocksize = args.blocksize 
    polygons, areas = wht2poly_from_fits(whtsname, block_size = blocksize)

    for icontour, area in enumerate(areas):
        poly    = polygons[icontour]
        ra, dec = poly.exterior.coords.xy
        tab     = Table() 
        tab['RA']  = ra
        tab['DEC'] = dec
        tab.write('survery%03d.reg'%icontour, format = 'ascii.fixed_width', delimiter = None,  overwrite=True)
    print('ploting') 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1, figsize = (10, 10))
    for icontour, area in enumerate(areas):
        tab = ascii.read('survery%03d.reg'%icontour ) 
        print(icontour, area )
        name    = '%s'%icontour
        ax.plot(tab['RA'], tab['DEC'], label= name)
        ax.text(tab['RA'][0], tab['DEC'][0], name)
    plt.savefig('survery_edge.png') 
    plt.show( )  
