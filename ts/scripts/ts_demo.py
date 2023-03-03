#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
As requested a quick demo of how I obtained some time series via GEE. I'm sure
 there is a better way of doing most of this, but it worked....

For the download I have used FAO polyogns available on GEE

For time series direct to polygon you will need a shapefile in 
ESPG:4326 and the geometry will need to befixed for errors - eg see QGIS for 
functionality in this area.

"""
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
import os
from eot import eotimeseries as et
from eot import utils as ut
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import ee
import geemap
import wxee
from glob import glob
import xarray as xr
from cdo import Cdo
import rioxarray


"""
Downloading - this will take a little while and is only possible to google 
drive at this scale upwards. Functions are in a lib of random GEE stuff I have
written called eot. 

WARNING! This will fill up you drive pretty quick (NDVI will be ~70gb)

"""

# WALES, well it is smallish
counties = ee.FeatureCollection("FAO/GAUL/2015/level1")
wales = counties.filterMetadata("ADM1_NAME", "equals", 'Wales')
poly = wales.geometry()

# Bare soil via GEOS3 method 

# You could either drip feed it as a set of tasks in sequence (uncomment)
# dates = [("2016-01-01", "2016-12-31"),
#          ("2017-01-01", "2017-12-31"), ("2018-01-01", "2018-12-31"),
#          ("2019-01-01", "2019-12-31"), ("2020-01-01", "2020-12-31"),
#          ("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")]

# for d in tqdm(dates):
#     et.bs_down(poly, d[0], d[1], 'bs'+d[0]+'.tif')
    
# Or do it in one. 
et.bs_down(poly, dates[0][0], dates[6][6], 'bs16to22.tif')

# Either way - check tasks in your GEE javascript interface to see if it is
# running (it should be)

# Assuming you have downloaded the bare soil tifs somewhere, you could make them into 
# a netcdf

fldr = ('myfolder/*.tif')

inlist = glob(fldr)
inlist.sort()

# clip the water 
inshp = ('Wales boundarywgs84.shp')

for i in tqdm(inlist):
    ut.clip_raster(i, inshp, i[:-4]+'_clp.tif')
    
clplist = glob(fldr)
clplist.sort()

# April is missing in 2016 - always worth checking each file
# xarray is very slow....gdf = gpd.read_file('polygontest.shp')
dt = pd.date_range("2016-01-01","2016-12-31",freq='M')
dt = dt.drop('2016-03-31')
ds16 = rioxarray.open_rasterio(clplist[0])
ds16['band'] = dt
ds16.to_netcdf(clplist[0][:-3]+'nc')

# sanity check if you want it
#ds16 = ds16.assign_coords(time=dt)
#ds16.to_netcdf(clplist[0][:-3]+'nc')
# every time I forget how this stupid lib works
# yip = ds16.sel(time="2016-04")
# yip.band_data.plot()

clplist.pop(0)
dates.pop(0)

dr = [pd.date_range(d[0], d[1], freq='M') for d in dates]

# rename the bands and produce a massive netcdf
for c, dts in tqdm(zip(clplist, dr)):
    xds = rioxarray.open_rasterio(c)
    # so the band dim is now dates
    xds['band'] = dts
    
    xds.to_netcdf(c[:-3]+'nc')

cd = Cdo()

pths = glob(fldr[:-3]+'nc')
pths.sort()

nm = ('bs2016-2022.nc')

# merge
ofile = cd.cat(input=pths, output=nm, options='-r')

# Can do the same with NDVI - this will fill your drive up! 
# This also does a harmonic smoothing of the whole TS
et.nd_down(poly, dates[0][0], dates[6][6], 'nd_2016-22.tif')

"""
# You can also drag TS straight through to your computer via polygon,
# but this is of course data volume limited. Will get away with up to 
# 1000s of polys in a compact area

unzip polygontest.zip to do this
"""
sd = '2019-01-01'
ed = '2022-01-01'

inshp = 'polygontest.shp'

gdf = gpd.read_file(inshp)
df = et.zonal_tseries(gdf, sd, ed, bandnm='NDVI', attribute='idx')

# merge results
merge = gdf.merge(df, on='idx')
# plot
et.plot_group(merge, 'idx', [0], 'n-', 
              freq='M', plotstat='mean',  fill=True)

# If you were using a CS / GMEP square, you could using the following to plot
# assuming you had already reprojected followed the above steps with such a file
# remeber to reproject to 4326, before zonal_tseries and merge
              #gdf     #column    # hab label            #prefix -n for ndvi
et.plot_group(merge, 'BROAD_H', ['Improved Grassland'], 'n-', 
              freq='M', plotstat='mean',  fill=True)










