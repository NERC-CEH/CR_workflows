#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict canopy cover from UAV derived data - you will need my other lib
geospatial_learn to do this unless you are happy to work directly with data/sklearn

This merely demonsrates how to do this - but obviously this is a tiny and and totally
rubbish training dataset in reality and scores will reflect this. 
Looks suprisingly convincing despite the above statment!

"""
from geospatial_learn import learning as l
from geospatial_learn import shape as s
import os
import geopandas as gpd

# Save the following files here
main_dir = "insert/dir/here"
os.chdir(main_dir)

# bianry mask of trees earlier derived from ML
trees = ('trees.tif')
# A meshcovering the image to write fractional cover
inshp=('Betws_mesh.shp')
# A temporal composite of S2 from the same month
inras = ('btwsOSGB10.tif')
# get the frac cover
s.zonal_stats(inshp, trees, 1, bandname='Ccvr', stat='perc', nodata_value=255)

# Next we need to extract exact values - and things never line up perfectly, 
# so convert the mesh to centroid points to records the corresponding S2 pixel
# values
gdf = gpd.read_file(inshp)
# make a copy of the gdf...
points = gdf.copy()
# change the geometry to centroids
points.geometry = points['geometry'].centroid
# copy the crs
points.crs = gdf.crs
# save it
trnshp = ('trnpoints.shp')
points.to_file(trnshp)

outtr = os.path.join('trn.gz')
del points

bands = ['b', 'g', 'r', 'nir']
nms = [1, 2, 3, 7]

for i, b in zip(nms, bands):
    s.zonal_point(trnshp, inras, b, band=i)

# now get the training data
df = gpd.read_file(trnshp)

train = df[['Ccvr', 'b', 'g', 'r', 'nir']].to_numpy()

# rf params - though change for other sklearn models
param_grid = {"n_estimators": [500],"max_features": ['sqrt', 'log2'],
                         "min_samples_split": [2,3,5],
                         "min_samples_leaf": [5,10,20],
                         "max_depth": [10, 15, 20]}
# save model
ootmodel = os.path.join('rfregress.gz')
results = l.create_model(train, ootmodel, regress=True, clf='rf',
                         params=param_grid)
# svae map - view in GIS
outMap = ('Ccvr.tif')
l.classify_pixel_bloc(ootmodel, inras, outMap, bands=[1,2,3,7],
                      blocksize=64, dtype=6)











