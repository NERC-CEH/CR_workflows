#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This will provide a rough idea of the workflow in a recent report, 
though much is open to experimentation/improvement so I leave it open to the user.
A similar approach was used for the other squares where the point cloud is classified
then the results post-processed.
 
The aim is to classify the points into classes that can be used 
for area indicators (e.g. woodland, hedges) and extract the information therein.  

At the outset it must be noted we do not possess a extensive set of ground
returns for most of the area (leaf on conditions August), 
so a Welsh Gov DTM should could be obtained (I have provided a subset for this example).

HOWEVER...in this instance it is of limited use because the 'ground' in this 
WGDTM is in fact the veg canopy for the most part. Interpolation is not viable
in the ravine in the centre of the area as it is filled with veg for the most part,
so one cannot interpolate to the valley bottom and out again.

When it comes to estimating volume the cut/fill method on a DSM is best here,
but an underestimate nonetheless.

The point cloud has already had both the flight lines removed, then thinned 
using the poisson method, as the original L1 pointcloud was far too dense 
to process. See preproc.py for this. 
"""

from pointutils import learning as l
from pointutils import utils as ut
import os
from glob import glob
import numpy as np
from pyntcloud import PyntCloud
import geopandas as gpd
import pandas as pd
import skimage.morphology as skm
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import skimage.morphology as skm
from matplotlib.colors import ListedColormap
import scipy.ndimage as nd


# Input variables - obtain from SLU/C2C

# replace as appropriate
maindir = os.getcwd()
wgdtm = os.path.join(maindir, 'wgdtmhells.tif')

rgbcld = os.path.join(maindir, 'HellsMouth0.5_clp.ply')

"""
Classify the pointcloud

 Using a 'classical' machine learning approach, having already labelled the 
 data using cloud compare, cgal polyhedron viewer or one of many labelling 
 apps on github. Using Cloud Compare is clumsy (the point segmentation tool),
 but the others will require you to be able to compile code/libs 
 on your machine (good luck on a CEH laptop....).

 The ML is from my lib, in turn derived from scikit learn. xgboost/erf are
 equally effective. A k-fold stratified grid search is used, but an evolutionary
 approach is also available in the lib 'learning' module.

 Please change the `maindir` variable below to where the pointcloud and WGDTM 
 reside or move them to where this notebook is and leave it as is.
"""


clsnms = ['Grass', 'Hedge', 'TallTree', 'Building', 'Road', 
          'LowTree', 'Bracken']

nt = -1
outtrain = os.path.join(maindir, 'hellstrn.gz')

"""
 A set of geometric and colour-based features are generated at multiple
 scales to train the model. These are generated on the fly as saving directly
 to disk creates very large files. The default is to use cgal-based features
 (most effective), though pdal or pyntcloud-based are also available. Please 
 consult the cgal website c++ api for further details

 A k-fold stratified grid search is run with a further held out test set, the accuracy of
 which will be plotted and saved. The results are summarised in the returned variable
 and the model saved as a .gz file.
"""

train, fnames = l.get_training_ply(rgbcld, label_field='training', rgb=True, 
                           outFile=outtrain)


param_grid = {"n_estimators": [100],"max_features": ['sqrt', 'log2'],
                         "min_samples_split": [2,3,5],
                         "min_samples_leaf": [5,10,20],
                         "max_depth": [10, 15, 20]}

modelerf = outtrain[:-3]+'_erf.gz'

reserf = l.create_model(train, modelerf, clf='erf',
                                        params=param_grid, cv=5, cores=
                                        nt, class_names=clsnms, ply=True)

outcld = rgbcld[:-4]+'_erf.ply'
l.classify_ply(rgbcld, modelerf, rgb=True, outcld=outcld)

# Should you wish, you could use the native cgal classifier, results
# vary slightly to sklearn - you decide which is best - it is tuned slightly differently
# to sklearn

#cgloot = rgbcld[:-4]+'_cgl.ply'
#cgmdl = os.path.join(maindir, 'cgalrf.mdl')
#l.create_model_cgal(rgbcld, cgmdl, clsnms, k=5, rgb=True, normal=True, 
#                    outcld=cgloot)

#split classes
ut.split_into_classes(outcld)

clsdr = os.path.join(maindir, 'sepCls')

# ### Trees 
# Cleanup & grid the trees
treecld1 = os.path.join(clsdr, 'HellsMouth0.5_clp_erf2.ply')
treecld2 = os.path.join(clsdr, 'HellsMouth0.5_clp_erf5.ply')
treemerge = os.path.join(clsdr, 'TreeMerge.ply')
# merge the 2 tree classes
# GMEP/ERAMMP indicates areas as woodland, but the may be mainly scrub....
ut.merge_cloud([treecld1, treecld2], treemerge)

treeoot = treemerge[:-4]+'_tree_cln.ply'

# SLOW until my quictker version is fixed
ut.nrm_point_dtm(treemerge, wgdtm, outcld=treeoot, reader='ply', writer='ply')

# Remove outliers
dist = 0.1
treefin = treeoot[:-4]+'_ol'+str(dist)+'.ply'
ut.cgal_outlier(treeoot, treefin, distance=dist)

# Convenience function to produce thematic layer - see BetwsDemo.py or look
# up function internals for details
treeras = os.path.join(maindir, 'tree.tif')
treepoly = os.path.join(maindir, 'tree.shp')
treegdf, totaltree = ut._lc_posproc(treefin, treeras, treepoly, area_close=4)
print('Tree area is', totaltree, 'ha')
#treegdf.plot(facecolor="green")

# ### Hedges
# Cleanup & grid the hedges 
hedgecld = os.path.join(clsdr, 'HellsMouth0.5_clp_erf1.ply')

# SLOW until my quicker version is fixed
hedgenrm = os.path.join(clsdr, 'Hedge_nrm.ply')
ut.nrm_point_dtm(hedgecld, wgdtm, outcld=hedgenrm, reader='ply', writer='ply')

# cln #1 # bear in mind the height is unlikely that meaningful
hedgedrp = os.path.join(clsdr, 'Hedge_drp.ply')
ut.drop_cld_value(hedgenrm, column='z', rule='<', val=0.05, outcld=hedgedrp)

#outliers
dist = 0.1 # in case you wish to change it
hedgecln = os.path.join(clsdr, 'Hedge_cln'+str(dist)+'.ply')
ut.cgal_outlier(hedgedrp, hedgecln, distance=dist)

hedgeras = os.path.join(maindir, 'hedge.tif')

#res 
res = 0.3
ut.grid_cloud(hedgecln, hedgeras, attribute='label', outtype='max', 
              spref="EPSG:27700", resolution=res, dtype='uint8')

ut.fill_nodata(hedgeras, maxSearchDist=1)
"""
 For the hedges a little more image processing is required - 
 this is not ideal/transferrable post proc, but to produce 
 reasonable results without manual intervention it is required.
 The reason is we require the medial axis of the hedge polygons to estimate their 
 length, which would not be possible with compact objects.
 Likely more training is required on many datasets to reduce this sort of thing,
 but the following will do the job. 
"""
img = ut.raster2array(hedgeras)
img[img==255]=0
#plt.figure(figsize = (6,6))
#plt.imshow(img, cmap='gray')
# Image morph should do the trick - some small holes need to be closed (trust me look in a GIS). 
img = skm.area_closing(img, area_threshold=4)

# Use a binary image prop to eliminate some of the non-hedges - far from
# perfect, but required to get a medial axis skeleton image that is largely 
# clean

# connected components
label, _ = nd.label(img)
# the attribute in question - solidity this time as the hedges are all joined
cllabel = ut.colorscale(label, prop='Solidity')
cllabel = np.uint8(cllabel*100) # vals for GUI

# napari is a bit slow 
ut.image_thresh(np.uint8(cllabel))

# **The chosen values** 
hdge = (cllabel < 84) & (cllabel > 0)
bush = (cllabel >= 90) & (cllabel > 0) #??

ut.array2raster(hdge, 1, hedgeras, hedgeras, 3)

# So we have the 'non linear' veg left over, this could be merged with the
# shrub class later - into a misc. category
bushras = os.path.join(maindir, 'bush'+str(res)+'.tif')
ut.array2raster(bush, 1, hedgeras, bushras, 3)
bushpoly = os.path.join(maindir, 'bush.shp')
ut.polygonize(bushras, bushpoly)

bushgdf = gpd.read_file(bushpoly)
bushgdf["Area"] = bushgdf['geometry'].area
totalbush = bushgdf.Area.sum() /10000 #ha
print('Bush/Shrub area is', totalbush, 'ha')
bushgdf.to_file(bushpoly)

# Now calculate the skeleton
skel = ut.raster_skel(hedgeras, hedgeras[:-4]+'_hdgline.tif', nodata=255,
                      prune_len=70)
#plt.figure(figsize = (6,6))
#plt.imshow(skel)

# Hedgerow length - count the pixels we know the resolution is 0.3m so...
hedgelength = np.count_nonzero(skel)*0.3/1000 #km
print('Total length is ', hedgelength, 'km')

# Hedge mask and area estimate 
hedgepoly = os.path.join(maindir, 'hedge.shp')
ut.polygonize(hedgeras, hedgepoly)
hedgegdf = gpd.read_file(hedgepoly)
#Area....
hedgegdf["Area"] = hedgegdf['geometry'].area
totalhedge = hedgegdf.Area.sum() /10000 #ha
print('Hedge area is', totalhedge, 'ha')
# average width
hedgegdf['Length'] = hedgegdf['geometry'].length
hedgegdf['AvWid'] = hedgegdf['Area'].divide(hedgegdf['Length']) * 4
hedgegdf.to_file(hedgepoly) 

# Bracken - there was bracken - so map it....
bracken = os.path.join(clsdr, 'HellsMouth0.5_clp_erf6.ply')

#nrm
brknrm = bracken = os.path.join(clsdr, 'Bracken_nrm.ply')
ut.nrm_point_dtm(bracken, wgdtm, outcld=brknrm, reader='ply', writer='ply')

# Remove outliers
dist = 0.1
brkfin = brknrm[:-4]+'_ol'+str(dist)+'.ply'
ut.cgal_outlier(brknrm, brkfin, distance=dist)

# Convenience function to produce thematic layer - see BetwsDemo.py or look
# up function internals for details
brkras = os.path.join(maindir, 'bracken.tif')
brkpoly = os.path.join(maindir, 'bracken.shp')
brkgdf, totalbrk = ut._lc_posproc(brkfin, brkras, brkpoly)
print('Bracken area is', totalbrk, 'ha')

ax = brkgdf.plot(facecolor="yellow")
hedgegdf.plot(ax=ax, facecolor="red")
treegdf.plot(ax=ax, facecolor="green")




