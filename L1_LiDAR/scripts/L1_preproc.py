#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The following details the two commands used to reduce the L1 point cloud for
classification and surface reconstruction. By installing my lib pointutils you
will have access to these. In addition you will need cloud compare to visualise
results and save the final .ply file.

The pointcloud is also re-coloured using the 
ortho-mosiac as the 'factory' colour illumination provided by the L1
is hopeless for machine learning. 

There is much that could be done to a point cloud prior to doing anything
with it. There is no silver bullet here so consult the lit/libs etc.
These are just three of many potential steps depending on the application
(e.g. https://doc.cgal.org/latest/Point_set_processing_3/index.html).

So you would need:

- the original dense pointcloud

- an orthomosaic generated from the L1 or P1 camera

"""
from WBT.whitebox_tools import WhiteboxTools
from pointutils import utils as ut
wbt = WhiteboxTools()

#the files are called Betws as it was just up the road....
# 190 ma points - far too big much info redundant
incld = 'BetwsOrig.las'

# the size of the square area to assess neighbouring points
res = 0.5
outcld = incld[:-4]+'_culled'+str(res)+'.las'
# the filter flag culls the overlap points. You will have to adjust
# resolution to suit the data you have e.g. estimate the overlap visually or
# inspect results by trial/error

wbt.classify_overlap_points(incld, outcld, resolution=0.5, filter=True)
# 90 ma points as a result of the above

# thin using poisson as we wish to preserve surface texture
rad = 0.5
thinned = incld[:-4]+'cull_thin'+str(rad)+'.las'

# this may take a while.....
# now reduced to 3.5 ma - much better....
ut.pdal_thin(outcld, thinned, method='filters.sample', radius=rad)

# the path to the orthomosaic
inras = 'BetwsOrtho.tif'

final = thinned[:-4]+'_rgb.las'
ut.colour_cloud(thinned, inras, final, writer='las')
"""
 A bug with pdal prevents the correct translation of rgb vals to .ply format
 YOU WILL NEED cloud compare for this either open the las and save via GUI
 to ply or with the following function which calls the CC cmd line
"""
ut.las2ply_cc(final)






