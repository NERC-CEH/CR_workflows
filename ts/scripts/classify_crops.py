#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some crop mapping experiments using machine learning based on some declarations

received from Defra and merged with a modified version of LCM.

In reality far too little data to make a good job of this, but works pretty
well despite this. 

You will need my lib geospatial_learn to do all this/see the internals of the
training etc. This is more or less as I did it - so not well organised. 

The files associated with all of this are in possession of C2C/SLU in a 
7zip or folder called Crops. 

"""

from geospatial_learn import learning as l
# plot group & crop now in here
from geospatial_learn import handyplots as hp
import geopandas as gpd
import pandas as pd
import os
from scipy.stats import randint as sp_randint


inShp = 'croplcm_RAW_joinedbuf30WGS84S10mbnds.shp'

plist = l.get_polars(inShp)

# make an integer label for classification
gdf = gpd.read_file(inShp)

"""
Factorize the labels for training
"""
gdf.Descriptio.value_counts()

# class labels
gdf['train'] = pd.factorize(gdf.Descriptio)[0]
#TODO!!!!
gdf['train'] = gdf['train']+1 
# These start from zero!! they must have 1 added to them as in geolearn
# zero gets scrubbed
gdf.to_file(inShp)


# get the crop types
# for ref
crops = gdf.Descriptio.unique().tolist()
# the types, these can be further subdivided if required
winter = gdf[gdf.Descriptio.str.contains('winter')]

spring = gdf[gdf.Descriptio.str.contains('spring')]


# this code equates to winter wheat
ww = gdf.loc[gdf['Descriptio'] == 'Wheat-winter']
## may as well show the SAR ratio (S1R)
polars = ['VV', 'VH']

ws = gdf.loc[gdf['Descriptio'] == 'Wheat-spring']

# this is a bit hard to look at in a loop on same plot....
#for p in polars:
hp.plot_group(gdf, 'key_0', ww['key_0'].to_list(), 'VV',  year=None, 
              title='Wheat-winter VV', fill=True, freq='M', plotstat='mean')

hp.plot_group(gdf, 'key_0', ww['key_0'].to_list(), 'VH',  year=None, 
              title='Wheat-winter  VH', fill=True, freq='M', plotstat='mean')

ow = gdf.loc[gdf['Descriptio'] == 'Oats-winter']

hp.plot_group(gdf, 'key_0', ow['key_0'].to_list(), 'VV',  year=None, 
              title='Oats-winter VV', fill=True, freq='M', plotstat='mean')

hp.plot_group(gdf, 'key_0', ow['key_0'].to_list(), 'VH',  year=None, 
              title='Oats-winter  VH', fill=True, freq='M', plotstat='mean')

# This pair underlines that NDVI may provide the discrimination required
# over the winter for better performance on spring variaties

hp.plot_group(gdf, 'key_0', ww['key_0'].to_list(), 'nd',  year=None, 
              title='Wheat-winter VV', fill=True, freq='M', plotstat='mean')

hp.plot_group(gdf, 'key_0', ws['key_0'].to_list(), 'nd',  year=None, 
              title='Wheat-spring ndvi', fill=True, freq='M', plotstat='mean')


"""
1. Use growing season(s)

Make a new file based on the growing season for training

Here for info as the the file is already made (bigShp)

"""
# Split by growing season
# functions to extract a growing season from the time series
# This is all a bit hap-hazard

# May be better to extract entire frame
start='2019-09-01'
end='2020-09-01'
# the vals of interest
vvsep = hp.extract_by_date(gdf, start, end, freq='M', band='VV')
vhsep = hp.extract_by_date(gdf, start, end, freq='M', band='VH')#
b = hp.extract_by_date(gdf, start, end, freq='M', band='B2')
g = hp.extract_by_date(gdf, start, end, freq='M', band='B3')
r = hp.extract_by_date(gdf, start, end, freq='M', band='B4')
re1 = hp.extract_by_date(gdf, start, end, freq='M', band='B5')
re2 = hp.extract_by_date(gdf, start, end, freq='M', band='B6')
re3 = hp.extract_by_date(gdf, start, end, freq='M', band='B7')
nir1 = hp.extract_by_date(gdf, start, end, freq='M', band='B8A')

ndsep = hp.extract_by_date(gdf, start, end, freq='M', band='nd')

vvmar = hp.extract_by_date(gdf, '2020-03-01', end, freq='M', band='VV')
vhmar = hp.extract_by_date(gdf, '2020-09-01', end, freq='M', band='VH')
ndmar = hp.extract_by_date(gdf, '2020-09-01', end, freq='M', band='nd')
# TODO Might it be better to drop the nondesirable dates 

nonsar = ['OGC_FID', 'gid', '_hist', '_mode', '_purity', '_conf',
       '_stdev', '_n', 'cmp', 'OBJECTID', 'Crop_Type', 'Descriptio', 'Source',
       'Date', 'Confidence', 'Area', 'geometry']

# forget merge or concat, this is far simpler
final = gdf[nonsar].join([vhsep, vvsep, b, g, r, re1, re2, re3, nir1, ndsep])

"""
Heres one I made earlier. Now we can test how effective our rather short time series
data works depending on the period chosen
"""

bigShp = 'croplcm_RAW_joinedbuf30WGS84S10mbnds_sep2sep.shp'
final.to_file(bigShp)


#spring
sp_crops = gdf[nonsar].join([vhmar, vvmar, ndmar])
#sp_crops['train'] = pd.factorize(sp_crops.Descriptio)[0]
sp_crops = sp_crops.loc[sp_crops.Descriptio.str.contains('spring')]
sprshp = ('croplcm_RAW_sprnall.shp')
sp_crops.to_file(sprshp)
# since winter has little discrim power
sp_crops.to_file('croplcm_RAW_mar2sepALLVAR.shp')


# factorize the crops
final['train'] = pd.factorize(final.Descriptio)[0]
final['train'] = final['train']+1 

sepshp = ('croplcm_RAW_sep2sep.shp')
final.to_file(sepshp)

# a quick look why not
# first a list of poss vals
final['Descriptio'].unique() 

['Wheat-winter', 'Onion', 'Oats-winter', 'Oats-spring',
       'Field beans-spring', 'Maize', 'Pea-spring', 'Oilseed-winter',
       'Wheat-spring', 'Barley-spring', 'Lucerne', 'Barley-winter',
       'Beet', 'Rye-winter', 'Potato', 'Linseed-spring',
       'Field beans-winter', 'Cabbage-spring', 'Lettuce', 'Clover',
       'Trees', 'Bracken', 'Grass', 'Apples']

cropt = 'Wheat-winter'

ow = final.loc[final['Descriptio'] == cropt]

hp.plot_group(final, 'gid', ow['gid'].to_list(), 'VV',  year=None, 
              title=cropt+' VV', fill=True, freq='M', plotstat='mean')

hp.plot_group(final, 'gid', ow['gid'].to_list(), 'VH',  year=None, 
              title=cropt+' VH', fill=True, freq='M', plotstat='mean')

hp.plot_group(final, 'gid', ow['gid'].to_list(), 'nd',  year=None, 
              title=cropt+' ndvi', fill=True, freq='M', plotstat='mean')

# had i thought this through I'd have just added columns rather than
# saving lots of shapefiles...
# lets do it 'blind' first on the seasonal data before chopping it up more
winter = final.loc[final.Descriptio.str.contains('winter')]
wshp = 'croplcm_RAW_sep2sepwinter.shp'
winter.to_file(wshp)

spring = final.loc[final.Descriptio.str.contains('spring')]
spshp = ('croplcm_RAW_sep2sepspring.shp')
spring.to_file(spshp)

# make one without season distinctions....
# Classes must be refactorised as they are being aggregated
naeseason = final.copy(deep=True)
naeseason.Descriptio = naeseason.Descriptio.str.replace("-winter", "")
naeseason.Descriptio = naeseason.Descriptio.str.replace("-spring", "")
naeseason.Descriptio.unique()
naeseason = naeseason[naeseason.Descriptio!='Apples']
naeseason['train'] = pd.factorize(naeseason.Descriptio)[0]
naeseason['train'] = naeseason['train']+1 
naeshp = ('croplcm_RAW_sep2sepnoseason.shp')
naeseason.to_file(naeshp)


# will not accept single class instance
final = final[final.Descriptio!='Apples']
applefree = 'croplcm_RAW_sep2sepwinter_naeapl.shp'
final.to_file(applefree)
"""
###############################################################################
SUMMARY #######################################################################
###############################################################################

Sep2Sep is the most useful time slice (mar2sep is always worse)

Removing seasonal crop component (eg winter spring) where present improves accuarcy

Tall crops (particularly cereal) can be mapped to >80% accuarcy (train & test),
Perhaps due to a 'uniform' appearance translating well to a single figure,
whereas short stuff like cabbage/onion is more spatially heterogeneous??

ndvi helps, but only improves accuarcy marginally.

Other bands only help cabbage precision! (hahaha)

Cereal only is very good all 80%+
Winter only is very good 80%+

In winter there is a huge hump representing moisture (I assume)

###############################################################################
"""

# TESTING #####################################################################
###############################################################################

# CABBAGE performs badly on all tactics! haha
# Dropping the seasonal variety improves accuracy (even with NDVI) suggesting
# the use of both SAR & NDVI

# SAR alone performs well on winter crops (86%) and OK on spring if separated 
# using Sep2Sep
# Performance is reduced using mar>sep on spring crops


# change to subdivision for testing
#testshp = sepshp
#testshp = wshp #winter sep -sep
#testshp = spshp #spring sep -sep
#testshp = sprshp
testshp = applefree #rid of single apple
testshp = naeshp
testshp = 'croplcm_RAW_sep2sep_Cereal.shp'
#testshp='croplcm_RAW_mar2sepALLVAR.shp'
#testshp = bigShp
_, tl = os.path.split(testshp)

plist = l.get_polars(testshp)
#ndfields
nds = ['nd-19-09', 'nd-19-10', 'nd-19-11',
       'nd-19-12', 'nd-20-01', 'nd-20-02', 'nd-20-03', 'nd-20-04', 'nd-20-05',
       'nd-20-06', 'nd-20-07', 'nd-20-08']
# add the other bands
#bandlist=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A' ]
#colstmp = []
#
#for b in bandlist:
#    tmp = [n.replace('nd', b) for n in nds]
#    colstmp.append(tmp)
    
# apperently quickest
#flds = plist + list(chain.from_iterable(colstmp))

#nds = ['nd-20-03', 'nd-20-04', 'nd-20-05',
#       'nd-20-06', 'nd-20-07', 'nd-20-08']

flds = plist+nds
del plist

traingz = os.path.join('training2019-20', 
                       tl[:-4]+'train.gz')

# arguably via OGR or pandas this is pointless repetition
train, rej = l.get_training_shp(testshp, label_field='train', 
                                feat_fields=flds, outFile=traingz)
# weirdly, train is appended to plist - weird python list thing again
# this happens in function, but is not cleaned up at the end....

#OR with pandas
#train = final[['train']+flds] 

# cereal only
#temp = final[final['Descriptio'].isin(['Wheat-winter','Oats-winter', 'Oats-spring',
#                     'Wheat-spring', 'Barley-spring', 'Barley-winter',
#                     'Rye-winter'])]



# train set is tiny so no trees not really important
oobmodel = os.path.join('models2019-20/',
                       tl[:-4]+'oob.gz')
#oob error is worst with only ndvi
results = l.RF_oob_opt(oobmodel, train.to_numpy(), 50, 1000, 10)

# better than last time levels out more predictably with all data used
# with winter a bit noisy up to 430 lowest oob
#ntrees = 520 #(all)
ntrees = 415 #(winter 410-20)
#ntrees = 550 # (spring mar-sep)
ntrees = 410 # all nae apple (cheaper), though 750 lowest

ntrees = 740 # all nae apple with ndvi, though 370 would be cheaper

ntrees = 590 # all nae season with ndvi, though sqrt,lg2 lower individually 

#ntrees = 540 # all vars using only mar2sep (since winter does not dicrim much with 
                # SAR) in theory
        
# grid 79.6, test p76, r76, f1 75 

#RF###########################################################################
param_grid = {"n_estimators": [ntrees],
          /home/ciaran/BareSoilMapping/    "max_features": ['sqrt', 'log2'],
              "min_samples_split": [2,3,5],
              "min_samples_leaf": [5,10,20],
              "max_depth": [10, 20, 30],
              "criterion": ["gini", "entropy"]}

modelrf = os.path.join('models2019-20/',
                       tl[:-4]+'_rf.gz')

classes = final.Descriptio.unique().tolist()
#classes = winter.Descriptio.unique().tolist()
#classes = spring.Descriptio.unique().tolist()

resrf = l.create_model(train.to_numpy(), modelrf, clf='rf',
                                        params=param_grid, cv=5, cores=
                                        20, class_names=classes)

modelerf = os.path.join('models2019-20/',
                       tl[:-4]+'_erf.gz')

reserf = l.create_model(train.to_numpy(), modelerf, clf='erf',
                                        params=param_grid, cv=5, cores=
                                        20, class_names=classes)

# After fixing the labels there is no improvement in performance on all
# All sep2sep 66% 
# All nae apple 66% 
# All +ndvi nae apple 68% , test F1 score 68% (p=70,r=69) so consistent
# All +ndvi nae season var, grid 78 , test p74, r74, f1 73
# winter sep2sep 83% p82 r81 f1 80
# spring sep2sep 75%
# spring mar2sep 68%
# All mar2sep grid 54, test f1 54 DIRE so winter info is vital 

rf1 = {'criterion': ['entropy'],
 'max_depth': [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], 
 'max_features': ['sqrt'],
 'min_samples_leaf': [5],
 'min_samples_split': [2],
 'n_estimators': [ntrees]}

rf1m= os.path.join('models2019-20/',
                       tl[:-4]+'_rf1.gz')
resrf1 = l.create_model(train.to_numpy(), rf1m, clf='rf',
                       params=rf1, cv=5, cores=20)
# no improvement on orig grid 78, test p74, r74, f1 73

# Interestingly depth is 19

rf2 = {'criterion': ['entropy'],
 'max_depth': [19], 
 'max_features': ['sqrt'],
 'min_samples_leaf': [2,3,5,7,10,15,20,25,30,35,40,50],
 'min_samples_split': [2],
 'n_estimators': [ntrees]}

rf2m= os.path.join('models2019-20/',
                       tl[:-4]+'_rf2.gz')

resrf2 = l.create_model(train.to_numpy(), rf2m, clf='rf', 
                       params=rf2, cv=5, cores=20)
# grid 79.2, test p76, r76, f1 76 !! pretty good (relatively)


rf3 = {'criterion': ['entropy'],
 'max_depth': [19], 
 'max_features': ['sqrt'],
 'min_samples_leaf': [2],
 'min_samples_split': [2,3,5,7,10,15,20,25,30,35,40,50],
 'n_estimators': [ntrees]}

rf3m= os.path.join('models2019-20/',
                       tl[:-4]+'_rf3.gz')

resrf3 = l.create_model(train.to_numpy(), rf3m, clf='rf', 
                       params=rf3, cv=5, cores=20)
# grid 79.6, test p76, r76, f1 75 

rf4 = {'criterion': ["gini", "entropy"],
 'max_depth': [19], 
 'max_features': ['sqrt', "log2"],
 'min_samples_leaf': [2],
 'min_samples_split': [5],
 'n_estimators': [ntrees]}

rf4m= os.path.join('models2019-20/',
                       tl[:-4]+'_rf4.gz')

resrf4 = l.create_model(train.to_numpy(), rf4m, clf='rf', 
                       params=rf4, cv=5, cores=20)
resrffin = {'criterion': 'entropy',
 'max_depth': 19,
 'max_features': 'sqrt',
 'min_samples_leaf': 2,
 'min_samples_split': 5,
 'n_estimators': 590}

# grid 79.6, test p76, r76, f1 75 no improvement on rnd 3

#random search#######################
#no cabbage not comprimise on trees
#ntrees = 890
rf_rnd = os.path.join('models2019-20/',
                       tl[:-4]+'_rndrf.gz')

rndp = param_grid = {"n_estimators": [ntrees],
                     "max_features": ['sqrt', 'log2'],
                     "max_depth": sp_randint(1, 30),
                     "min_samples_split": sp_randint(1, 20),
                     "min_samples_leaf": sp_randint(1, 20),
                     "criterion": ["gini", "entropy"]}

resrfrnd = l.create_model(train.to_numpy(), rf_rnd, clf='rf', random=True,
                       params=rndp, cv=5, cores=20)
# All sep2sep 67% 
# All with ndvi 66%
# All no cabbage, no season var grid 79.1, test p78, r78, f1 78
# winter sep2sep 84%
# spring sep2sep 75%
# spring mar2sep 69%


# Done with SAR+NDVI and seasonal vars#########################################
#The long way or is it the input data.....#####################################
rfps = {'criterion': 'entropy',
 'max_depth': 20,
 'max_features': 'sqrt',
 'min_samples_leaf': 5,
 'min_samples_split': 2,
 'n_estimators': 740} #68%

rndps = {'criterion': 'gini',
 'max_depth': 16,
 'max_features': 'sqrt',
 'min_samples_leaf': 3,
 'min_samples_split': 19,
 'n_estimators': 740} #66%

# Attempt 1 tune depth (one below the above to start it)
rnd1 = {'criterion': ['gini'],
 'max_depth': [15,20,25,30,35,40], 
 'max_features': ['sqrt'],
 'min_samples_leaf': [3],
 'min_samples_split': [19],
 'n_estimators': [740]}

rnd1res = l.create_mod/home/ciaran/BareSoilMapping/el(train.to_numpy(), rf_rnd[:-2]+'rnd1.gz', clf='rf',
                       params=rnd1, cv=5, cores=20)
# test p70, r70, f168
#depth not helping oddly - set lower?
rnd2 = {'criterion': ['gini'],
 'max_depth': [3,5,7,9,11,13,15], 
 'max_features': ['sqrt'],
 'min_samples_leaf': [3],
 'min_samples_split': [19],
 'n_estimators': [740]}

rnd2res = l.create_model(train.to_numpy(), rf_rnd[:-2]+'rnd2.gz', clf='rf',
                       params=rnd2, cv=5, cores=20)
# test p70, r70, f168 with 15
# NOPE 15 is the best....0.680

rnd3 = {'criterion': ['gini'],
 'max_depth': [15], 
 'max_features': ['sqrt'],
 'min_samples_leaf': [2,3,5,7,10,15,20,25,30,35,40,50],
 'min_samples_split': [19],
 'n_estimators': [740]}

rnd3res = l.create_model(train.to_numpy(), rf_rnd[:-2]+'rnd3.gz', clf='rf',
                       params=rnd3, cv=5, cores=20)
# NOPE 3 is the best...like orig 0.680

rnd4 = {'criterion': ['gini'],
 'max_depth': [15], 
 'max_features': ['sqrt'],
 'min_samples_leaf': [3],
 'min_samples_split': [3,4,5,6,7,8,9,10,15,19,20,25,30,40],
 'n_estimators': [740]}

# added < 15 as 
rnd4res = l.create_model(train.to_numpy(), rf_rnd[:-2]+'rnd4.gz', clf='rf',
                       params=rnd4, cv=5, cores=20)
# test p70, r70, f1 69 # wow 1% improvement on test, but none
#  0.689 

# Right vary the info theory params
rnd5 = {'criterion': ["gini", "entropy"],
 'max_depth': [15], 
 'max_features': ['sqrt', "log2"],
 'min_samples_leaf': [3],
 'min_samples_split': [9],
 'n_estimators': [740]}

rnd5res = l.create_model(train.to_numpy(), rf_rnd[:-2]+'rnd5.gz', clf='rf',
                       params=rnd5, cv=5, cores=20)
# test p71, r70, f1 69, grid 0.689

# final criteria then
rnd6 = {'criterion': 'gini',
 'max_depth': 15,
 'max_features': 'sqrt',
 'min_samples_leaf': 3,
 'min_samples_split': 9,
 'n_estimators': 740}

#XGB###########################################################################

#classes = ['Wheat-winter','Oats-winter', 'Oats-spring',
#                     'Wheat-spring', 'Barley-spring', 'Barley-winter',
#                     'Rye-winter']

# This serach seems to be always a winner in terms of accuracy - autosk does 
# not improve on it. 

xgbparams= {'n_estimators': [50, 100, 200, 300],
                        'learning_rate': [0.1, 0.075, 0.05, 0.025, 0.01], 
                        'max_depth': [2, 4, 6, 8, 10],
                        'colsample_bytree': [0.2, 0.4, 0.6, 0.8]}

modelxgb = os.path.join('models2019-20/',
                       tl[:-4]+'_xgb.gz')

resxgb = l.create_model(train.to_numpy(), modelxgb, clf='xgb', 
                        params=xgbparams, class_names=classes, cv=5, cores=16)

# cereal sep2sep grid - 82%  test p83, r83, f1 83
# winter sep2sep grid - 84%  test p85, r85, f1 84
# spring sep2sep 77%
# spring mar2sep grid=80 test=p77, r77, f1 77 #pretty good 
# nae season sep2sep grid=73 test=p753, r73, f1 73
#{'colsample_bytree': 0.8,
# 'learning_rate': 0.075,
# 'max_depth': 8,
# 'n_estimators': 300}

# Just set this up in learning
automdl =  ('models2019-20/'
            'croplcm_RAW_sep2sep_autosk.gz')

classes = gdf.Descriptio.unique().tolist()

# time is in seconds
autores = l.create_model_autosk(train.to_numpy(), automdl, cores=16,
                                res_args={'cv' : 5}, class_names=classes,
                                total_time=480*60)
# Not bad with 10minutes, but slightly inferior to my usual methods on training
# set
# nae season sep2sep grid=74 test=p76, r76, f1 76

# Final question is whether there is a NN solution worthwhile (shortcut is 
# autopytorch or otherwise)

# Multi-temp CNN would be too hefty a code/machine burden, leaving a MLP type
# thing as the alternative


                         

############################## TPOT ###########################################
# will not accept single class instance so use naeapl one above in prep

# not improving - due to params being limited??
#paramst = {'sklearn.ensemble.RandomForestClassifier': {"n_estimators": [520],
#                         "max_features": ['sqrt', 'log2'],
#                         "min_samples_split": [2,3,5],
#                         "min_samples_leaf": [5,10,20],
#                         "max_depth": [10, 20, 30],
#                         "criterion": ["gini", "entropy"]
#},
#
#'xgboost.sklearn.XGBClassifier': {
#    'n_estimators': [50, 100, 200, 300],
#                        'learning_rate': [0.1, 0.2, 0.4, 0.8], 
#                        'max_depth': [4, 6, 8, 10],
#                        'colsample_bytree': [0.4, 0.6, 0.8]
#}}

# does this mean all params are free?? Not sure.......
# The usual more effective (??) suspects. Using all kept yielding ExtraTrees
# at the top so have just gone with the tree-based ones here
paramst = {'sklearn.ensemble.RandomForestClassifier':{},
           'xgboost.sklearn.XGBClassifier':{},
           'sklearn.ensemble.ExtraTreesClassifier':{}}
#paramst=None

outscript = ('models2019-20/rfgbtpotrf.py')

# Gens & Pop will need to be bigger (100 is recommended really), to get
# something decent (smaller has proved fruitless)

# Should really have train/test split the data 

nt = 30 # fine if leaving it!!
tmodel, testscore = l.create_model_tpot(train.to_numpy(), outscript, cv=5, 
                             cores=nt,  gen=100, popsize=100,
                             regress=False, params=paramst,
                             scoring=None) 

# with 100 pop & gen, /home/ciaran/BareSoilMapping/21 generations in with the above has not 
# yielded more a single decimalb improvement
# 73 gen from 66 > 68% (when stopped it was on 69.1 seemingly at gen 78), which
# I reached by 'hand' tuning in 5 steps above (albeit with ndvi) 

#If 100 does not produce a decent model/test score, will have to try and clean
# up data (this could mean increasing freq of SAR observations) or add optical

# moisture contributions to SAR a consideration but likely a paper in itself

# Results inspection###########################################################
fnames_mar = ['VV-20-03', 'VV-20-04', 'VV-20-05',
       'VV-20-06', 'VV-20-07', 'VV-20-08']

# Lets have a look at how the model 'views' the features
fnames_s2s= ['VH-19-09', 'VH-19-10', 'VH-19-11',
       'VH-19-12', 'VH-20-01', 'VH-20-02', 'VH-20-03', 'VH-20-04', 'VH-20-05',
       'VH-20-06', 'VH-20-07', 'VH-20-08', 'VV-19-09', 'VV-19-10', 'VV-19-11',
       'VV-19-12', 'VV-20-01', 'VV-20-02', 'VV-20-03', 'VV-20-04', 'VV-20-05',
       'VV-20-06', 'VV-20-07', 'VV-20-08']

s2sspring = ('models2019-20/'
 'croplcm_RAW_sep2sepspring_rf.gz')

s2swinter = ('models2019-20/'
             'croplcm_RAW_sep2sepwinterrf.gz')

sep2sepmdl = ('models2019-20/', 
              'croplcm_RAW_sep2sep_rf.gz')


#all
l.plot_feature_importances(modelrf, fnames_s2s)
# most important is during growing season so makes sense!
l.plot_feature_importances(s2sspring, fnames_s2s)
# likewise here, some difference in winter
l.plot_feature_importances(s2swinter, fnames_s2s)
# this model is the naeseason one
l.plot_feature_importances(modelxgb, flds, model_type='xgb')



# With the variable results, it is all pointing to trying the ceasium method

# before, quickly test on the wimpole data (too small but hey...)

fnames = fnames_s2s+nds
model='models2019-20/croplcm_RAW_sep2sepwinter_xgb.gz'
l.classify_object(model, testshp, fnames, field_name='classif', write='gpd')

















