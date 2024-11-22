import os
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
!pip install hmmlearn
from hmmlearn import hmm

!pip install scikit-gstat
import skgstat as skg
from skgstat import Variogram

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def make_lczdict():
  data_lcz = pd.read_csv('/content/drive/MyDrive/Research/Morphology/Data/LCZ_dict.csv', header=None).set_index(0)
  LCZ_dict = data_lcz.to_dict(orient='dict')
  data_val = pd.read_csv('/content/drive/MyDrive/Research/Morphology/Data/val_areas_lcz.csv').set_index('id')
  val_lcz_dict = data_val.to_dict(orient="dict")
  val_lcz_dict=val_lcz_dict['lcz']
  lcz_rec = []
  for id in range(28):
    lcz_rec.append(LCZ_dict[1][val_lcz_dict[id]])
  return val_lcz_dict, LCZ_dict

def stack_binData(IDs):
  i=0
  for id in IDs:
    if i==0:
      gt_bin = gtbin_construct(gt_mod, id, dates)
      hmm_bin = hmmbin_construct(cnn_mod, id, dates)
      id_log = np.repeat(id, gtbin_construct(gt_mod, id, dates).shape[0])
    else:
      gt_bin1 = gtbin_construct(gt_mod, id, dates)
      hmm_bin1 = hmmbin_construct(cnn_mod, id, dates)
      gt_bin = np.append(gt_bin, gt_bin1, axis=0)
      hmm_bin = np.append(hmm_bin, hmm_bin1, axis=0)
      id_log = np.append(id_log, np.repeat(id, gtbin_construct(gt_mod, id, dates).shape[0]))
    i=i+1
  return gt_bin, hmm_bin, id_log

def gtbin_construct1(gt_bin, id, dates):
  #Returns arrays of n cells by num. years

  #Validation Areas
  crs = 32736
  validation_gdf = gpd.read_file("/content/drive/MyDrive/Research/Morphology/Data/Validation Areas/validationAreas (1).shp").to_crs(crs)

  i=0
  dates = np.flip(dates) #Start from last available year to initialize array
  for code in dates:
    year = code[:4]
    gt_parquet = '/content/drive/MyDrive/Research/Morphology/Data/grid_dfs/'+str(year)+'_'+str(id)+'_kigali_gt_raster10'
    if os.path.exists(gt_parquet):
      gt = gpd.read_parquet(gt_parquet)
      gt = gt[gt.intersects(validation_gdf.loc[id,:].geometry)].fillna(0)
      gt = gt.rename(columns={"value": year})
      if i<1:
        gt_all = gt
      else:
        gt_all = gt_all.sjoin(gt).drop(columns = ["index_right"])
    else: #Handle Empty Parquets (Sparsly Built with no geometry in early years)
      gt = gpd.GeoDataFrame(columns=['geometry', 'value'], index=range(625), geometry='geometry', crs=crs)
      gt = gt.rename(columns={"value": year})
      if i<1:
        gt_all = gt
      else:
        gt_all[year] = 0
    i=i+1
  return gt_all.drop(columns = ["geometry"]).to_numpy().astype('int')[::-1]

def gtbin_construct(gt_mod, id, dates):
  crs = 32736
  i=0
  for code in dates:
    year = code[:4]
    conf = gt_mod[year][id].set_crs(crs)
    conf = conf.rename(columns={"value": year})
    if i<1:
      conf_all = conf
    else:
      conf_test = conf_all.sjoin(conf).drop(columns = ["index_right"])
      if conf_test.shape[0]==0:
        conf_all[year] = 0
      else:
        conf_all = conf_test
    i=i+1
  return conf_all.drop(columns = ["geometry"]).to_numpy()


def hmmbin_construct(cnn_mod, id, dates):
  #Bug: ID 2 results in 0 cells in 2017 (because 2017 is too small, may need to recollect data)
  #Returns arrays of n cells by num. years

  i=0
  for code in dates:
    year = code[:4]
    conf = cnn_mod[year][id]
    conf = conf.rename(columns={"value": year})
    if i<1:
      conf_all = conf
    else:
      conf_all = conf_all.sjoin(conf).drop(columns = ["index_right"])
    i=i+1
  return conf_all.drop(columns = ["geometry"]).to_numpy()/100

def loadGT(dates):
  #Loads dataframe of all Ground Truth Data for all validation areas as a dictionary indexed [Year][ID]

  pixel_df = {}
  #Validation Areas
  crs = 32736
  validation_gdf = gpd.read_file("/content/drive/MyDrive/Research/Morphology/Data/Validation Areas/validationAreas (1).shp").to_crs(crs)

  for code in dates:
    year = code[:4]
    pixel_df[year] = {}
    for id in [ 3,  4,  5,  6,  7,  9, 11, 12, 13, 14, 18, 20, 21, 23]:
      #gt_parquet = '/content/drive/MyDrive/Research/Morphology/Data/grid_dfs/'+str(year)+'_'+str(id)+'_kigali_gt_raster10'
      gt_parquet = f"/content/drive/MyDrive/Research/Morphology/Data/grid_dfs/{year}_{id}_grid10"
      if os.path.exists(gt_parquet):
        cnn_mod = gpd.read_parquet(gt_parquet)

      else:
        #Check next year for valid geometry
        code = dates[1]
        year1 = code[:4]
        #gt_parquet = '/content/drive/MyDrive/Research/Morphology/Data/grid_dfs/'+str(year1)+'_'+str(id)+'_kigali_gt_raster10'
        gt_parquet = f"/content/drive/MyDrive/Research/Morphology/Data/grid_dfs/{year1}_{id}_grid10"
        if os.path.exists(gt_parquet):
          cnn_mod = gpd.read_parquet(gt_parquet)
          cnn_mod['value'] = 0
      pixel_df[year][id] = cnn_mod[cnn_mod.intersects(validation_gdf.loc[id,:].geometry)].fillna(0)
  return pixel_df


def loadCNNconf(dates):
  #Loads dataframe of all CNN Model outputs for all validation areas as a dictionary indexed [Year][ID]

  pixel_df = {}
  #Validation Areas
  crs = 32736
  validation_gdf = gpd.read_file("/content/drive/MyDrive/Research/Morphology/Data/Validation Areas/validationAreas (1).shp").to_crs(crs)
  for code in dates:
    year = code[:4]
    pixel_df[year] = {}
    cnn_mod = gpd.read_parquet("/content/drive/MyDrive/Research/Morphology/Data/grid_dfs/"+year+"_conf_grid10")
    for id in range(28):
      pixel_df[year][id] = cnn_mod[cnn_mod.intersects(validation_gdf.loc[id,:].geometry)].fillna(0)
  return pixel_df

## Calibration
def calibrateHMM_dk_parametric(mat, threshold, t01, t10, e11, e00):
  X=mat.ravel().reshape(-1, 1)
  lengths = np.ones((1, mat.shape[0]))*mat.shape[1]
  lengths = lengths[0].astype('int').tolist()
  model_gt = hmm.CategoricalHMM(n_components=2, startprob_prior=0, init_params="", params = "", random_state=9).fit(X, lengths)
  model_gt.emissionprob_ = np.array([[e00, 1-e00],[1-e11, e11]])
  model_gt.transmat_ = np.array([[1-t01, t01],[t10, 1-t10]])
  model_gt.startprob_ = np.array([.9, .1])

  return model_gt

def calibrateHMM_dk_OSLimit(mat, threshold):
  #Test the Application: Limit Case where emission = 1
  X=mat.ravel().reshape(-1, 1)
  lengths = np.ones((1, mat.shape[0]))*mat.shape[1]
  lengths = lengths[0].astype('int').tolist()
  model_gt = hmm.CategoricalHMM(n_components=2, init_params="s", params = "s", random_state=9)
  model_gt.emissionprob_ = np.array([[1, 0],[0, 1]]) # Perfect Emission
  model_gt.transmat_ = np.array([[.3, .7],[.3, .7]])
  model_gt.fit(X, lengths)

  return model_gt

def calibrateHMM_dk(mat, threshold):
  X=mat.ravel().reshape(-1, 1)
  lengths = np.ones((1, mat.shape[0]))*mat.shape[1]
  lengths = lengths[0].astype('int').tolist()
  model_gt = hmm.CategoricalHMM(n_components=2, init_params="s", params = "s", random_state=9)
  model_gt.emissionprob_ = np.array([[.5, .5],[.5, .5]])
  model_gt.transmat_ = np.array([[.3, .7],[.3, .7]])
  model_gt.fit(X, lengths)

  return model_gt

def calibrateHMM_gt(mat, mat_cnn, threshold):
  mat = mat.astype('int')
  mat_cnn = (mat_cnn>threshold).astype('int')
  X=mat.ravel().reshape(-1, 1)
  lengths = np.ones((1, mat.shape[0]))*mat.shape[1]
  lengths = lengths[0].astype('int').tolist()

  model_gt = hmm.CategoricalHMM(n_components=2, init_params="", params = "", random_state=9).fit(X, lengths)
  # Calculate Emission Probabilities
  mat_diff = np.subtract(2*mat, mat_cnn)
  unique, counts = np.unique(mat_diff, return_counts=True)
  e00 = counts[1]/(counts[0]+counts[1])
  e01 = counts[0]/(counts[0]+counts[1])
  e10 = counts[3]/(counts[2]+counts[3])
  e11 = counts[2]/(counts[2]+counts[3])
  model_gt.emissionprob_ = np.array([[e00, e01],[e10, e11]]) #This is the expected emission probability of the model (We can calibrate this with the model though)
  #Calculate transition probabilities
  mat_diff = np.subtract(2*mat[:,:-1], mat[:,1:])
  unique, counts = np.unique(mat_diff, return_counts=True)
  t00 = counts[1]/(counts[0]+counts[1])
  t01 = counts[0]/(counts[0]+counts[1])
  t10 = counts[3]/(counts[2]+counts[3])
  t11 = counts[2]/(counts[2]+counts[3])
  model_gt.transmat_ = np.array([[t00, t01],[t10, t11]])
  #Calculate Start Probabilities
  unique, counts = np.unique(mat[:,0], return_counts=True)
  model_gt.startprob_ = np.array([counts[0]/(counts[0]+counts[1]), counts[1]/(counts[0]+counts[1])])


  model_gt_best = model_gt
  score_best = model_gt.score(X, lengths)

  return model_gt_best

def calibrateHMM(mat, threshold):
  X=mat.ravel().reshape(-1, 1)
  lengths = np.ones((1, mat.shape[0]))*mat.shape[1]
  lengths = lengths[0].astype('int').tolist()

  model_CNN_best = hmm.CategoricalHMM(n_components=2, emissionprob_prior=np.array([[.9, .1],[.1, .9]]), transmat_prior=np.array([[.9, .1],[.1, .9]])).fit(X, lengths)
  score_best = model_CNN_best.score(X, lengths)
  for i in range(10):
    model_CNN = hmm.CategoricalHMM(n_components=2, emissionprob_prior=np.array([[.9, .1],[.1, .9]]), transmat_prior=np.array([[.75, .25],[.1, .9]]), random_state=i)
    model_CNN.fit(X, lengths)
    score = model_CNN.score(X, lengths)
    if score > score_best:
      model_CNN_best = model_CNN
      score_best = score
  return model_CNN_best

def calibrateHMM_Gaussian(mat, threshold):
  #This model retains CNN confidence information
  X= mat.ravel().reshape(-1, 1)
  X = np.column_stack([X, 1-X])
  lengths = np.ones((1, mat.shape[0]))*mat.shape[1]
  lengths = lengths[0].astype('int').tolist()

  model_CNN_best = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000).fit(X, lengths)
  score_best = model_CNN_best.score(X, lengths)
  for i in range(10):
    model_CNN = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000, random_state=i)
    model_CNN.fit(X, lengths)
    score = model_CNN.score(X, lengths)
    if score > score_best:
      model_CNN_best = model_CNN
      score_best = score
  return model_CNN_best



def calibrateHMM_mle(mat, threshold):
  conv_arr= (mat.values>threshold).astype("int")
  X=conv_arr.ravel().reshape(-1, 1)
  lengths = np.ones((1, conv_arr.shape[0]))*mat.shape[1]
  lengths = lengths[0].astype('int').tolist()

  model_gt = hmm.CategoricalHMM(n_components=2, init_params="st", params = "st", random_state=9)
  model_gt.emissionprob_ = np.array([[.8, .2],[.4, .6]])
  model_gt.fit(X, lengths)
  model_gt_best = model_gt
  score_best = model_gt.score(X, lengths)

  for i in range(10):
    model_gt = hmm.CategoricalHMM(n_components=2, init_params="st", params = "st", random_state=i)
    model_gt.emissionprob_ = np.array([[.99, .01],[.01, .99]])
    model_gt.fit(X, lengths)
    score = model_gt.score(X, lengths)
    if score > score_best:
      model_gt_best = model_gt
      score_best = score
  return model_gt_best

## Prediction
def predictHMM(model, mat, threshold):
  X=mat.ravel().reshape(-1, 1)
  lengths = np.ones((1, mat.shape[0]))*mat.shape[1]
  lengths = lengths[0].astype('int').tolist()

  Z_model = model.predict_proba(X, lengths)
  #Convert from probability sets to probability x=1
  Z_mod1 = np.zeros([len(Z_model),1])
  for i in range(len(Z_model)):
    Z_mod1[i] = Z_model[i][1]
  mat_pred = {}
  mat_pred['probs'] = Z_mod1.reshape((int(X.shape[0]/len(dates)),len(dates)))

  Z_model = model.predict(X, lengths)
  mat_pred['map'] = Z_model.reshape((int(X.shape[0]/len(dates)),len(dates)))
  return mat_pred

def predictHMM_Gaussian(model, mat, threshold):
  X=mat.ravel().reshape(-1, 1)
  lengths = np.ones((1, mat.shape[0]))*mat.shape[1]
  lengths = lengths[0].astype('int').tolist()

  Z_model = model.predict_proba(X, lengths)
  #Convert from probability sets to probability x=1
  Z_mod1 = np.zeros([len(Z_model),1])
  for i in range(len(Z_model)):
    Z_mod1[i] = Z_model[i][1]
  mat_pred = {}
  mat_pred['probs'] = Z_mod1.reshape((int(X.shape[0]/len(dates)),len(dates)))

  Z_model = model.predict(X, lengths)
  mat_pred['map'] = Z_model.reshape((int(X.shape[0]/len(dates)),len(dates)))
  return mat_pred

def calibrateHMMs(hmm_conf, gt_bin):
  hmm = {}
  model_record = []
  threshold = 0.1 #Confidence Threshold for binary hmm
  hmm_bin = (hmm_conf>threshold).astype('int')

  #Baseline - One-Shot Model Confidence
  hmm['os'] = {}
  hmm['os']['probs'] = hmm_conf
  model_record.append([id, "os", 0, 0]) # Record Model Parameters

  #Baseline Hidden Markov Model - Limit Case to One-Shot
  model_dk_OSLimit = calibrateHMM_dk_OSLimit(gt_bin, threshold)
  hmm['hmm_os'] = predictHMM(model_dk_OSLimit, hmm_bin, threshold)
  model_record.append([id, "HMM-OS", model_dk_OSLimit.transmat_[0][1], model_dk_OSLimit.transmat_[1][1]]) # Record Model Parameters

  #Hidden Markov Model - Domain Knowledge
  #Test a number of different cases
  #High Growth, High Emission
  t01 = 0.6
  t10 = 0.2
  e11 = 0.6
  e00 = 0.6

  model_dk = calibrateHMM_dk_parametric(hmm_bin, threshold, t01, t10, e11, e00)
  hmm['dk_hghe'] = predictHMM(model_dk, hmm_bin, threshold)
  model_record.append([id, "DK-hghe", model_dk.transmat_[0][1], model_dk.transmat_[1][1]]) # Record Model Parameters

  #High Growth, High Emission
  t01 = 0.6
  t10 = 0.2
  e11 = 0.9
  e00 = 0.9

  model_dk = calibrateHMM_dk_parametric(hmm_bin, threshold, t01, t10, e11, e00)
  hmm['dk_hghe'] = predictHMM(model_dk, hmm_bin, threshold)
  model_record.append([id, "DK-hghe", model_dk.transmat_[0][1], model_dk.transmat_[1][1]]) # Record Model Parameters

  #Medium Growth, High Emission
  t01 = 0.4
  t10 = 0.1
  e11 = 0.8
  e00 = 0.7
  model_dk = calibrateHMM_dk_parametric(hmm_bin, threshold, t01, t10, e11, e00)
  hmm['dk_mghe'] = predictHMM(model_dk, hmm_bin, threshold)
  model_record.append([id, "DK-mghe", model_dk.transmat_[0][1], model_dk.transmat_[1][1]]) # Record Model Parameters

  #model_dk = calibrateHMM_dk(gt_bin, threshold)
  #hmm['dk'] = predictHMM(model_dk, hmm_bin, threshold)
  #model_record.append([id, "DK", model_dk.transmat_[0][1], model_dk.transmat_[1][1]]) # Record Model Parameters

  #Hidden Markov Model - Calibrated on Ground Truth
  model_gt = calibrateHMM_gt(gt_bin, hmm_bin, threshold)
  hmm['gt'] = predictHMM(model_gt, hmm_bin, threshold)
  model_record.append([id, "GT", model_gt.transmat_[0][1], model_gt.transmat_[1][1]]) # Record Model Parameters

  #Hidden Markov Model - Emission is a function of confidence

  # #Model-HMM (Fixed Emission)
  # model_modMLE = calibrateHMM_mle(hmm_bin, threshold)
  # hmm['modMLE'] = predictHMM(model_modMLE, hmm_bin, threshold)
  # model_record.append([id, "HMM-MLE", model_modMLE.transmat_[0][1], model_modMLE.transmat_[1][1]]) # Record Model Parameters

  #Model-HMM-EM
  model_hmm = calibrateHMM(hmm_bin, threshold)
  hmm['modEM'] = predictHMM(model_hmm, hmm_bin, threshold)
  model_record.append([id, "HMM-EM`", model_hmm.transmat_[0][1], model_hmm.transmat_[1][1]]) # Record Model Parameters

  #Model-HMM-Gaussian (EM)
  model_hmmG = calibrateHMM_Gaussian(hmm_bin, threshold)
  hmm['modG'] = predictHMM_Gaussian(model_hmmG, hmm_bin, threshold)
  model_record.append([id, "HMM-Gaussian-EM", model_hmmG.transmat_[0][1], model_hmmG.transmat_[1][1]]) # Record Model Parameters

  return hmm, model_record

def scoreHMM(gt_bin, hmm_pred, id, modType):
  loss1=pd.DataFrame(columns = ['ID', 'Type', 'Year', 'Score', 'Score-Built', 'Score-Empty'])
  baseline = 625

  score = (np.abs(gt_bin-hmm_pred[modType]['probs'])).sum(axis=0)/baseline
  score_built = ((np.abs(gt_bin-hmm_pred[modType]['probs']))*gt_bin).sum(axis=0)/((gt_bin).sum(axis=0)) #Only GT Built Cells
  score_empty = ((np.abs(gt_bin-hmm_pred[modType]['probs']))*np.abs(1-gt_bin)).sum(axis=0)/((np.abs(1-gt_bin)).sum(axis=0)) #Only GT Empty Cells

  loss1['Year'] = years
  loss1['Score'] = score
  loss1['Score-Built'] = score_built
  loss1['Score-Empty'] = score_empty
  loss1['ID'] = id
  loss1['Type'] = modType
  return loss1
  
