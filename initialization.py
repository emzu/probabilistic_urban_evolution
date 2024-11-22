#Validation Areas
crs = 32736
validation_gdf = gpd.read_file("/content/drive/MyDrive/Research/Morphology/Data/Validation Areas/validationAreas (1).shp").to_crs(crs)
#Load Pleides Building Footprint Data
pleides_ref = {}
gt_parquet = '/content/drive/MyDrive/Research/Morphology/Data/grid_dfs/2015_pleides_grid'
if os.path.exists(gt_parquet):
  gt = gpd.read_parquet(gt_parquet)
for id in range(28):
  pleides_ref[id] = gt[gt.intersects(validation_gdf.loc[id,:].geometry)].fillna(0)
  pleides_ref[id]['bin'] = (pleides_ref[id]['value']>0).astype('int')

validation_gdf = gpd.read_file("/content/drive/MyDrive/Research/Morphology/Data/Validation Areas/validationAreas (1).shp").to_crs(crs)
googleT_ref = {}
#Load Google Building Footprint Data
for year in range(2016, 2024):
  googleT_ref[year] = {}
  gt_parquet = f"/content/drive/MyDrive/Research/Morphology/Data/grid_dfs/googleTemporal_{year}_grid10"
  if os.path.exists(gt_parquet):
    gt_google = gpd.read_parquet(gt_parquet).to_crs(crs)
    for id in range(28):
      googleT_ref[year][id] = gpd.clip(gt_google, validation_gdf.loc[id,:].geometry).fillna(0)
      googleT_ref[year][id]['bin'] = (googleT_ref[year][id]['value']>0).astype('int') 

validation_gdf = gpd.read_file("/content/drive/MyDrive/Research/Morphology/Data/Validation Areas/validationAreas (1).shp").to_crs(crs)
#Load Ground Truth Building Footprint Data
gt_ref = {}

for id in range(28):
  gt_parquet = f"/content/drive/MyDrive/Research/Morphology/Data/grid_dfs/2017_{id}_grid10"
  if os.path.exists(gt_parquet):
    gt_hand = gpd.read_parquet(gt_parquet)
    gt_ref[id] = gt_hand[gt_hand.intersects(validation_gdf.loc[id,:].geometry)].fillna(0)
    gt_ref[id]['bin'] = (gt_ref[id]['value']>0).astype('int')

validation_gdf = gpd.read_file("/content/drive/MyDrive/Research/Morphology/Data/Validation Areas/validationAreas (1).shp").to_crs(crs)
#Load Google Building Footprint Data
google_ref = {}
gt_parquet = '/content/drive/MyDrive/Research/Morphology/Data/grid_dfs/2022_google_grid10'
if os.path.exists(gt_parquet):
  gt_google = gpd.read_parquet(gt_parquet)
for id in range(28):
  google_ref[id] = gt_google[gt_google.intersects(validation_gdf.loc[id,:].geometry)].fillna(0)
  google_ref[id]['bin'] = (google_ref[id]['value']>0).astype('int')

## Load CNN and GT
dates = ["2006_0225", "2011_0620", "2017_0713", "2022_0126"]
years = [2006, 2011, 2017, 2022]

 #Load all CNN Parquet files into gpd dictionary
cnn_mod = loadCNNconf(dates)
gt_mod = loadGT(dates)
val_lcz_dict, LCZ_dict = make_lczdict()

valid_IDs = np.array([4,  5,  6,  7,  8, 10, 12, 13, 14, 15, 19, 21, 22, 24])-1

relative_score = []
score_hmm = []
model_log = {}

#Record Scores
score_baseline = {}
score_baseline_binary = {}
score_hmm = {}
rel_score = {}
