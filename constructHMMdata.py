import geopandas as gpd

def gtbin_construct(id, dates):
  #Returns arrays of n cells by num. years

  #Validation Areas
  crs = 32736
  validation_gdf = gpd.read_file("/content/drive/MyDrive/Research/Morphology/Data/Validation Areas/validationAreas (1).shp").to_crs(crs)

  i=0
  for code in dates:
    year = code[:4]
    gt_parquet = '/content/drive/MyDrive/Research/Morphology/Data/grid_dfs/'+str(year)+'_'+str(id)+'_kigali_gt_raster10'
    gt = gpd.read_parquet(gt_parquet)
    gt = gt[gt.intersects(validation_gdf.loc[id,:].geometry)].fillna(0)
    gt = gt.rename(columns={"value": year})
    if i<1:
      gt_all = gt
    else:
      gt_all = gt_all.sjoin(gt).drop(columns = ["index_right"])
    i=i+1
  return gt_all
