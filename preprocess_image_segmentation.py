# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:02:52 2023

@author: franc
"""

import geopandas as gpd
from shapely.geometry import box
import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import os
import subprocess
from shapely.geometry import Polygon
#from geocube.api.core import make_geocube
import glob
import math
from geocube.api.core import make_geocube


def getXY(pt):
    return (pt.x, pt.y)

def box_coords(point, width, height, crs_in):
    crop_layer = gpd.GeoDataFrame(
        geometry=[
            box(point[0]-int(width/2), point[1]+int(height/2), point[0]+int(width/2), point[1]-int(height/2))
        ],
        crs='EPSG:'+str(crs_in)
    )
    return(crop_layer)

def polygonize(infile, driver='GeoJSON'):
    polygscript_path = r'C:\Users\franc\anaconda3\envs\gudenv\Scripts\gdal_polygonize.py'
    if driver == 'GeoJSON':
        outfile = infile[:-4] + '.geojson'
        if os.path.isfile(outfile):
            os.remove(outfile)
    if driver == 'ESRI Shapefile':
        outfile = infile[:-4] + '.shp'
        if os.path.isfile(outfile):
            os.remove(outfile)
    subprocess.call(['python', polygscript_path, infile, '-f', driver, outfile])
    return(outfile)

def generatefootprint(image_path):
    tc = rxr.open_rasterio(image_path, chunks=True)
    imname = os.path.basename(image_path).split('.')[0]
    imdir = os.path.dirname(image_path)
    footppath = os.path.join(imdir, imname+'_footp.tif' )
    outpath_footp_vect = footppath.replace('tif', 'geojson')

    if not os.path.isfile(outpath_footp_vect):
        print('Compute footp ' + footppath)
        footp = xr.where(np.logical_or(tc[1].values > 0, tc[2].values > 0, tc[3].values > 0), 1, 0)
        footp_xr = xr.DataArray(footp, coords={'y': tc.y,'x': tc.x}, dims=["y", "x"])
        #footp = xr.where(np.logical_or(red > 0, nir > 0), 1, 0) #several bands in different files
        #footp = xr.where(nir.values > 0, 1, 0) #One band
        footp_xr = footp_xr.astype(np.int8)
        footp_xr = footp_xr.rio.write_crs(tc.rio.crs.to_epsg())
        footp_xr.rio.to_raster(footppath)
        #Polygonize footp
        footpvectpath = polygonize(footppath)
        footp_gdf = gpd.read_file(footpvectpath)
        #Select records by DN and minimum area
        footp_gdf_clean = footp_gdf[(footp_gdf.DN == 1) & (footp_gdf.area > 10000)]
        if len(footp_gdf_clean) != 1:
            print('Revisar footp ' + footpvectpath)
        #Remove holes
        xport = gpd.GeoDataFrame(columns=footp_gdf_clean.columns)
        xport.loc[0, 'geometry'] = Polygon(list(footp_gdf_clean.iloc[0]['geometry'].exterior.coords))
        xport.loc[0, 'imagepath'] = image_path
        xport['DN'] = 1
        xport.crs = footp_gdf_clean.crs
        #Save footprint vector cleaned
        xport.to_file(outpath_footp_vect, driver ='GeoJSON')
        #rasterize 
        # boundary = make_geocube(
        #     vector_data=xport,
        #     #resolution=(-0.0001, 0.0001),
        #     like=footp_xr, 
        #     fill=0,
        # )
        # boundary.rio.to_raster(footppath)
        # #Remove tif footprint
        os.remove(footppath)

    return(outpath_footp_vect)

def manage_footprints(workingdir, zonaname):
    
    #workingdir = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_ValldEbo'
    images = glob.glob(os.path.join(workingdir, '*.tif'))
    footp = [generatefootprint(im) for im in images]
    footp_gpd = [gpd.read_file(footpvectpath) for footpvectpath in footp]
    union = gpd.GeoDataFrame(pd.concat(footp_gpd))
    union = union.to_crs(32630)
    outpath = os.path.join(workingdir, zonaname+'_union_footp.geojson')
    #if not os.path.isfile(outpath):
    union.to_file(outpath, driver ='GeoJSON')
    return(outpath)

def reprojectraster(fcgpd, raster):
    proj = raster.rio.crs.to_proj4()
    reproj = fcgpd.to_crs(proj)
    return reproj

def open_clean_band(band_path, crop_layer=None):
    if crop_layer is not None:
        try:
            clip_bound = crop_layer
            cleaned_band = rxr.open_rasterio(band_path, chunks={'x':512, 'y':512}, masked=True).rio.clip(clip_bound.geometry, from_disk=True).squeeze()
            #cleaned_band = rxr.open_rasterio(band_path, masked=True).rio.clip(clip_bound, from_disk=True).squeeze()
        except Exception as err:
            print("Oops, I need a geodataframe object for this to work.")
            print(err)
    else:
        cleaned_band = rxr.open_rasterio(band_path, chunks={'x':512, 'y':512}, masked=True).squeeze()
        #cleaned_band = rxr.open_rasterio(band_path, masked=True).squeeze()
    return cleaned_band

def closer_corner(coords, origX, origY, step):
    nx = int((coords[0]-origX)/step)
    ny = int((coords[1]-origY)/step)
    UL = (nx*step, (ny+1)*step)
    UR = ((nx+1)*step, (ny+1)*step)
    LL = (nx*step, ny*step)
    LR = ((nx+1)*step, ny*step)
    corners = [UL, UR, LL, LR]
    dist = [math.sqrt((p[0] - coords[0])**2 + (p[1] - coords[1])**2) for p in corners]
    min_index = np.array(dist).argmin()
    return(corners[min_index])

def check_valid_centroid(centr, width, height, res, polyg, crs_in):
    bbox = gpd.GeoDataFrame(
        geometry=[
            box(centr[0]-(width*res)/2, centr[1]+(height*res)/2, centr[0]+(width*res)/2, centr[1]-(height*res)/2)
        ],
        crs='EPSG:'+str(crs_in)
    )
    polyg_df = gpd.GeoDataFrame(geometry=[polyg.geometry], crs='EPSG:'+str(crs_in))
    inters = gpd.overlay(polyg_df, bbox, how='intersection', keep_geom_type=None, make_valid=True)
    if len(inters)>0:
        return (True)
    else:
        return(False)

def compute_centroids_bbox(polyg, width, height, res, crs_in):
    bbox_polyg = polyg.geometry.bounds
    dif_x = bbox_polyg[2] - bbox_polyg[0]
    dif_y = bbox_polyg[3] - bbox_polyg[1]
    num_x = math.ceil(((dif_x)/res)/width)
    num_y = math.ceil(((dif_y)/res)/height)
    valid_centr = []
    for j in range(num_y):
        for i in range(num_x):
            centr = (bbox_polyg[0]+(i*dif_x/num_x)+(width*res/2), (bbox_polyg[3]-(j*dif_y/num_y)-(height*res/2)))
            if check_valid_centroid(centr, width, height, res, polyg, crs_in):
                valid_centr.append(centr)
    return(valid_centr)

def select_ortho(box_polyg, ortho_footp_path, crs_out):
    ortho_footp = gpd.read_file(ortho_footp_path)
    if not crs_out == ortho_footp.crs.to_epsg():
        ortho_footp = ortho_footp.to_crs(crs_out)
    inters = gpd.overlay(ortho_footp, box_polyg, how='intersection', keep_geom_type=None, make_valid=True)
    inters['area'] = inters['geometry'].area
    return(inters.loc[inters['area'].idxmax()]['imagepath'])

def generate_imagesamples_patches(samplespath, dir_image_patches, dir_sample_patches, width, height, res, ortho_footp_path, crs_out, origX, origY):
    #
    samples = gpd.read_file(samplespath)
    samples = samples.explode()
    crs_in = samples.crs.to_epsg()
    if not crs_out == crs_in:
        samples = samples.to_crs(crs_out)

    centroidseries = samples['geometry'].centroid
    centr_coords = list((map(getXY, centroidseries)))
    cont = 0
    step = res
    for i in range(len(samples)):
        valid_centr = compute_centroids_bbox(samples.iloc[i], width, height, res, crs_in)
        for j, centr_coords in enumerate(valid_centr):
            coords_grid = closer_corner(centr_coords, origX, origY, step)
            box_polyg = box_coords(coords_grid, int(width*res), int(height*res), crs_in)
            #Select ortho image that cover the sample
            orto_path = select_ortho(box_polyg, ortho_footp_path, crs_out)
            im = open_clean_band(orto_path, crop_layer=box_polyg)
            im_path = os.path.join(dir_image_patches, str(cont).zfill(5)+'.tif')
            im.rio.to_raster(im_path)
            
            #Create sample ground truth
           # sample_df = samples.loc[i]
            sample_df = samples#.loc[i]
            sample_path = os.path.join(dir_sample_patches, str(cont).zfill(5)+'.tif')
            boundary = make_geocube(
                vector_data=sample_df,
                measurements=['Class'],
                #resolution=(-0.0001, 0.0001),
                like=im[0], 
                fill=0,
            )
            boundary = boundary.astype(np.int8)
            boundary.rio.to_raster(sample_path)
            cont=cont+1


#csv file containing paths and names of the different zones
zonas_paths = r"C:\PROYECTOS\HUELLA_INCENDIO_112\zonas_rutas_sigpac.csv"
df_zonas = pd.read_csv(zonas_paths, sep=';')

#Dimensions of image patch in px
width = 512
height = 512
res = 0.25
ortho_footp_path = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_ValldEbo\ValldEbo_union_footp.geojson'
crs_out = 32630
origX = 0
origY = 0

for i in range(len(df_zonas)):
    workingdir = df_zonas['ortos_dir'][i]
    zonaname = df_zonas['zona'][i]
    manage_footprints(workingdir, zonaname)
    samplespath = df_zonas['samplespath'][i]
    dir_image_patches = df_zonas['dir_image_patches'][i]
    dir_sample_patches = df_zonas['dir_sample_patches'][i]
    generate_imagesamples_patches(samplespath, dir_image_patches, dir_sample_patches, width, height, res, ortho_footp_path, crs_out, origX, origY)


def recode_samples(samples_path):
    samples_path = df_zonas['samplespath'][i]
    samples = gpd.read_file(samples_path)
    #Recode 1, 2, 3 to 1
    samples.loc[((samples['Class'] == 2) | (samples['Class'] ==3)), ['Class']] =1    
    samples.loc[((samples['Class'] == 5) | (samples['Class'] ==6)), ['Class']] =2    
    samples.loc[(samples['Class'] == 7), ['Class']] =3
    path_out = 'D:/GUD/HUELLA_INCENDIO_112/Data_Classif/samples/samples_ValldEbo_EPSG4326_recoded.shp'
    samples.to_file(path_out)
    pass

# samplespath = r'C:/PROYECTOS/HUELLA_INCENDIO_112/SipPac_ValldEbo/dummy_noZ_preprocess_sigpac_ValldEbo.geojson'
# dir_image_patches = r"D:\GUD\HUELLA_INCENDIO_112\Data_Classif\image_patches"
# dir_sample_patches = r"D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sample_patches"

# generate_imagesamples_patches(samplespath, dir_image_patches, dir_sample_patches, width, height, res, ortho_footp_path, crs_in, origX, origY)
