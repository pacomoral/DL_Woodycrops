# -*- coding: utf-8 -*-
"""
Created on 15/04/2023

@author: pacomoral
"""

import os
import rioxarray
import xarray as xr
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
import glob
from zipfile import ZipFile
from joblib import Parallel, delayed
from rasterio.enums import Resampling
from datetime import datetime, timedelta
from scipy import stats
import math
import seaborn as sns
from rasterstats import zonal_stats
import subprocess

def reprojectraster(fcgpd, raster):
    proj = raster.rio.crs.to_proj4()
    reproj = fcgpd.to_crs(proj)
    return reproj

def scaledata(dataset, maxinitial, mininitial, maxfinal, minfinal):
    scaled = ((dataset.astype(float) - mininitial) * (maxfinal-minfinal)/(maxinitial-mininitial)).astype(np.uint8)
    #scaled = scaled.rio.write_crs(dataset.rio.crs.to_epsg())
    return (scaled)

#Function for clipping maximum values for Datarrays. Default values set for Sentinel-2 ([0, 10000])
def clipmaxvalue(da, maxvalue=10000, clipvalue=10000):
    clipped = xr.where(da <= maxvalue, da, clipvalue)
    #clipped = clipped.rio.write_crs(da.rio.crs.to_epsg())
    return (clipped)

def selectbands(folder, bandlist):
    filelist = []
    #Level 1C S2 image path
    imsfolder = 'GRANULE\\*\\IMG_DATA\\'
    for band in bandlist:
        filelist.append(''.join(glob.glob(os.path.join(folder, imsfolder,'*' + band + '.jp2'))))
    return(filelist)

def unzipSentinel(zipfile, diroutput, bands=None):
    dirunzip = os.path.join(diroutput, os.path.basename(zipfile).replace('.zip', '.SAFE'))
    #Check if unzipped directory exists (if not it will unzip)
    if not os.path.isdir(dirunzip):
        with ZipFile(zipfile, 'r') as zipObj:
            #If no optional arguments are provided, it will unzip everything
            if bands == None:
                zipObj.extractall(diroutput)
            else:
                namelistzip = zipObj.namelist()
                for tag in bands:
                    [zipObj.extract(filezip, diroutput) for filezip in namelistzip if tag in filezip]
    return(dirunzip)

def open_clean_band(band_path, crop_layer=None):
    if crop_layer is not None:
        try:
            clip_bound = crop_layer.geometry
            cleaned_band = rioxarray.open_rasterio(band_path, chunks={'x':512, 'y':512}, masked=True).rio.clip(clip_bound, from_disk=True).squeeze()
            #cleaned_band = rxr.open_rasterio(band_path, masked=True).rio.clip(clip_bound, from_disk=True).squeeze()
        except Exception as err:
            print("Oops, I need a geodataframe object for this to work.")
            print(err)
    else:
        cleaned_band = rioxarray.open_rasterio(band_path, chunks={'x':512, 'y':512}, masked=True).squeeze()
        #cleaned_band = rxr.open_rasterio(band_path, masked=True).squeeze()
    return cleaned_band
   
#Loadbands en este caso se carga un índice en distintas fechas.
def loadbands(paths, timelist, crop_layer=None):
    #https://www.earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/landsat-in-Python/replace-raster-cell-values-in-remote-sensing-images-in-python/
    all_bands = []
    for i, band_path in enumerate(paths):
        cleaned = open_clean_band(band_path, crop_layer)
        cleaned['time'] = timelist[i]
        all_bands.append(cleaned)
        
    return xr.concat(all_bands, dim='time')

def reproject(fcgpd, raster):
    proj = raster.rio.crs.to_proj4()
    reproj = fcgpd.to_crs(proj)
    return reproj

def scaledata(dataset, maxinitial, mininitial, maxfinal, minfinal):
    scaled = (dataset - mininitial) * (maxfinal-minfinal)/(maxinitial-mininitial)
    scaled = xr.where(scaled > maxfinal, maxfinal, scaled)
    scaled = scaled.rio.write_crs(dataset.rio.crs.to_epsg())
    return (scaled)

def computeindices(inputim, images, geodf, outdir, crs_frame, norm=True):
    print(inputim)
    print('normalization: '+str(norm))
    imname = os.path.basename(inputim)
    dateim = imname[imname.find('MSIL')+7:imname.find('MSIL')+15]
    imB08 = glob.glob(os.path.join(inputim, '*\\*\\*\\*B08*'))[0]
    xds = open_clean_band(imB08, crop_layer=geodf)
    crs_im = xds.rio.crs.to_epsg()

#   Control CRS
    if crs_im != crs_frame:
        geodf = reproject(geodf, xds)
    bluepath = glob.glob(os.path.join(inputim, '*\\*\\*\\*B02*'))[0]
    greenpath = glob.glob(os.path.join(inputim, '*\\*\\*\\*B03*'))[0]
    redpath = glob.glob(os.path.join(inputim, '*\\*\\*\\*B04*'))[0]
    nirpath = glob.glob(os.path.join(inputim, '*\\*\\*\\*B08*'))[0]
    swir1path = glob.glob(os.path.join(inputim, '*\\*\\*\\*B11*'))[0]
    
    red = open_clean_band(redpath, crop_layer=geodf)
    green = open_clean_band(greenpath, crop_layer=geodf)
    blue = open_clean_band(bluepath, crop_layer=geodf)
    nir = open_clean_band(nirpath, crop_layer=geodf)
    swir1 = open_clean_band(swir1path, crop_layer=geodf)
    swir1 = swir1.rio.reproject(swir1.rio.crs, shape=(nir.rio.height, nir.rio.width), resampling=Resampling.nearest)

    ndvi = np.zeros(red.shape, dtype=float)
    ndvi = (nir.astype(rasterio.float32)- red.astype(rasterio.float32))/(nir + red)
    ndvi = xr.where((nir + red) == 0, 0, ndvi)
    ndvi = ndvi.rio.write_crs(red.rio.crs.to_epsg())
    ndvipath = os.path.join(outdir, dateim+'_ndvi.tif')

    ndwi = np.zeros(nir.shape, dtype=float)
    ndwi = (nir.astype(rasterio.float32)- swir1.astype(rasterio.float32))/(nir + swir1)
    ndwi = xr.where((nir + swir1) == 0, 0, ndwi)
    ndwi = ndwi.rio.write_crs(nir.rio.crs.to_epsg())
    ndwipath = os.path.join(outdir, dateim+'_ndwi.tif')
    
    sci = np.zeros(nir.shape, dtype=float)
    sci = (swir1.astype(rasterio.float32)- nir.astype(rasterio.float32))/(swir1+nir)
    sci = xr.where((nir + swir1) == 0, 0, sci)
    sci = sci.rio.write_crs(nir.rio.crs.to_epsg())
    scipath = os.path.join(outdir, dateim+'_sci.tif')
    
    alb = np.zeros(nir.shape, dtype=float)
    alb = (blue.astype(rasterio.float32) + green.astype(rasterio.float32) + red.astype(rasterio.float32))/3
    alb = alb.rio.write_crs(nir.rio.crs.to_epsg())
    albpath = os.path.join(outdir, dateim+'_alb.tif')
    
    if norm==True:
        ndvi = scaledata(ndvi, 1, -1, 255, 0)
        ndwi = scaledata(ndwi, 1, -1, 255, 0)
        sci = scaledata(sci, 1, -1, 255, 0)
        alb = scaledata(alb, 10000, 0, 255, 0)

    ndvi.astype(np.uint8).rio.to_raster(ndvipath)
    ndwi.astype(np.uint8).rio.to_raster(ndwipath)
    sci.astype(np.uint8).rio.to_raster(scipath)
    alb.astype(np.uint8).rio.to_raster(albpath)

    return(ndvipath, ndwipath, scipath, albpath)

def filterdfbydaymonth(dirstoreindices, indextag, dateupdate, rangedays):
    fromdate = dateupdate - timedelta(days=rangedays)
    storedates = [datetime.strptime(os.path.basename(im)[:8], '%Y%m%d') for im in glob.glob(os.path.join(dirstoreindices, '*'+indextag+'*'))]
    df_dates = pd.DataFrame(storedates, columns=['dates'])
    years = set([storedates[i].year for i in range(len(storedates))])
    ims_filtered = pd.DataFrame()
    for year in years:
        start_date = pd.to_datetime(str(year) + str(fromdate.month).zfill(2) + str(fromdate.day).zfill(2))
        end_date = start_date + timedelta(days=2*rangedays)
        df_sel = df_dates[df_dates['dates'].between(start_date, end_date)]
        ims_filtered = pd.concat([ims_filtered, df_sel], ignore_index=True)
    return(ims_filtered)

def returnpercentile(lista, valor):
    #TODO Valorar si implementar kde para Probablity Density Function
    score = stats.percentileofscore(lista, valor, kind='weak')
    return(score)

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

def envelope_corners(boxcoords, origX, origY, step):
    nxUL = int((boxcoords.minx[0]-origX)/step)
    nyUL = int((boxcoords.maxy[0]-origY)/step)
    nxLR = int((boxcoords.maxx[0]-origX)/step)
    nyLR = int((boxcoords.miny[0]-origY)/step)
    ULx = nxUL*step
    ULy = (nyUL+1)*step
    LRx = (nxLR+1)*step
    LRy = (nyLR-1)*step
    return(ULx, ULy, LRx, LRy)

def preprocess_sentinel2(bound_path, crs_frame, workingdir, dirzippedimages):
    numcores = 7
    origX = 0
    origY = 0
    step = 60
    bound = gpd.read_file(bound_path)
    bound = bound.to_crs(crs_frame)
    boxcoords = bound.bounds
    #SUBZONA VERY IMPORTANT: COORDINATES SHOULD BE MULTIPLE OF 10, 20, 60 DEPENDING ON BANDS USED
    ULx, ULy, LRx, LRy = envelope_corners(boxcoords, origX, origY, step)
    #Set area for subset
    geodf = gpd.GeoDataFrame(
        geometry=[
            box(ULx, LRy, LRx, ULy)
        ],
        crs='EPSG:'+str(crs_frame)
    )
    dirstoreindices = os.path.join(workingdir, 'INDICES')
    if not os.path.isdir(dirstoreindices):
        os.mkdir(dirstoreindices)
    bandlist = ['B02', 'B03','B04', 'B08', 'B11']

    imagefolders = glob.glob(os.path.join(dirzippedimages,'*.SAFE'))
    
    #Compute indices and extract info from samples
    indices_paths = Parallel(n_jobs = numcores)(delayed(computeindices)(im, imagefolders, geodf, dirstoreindices, crs_frame) for i, im in enumerate(imagefolders))
    namesindices = ['ndvi', 'ndwi', 'sci', 'alb']
    paths_out = []
    for i, ind in enumerate(namesindices):
        image_list = []
        date_list = []
        ds = []
        all_stats = []
        for j in range(len(indices_paths)):
            image_list.append(indices_paths[j][i])
            date_list.append(os.path.basename(indices_paths[j][i])[:8])
        #open time series for every index
            ds.append(open_clean_band(indices_paths[j][i], crop_layer=None))
                      
        time_varformat = xr.Variable('time', [datetime.strptime(str(time), '%Y%m%d') for time in date_list])
        ds_index = xr.concat(ds, dim=time_varformat)

        #INTERPOLAR DATOS
        #https://docs.dea.ga.gov.au/notebooks/Frequently_used_code/Working_with_time.html
        interpdates = ['2021-09-01', '2021-11-01', '2022-01-01', '2022-03-01', '2022-05-01', '2022-07-01']
        ds_interp = ds_index.interp(time=interpdates)
        #ds_pathout = os.path.join(dirstoreindices, ind+'_interp.tif')
        all_stats.append(ds_interp.mean(dim='time'))
        #.rio.to_raster(os.path.join(dirstoreindices, ind+'_mean.tif'))
        all_stats.append(ds_interp.std(dim='time'))
        all_stats.append(ds_interp.max(dim='time'))
        all_stats.append(ds_interp.min(dim='time'))
        out = os.path.join(dirstoreindices, ind+'_mean_std_max_min.tif')
        paths_out.append(out)
        xr.concat(all_stats, dim='stat').astype(np.uint8).rio.to_raster(out)
        
        #ds_interp.rio.to_raster(ds_pathout)
        #FIN INTERPOLAR DATOS
    return(paths_out)
        
def getXY(pt):
    return (pt.x, pt.y)

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

def box_coords(point, width, height, crs_in):
    crop_layer = gpd.GeoDataFrame(
        geometry=[
            box(point[0]-int(width/2), point[1]+int(height/2), point[0]+int(width/2), point[1]-int(height/2))
        ],
        crs='EPSG:'+str(crs_in)
    )
    return(crop_layer)

def select_ortho(box_polyg, ortho_footp_path):
    inters = gpd.overlay(gpd.read_file(ortho_footp_path), box_polyg, how='intersection', keep_geom_type=None, make_valid=True)
    inters['area'] = inters['geometry'].area
    return(inters.loc[inters['area'].idxmax()]['imagepath'])

def generate_imagesamples_patches(samplespath, dir_image_patches, width, height, res, S2_stats_path, crs_out, origX, origY):
    #
    samples = gpd.read_file(samplespath)
    samples = samples.explode()
    crs_in = samples.crs.to_epsg()
    if not crs_out == crs_in:
        samples = samples.to_crs(crs_out)

    centroidseries = samples['geometry'].centroid
    centr_coords = list((map(getXY, centroidseries)))

    step = res
    for i in range(len(samples)):
        valid_centr = compute_centroids_bbox(samples.iloc[i], width, height, res, crs_in)
        for j, centr_coords in enumerate(valid_centr):
            #For Sentinel-2, step=10, extent of 140 m (rounding up 128m (=512*0.25) considering max resolution of 20 m)
            coords_grid = closer_corner(centr_coords, origX, origY, 10)
            box_polyg = box_coords(coords_grid, 140, 140, crs_in)
            #Select image that covers the sample
            im = []
            for feat in S2_stats_path:
                #orto_path = select_ortho(box_polyg, ortho_footp_path)
                im.append(open_clean_band(feat, crop_layer=box_polyg))
            
            im_path = os.path.join(dir_image_patches, str(i).zfill(4)+'_'+str(j).zfill(3)+'.tif')
            imS2 = xr.concat(im, dim='band')
            
            s2params = (np.asarray(imS2.spatial_ref.GeoTransform.split()).astype(float)).tolist()
            orto_path = im_path.replace('S2_patches', 'image_patches')
            orto = open_clean_band(orto_path)
            orto_params = (np.asarray(orto.spatial_ref.GeoTransform.split()).astype(float)).tolist()
            #Apply resampling and clipping for getting the same dimensions of the S2 image patch with respect to the ortho image patch
            scale_factor = s2params[1]/orto_params[1]
            imS2resampl = imS2.rio.reproject(imS2.rio.crs, shape=(int(imS2.shape[1]*scale_factor), int(imS2.shape[2]*scale_factor)), resampling=Resampling.nearest)
            
            ULx = orto_params[0]
            ULy = orto_params[3]
            LRx = orto_params[0] + orto_params[1]*orto.shape[2]
            LRy = orto_params[3] + orto_params[5]*orto.shape[1]
            imS2resampl = imS2resampl.rio.clip_box(minx=ULx, miny=LRy, maxx=LRx, maxy=ULy)
            
            orto_S2 = xr.concat([orto, imS2resampl], dim='band').rio.to_raster(im_path)
            

#Statistics exploratory analysis function
def getstats_fromimages(samplespath, multibandimagepath, bandlist, croplayer=None):
    #Sample points in geojson format, composed by points/polygons with a column of class 
    samples = gpd.read_file(samplespath)
    #En caso de que el fichero sea MULTIPOINT/MULTIPOLYGON
    samples = samples.explode(index_parts=True)

    multibandimagesdir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\S2_patches'
    samplesdir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sample_patches'
    imagesfiles = glob.glob(multibandimagesdir+'\*.tif')
    samplesfiles = glob.glob(samplesdir+'\*.tif')
    
    stats_list = ['mean']
    classdata = []
    stats_polygs= []
    for samplepath in samplesfiles:
        sample_vect_path = polygonize(samplepath, driver='GeoJSON')
        sample = gpd.read_file(sample_vect_path)
        image_path = samplepath.replace('sample_patches', 'S2_patches')
        
        ds_imgdata = xds = open_clean_band(image_path)
        zs = []
        for i in range(len(ds_imgdata)):
            #imgdata.append(ds_imgdata[datavar].sel(y=samples.geometry.iloc[i].y, x=samples.geometry.iloc[i].x, method="nearest").data.item())
            #imgdata.append(ds_imgdata[datavar].sel(y=samples.geometry.iloc[i].y, x=samples.geometry.iloc[i].x, method="nearest").data)
            temp_path = image_path.replace('.tif', str(i)+'.tif')
            ds_imgdata[i].rio.to_raster(temp_path)
            zs.append(zonal_stats(sample_vect_path, temp_path, stats=stats_list, all_touched=True))
            #delete temp file
            os.remove(temp_path)
        stats_polygs.append([zs[i][0].get('mean') for i in range(len(zs))])
        classdata.append(sample.DN[0])
        os.remove(sample_vect_path)

    #Pasamos todos los datos leídos de las imágenes a un DF y sacamos gráficas
    samplesdatalist = []
    for i in range(len(stats_polygs)):
        rowlista = []
        for j in range(len(stats_polygs[i])):
            rowlista.append(stats_polygs[i][j])
        samplesdatalist.append(rowlista)
    df = pd.DataFrame(samplesdatalist)
    df['Clase'] = classdata
    sns.pairplot(df, vars=df.columns[:-1], hue="Clase")
    stats_samples = df.groupby('Clase').describe()
    df.to_csv(r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\samples_stats.csv', sep=';')

crs_frame = 32630


#csv file containing paths and names of the different zones
zonas_paths = r"C:\PROYECTOS\HUELLA_INCENDIO_112\zonas_rutas_sigpac.csv"
df_zonas = pd.read_csv(zonas_paths, sep=';')

width = 512
height = 512
res = 0.25
ortho_footp_path = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_ValldEbo\ValldEbo_union_footp.geojson'
crs_in = 32630
origX = 0
origY = 0

for i in range(len(df_zonas)):
    workingdir = df_zonas['S2_dir'][i]
    zonaname = df_zonas['zona'][i]
    bound_path = df_zonas['boundary'][i]
    dirzippedimages = df_zonas['S2_unzipped_dir'][i]
    #manage_footprints(workingdir, zonaname)
    S2_stats_path = preprocess_sentinel2(bound_path, crs_frame, workingdir, dirzippedimages)
    samplespath = df_zonas['samplespath'][i]
    dir_image_patches = df_zonas['S2_patches_dir'][i]
    generate_imagesamples_patches(samplespath, dir_image_patches, width, height, res, S2_stats_path, crs_in, origX, origY)
#Dimensions of image patch in px


samplespath = r'C:/PROYECTOS/HUELLA_INCENDIO_112/SipPac_ValldEbo/dummy_noZ_preprocess_sigpac_ValldEbo.geojson'
dir_image_patches = r"D:\GUD\HUELLA_INCENDIO_112\Data_Classif\image_patches"
dir_sample_patches = r"D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sample_patches"

generate_imagesamples_patches(samplespath, dir_image_patches, width, height, res, ortho_footp_path, crs_in, origX, origY)
