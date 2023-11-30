# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:36:41 2023

@author: franc
"""
import geopandas as gpd
import glob
import os
import numpy as np
import pandas as pd
from geocube.api.core import make_geocube
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterstats import zonal_stats
import rasterio
from joblib import Parallel, delayed
import xarray as xr
from xrspatial import zonal_stats as xrzonal
import regionmask
import timeit
import zipfile

def reprojectgpd(fcgpd, epsg):
    #proj = raster.rio.crs.to_proj4()
    reproj = fcgpd.to_crs(epsg)
    return reproj

def cleanSigPac(sigpac_path, epsg):
    
    dltfields = ['provincia', 'municipio', 'agregado', 'zona', 'poligono', 'parcela', 'recinto', 'dn_surface', 'dn_perimeter']
    sigpac = gpd.read_file(sigpac_path, layer='recinto')
    #Clean fields
    sigpac = sigpac.drop(columns=dltfields)
    usos = ['CF', 'CS', 'CV', 'FF', 'OC', 'CI', 'FY', 'FS', 'FL', 'FV', 'OV','OF', 'VI', 'VF', 'VO',]
    #usos = ['CF', 'CS', 'CV', 'FF', 'OC', 'CI', 'FY', 'FS', 'FL', 'FV', 'OV','OF', 'VI', 'VF', 'VO', 'TA', 'TH', 'IV']
    cleaned = sigpac[sigpac['uso_sigpac'].isin(usos)]
    cleaned['incidencias'].astype(str)
    cleaned = cleaned.fillna(0)
    cleaned = cleaned.replace('', 0)
    #Fix geometries
    cleaned['geometry'] = cleaned.geometry.buffer(0)
    if cleaned.crs.to_epsg() != epsg:
        cleaned = reprojectgpd(cleaned, epsg)
    cleaned=cleaned.explode()
    return(cleaned)

def recodeSigPac(sigpac_path, epsg):
    
    dltfields = ['provincia', 'municipio', 'agregado', 'zona', 'poligono', 'parcela', 'recinto', 'dn_surface', 'dn_perimeter', 'pendiente_media', 'coef_admisibilidad', 'coef_regadio', 'incidencias', 'region']
    sigpac = gpd.read_file(sigpac_path, layer='recinto')
    #Clean fields
    sigpac = sigpac.drop(columns=dltfields)
    
    
    tierra = ['TA', 'TH', 'IV']
    lenoso = ['CF', 'CS', 'CV', 'FF', 'OC', 'CI', 'FY', 'FS', 'FL', 'FV', 'OV','OF', 'VI', 'VF', 'VO',]
    forestal = ['FO', 'MT', 'PS', 'PR', 'PA']
    no_agri = ['AG','ED','EP','MI','CA','ZU', 'ZV', 'ZC', 'IM']
    
    recoded = sigpac.copy()
    
    recoded.loc[recoded['uso_sigpac'].isin(tierra), 'recoded'] = int(1)
    recoded.loc[recoded['uso_sigpac'].isin(lenoso), 'recoded'] = int(2)
    recoded.loc[recoded['uso_sigpac'].isin(forestal), 'recoded'] = int(3)
    recoded.loc[recoded['uso_sigpac'].isin(no_agri), 'recoded'] = int(4)

    recoded = recoded.fillna(0)
    recoded = recoded.replace('', 0)
    #Fix geometries
    recoded['geometry'] = recoded.geometry.buffer(0)
    if recoded.crs.to_epsg() != epsg:
        recoded = reprojectgpd(recoded, epsg)
    recoded=recoded.explode()
    return(recoded)




#csv file containing paths and names of the different zones
zonas_paths = r"C:\PROYECTOS\HUELLA_INCENDIO_112\zonas_rutas_sigpac.csv"
df_zonas = pd.read_csv(zonas_paths, sep=';')

epsg = 32630


for i in range(len(df_zonas)):
    workingdir = df_zonas['sigpac_dir'][i]
    boundarypath = df_zonas['boundary_orto'][i]
    name = df_zonas['zona'][i]
    bound = gpd.read_file(boundarypath)
    ortosdir = df_zonas['ortos_dir'][i]
    if bound.crs.to_epsg() != epsg:
        bound = reprojectgpd(bound, epsg)

    sigpacs = glob.glob(os.path.join(workingdir, '*gpkg.zip'))
    
    for j, sigpac in enumerate(sigpacs):
        sigpac1 = cleanSigPac(sigpac, epsg)
        if j==0:
            temp = sigpac1.copy()
        else:
            temp = pd.concat([sigpac1, temp])


    for j, sigpac in enumerate(sigpacs):
        sigpac1 = recodeSigPac(sigpac, epsg)
        if j==0:
            temp2 = sigpac1.copy()
        else:
            temp2 = pd.concat([sigpac1, temp2])

    union_recoded = gpd.clip(temp2, bound)
    #Arregla geometrias
    union_recoded['geometry'] = union_recoded.geometry.buffer(0)
    union_recoded.to_file(os.path.join(workingdir, 'preprocess_sigpac_recoded_'+name+'.shp'))
    
    join_sigpac_samples()
    
    sigpac_y_samples = gpd.read_file('D:/GUD/HUELLA_INCENDIO_112/Data_Classif/sigpac_con_samples/preprocess_sigpacsamples_ValldEbo.shp')
    sigpac_y_samples['code'] = sigpac_y_samples['recoded_ri'].fillna(sigpac_y_samples['recoded_le'])
    sigpac_y_samples['code'] = pd.to_numeric(sigpac_y_samples['code'], downcast='integer')
    sigpac_y_samples.to_file(os.path.join('D:/GUD/HUELLA_INCENDIO_112/Data_Classif/sigpac_con_samples', 'preprocess_sigpacupdated_'+name+'.shp'))

    #sigpac_y_samples = gpd.read_file('D:\GUD\HUELLA_INCENDIO_112\SigPac_Bejis\preprocess_sigpac_recoded_Bejis.shp')

   # sigpac_y_samples['recoded'] =int(sigpac_y_samples['recoded_le'])
    #ortosdir = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_0_5_ICV_2022_Bejis'
    ortosdir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\ortos_0_5m_ValldEbo'
    ortos = glob.glob(os.path.join(ortosdir, '*.tif'))
    for ortopath in ortos:
        orto = rxr.open_rasterio(ortopath)
        samplemap = make_geocube(
            vector_data=sigpac_y_samples,
            measurements=['code'],
            like=orto, 
            fill=0,
        )
        samplemap = samplemap.astype(np.int8)
        #samplemap_outpath = os.path.join(r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\sigpac_masks', os.path.basename(ortopath))
        samplemap_outpath = os.path.join(r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\sigpac_masks_0_5_ValldEbo', os.path.basename(ortopath))

        samplemap.rio.set_crs(orto.rio.crs, inplace=True)
        samplemap.rio.to_raster(samplemap_outpath)

    union_clipped = gpd.clip(temp, bound)

    #Selección de parcelass con incidencia de SigPac '199' Recinto inactivo
    incid_199 = []
    for k in range(len(union_clipped)):
        if '199' in str(union_clipped['incidencias'].iloc[k]):
            incid_199.append('1')
        else:
            incid_199.append('0')
    union_clipped['incid_199'] = incid_199
    union_clipped.to_file(os.path.join(workingdir, 'preprocess_sigpac_'+name+'.shp'))


def join_sigpac_samples():
    
    samplespath = r'D:/GUD/HUELLA_INCENDIO_112/Data_Classif/samples/samples_ValldEbo_EPSG4326.shp'
    samples = gpd.read_file(samplespath)
    samples['geometry'] = samples.geometry.buffer(0)

    #samples_recoded = samples.copy()
    samples.loc[(samples['Class']==1) |(samples['Class']==2) | (samples['Class']==3), 'recoded'] = 2
    samples.loc[(samples['Class']==5) | (samples['Class']==6), 'recoded'] = 3
    samples.loc[samples['Class']==7, 'recoded'] = 1

    sigpacrecodedpath = r'D:/GUD/HUELLA_INCENDIO_112/SipPac_ValldEbo/preprocess_sigpac_recoded_ValldEbo.shp'
    sigpac = gpd.read_file(sigpacrecodedpath)
    sigpac['geometry'] = sigpac.geometry.buffer(0)

    if samples.crs.to_epsg() != sigpac.crs.to_epsg():
        samples = samples.to_crs(sigpac.crs.to_epsg())
        
    sample_centr_series = samples.representative_point()
    #sample_centr_series['recoded'] = samples['recoded']
    sample_centr = gpd.GeoDataFrame(geometry=sample_centr_series)
    sample_centr['recoded'] = samples['recoded']

    sample_centr.set_crs('epsg:'+str(samples.crs.to_epsg()))
    sample_centr.to_file(os.path.join('D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples', 'preprocess_sigpacsamplespoints_'+name+'.shp'))
    #difference_result = gpd.overlay(sigpac, samples, how = 'difference')
    #union_result = gpd.overlay(sigpac, difference_result, how = 'union')
    union_result = sigpac.sjoin(sample_centr, how="left")
    
    union_result.to_file(os.path.join('D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples', 'preprocess_sigpacsamples_'+name+'.shp'))


def resample_orthos(ortosdir, outdir):
    #ortosdir=r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_ValldEbo'
    ortosdir=r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_Bejis'
    ortosdir=r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_Costur'
    ortos = glob.glob(os.path.join(ortosdir, '*.tif'))
    upscale_factor = 0.5
    #outdir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\ortos_0_5m_ValldEbo'
    outdir = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_0_5_ICV_2022_Bejis'
    outdir = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_0_5_ICV_2022_Costur'
    for ortopath in ortos:
        ortoname = os.path.basename(ortopath)
        ortopathout = os.path.join(outdir, ortoname)
        if not os.path.isfile(ortopathout):
            orto = rxr.open_rasterio(ortopath)
            new_width = int(orto.rio.width * upscale_factor)
            new_height = int(orto.rio.height * upscale_factor)
            orto_res = orto.rio.reproject(orto.rio.crs, shape=(new_height, new_width), resampling=Resampling.nearest)
            orto_res.rio.write_nodata(None, inplace=True)
            #Fix fill values
            #dateset.set_fill_off()
            
            orto_res.rio.to_raster(ortopathout)


def select_ortho(box_polyg, ortho_footp_path, crs_out):
    ortho_footp = gpd.read_file(ortho_footp_path)
    if not crs_out == ortho_footp.crs.to_epsg():
        ortho_footp = ortho_footp.to_crs(crs_out)
    inters = gpd.overlay(ortho_footp, box_polyg, how='intersection', keep_geom_type=None, make_valid=True)
    inters['area'] = inters['geometry'].area
    return(inters.loc[inters['area'].idxmax()]['imagepath'])

def box_coords(geoseries, crs_in):
    crop_layer = gpd.GeoDataFrame(
        geometry=[
            box(point[0]-int(width/2), point[1]+int(height/2), point[0]+int(width/2), point[1]-int(height/2))
        ],
        crs='EPSG:'+str(crs_in)
    )
    return(crop_layer)


# def intersect_map_union(boundary_gpd, big_data_gpd):
#     # union = gpd.GeoDataFrame()
#     # for data in list_data_gpd:
#     #     if boundary_gpd.crs != data.crs:
#     #         boundary_gpd = reprojectgpd(boundary_gpd, data.crs)
#     partial_data = big_data_gpd.clip(boundary_gpd)
#     #partial_data = gpd.overlay(boundary_gpd, data, how='intersection')
#     if len(partial_data)>0:
#         union = pd.concat([union, partial_data])
#     delete_cols =[col for col in list(union) if col not in list(data)]
#     union = union.drop(columns=delete_cols)
#     return(union)

def zonal_probs(prob, polygs_gpd, bandindex, colname):
    arr = prob.read(bandindex+1)
    affine = prob.transform
    stats = zonal_stats(polygs_gpd, arr, stats='mean', affine=affine)
    stats_df = pd.DataFrame(stats)
    stats_df.rename(columns={'mean':colname}, inplace=True)
    polygs_gpd[translate_codes[k]] = pd.Series(stats_df[translate_codes[k]])
    return(stats_df)

def extrae_probs_DL(sigpac_path, probs_dir, footp_path, outfile_path):
    
    from rasterio.merge import merge

    

   #  sigpac_path = r'D:/GUD/HUELLA_INCENDIO_112/SipPac_ValldEbo/preprocess_sigpac_recoded_ValldEbo.shp'
   #  footp_path = r'D:/GUD/HUELLA_INCENDIO_112/ortofoto_ICV_2022_ValldEbo/ValldEbo_union_footp.geojson'
   #  #probs_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_ValldEbo_modelovalldebo'
   #  #probs_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_Costur_modeloBejis'
   #  probs_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_Bejis_modeloBejis'
   #  #zonaname = 'Costur'
   #  zonaname = 'Bejis'

   #  #Mosaic all output probability files
   #  rasters = glob.glob(os.path.join(probs_dir, '*.tif'))


   #  mosaic_files = [] ##make an empty list


   #  src = rasterio.open(rasters[0])
   #  mosaicpath = os.path.join(probs_dir, zonaname+'.tif')
   # #mosaiccodedpath = os.path.join(outdir, os.path.basename(predict_im_name)+'_class.tif')

   # # mosaic, out_trans = merge(tilefiles, method='max')
   #  mosaic, out_trans = merge(rasters)

   #  out_meta = src.meta.copy()
   #  out_meta.update({"driver": "GTiff",
   #                   "height": mosaic.shape[1],
   #                   "width": mosaic.shape[2],
   #                   "transform": out_trans,
   #                   "crs": src.crs
   #                   }
   #                  )
   #  with rasterio.open(mosaicpath, "w", **out_meta) as dest:
   #      dest.write(mosaic)
        
   #  src.close()

# Proba extrae por footprint FUNCIONA
    zonas_paths = r"C:\PROYECTOS\HUELLA_INCENDIO_112\zonas_rutas_sigpac.csv"
    df_zonas = pd.read_csv(zonas_paths, sep=';')
    translate_codes = {0:'background', 1:'arable', 2:'leñoso', 3:'forestal', 4:'no_agric'}
    idgud = 'idgud'
    
    #Select ZONA
    for i in range(len(df_zonas)):
        tic = timeit.default_timer()
        footp_path = glob.glob(df_zonas.iloc[i]['ortos_dir']+'\*union*')[0]
        #footp_path = r'D:/GUD/HUELLA_INCENDIO_112/ortofoto_ICV_2022_ValldEbo/ValldEbo_union_footp.geojson'
        name = df_zonas['zona'][i]
        workingdir = df_zonas['sigpac_dir'][i]
        probs_dir = df_zonas['dir_probabil'][i]
        clip_bound = gpd.read_file(df_zonas['boundary'][i])
        sigpacs = [gpd.read_file(f, layer='recinto')for f in glob.glob(os.path.join(workingdir, '*gpkg.zip'))]
        #Join sigpacs
        union = pd.concat(sigpacs)
        union['geometry'] = union.geometry.buffer(0)
        #union = union.explode()
        union[idgud] = range(len(union))
        
        if union.crs != clip_bound.crs:
            clip_bound = reprojectgpd(clip_bound, union.crs)
        union_clip = gpd.clip(union, clip_bound)

        footpmap = gpd.read_file(footp_path)
        sigpacs_updated = []

        for j in range(len(footpmap)):
            prob = rasterio.open(os.path.join(probs_dir, os.path.basename(footpmap.iloc[[j]]['imagepath'].values[0])))
            #sigpac_footp = intersect_map_union(footpmap.iloc[[j]], union)
            sigpac_footp = footpmap.iloc[[j]]
            nbands = prob.count
            if sigpac_footp.crs != prob.crs:
                sigpac_footp = reprojectgpd(sigpac_footp, prob.crs)
            if union_clip.crs != prob.crs:
                union_clip = reprojectgpd(union_clip, prob.crs)
            sigpac_footp['geometry'] = sigpac_footp.geometry.buffer(0)
            sigpac_footp = union_clip.clip(sigpac_footp)
            sigpac_upd = pd.DataFrame()
            #numcores = nbands
            #sigpacs_updated = Parallel(n_jobs = numcores)(delayed(zonal_probs)(prob, sigpac_footp, k, translate_codes[k]) for k in range(nbands))
    #indices_paths = Parallel(n_jobs = numcores)(delayed(recode_probs_DL)(prob, outdir) for prob in probs)

            for k in range(nbands):
                arr = prob.read(k+1)
                affine = prob.transform
                stats = zonal_stats(sigpac_footp, arr, stats='mean', affine=affine, nodata=prob.nodata)
                stats_list = [list(stats[m].values())[0] for m in range(len(stats))]
                #stats_list = [round(stats_list[c]) for c in range(len(stats_list))]
                stats_list = [round(stats_list[c]) if isinstance(stats_list[c], float) else 0 for c in range(len(stats_list))]
                sigpac_footp['prob_'+translate_codes[k]] = stats_list

                # stats_df = pd.DataFrame(stats)
                # stats_df.rename(columns={'mean':'prob_'+translate_codes[k]}, inplace=True)
                #sigpac_footp['prob_'+translate_codes[k]] = pd.Series(stats_df['prob_'+translate_codes[k]])
                #sigpac_upd = pd.concat([sigpac_footp, stats_df], axis=1)
            sigpacs_updated.append(sigpac_footp)
                
        sigpac_upd_zona = pd.concat(sigpacs_updated)
        #sigpac_upd_zona_diss = sigpac_upd_zona.dissolve(by=sigpac_upd_zona.index, aggfunc='max')
        #Arregla geometrias
        sigpac_upd_zona['geometry'] = sigpac_upd_zona.geometry.buffer(0)
        #Manage splitted polygons
        
        result = sigpac_upd_zona.dissolve(by=idgud)
        result = result.explode()
        result['Clase'] = result.apply(lambda x: 
                              translate_codes[np.argmax(np.array([
                                  x['prob_background'],
                                  x['prob_arable'], 
                                  x['prob_leñoso'], 
                                  x['prob_forestal'],
                                  x['prob_no_agric']
                                  ]))], axis=1)

        #result = result.drop(idgud, axis=1)
        result2 = reprojectgpd(result, union.crs)

        result2.to_file(os.path.join(probs_dir, 'sigpac_clases_'+name+'.geojson'))
        
        
        toc = timeit.default_timer()
        print('tiempo de zona: ' + name + ': ' + str(toc - tic))

        

# Fin Proba extrae por footprint


    footpmap = gpd.read_file(footp_path)
    
    sigpac = gpd.read_file(sigpac_path)
    rasters = glob.glob(os.path.join(probs_dir, '*.tif'))
    
    
    
    if sigpac.crs != footpmap.crs:
        sigpac = reprojectgpd(sigpac, footpmap.crs)
        
    stats_list = 'mean'

    probname = ['background', 'arable', 'leñoso', 'forest', 'noagric']
    for col in probname:
        sigpac[col] = None
        
    store = []
    
    for i in range(len(sigpac)):
        sigpac.iloc[[i]]
        footp_ref_gdf = gpd.overlay(sigpac.iloc[[i]], footpmap, how='intersection')
        footp_ref = os.path.basename(footp_ref_gdf.imagepath[0])
        
        prob_im_ref = os.path.join(probs_dir, footp_ref)
        
        dataset = rasterio.open(prob_im_ref)
        if dataset.crs != sigpac.crs:
            sigpac2 = reprojectgpd(sigpac, dataset.crs)
            
        nbands = dataset.count
        statsbands = []
        for j in range(nbands):
            arr = dataset.read(j+1)
            affine = dataset.transform
            stats = zonal_stats(sigpac2.iloc[[i]], arr, affine=affine)
            #print(stats)
            statsbands.append(stats[0].get('mean'))
        #Fill sigpac
        store.append(statsbands)
        # sigpac.iloc[i]['arable'] = statsbands[1]
        # sigpac.iloc[i]['leñoso'] = statsbands[2]
        # sigpac.iloc[i]['forest'] = statsbands[3]
        # sigpac.iloc[i]['noagric'] = statsbands[4]

        #zs = zonal_stats(sigpac.iloc[[i]], arr, stats=stats_list, all_touched=True)



def recode_probs_DL(prob_path, outdir):
    
    mask_out = os.path.join(outdir, os.path.basename(prob_path))
    if not os.path.isfile(mask_out):
        mask = rxr.open_rasterio(prob_path)
        flat_pred = mask.argmax(dim='band')
        flat_pred.astype(np.int8).rio.to_raster(mask_out)
        return(mask_out)
    


def parallel_recode_probs_DL(probs_dir):
    probs_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_Bejis_modeloBejis'
    probs_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_ValldEbo_modelovalldebo'
    probs_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_Costur_modelovalldebo'
    probs = glob.glob(os.path.join(probs_dir, '*.tif'))
    outdir = probs_dir+'_recoded'
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    

    numcores = 6
    indices_paths = Parallel(n_jobs = numcores)(delayed(recode_probs_DL)(prob, outdir) for prob in probs)


    pass




   #  sigpac_path = r'D:/GUD/HUELLA_INCENDIO_112/SipPac_ValldEbo/preprocess_sigpac_recoded_ValldEbo.shp'
   #  footp_path = r'D:/GUD/HUELLA_INCENDIO_112/ortofoto_ICV_2022_ValldEbo/ValldEbo_union_footp.geojson'
   #  #probs_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_ValldEbo_extended'
   #  #probs_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_Costur_modeloBejis'
   #  probs_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_Bejis_modeloBejis'
   #  #zonaname = 'Costur'
   #  zonaname = 'Bejis'

   #  #Mosaic all output probability files
   #  rasters = glob.glob(os.path.join(probs_dir, '*.tif'))


   #  mosaic_files = [] ##make an empty list


   #  src = rasterio.open(rasters[0])
   #  mosaicpath = os.path.join(probs_dir, zonaname+'.tif')
   # #mosaiccodedpath = os.path.join(outdir, os.path.basename(predict_im_name)+'_class.tif')

    
   #  pass


import cv2
import mahotas as mt

def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean

img=r"D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_Costur\020201_2022CVAL0025_25830_8bits_RGBI_0592_8-8.tif"

image = cv2.imread(img)


# extract haralick texture from the image
features = extract_features(image)
    features = mahotas.features.haralick(image).mean(axis=0)
    
    
def textures_skimage():
    
    #conda activate skimage-dev
    #https://scikit-image.org/docs/stable/user_guide/install.html#install-via-conda
    from skimage.feature import graycomatrix, graycoprops
    from skimage import data
    image = data.camera()
    
    img=r"D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_Costur\020201_2022CVAL0025_25830_8bits_RGBI_0592_8-8.tif"

    from skimage import io
    import numpy as np
    
    image = io.imread(img)
    rows, cols, bands = image.shape
    
    
    glcm = graycomatrix(image[:,:,0], distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    
    
    
    xs = []
    ys = []
 
    xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(graycoprops(glcm, 'correlation')[0, 0])
    
    distances = [1]
    angles = [0, np.pi/2]
    props = ['contrast', 'dissimilarity', 'homogeneity']
    dim = len(distances)*len(angles)*len(props)
    
    band=3
    feats = np.zeros(shape=(rows, cols, dim))
    for row in range(rows):
        for col in range(cols):
            pixel_feats = []
            glcm = graycomatrix(image[:, :, band], distances=distances, angles=angles)
            pixel_feats.extend([graycoprops(glcm, prop).ravel() for prop in props])
            feats[row, col, :] = np.concatenate(pixel_feats)
