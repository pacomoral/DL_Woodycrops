# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:15:40 2023

@author: franc
"""

#https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html

'''
Instalar en environment de conda:
conda create -n proba python=3.9 torchgeo pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia rasterio spyder xarray


'''

import os
import xarray as xr
import glob
import torch
from torch.utils.data import DataLoader, random_split
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler, Units, GridGeoSampler
from torchvision.models.segmentation import deeplabv3_resnet50
from typing import List, Optional, Callable
from sklearn.metrics import jaccard_score
import rasterio
from rasterio.merge import merge
import rioxarray as rxr
import numpy as np
from joblib import Parallel, delayed
import shutil
import timeit

    
#intento 1
#movefiles(trainSet_im.dataset, trainSet_im.indices, train_im_dir)

def movefiles(list_files, indices_list, outDir):
    numcores = 6
    indices_paths = Parallel(n_jobs = numcores)(delayed(shutil.copy)(list_files[ind], os.path.join(outDir, os.path.basename(list_files[ind]))) for ind in indices_list)
#delayed(sqrt)(i**2) for i in range(10)
    # for ind in indices_list:
    #     namefile = os.path.basename(list_files[ind])
    #     shutil.copy(list_files[ind], os.path.join(outDir, namefile))
    
    
def glob_multiple(list_dirs, tag='/*.tif'):
    files = []
    for dire in list_dirs:
        [files.append(file) for file in glob.glob(dire+tag)]
    return(files)

TRAIN_VAL_SPLIT = 0.6

basedir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples'

#MUY IMPORTANTE LAS CLASES DE LAS LABELS TIENEN QUE EMPEZAR EN 0 CORRELATIVAMENTE
#image_dataset_dir =r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_0_5_ICV_2022_Bejis'
image_dataset_dir =r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_ValldEbo'
#image_dataset_dir =r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_Bejis'

#mask_dataset_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_masks'
mask_dataset_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\sigpac_mask_ValldEbo'
#mask_dataset_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\sigpac_masks_0_5_Bejis'
#mask_dataset_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\sigpac_masks_Bejis'

multiple_ims_dirs = ['D:\GUD\HUELLA_INCENDIO_112\ortofoto_0_5_ICV_2022_Bejis', 
                     'D:\GUD\HUELLA_INCENDIO_112\ortofoto_0_5_ICV_2022_ValldEbo']
multiple_masks_dirs = ['D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\sigpac_masks_0_5_Bejis', 
                     'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\sigpac_masks_0_5_ValldEbo']

imagePaths = glob.glob(image_dataset_dir+'/*.tif')
maskPaths = glob.glob(mask_dataset_dir+'/*.tif')
# imagePaths = glob_multiple(multiple_ims_dirs)
# maskPaths = glob_multiple(multiple_masks_dirs)

if len(imagePaths) == len(maskPaths):
    print('número de ficheros en imágenes y máscaras coincidente')
else:
    print('ERROR número de ficheros en imágenes y máscaras NO coincidente')


num_samples_train = int(TRAIN_VAL_SPLIT*len(imagePaths))
num_samples_val = len(imagePaths) -num_samples_train

paths = list(zip(imagePaths, maskPaths))

split = int(TRAIN_VAL_SPLIT*len(imagePaths))
trainSet_im, validSet_im = random_split(imagePaths, [split, len(imagePaths)-split])


train_im_dir = os.path.join(basedir, 'train_im')
train_mask_dir = os.path.join(basedir, 'train_mask')
val_im_dir = os.path.join(basedir, 'val_im')
val_mask_dir = os.path.join(basedir, 'val_mask')

for direc in [train_im_dir, train_mask_dir, val_im_dir, val_mask_dir]:
    if not os.path.isdir(direc):
        os.makedirs(direc)

movefiles(trainSet_im.dataset, trainSet_im.indices, train_im_dir)
movefiles(maskPaths, trainSet_im.indices, train_mask_dir)
movefiles(validSet_im.dataset, validSet_im.indices, val_im_dir)
movefiles(maskPaths, validSet_im.indices, val_mask_dir)

# train_im_dir = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_ValldEbo'
# train_mask_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_masks'
# val_im_dir = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_ValldEbo'
# val_mask_dir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_masks'



train_imgs = RasterDataset(root=train_im_dir, crs='epsg:25830', res=0.25)
train_msks = RasterDataset(root=train_mask_dir, crs='epsg:25830', res=0.25)
valid_imgs = RasterDataset(root=val_im_dir, crs='epsg:25830', res=0.25)
valid_msks = RasterDataset(root=val_mask_dir, crs='epsg:25830', res=0.25)

# train_imgs = RasterDataset(root=train_im_dir, crs='epsg:25830', res=0.5)
# train_msks = RasterDataset(root=train_mask_dir, crs='epsg:25830', res=0.5)
# valid_imgs = RasterDataset(root=val_im_dir, crs='epsg:25830', res=0.5)
# valid_msks = RasterDataset(root=val_mask_dir, crs='epsg:25830', res=0.5)

# IMPORTANT
train_msks.is_image = False
valid_msks.is_image = False

train_dset = train_imgs & train_msks
valid_dset = valid_imgs & valid_msks

train_sampler = RandomGeoSampler(train_imgs, size=256, length=10000, units=Units.PIXELS)
valid_sampler = RandomGeoSampler(valid_imgs, size=256, length=10000, units=Units.PIXELS)
# train_sampler = RandomGeoSampler(train_imgs, size=256, length=10000, units=Units.PIXELS)
# valid_sampler = RandomGeoSampler(valid_imgs, size=256, length=10000, units=Units.PIXELS)


bbox = next(iter(train_sampler))
bbox

sample = train_dset[bbox]
sample.keys()

sample['image'].shape, sample['mask'].shape

"""Notice we have now patches of same size (..., 512 x 512)

## Creating Dataloaders

Creating a `DataLoader` in TorchGeo is very straightforward, 
just like it is with Pytorch (we are actually using the same class). 
Note below that we are also using the same samplers already defined. 
Additionally we inform the dataset that the dataloader will use to pull data from,
 the batch_size (number of samples in each batch) and a collate function 
 that specifies how to “concatenate” the multiple samples into one single batch.

Finally, we can iterate through the dataloader to grab batches from it. To test it, we will get the first batch.
"""
#CREATING DATALOADER

train_dataloader = DataLoader(train_dset, sampler=train_sampler, batch_size=10, collate_fn=stack_samples)
valid_dataloader = DataLoader(valid_dset, sampler=valid_sampler, batch_size=10, collate_fn=stack_samples)

train_batch = next(iter(train_dataloader))
valid_batch = next(iter(valid_dataloader))
train_batch.keys(), valid_batch.keys()

#Model definition
model = deeplabv3_resnet50(weights=None, num_classes=5)
backbone = model.get_submodule('backbone')
conv = torch.nn.modules.conv.Conv2d(
    in_channels=4,
    out_channels=64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False
)
backbone.register_module('conv1', conv)

pred = model(torch.randn(3, 4, 512, 512))
pred['out'].shape


"""## Training Loop

The training function should receive the number of epochs, 
the model, the dataloaders, the loss function (to be optimized) 
the accuracy function (to assess the results), 
the optimizer (that will adjust the parameters of the model in the correct 
direction) and the transformations to be applied to each batch.
"""

def train_loop(
    epochs: int,
    train_dl: DataLoader,
    val_dl: Optional[DataLoader],
    model: torch.nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    acc_fns: Optional[List]=None,
    batch_tfms: Optional[Callable]=None
):
    # size = len(dataloader.dataset)
    cuda_model = model.cuda()

    for epoch in range(epochs):
        accum_loss = 0
        for batch in train_dl:
            
            if batch_tfms is not None:
                batch = batch_tfms(batch)

            X = batch['image'].cuda()
            y = batch['mask'].type(torch.long).cuda()
            pred = cuda_model(X)['out']
            loss = loss_fn(pred, y)

            # BackProp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the accum loss
            accum_loss += float(loss) / len(train_dl)

        # Testing against the validation dataset
        if acc_fns is not None and val_dl is not None:
            # reset the accuracies metrics
            acc = [0.] * len(acc_fns)

            with torch.no_grad():
                for batch in val_dl:
                    if batch_tfms is not None:
                        batch = batch_tfms(batch)

                    X = batch['image'].type(torch.float32).cuda()
                    y = batch['mask'].type(torch.long).cuda()

                    pred = cuda_model(X)['out']

                    for i, acc_fn in enumerate(acc_fns):
                        acc[i] = float(acc[i] + acc_fn(pred, y)/len(val_dl))

            # at the end of the epoch, print the errors, etc.
            print(f'Epoch {epoch}: Train Loss={accum_loss:.5f} - Accs={[round(a, 3) for a in acc]}')
        else:
            print(f'Epoch {epoch}: Train Loss={accum_loss:.5f}')

"""## Loss and Accuracy Functions

For the loss function, normally the Cross Entropy Loss should work, but it requires the mask to have shape (N, d1, d2). In this case, we will need to squeeze our second dimension manually.
"""


def oa(pred, y):
    flat_y = y.squeeze()
    flat_pred = pred.argmax(dim=1)
    acc = torch.count_nonzero(flat_y == flat_pred) / torch.numel(flat_y)
    return acc

def iou(pred, y):
    flat_y = y.cpu().numpy().squeeze()
    flat_pred = pred.argmax(dim=1).detach().cpu().numpy()
    return jaccard_score(flat_y.reshape(-1), flat_pred.reshape(-1), average='weighted', zero_division=1.)


def loss(p, t):
    return torch.nn.functional.cross_entropy(p, t.squeeze())
#nn.CrossEntropyLoss()

"""## Training

> To train the model it is important to have CUDA GPUs available. In Colab, it can be done by changing the runtime type and re-running the notebook.
"""

#tfms = train_imgs


#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.01)


def arregla_masks():
    for dirmasks in [train_mask_dir, val_mask_dir]:
        maskfiles = glob.glob('D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\sigpac_masks'+'/*.tif')
        for maskfile in maskfiles:
            outdir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\sigpac_masks_recoded'
            maskfilename = os.path.basename(maskfile)
            mask = rxr.open_rasterio(maskfile)
            unique, counts = np.unique(mask, return_counts=True)
            print(f'file {maskfile}: unique={unique}')
            mask2 = xr.where(mask < 0, 4, mask)
            mask.close()
            mask2 = mask2.rio.write_crs(mask.rio.crs.to_epsg())
            mask2.rio.to_raster(os.path.join(outdir, maskfilename))
    pass

lrates = [0.001, 0.0001]
#l=0.0001
nepochs = 30
for l in lrates:
    tic = timeit.default_timer()
    optimizer = torch.optim.Adam(model.parameters(), lr=l, weight_decay=0.01)
    train_loop(nepochs, train_dataloader, valid_dataloader, model, loss, optimizer,
           acc_fns=[oa, iou])#, batch_tfms=tfms)
    toc = timeit.default_timer()
    print('tiempo de lr: ' + str(l) + ': ' + str(toc - tic))

model_path = r"D:\GUD\HUELLA_INCENDIO_112\Data_Classif\model_valldebo_samplessigpac_selected.pth"
model_path = r"D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\checkpoints\model_bejis_0_5_sigpac_completo.pth"
model_path = r"D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\checkpoints\model_bejisvalldebo_0_5_sigpac_completo.pth"
model_path = r"D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\checkpoints\model_bejis_sigpac_completo.pth"

torch.save(model.state_dict(), model_path)



#INFERENCE

model2 = deeplabv3_resnet50(weights=None, num_classes=5)
backbone = model2.get_submodule('backbone')
conv = torch.nn.modules.conv.Conv2d(
    in_channels=4,
    out_channels=64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False
)
backbone.register_module('conv1', conv)



model2.load_state_dict(torch.load(model_path))
model2.eval()


def custom_merge(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
    old_data[:] = np.average(old_data, new_data)  # <== NOTE old_data[:] updates the old data array *in place*


def custom_method_avg(merged_data, new_data, merged_mask, new_mask, **kwargs):
    """Returns the average value pixel."""
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    np.nanmean([merged_data, new_data], axis=0, out=merged_data, where=mask)
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def make_predictions(
    i: str,
    predict_im: str,
    model: torch.nn.Module,
    tempdir:str,
    outdir: str,
    batch_tfms: Optional[Callable]=None
    ):
    #predict_im,model,tempdir,outdir,batch_tfms=predict_im,model2,tempdir,outputdir,None
    time0 = timeit.default_timer()
    tempdir = tempdir + '{:0>4}'.format(i)

    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)
    os.makedirs(tempdir)
    tempdirbatches = os.path.join(tempdir, 'batches')
    #tempdirbatchescoded = os.path.join(tempdir, 'batches_coded')

    #if not os.path.exists(tempdirbatches):
    os.makedirs(tempdirbatches)
    # else:
    #     shutil.rmtree(tempdir)
    # if not os.path.exists(tempdirbatchescoded):
    #     os.makedirs(tempdirbatchescoded)
    
    predict_im_name = os.path.basename(predict_im)
    temp_im = os.path.join(tempdir, predict_im_name)
    shutil.copy(predict_im, temp_im)
    
    #ref_im = rxr.open_rasterio(predict_im)
    with xr.open_dataset(temp_im) as ref_im:
        crs_ref = ref_im.rio.crs
        res_ref_x = ref_im.rio.resolution()[0]
        res_ref_y = ref_im.rio.resolution()[1]
    #del ref_im
    
    img = RasterDataset(root=tempdir, crs=crs_ref, res=res_ref_x)
    sampler = GridGeoSampler(img, size=256, stride=256, units=Units.PIXELS)
    dataloader = DataLoader(img, sampler=sampler, collate_fn=stack_samples)
    
    time1 = timeit.default_timer()
    cuda_model = model.cuda()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            #In case of augmentations
            if batch_tfms is not None:
                batch = batch_tfms(batch)

            image = batch['image'].cuda()
            pred = cuda_model(image)['out']
            pred = torch.softmax(pred, dim=1)
            #predcoded = torch.argmax(pred, dim = 1)
            mask = pred.squeeze(0).cpu().detach().numpy()
            #mask_coded = predcoded.squeeze(0).cpu().detach().numpy()
            mask_outpath = os.path.join(tempdirbatches, '{:0>4}'.format(i)+'.tif')
            #mask_coded_outpath = os.path.join(tempdirbatchescoded, '{:0>4}'.format(i)+'.tif')

            xs = np.arange(batch['bbox'][0][0], batch['bbox'][0][1], res_ref_x)
            ys = np.arange(batch['bbox'][0][2], batch['bbox'][0][3], -res_ref_y)
            ys = ys[::-1]
            mask = np.round(mask*100)
            mask = mask.astype(np.int8)
            
            
            #mask_coded = mask_coded.astype(np.int8)

            data_xr = xr.DataArray(mask, coords={'y': ys,'x': xs}, dims=['pred', "y", "x" ])
            data_xr = data_xr.rio.set_crs(crs_ref, inplace=True)
            data_xr.rio.to_raster(mask_outpath)
            data_xr.close()
            del data_xr
            #data_xr_coded = xr.DataArray(mask_coded, coords={'y': ys,'x': xs}, dims=["y", "x" ])
            #data_xr_coded = data_xr_coded.rio.set_crs(crs_ref, inplace=True)
            #data_xr_coded.rio.to_raster(mask_coded_outpath)
    #del img, sampler, dataloader
    print(f'image: {predict_im} time: {timeit.default_timer()-time1}')

    tilefiles = glob.glob(tempdirbatches+'/*.tif')
    #tilefiles_coded = glob.glob(tempdirbatchescoded+'/*.tif')

    src = rasterio.open(tilefiles[0])
    mosaicpath = os.path.join(outdir, predict_im_name)
   #mosaiccodedpath = os.path.join(outdir, os.path.basename(predict_im_name)+'_class.tif')

   # mosaic, out_trans = merge(tilefiles, method='max')
    mosaic, out_trans = merge(tilefiles, method=custom_method_avg)

    # mosaic = torch.softmax(mosaic, dim=1)
    # mosaic2 = torch.softmax(torch.from_numpy(mosaic).float(), dim=1)

    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": src.crs
                     }
                    )
    with rasterio.open(mosaicpath, "w", **out_meta) as dest:
        dest.write(mosaic)
        
    src.close()
    # mosaic_coded, out_trans_coded = merge(tilefiles_coded, method='last')
    # src_coded = rasterio.open(tilefiles_coded[0])
    # out_meta_coded = src_coded.meta.copy()
    # out_meta_coded.update({"driver": "GTiff",
    #                  "height": mosaic_coded.shape[1],
    #                  "width": mosaic_coded.shape[2],
    #                  "transform": out_trans_coded,
    #                  "crs": src.crs
    #                  }
    #                 )

    # with rasterio.open(mosaiccodedpath, "w", **out_meta_coded) as dest2:
    #     dest2.write(mosaic_coded)
    
    print(f'image: {predict_im} time total: {timeit.default_timer()-time0}')
    #Clean temp files/dirs
    #shutil.rmtree(tempdir)

    return(mosaicpath, tempdir)
        
    
#predict_im_dir = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_Bejis'
#predict_im_dir = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_ValldEbo'
#predict_im_dir = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_Bejis'
predict_im_dir = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_Costur'
predict_im_dir = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_0_5_ICV_2022_Costur'
predict_ims = glob.glob(predict_im_dir+'/*.tif')
tempdir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\temp'
# if not os.path.exists(tempdir):
#     os.makedirs(tempdir)
#outputdir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_Bejis_extended'
#outputdir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_Bejis_modeloBejis'
outputdir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_Costur_modelobejisvalldebo_0_5'
#outputdir = r'D:\GUD\HUELLA_INCENDIO_112\Data_Classif\sigpac_con_samples\output_ValldEbo_modelovalldebo'


if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    
#predict_im = r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_Bejis\020201_2022CVAL0025_25830_8bits_RGBI_0639_4-5.tif'
#predict_im=r'D:\GUD\HUELLA_INCENDIO_112\ortofoto_ICV_2022_Bejis\020201_2022CVAL0025_25830_8bits_RGBI_0639_5-3.tif'
for i, predict_im in enumerate(predict_ims):
    pred = make_predictions(
        i,
        predict_im,
        model2,
        tempdir,
        outputdir
       )


