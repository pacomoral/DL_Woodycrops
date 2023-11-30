# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:34:21 2023

@author: franc
"""

#!pip install -q torchgeo
# !pip install -q git+https://github.com/microsoft/torchgeo.git
# !pip install -q GPUtil


import torch
from GPUtil import showUtilization as gpu_usage

torch.cuda.empty_cache()
gpu_usage()


import lightning.pytorch as pl # Instead of import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples, RasterDataset
from torchgeo.datasets.splits import random_bbox_assignment
from torchgeo.samplers import RandomGeoSampler, RandomBatchGeoSampler, GridGeoSampler
import os
import matplotlib.pyplot as plt
import numpy as np
from torchgeo.trainers import SemanticSegmentationTask
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import ssl
import multiprocessing as mp
from torchgeo.datamodules import GeoDataModule
from typing import Type
import albumentations as A
import timeit
import torch
import numpy as np
from rasterio.plot import show
from rasterio.merge import merge
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.crs import CRS
from rasterio.io import MemoryFile

#import shutil
#shutil.rmtree("/kaggle/working/")


# Defining class config
"""
Complete 13-class dataset.

    This version of the dataset is composed of 13 classes:

    0. No Data: Background values
    1. Water: All areas of open water including ponds, rivers, and lakes
    2. Wetlands: Low vegetation areas located along marine or estuarine regions
    3. Tree Canopy: Deciduous and evergreen woody vegetation over 3-5 meters in height
    4. Shrubland: Heterogeneous woody vegetation including shrubs and young trees
    5. Low Vegetation: Plant material less than 2 meters in height including lawns
    6. Barren: Areas devoid of vegetation consisting of natural earthen material
    7. Structures: Human-constructed objects made of impervious materials
    8. Impervious Surfaces: Human-constructed surfaces less than 2 meters in height
    9. Impervious Roads: Impervious surfaces that are used for transportation
    10. Tree Canopy over Structures: Tree cover overlapping impervious structures
    11. Tree Canopy over Impervious Surfaces: Tree cover overlapping impervious surfaces
    12. Tree Canopy over Impervious Roads: Tree cover overlapping impervious roads
    13. Aberdeen Proving Ground: U.S. Army facility with no labels
"""

names=[
    'No Data',
    'Water',
    'Wetlands',
    'Tree Canopy',
    'Shrubland',
    'Low Vegetation',
    'Barren',
    'Structures',
    'Impervious Surfaces',
    'Impervious Roads',
    'Tree Canopy over Structures',
    'Tree Canopy over Impervious Surfaces', 
    'Tree Canopy over Impervious Roads',
    'Aberdeen Proving Ground',
]

# subclasses use the 13 class cmap by default
cmap = [
    (0, 0, 0),
    (0, 197, 255),
    (0, 168, 132),
    (38, 115, 0),
    (76, 230, 0),
    (163, 255, 115),
    (255, 170, 0),
    (255, 0, 0),
    (156, 156, 156),
    (0, 0, 0),
    (115, 115, 0),
    (230, 230, 0),
    (255, 255, 115),
    (197, 0, 255),
]


# Defining some parameters.

OUTPUT_DIR = '/kaggle/working/'
INPUT_DIR = '/kaggle/input/naip-chesapeake-sample'
EPOCHS = 15
LR = 1e-4

IN_CHANNELS = 4 # NAIP dataset has 4 bands
NUM_CLASSES = len(names) # Chesapeake dataset has 13 classes
IMG_SIZE = 256
BATCH_SIZE = 8
SAMPLE_SIZE = 500

PATIENCE = 5
SEGMENTATION_MODEL = 'deeplabv3+' # only supports 'unet', 'deeplabv3+' and 'fcn'
#BACKBONE = 'se_resnet50'
BACKBONE = 'resnet50' # supports TIMM encoders (https://smp.readthedocs.io/en/latest/encoders_timm.html)
WEIGHTS = 'imagenet'
LOSS = 'focal' # supports ‘ce’, ‘jaccard’ or ‘focal’ loss


DEVICE, NUM_DEVICES = ("cuda", torch.cuda.device_count()) if torch.cuda.is_available() else ("cpu", mp.cpu_count())
WORKERS = mp.cpu_count()
print(f'Running on {NUM_DEVICES} {DEVICE}(s)')

# Defining our task, logger and checkpoint callbacks
ssl._create_default_https_context = ssl._create_unverified_context

TEST_DIR = os.path.join(OUTPUT_DIR, "test")
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)
    
logger = CSVLogger(
    TEST_DIR, 
    name='torchgeo_logs'
)

checkpoint_callback = ModelCheckpoint(
    every_n_epochs=1,
    dirpath=TEST_DIR,
    filename='torchgeo_trained'
)

task = SemanticSegmentationTask(
    model = SEGMENTATION_MODEL,
    backbone = BACKBONE,
    weights = True, # to use imagenet. Before we should define weights='imagenet'
    in_channels = IN_CHANNELS,
    num_classes = NUM_CLASSES,
    loss = LOSS,
    ignore_index = None,
    learning_rate = LR,
    learning_rate_schedule_patience = PATIENCE, 
)

# Defining our trainer
trainer = pl.Trainer(
        accelerator=DEVICE,
        devices=NUM_DEVICES,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, ],
        logger=logger,
    )

# Subclassing RasterDataset to create our datasets. One for the NAIP images and one for the Chesapeake labels.

class NAIPImages(RasterDataset):
    filename_glob = "m_*.tif"
    is_image = True
    separate_files = False
    
class ChesapeakeLabels(RasterDataset):
    filename_glob = "m_*.tif"
    is_image = False
    separate_files = False
    

# Testing if our datasets are working properly.
data_augmentation_transform = A.Compose([
    A.Flip(),
    A.ShiftScaleRotate(),
    A.OneOf([
        A.RandomBrightness(),
        A.RandomGamma(),
    ]),
    A.CoarseDropout(max_height=32, max_width=32, max_holes=5)
])

naip_root = os.path.join(INPUT_DIR, 'naip_images')
naip_images = NAIPImages(
    root=naip_root,
    #transforms=data_augmentation_transform,
)
print(naip_images)

chesapeake_root = os.path.join(INPUT_DIR, "chesapeake_labels")
chesapeake_labels = ChesapeakeLabels(
    root=chesapeake_root,
    #transforms=data_augmentation_transform,
)
print(chesapeake_labels)

# Creating an intersection dataset and a CustomGeoDataModule
dataset = naip_images & chesapeake_labels # this means I'm creating an IntersectionDataset
sampler = RandomGeoSampler(dataset, size=IMG_SIZE, length=SAMPLE_SIZE)
dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

class CustomGeoDataModule(GeoDataModule):
    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = self.dataset_class(**self.kwargs)
        
        generator = torch.Generator().manual_seed(0)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_bbox_assignment(dataset, [0.6, 0.2, 0.2], generator)
        
        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )
            
datamodule = CustomGeoDataModule(
    dataset_class = type(dataset), # GeoDataModule kwargs
    batch_size = BATCH_SIZE, # GeoDataModule kwargs
    patch_size = IMG_SIZE, # GeoDataModule kwargs
    length = SAMPLE_SIZE, # GeoDataModule kwargs
    num_workers = WORKERS, # GeoDataModule kwargs
    dataset1 = naip_images, # IntersectionDataset kwargs
    dataset2 = chesapeake_labels, # IntersectionDataset kwargs
    collate_fn = stack_samples, # IntersectionDataset kwargs
)

# Defining a colormap to plot our predictions and creating a function to apply this colormap to our predicted tensors
# Perform colour coding on the outputs
def colour_code_segmentation(image, colors):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        colormap: the list os rgb colors for each class

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(colors)
    return colour_codes[image]


# Helper function for data visualization
import torchvision.transforms as transforms
reverse_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=IMG_SIZE),
        ])

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
n = 0
for sample in dataloader:
    if n == 10:
        break
    image, gt_mask = sample['image'], sample['mask']
    
    gt_mask = colour_code_segmentation(gt_mask, cmap)
        
    visualize(
        image=reverse_transform(image.squeeze()[:3]),
        ground_truth = gt_mask.squeeze(),
    )
    n += 1
    
# Running our training based on a previously saved checkpoint if available, otherwise we start from scratch. This is also useful to load our model from a checkpoint.
start = timeit.default_timer() # Measuring the time

checkpoint_file = os.path.join(TEST_DIR, 'torchgeo_trained.ckpt') 

if os.path.isfile(checkpoint_file):
    print('Resuming training from previous checkpoint...')
    trainer.fit(
        model=task, 
        datamodule=datamodule,
        ckpt_path=checkpoint_file
    )
else:
    print('Starting training from scratch...')
    trainer.fit(
        model=task,
        datamodule = datamodule,
    )
    
print("The time taken to train was: ", timeit.default_timer() - start)

# Testing our model
trainer.test(
    model=task,
    datamodule=datamodule
)

# Creating a function to georreference in-memory the predicted chips passed through our model
def create_in_memory_geochip(predicted_chip, geotransform, crs, color_coded=False):
    """
    Apply georeferencing to the predicted chip.
    
    Parameters:
        predicted_chip (numpy array): The predicted segmentation chip (e.g., binary mask).
        geotransform (tuple): A tuple containing the geotransformation information of the chip (x-coordinate of the top left corner, x and y pixel size, rotation, y-coordinate of the top left corner, and rotation).
        crs (str): Spatial Reference System (e.g., EPSG code) of the chip.
        color_coded (bool): If the chip is color coded, it has a shape [H, W, C], othersize has a shape [1, H, W]

    Return:
        A rasterio dataset that is georreferenced.
    """
    if color_coded:
        predicted_chip = np.rollaxis(predicted_chip, axis=2) # putting the bands first
        photometric = 'RGB'
    else:
        photometric = 'MINISBLACK'
        
    memfile = MemoryFile()
    dataset = memfile.open(
        driver='GTiff',
        height=predicted_chip.shape[1],
        width=predicted_chip.shape[2],
        count=predicted_chip.shape[0],  # Number of bands
        dtype=np.uint8,
        crs=crs,
        transform=geotransform,
        photometric=photometric,
    )
    
    dataset.write(predicted_chip)
    return dataset

# Creating a in-memory georreferenced chip generator.
def georreferenced_chip_generator(dataloader, model, crs, pixel_size, colors, color_coded=False):
    """
    Apply georeferencing to the predicted chip.
    
    Parameters:
        dataloader (torch.utils.data.Dataloader): Dataloader with the data to be predicted.
        model (an https://github.com/qubvel/segmentation_models.pytorch model): model used for inference.
        crs (str): Spatial Reference System (e.g., EPSG code) of the chip.
        pixel_size (float): Pixel dimensoion in map units.

    Returns:
        A list of georeferenced numpy arrays of the predicted outputs.
    """
    georref_chips_list = []
    for i, sample in enumerate(dataloader):
        image, gt_mask, bbox = sample['image'], sample['mask'], sample['bbox'][0]

        image = image/255. # as I'm not using a GeoDatamodule, I need to divide de images by 255 manually

        prediction = model.predict(image)
        prediction = torch.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, dim = 1)

        geotransform = from_origin(bbox.minx, bbox.maxy, pixel_size, pixel_size)
        
        if color_coded:
            # applying the original colors
            prediction = colour_code_segmentation(prediction, colors)
            georref_chips_list.append(create_in_memory_geochip(prediction.squeeze(), geotransform, crs, color_coded))
        else:
            georref_chips_list.append(create_in_memory_geochip(prediction, geotransform, crs, color_coded))
    return georref_chips_list


# Function to merge the georreferenced predicted chips
def merge_georeferenced_chips(chips_list, output_path):
    """
    Merge a list of georeferenced chips into a single GeoTIFF file.

    Parameters:
        chips_generator (generator): A generator of Rasterio datasets representing the georeferenced chips.
        output_path (str): The path where the merged GeoTIFF file will be saved.

    Returns:
        None
    """
    # Merge the chips using Rasterio's merge function
    merged, merged_transform = merge(chips_list)
    
    # Calculate the number of rows and columns for the merged output
    rows, cols = merged.shape[1], merged.shape[2]

    # Update the metadata of the merged dataset
    merged_metadata = chips_list[0].meta
    merged_metadata.update({
        'height': rows,
        'width': cols,
        'transform': merged_transform
    })

    # Write the merged array to a new GeoTIFF file
    with rasterio.open(output_path, 'w', **merged_metadata) as dst:
        dst.write(merged)
        
    for chip in chips_list:
        chip.close()
        
# Making predictions, applying the correct georrefence to them and merging them in one file that is saved to disk.
test_dataset = datamodule.test_dataset
test_sampler = GridGeoSampler(test_dataset, 2048, 2048)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, collate_fn=stack_samples)

pixel_size = test_dataset.res
crs = test_dataset.crs.to_epsg()

start = timeit.default_timer() # Measuring the time
chips_generator = georreferenced_chip_generator(test_dataloader, task.model, crs, pixel_size, cmap, color_coded=False)
print("The time taken to predict was: ", timeit.default_timer() - start)

start = timeit.default_timer() # Measuring the time
file_name = os.path.join(OUTPUT_DIR, 'merged_prediction.tif')
merge_georeferenced_chips(chips_generator, file_name)
print("The time taken to generate a georrefenced image and save it was: ", timeit.default_timer() - start)


# Visualizing the merged georreferenced prediction
output_filepath = os.path.join(OUTPUT_DIR, 'merged_prediction.tif')
src = rasterio.open(output_filepath)
show(src.read(), transform=src.transform)


