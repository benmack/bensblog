+++
title = "Comparing USGS ShakeMap versions"
description = "The M 7.1 - 72 km ENE of Namie, Japan earthquake."
date = "2021-02-27"
categories = ["Python"]
[[images]]
  src = "img/blog/2021-02-27-1_comparing_usgs_shakemap_versions/output_26_0.png"
  alt = "output_26_0.png"
  stretch = "v"
+++

# Overview

Recently, I came across the question if and how it is possible to download superseded versions of [ShakeMaps](https://earthquake.usgs.gov/data/shakemap/) distributed by the [USGS Earthquake Hazards](https://www.usgs.gov/natural-hazards/earthquake-hazards) program.
This post comprises what I encountered on my first walk through the rich resources offered by the USGS.

In a nutshell, we will
* use the [libcomcat](https://github.com/usgs/libcomcat) Python library to 
    * query the history of product submissions (e.g. ShakeMaps) related to an event
    * download different versions of a product (here the raster.zip)
* reprocess rasters to have common extend and pixel alignment
* visualize how the [Modified Mercalli Intensity](https://www.usgs.gov/natural-hazards/earthquake-hazards/science/modified-mercalli-intensity-scale?qt-science_center_objects=0#qt-science_center_objects) (MMI) changes over different product versions for a given event. 

We will walk through these steps using the [*M 7.1 - 72 km ENE of Namie, Japan*](https://earthquake.usgs.gov/earthquakes/eventpage/us6000dher/executive) event.

# Data and Tools

The [USGS Earthquake Hazards Program > Data and Tools](https://www.usgs.gov/natural-hazards/earthquake-hazards/data-tools) webpage provides an overview of available data and different ways to access them.
The [Real-time Notifications, Feeds, and Web Services](https://earthquake.usgs.gov/earthquakes/feed/) webpage lists different ways to access real-time notifications as well as some links for developers.

We can find a link to the web service [API Documentation - Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/) there.
But we can also click our way through the [Developers Corner](https://github.com/usgs/devcorner) to the [Python libcomcat](https://github.com/usgs/libcomcat) project that provides *a Python equivalent to the ANSS ComCat search API*, i.e. the [API Documentation - Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/).

In this post, we will use the [libcomcat](https://github.com/usgs/libcomcat) Python API.
The [Python libcomcat](https://github.com/usgs/libcomcat) *README.md* contains links to the API Documentation, the Command Line Interface Documentation, and several great Jupyter notebooks that make it easy to get started with the API.
Most of the code snippets in this post are taken from there.

# Libraries


```python
from datetime import timezone
from libcomcat.dataframes import get_history_data_frame, split_history_frame
from libcomcat.search import get_event_by_id
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import zipfile

pd.set_option('display.max_columns', None)
```

# Event Details

Assume we already know the event ID of an earthquake of interest.
Then we can get a `DetailEvent` instance for it.


```python
quake_id = 'us6000dher'

event = get_event_by_id(quake_id, includesuperseded=True)
event
```




    us6000dher 2021-02-13 14:07:50.397000 (37.745,141.749) 49.9 km M7.1



With this object, we can derive the history of the event's products in a pandas data frame with the `get_history_data_frame` function.
We need to create the object with `includesuperseded=True` to get all the versions of the event's product ([History Dataframe](https://github.com/usgs/libcomcat/blob/master/docs/api.md#history-dataframe)).


```python
df_products, event = get_history_data_frame(event)
display(df_products.head(3))
df_products['Product'].value_counts()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Update Time</th>
      <th>Product</th>
      <th>Authoritative Event ID</th>
      <th>Code</th>
      <th>Associated</th>
      <th>Product Source</th>
      <th>Product Version</th>
      <th>Elapsed (min)</th>
      <th>URL</th>
      <th>Comment</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>313</th>
      <td>2021-02-13 14:19:04.892</td>
      <td>origin</td>
      <td>us6000dher</td>
      <td>pt21044000</td>
      <td>True</td>
      <td>pt</td>
      <td>1</td>
      <td>11.2</td>
      <td>https://earthquake.usgs.gov/archive/product/or...</td>
      <td></td>
      <td>Magnitude# 7.1|Time# 2021-02-13 14:07:00 |Time...</td>
    </tr>
    <tr>
      <th>311</th>
      <td>2021-02-13 14:20:10.768</td>
      <td>origin</td>
      <td>us6000dher</td>
      <td>at00qoh0l1</td>
      <td>True</td>
      <td>at</td>
      <td>1</td>
      <td>12.3</td>
      <td>https://earthquake.usgs.gov/archive/product/or...</td>
      <td></td>
      <td>Magnitude# 7.1|Time# 2021-02-13 14:07:52 |Time...</td>
    </tr>
    <tr>
      <th>312</th>
      <td>2021-02-13 14:20:10.774</td>
      <td>origin</td>
      <td>us6000dher</td>
      <td>at00qoh0l1</td>
      <td>True</td>
      <td>at</td>
      <td>2</td>
      <td>12.3</td>
      <td>https://earthquake.usgs.gov/archive/product/or...</td>
      <td></td>
      <td>Magnitude# 7.1|Time# 2021-02-13 14:07:52 |Time...</td>
    </tr>
  </tbody>
</table>
</div>





    dyfi              285
    moment-tensor      10
    ground-failure      8
    losspager           7
    shakemap            7
    origin              7
    phase-data          4
    finite-fault        1
    Name: Product, dtype: int64



The value counts on the `Products` column show that there are seven ShakeMap versions.
We can subset the dataset by hand or a dedicated helper function.
The helper function will subset the data, reset the row index and change the columns of the data frame.


```python
df_shakemap = split_history_frame(df_products, product='shakemap')
df_shakemap
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Update Time</th>
      <th>Product</th>
      <th>Authoritative Event ID</th>
      <th>Code</th>
      <th>Associated</th>
      <th>Product Source</th>
      <th>Product Version</th>
      <th>Elapsed (min)</th>
      <th>URL</th>
      <th>Comment</th>
      <th>MaxMMI</th>
      <th>Instrumented</th>
      <th>DYFI</th>
      <th>Fault</th>
      <th>GMPE</th>
      <th>Mag</th>
      <th>Depth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-02-13 14:28:21.006</td>
      <td>shakemap</td>
      <td>us6000dher</td>
      <td>us6000dher</td>
      <td>True</td>
      <td>us</td>
      <td>1</td>
      <td>20.5</td>
      <td>https://earthquake.usgs.gov/archive/product/sh...</td>
      <td></td>
      <td>5.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>[AtkinsonBoore2003SInter],[ZhaoEtAl2016SInter]...</td>
      <td>7.0</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-02-13 15:10:42.406</td>
      <td>shakemap</td>
      <td>us6000dher</td>
      <td>us6000dher</td>
      <td>True</td>
      <td>us</td>
      <td>2</td>
      <td>62.9</td>
      <td>https://earthquake.usgs.gov/archive/product/sh...</td>
      <td></td>
      <td>6.4</td>
      <td>4.0</td>
      <td>23.0</td>
      <td>NaN</td>
      <td>[AtkinsonBoore2003SInter],[ZhaoEtAl2016SInter]...</td>
      <td>7.0</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-02-13 16:11:24.776</td>
      <td>shakemap</td>
      <td>us6000dher</td>
      <td>us6000dher</td>
      <td>True</td>
      <td>us</td>
      <td>3</td>
      <td>123.6</td>
      <td>https://earthquake.usgs.gov/archive/product/sh...</td>
      <td></td>
      <td>6.5</td>
      <td>4.0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>[AtkinsonBoore2003SInter],[ZhaoEtAl2016SInter]...</td>
      <td>7.1</td>
      <td>50.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-02-13 22:11:55.217</td>
      <td>shakemap</td>
      <td>us6000dher</td>
      <td>us6000dher</td>
      <td>True</td>
      <td>us</td>
      <td>4</td>
      <td>484.1</td>
      <td>https://earthquake.usgs.gov/archive/product/sh...</td>
      <td></td>
      <td>7.0</td>
      <td>4.0</td>
      <td>43.0</td>
      <td>NaN</td>
      <td>[AbrahamsonEtAl2014],[BooreEtAl2014],[Campbell...</td>
      <td>7.1</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-02-14 00:17:56.940</td>
      <td>shakemap</td>
      <td>us6000dher</td>
      <td>us6000dher</td>
      <td>True</td>
      <td>us</td>
      <td>5</td>
      <td>610.1</td>
      <td>https://earthquake.usgs.gov/archive/product/sh...</td>
      <td></td>
      <td>7.6</td>
      <td>589.0</td>
      <td>43.0</td>
      <td>NaN</td>
      <td>[AbrahamsonEtAl2014],[BooreEtAl2014],[Campbell...</td>
      <td>7.1</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021-02-14 14:12:31.505</td>
      <td>shakemap</td>
      <td>us6000dher</td>
      <td>us6000dher</td>
      <td>True</td>
      <td>us</td>
      <td>6</td>
      <td>1444.7</td>
      <td>https://earthquake.usgs.gov/archive/product/sh...</td>
      <td></td>
      <td>8.0</td>
      <td>586.0</td>
      <td>49.0</td>
      <td>NaN</td>
      <td>[AbrahamsonEtAl2014],[BooreEtAl2014],[Campbell...</td>
      <td>7.1</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2021-02-14 22:08:57.989</td>
      <td>shakemap</td>
      <td>us6000dher</td>
      <td>us6000dher</td>
      <td>True</td>
      <td>us</td>
      <td>7</td>
      <td>1921.1</td>
      <td>https://earthquake.usgs.gov/archive/product/sh...</td>
      <td></td>
      <td>8.0</td>
      <td>586.0</td>
      <td>50.0</td>
      <td>NaN</td>
      <td>[AbrahamsonEtAl2014],[BooreEtAl2014],[Campbell...</td>
      <td>7.1</td>
      <td>49.9</td>
    </tr>
  </tbody>
</table>
</div>



# ShakeMap Product

## Inspect contents

The ShakeMap product comprises several contents or files.
The `Product` class instance has methods that make it easy to list and download specific contents or files.
Let us look at the `Product` instance of the *preferred* version of the ShakeMap product.


```python
products = event.getProducts('shakemap', version='preferred')
print(products) # list of products but we only expect one due to version='preferred'

product = products[0]
print(f'Contents: {product.contents}')

```

    [Product shakemap from us updated 2021-02-13 15:10:42.406000 containing 58 content files.]
    Contents: ['contents.xml', 'download/attenuation_curves.json', 'download/cont_mi.json', 'download/cont_mmi.json', 'download/cont_pga.json', 'download/cont_pgv.json', 'download/cont_psa0p3.json', 'download/cont_psa1p0.json', 'download/cont_psa3p0.json', 'download/coverage_mmi_high_res.covjson', 'download/coverage_mmi_low_res.covjson', 'download/coverage_mmi_medium_res.covjson', 'download/coverage_pga_high_res.covjson', 'download/coverage_pga_low_res.covjson', 'download/coverage_pga_medium_res.covjson', 'download/coverage_pgv_high_res.covjson', 'download/coverage_pgv_low_res.covjson', 'download/coverage_pgv_medium_res.covjson', 'download/coverage_psa0p3_high_res.covjson', 'download/coverage_psa0p3_low_res.covjson', 'download/coverage_psa0p3_medium_res.covjson', 'download/coverage_psa1p0_high_res.covjson', 'download/coverage_psa1p0_low_res.covjson', 'download/coverage_psa1p0_medium_res.covjson', 'download/coverage_psa3p0_high_res.covjson', 'download/coverage_psa3p0_low_res.covjson', 'download/coverage_psa3p0_medium_res.covjson', 'download/grid.xml', 'download/info.json', 'download/intensity.jpg', 'download/intensity.pdf', 'download/intensity_overlay.png', 'download/intensity_overlay.pngw', 'download/mmi_legend.png', 'download/mmi_regr.png', 'download/pga.jpg', 'download/pga.pdf', 'download/pga_regr.png', 'download/pgv.jpg', 'download/pgv.pdf', 'download/pgv_regr.png', 'download/pin-thumbnail.png', 'download/psa0p3.jpg', 'download/psa0p3.pdf', 'download/psa0p3_regr.png', 'download/psa1p0.jpg', 'download/psa1p0.pdf', 'download/psa1p0_regr.png', 'download/psa3p0.jpg', 'download/psa3p0.pdf', 'download/psa3p0_regr.png', 'download/raster.zip', 'download/rupture.json', 'download/shake_result.hdf', 'download/shakemap.kmz', 'download/shape.zip', 'download/stationlist.json', 'download/uncertainty.xml']


This list shows 58 files associated with a specific version of the ShakeMap product.
These can also be found and downloaded under the respective [event page](https://earthquake.usgs.gov/earthquakes/eventpage/us6000dher/executive).

## Download content

Now, we will download the *raster.zip*.
Since we will download different versions, we build a destination directory containing the product version.
Then we extract the zip file.


```python
path_zip = Path(f'data/{quake_id}_{product.source}_{product.version:02d}_raster.zip')
path_zip.parent.mkdir(parents=True, exist_ok=True)
print('Zip file local:', path_zip)
print('Source URL:')
print(product.getContent('raster.zip', path_zip))

with zipfile.ZipFile(path_zip, 'r') as zip_ref:
    zip_ref.extractall(path_zip.parent / path_zip.stem)
```

    Zip file local: data/us6000dher_us_02_raster.zip
    Source URL:
    https://earthquake.usgs.gov/archive/product/shakemap/us6000dher/us/1613229042406/download/raster.zip


Now we can do the same thing for all versions.

We could also do this via the CLI with

    getproduct shakemap raster.zip -i us6000dher --get-version all


```python
def download_and_extract_raster(product, quake_id, basedir, overwrite=False):
    
    path_zip = Path(f'{str(basedir)}/{quake_id}_{product.source}_{product.version:02d}_raster.zip')
    path_zip.parent.mkdir(parents=True, exist_ok=True)
    
    print('Zip file local:', path_zip)
    if not path_zip.exists() or overwrite:
        print('Source URL:', 
              product.getContent('raster.zip', path_zip))
    
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(path_zip.parent / path_zip.stem)

for product in event.getProducts('shakemap', version='all'):
    print(f'****** {product.version} ******')
    download_and_extract_raster(product, quake_id, basedir='data', overwrite=False)
```

    ****** 1 ******
    Zip file local: data/us6000dher_us_01_raster.zip
    Source URL: https://earthquake.usgs.gov/archive/product/shakemap/us6000dher/us/1613226501006/download/raster.zip
    ****** 2 ******
    Zip file local: data/us6000dher_us_02_raster.zip
    ****** 3 ******
    Zip file local: data/us6000dher_us_03_raster.zip
    Source URL: https://earthquake.usgs.gov/archive/product/shakemap/us6000dher/us/1613232684776/download/raster.zip
    ****** 4 ******
    Zip file local: data/us6000dher_us_04_raster.zip
    Source URL: https://earthquake.usgs.gov/archive/product/shakemap/us6000dher/us/1613254315217/download/raster.zip
    ****** 5 ******
    Zip file local: data/us6000dher_us_05_raster.zip
    Source URL: https://earthquake.usgs.gov/archive/product/shakemap/us6000dher/us/1613261876940/download/raster.zip
    ****** 6 ******
    Zip file local: data/us6000dher_us_06_raster.zip
    Source URL: https://earthquake.usgs.gov/archive/product/shakemap/us6000dher/us/1613311951505/download/raster.zip
    ****** 7 ******
    Zip file local: data/us6000dher_us_07_raster.zip
    Source URL: https://earthquake.usgs.gov/archive/product/shakemap/us6000dher/us/1613340537989/download/raster.zip


With the `update_time` we can generate the number in the URL leading to the respective versions.
For example, in the case of the last product we downloaded:


```python
str(product.update_time.replace(tzinfo=timezone.utc).timestamp()).replace('.','')
```




    '1613340537989'



# Reproject and stack rasters

For intended visualization, we need the different rasters to fit.
To see if this is the case, we can investigate the metadata of different raster files.

First, let us get a list of files we want to compare and see if we need to reproject them.


```python
paths_mmi = list(Path('data').rglob('mmi_mean.flt'))
paths_mmi.sort()
paths_mmi
```




    [PosixPath('data/us6000dher_us_01_raster/mmi_mean.flt'),
     PosixPath('data/us6000dher_us_02_raster/mmi_mean.flt'),
     PosixPath('data/us6000dher_us_03_raster/mmi_mean.flt'),
     PosixPath('data/us6000dher_us_04_raster/mmi_mean.flt'),
     PosixPath('data/us6000dher_us_05_raster/mmi_mean.flt'),
     PosixPath('data/us6000dher_us_06_raster/mmi_mean.flt'),
     PosixPath('data/us6000dher_us_07_raster/mmi_mean.flt')]




```python
rasterio.open(paths_mmi[0]).meta
```




    {'driver': 'EHdr',
     'dtype': 'float32',
     'nodata': 999.0,
     'width': 498,
     'height': 394,
     'count': 1,
     'crs': None,
     'transform': Affine(0.016666666666666666, 0.0, 138.025,
            0.0, -0.016666666666666666, 40.891666666666666)}




```python
rasterio.open(paths_mmi[-1]).meta
```




    {'driver': 'EHdr',
     'dtype': 'float32',
     'nodata': 999.0,
     'width': 511,
     'height': 427,
     'count': 1,
     'crs': None,
     'transform': Affine(0.016666666666666666, 0.0, 135.49166666666667,
            0.0, -0.016666666666666666, 41.208333333333336)}



The rasters of the different versions do not match, as we can see from the width, height, and transform values.
Therefore reprojection is necessary.
We will use the last version to define our template or destination array and reproject all the other versions to match it.

The implementation here follows [rasterio's Reprojection documentation](https://rasterio.readthedocs.io/en/latest/topics/reproject.html#reprojection).
We directly reproject raster values loaded into a numpy array object.
The documentation also describes and references ways to reproject raster files.

Let us first configure the destination array:


```python
mmi_template = rasterio.open(paths_mmi[-1])

dst_shape = mmi_template.shape
dst_transform = mmi_template.transform
dst_crs = {'init': 'EPSG:4326'}
destination = np.zeros(mmi_template.shape, mmi_template.meta['dtype'])
```

With that, we can reproject the data of the other versions to match it and collect all versions in a three-dimensional array.


```python
mmi_mean_version_stack = []

for i, path_mmi_i in enumerate(paths_mmi[:-1]):
    mmi_i = rasterio.open(path_mmi_i)
    src_shape = mmi_i.shape
    src_transform = mmi_i.transform
    src_crs = {'init': 'EPSG:4326'}
    source = mmi_i.read()

    mmi_mean_version_stack.append(
        reproject(
            source,
            destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)[0]  # returns tuple (arrray(...), Affine(...))
    )
mmi_mean_version_stack.append(mmi_template.read()[0, :, :])
mmi_mean_version_stack = np.stack(mmi_mean_version_stack, axis=2)

mmi_mean_version_stack.shape
```




    (427, 511, 7)



# Visual comparison of two ShakeMap versions

Finally, we have the different raster versions in a format that allows us to easily analyze how ShakeMap versions differ.

For example, the following plot shows the difference between the first and the last versions of *mmi_mean*.


```python
fig, axes = plt.subplots(1,1, figsize=(12, 8))

diff_v7v1 = mmi_mean_version_stack[:, :, -1] - mmi_mean_version_stack[:, :, 0]

img1 = axes.imshow(
    diff_v7v1, 
    cmap ='RdBu', 
    vmin=-1.3,
    vmax=1.3,
    interpolation ='nearest', 
    origin ='lower')
axes.set_title('Difference of mmi_mean versions 7 and 1 ')
axes.axis('off')
cbar = fig.colorbar(img1)
```
    
![output_26_0.png](/img/blog/2021-02-27-1_comparing_usgs_shakemap_versions/output_26_0.png)
