+++
title = "Satellite imagery classification - II"
description = "Downloading and preparing OpenStreetMap data"
date = "2019-10-26"
categories = ["Open Street Map", "Land Use/Land Cover", "Python", "osmnx"]
[[images]]
  src = "img/blog/2019-10-26-1_federsee-blog-series_part-2_osm/output_39_0.png"
  alt = "output_39_0.png"
+++

# Context and Content

This is the second part of a three parts series about using remote sensing data to classify the earth's surface.
Please read the [first part of the series]({{< ref "2019-09-29-1_federsee-blog-series_part-1_cog.md" >}}) if you are interested in an introduction to the post series and / or in the first part where we downloaded parts of Landsat scenes by leveraging the Cloud Optimized GeoTIFF format.

In this part, after a short introduction to OpenStreetMap (OSM), we will download and prepare land use and land cover data sourced from OSM. 
The data will be used to create labeled data for the supervised classification task in the next post.

# What is OSM?

Let us start with two quotes from the [OpenStreetMap Wiki - About OpenStreetMap](https://wiki.openstreetmap.org/wiki/About_OpenStreetMap):

*OpenStreetMap is a free, editable map of the whole world that is being built by volunteers largely from scratch and released with an open-content license.*

*The [OpenStreetMap License](https://www.openstreetmap.org/copyright) allows free (or almost free) access to our map images and all of our underlying map data. The project aims to promote new and interesting uses of this data. 
See ["Why OpenStreetMap?"](https://wiki.openstreetmap.org/wiki/Why_OpenStreetMap%3F) for more details about why we want an open-content map and for the answer to the question we hear most frequently: Why not just use Google maps?*

## OSM data model

According to the OSM Wiki [Features](https://wiki.openstreetmap.org/wiki/Features) are mappable physical landscape elements.
[Elements](https://wiki.openstreetmap.org/wiki/Elements) allow us to define where these features are in space and which of and how the elements belong together.
[Tags](https://wiki.openstreetmap.org/wiki/Tags) tell us more about the elements.

### Elements

The OSM database is a collection of the following [elements](https://wiki.openstreetmap.org/wiki/Elements)

    nodes    : points in space defined by latitude and longitude coordinates
    ways     : linear features and area boundaries defined by multiple nodes
    relations: sometimes used to explain how other elements work together

### Tags

Tags are described in the Wiki as follows: 

*A tag consists of two items, a key and a value. Tags describe specific features of map elements (nodes, ways, or relations) [...].* [OpenStreetMap Wiki - Tags](https://wiki.openstreetmap.org/wiki/Tags) 

The [OpenStreetMap Wiki - Map Features](https://wiki.openstreetmap.org/wiki/Map_Features) page lists such key value pairs.
In this list we can find - among many others - the key *natural* and *landuse*. 
Later we will download the data for our AOI which has been been tagged with one of these two keys.
We use the tags' values as labels for the land use and land cover classification in the next post.
Note that also other keys contain useful information for a land use and land cover classification but for this post series we only select two to keep it simple.

### Example Federsee

Let us see how the abstract description above looks like in practice.

The following screenshot shows [OSM representation of the Federsee on www.openstreetmap.org](https://www.openstreetmap.org/relation/8387767#map=14/48.0827/9.6327).
The *Federsee (8387767)* is a relation which consists of two ways (see *Members*), an outer and an inner one since there is an island in the lake.
If we click on the IDs of the ways in the lower left corner we would see that each of them is made up by several nodes.
These elements allow to draw the feature on the map.
Besides that, additional information about the feature is available via the tags, e.g. the *name* is *Federsee*, value for the key *natural* is *water*, etc. 
Note that the tags are also used for selecting the style in which to draw a feature.

![screenshot_osm_federsee_from_2019-10-24_16-45-57.png](/img/blog/2019-10-26-1_federsee-blog-series_part-2_osm/screenshot_osm_federsee_from_2019-10-24_16-45-57.png)



## Download options for OpenStreetMap data

There are several ways how you can get the OSM data.
One possibility is to navigate to the [AOI in the Export Tab of the OpenStreetMap Webpage](https://www.openstreetmap.org/export#map=14/48.0823/9.6069).
From there you can download the data of an area of interest (AOI) as a [OSM XML](https://wiki.openstreetmap.org/wiki/OSM_XML) file. 

You can also use the [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API) which *is a read-only API [...] optimized for data consumers that need a few elements within a glimpse or up to roughly 10 million elements*. 
A nice post for getting started is [Loading Data from OpenStreetMap with Python and the Overpass API](https://towardsdatascience.com/loading-data-from-openstreetmap-with-python-and-the-overpass-api-513882a27fd0) which also references [Overpass Torbo](https://overpass-turbo.eu/).
The latter is is practical to evaluate queries in the browser.
The whole data for the AOI can be queried as follows:

    /*
    This is a simple map call.
    It returns all data in the bounding box.
    */
    [out:xml];
    (
      node(48.0680, 9.5430, 48.1068, 9.6744);
      <;
    );
    out meta;

[**overpy**](https://python-overpy.readthedocs.io/en/latest/) and [**overpass**](https://github.com/mvexel/overpass-api-python-wrapper) are two Python APIs for accessing the Overpass API.
On October 3rd, 2019 **overpy** had 110 stars 41 forks and **overpass** 186 stars, 66 forks.

In this post we use yet another option, i.e. the **osmnx** package.
This package is very nice because it allows us to download the data of our AOI with a *landuse* or *natural* key and convert it in a ``geopandas.GeoDataFrame`` in one line of code.

Since we now know better what OSM data is, which part of the data we want to download and which framework we want to use, we can get start with the hands-on part of this post.

# Downloading and preparing OSM data with osmnx and pandas

## Libraries

Following libraries will be used in this post:


```python
import geopandas as gpd
from matplotlib.patches import Patch
import numpy as np
import osmnx as ox
import pandas as pd
```

## Area of interest

We load the AOI from the GeoJSON *./data_federsee/aoi_federsee_4326.geojson* which we stored in the [first part of the series]({% post_url 2019-09-29-1_federsee-blog-series_part-1_cog %}).
The content of the file looks as follows:


```bash
%%bash

cat ./data_federsee/aoi_federsee_4326.geojson
```

    {
    "type": "FeatureCollection",
    "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
    "features": [
    { "type": "Feature", "properties": { "ID": 1, "Name": "Federsee" }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 9.543, 48.1068 ], [ 9.6744, 48.1068 ], [ 9.6744, 48.068 ], [ 9.543, 48.068 ], [ 9.543, 48.1068 ] ] ] } }
    ]
    }



```python
gdf_aoi_4326 = gpd.read_file(
    "./data_federsee/aoi_federsee_4326.geojson"
)
gdf_aoi_4326
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
      <th>ID</th>
      <th>Name</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Federsee</td>
      <td>POLYGON ((9.54300 48.10680, 9.67440 48.10680, ...</td>
    </tr>
  </tbody>
</table>
</div>



## Download OSM data with osmnx

With the **osmnx** function [``create_footprints_gdf``](https://osmnx.readthedocs.io/en/latest/osmnx.html#osmnx.footprints.create_footprints_gdf) it does not take more than a line of code to download the data for a specific AOI and footprint type (or OSM tag key) such as 'building', 'landuse', 'place', etc. *and* convert the data into a ``geopandas.GeoDataFrame``.

Since we are interested in land use and land cover information we download the OSM data of the *footprint types* 'natural' and 'landuse'. 

### Download and prepare 'natural'

Let us download the data and see what we get.


```python
gdf_natural = ox.create_footprints_gdf(
    polygon=gdf_aoi_4326.loc[0, "geometry"],
    footprint_type="natural",
    retain_invalid=True,
)

print(
    "Number of (rows, columns) of the Geodataframe:",
    gdf_natural.shape,
)

print("\nFirst two rows of the GeoDataFrame:")
display(gdf_natural.head(2))

print(
    "\nTransposed description of the GeoDataFrame without geometry, nodes:"
)

# describe() does not work on the three columns we drop here first
gdf_natural.drop(
    ["geometry", "nodes", "members"], axis=1
).describe().transpose()
```

    Number of (rows, columns) of the Geodataframe: (102, 22)
    
    First two rows of the GeoDataFrame:



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
      <th>nodes</th>
      <th>natural</th>
      <th>geometry</th>
      <th>water</th>
      <th>wetland</th>
      <th>leaf_cycle</th>
      <th>leaf_type</th>
      <th>description</th>
      <th>intermittent</th>
      <th>man_made</th>
      <th>...</th>
      <th>operator</th>
      <th>species</th>
      <th>members</th>
      <th>type</th>
      <th>TMC:cid_58:tabcd_1:Class</th>
      <th>TMC:cid_58:tabcd_1:LCLversion</th>
      <th>TMC:cid_58:tabcd_1:LocationCode</th>
      <th>name</th>
      <th>wikidata</th>
      <th>wikipedia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>31829954</td>
      <td>[356416115, 3959793379, 356416144, 2486677463,...</td>
      <td>water</td>
      <td>POLYGON ((9.54352 48.10559, 9.54360 48.10561, ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>31829956</td>
      <td>[356416148, 2486677471, 2486677470, 356416186,...</td>
      <td>water</td>
      <td>POLYGON ((9.54280 48.10632, 9.54273 48.10634, ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 22 columns</p>
</div>


    
    Transposed description of the GeoDataFrame without geometry, nodes:





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
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>natural</td>
      <td>102</td>
      <td>6</td>
      <td>wetland</td>
      <td>27</td>
    </tr>
    <tr>
      <td>water</td>
      <td>13</td>
      <td>2</td>
      <td>pond</td>
      <td>12</td>
    </tr>
    <tr>
      <td>wetland</td>
      <td>19</td>
      <td>4</td>
      <td>wet_meadow</td>
      <td>8</td>
    </tr>
    <tr>
      <td>leaf_cycle</td>
      <td>12</td>
      <td>1</td>
      <td>deciduous</td>
      <td>12</td>
    </tr>
    <tr>
      <td>leaf_type</td>
      <td>14</td>
      <td>1</td>
      <td>broadleaved</td>
      <td>14</td>
    </tr>
    <tr>
      <td>description</td>
      <td>1</td>
      <td>1</td>
      <td>Regenüberlaufbecken</td>
      <td>1</td>
    </tr>
    <tr>
      <td>intermittent</td>
      <td>1</td>
      <td>1</td>
      <td>yes</td>
      <td>1</td>
    </tr>
    <tr>
      <td>man_made</td>
      <td>1</td>
      <td>1</td>
      <td>basin</td>
      <td>1</td>
    </tr>
    <tr>
      <td>attraction</td>
      <td>1</td>
      <td>1</td>
      <td>animal</td>
      <td>1</td>
    </tr>
    <tr>
      <td>barrier</td>
      <td>1</td>
      <td>1</td>
      <td>fence</td>
      <td>1</td>
    </tr>
    <tr>
      <td>operator</td>
      <td>1</td>
      <td>1</td>
      <td>Metallbau Knoll</td>
      <td>1</td>
    </tr>
    <tr>
      <td>species</td>
      <td>1</td>
      <td>1</td>
      <td>chicken, goat, sheep</td>
      <td>1</td>
    </tr>
    <tr>
      <td>type</td>
      <td>4</td>
      <td>1</td>
      <td>multipolygon</td>
      <td>4</td>
    </tr>
    <tr>
      <td>TMC:cid_58:tabcd_1:Class</td>
      <td>1</td>
      <td>1</td>
      <td>Area</td>
      <td>1</td>
    </tr>
    <tr>
      <td>TMC:cid_58:tabcd_1:LCLversion</td>
      <td>1</td>
      <td>1</td>
      <td>9.00</td>
      <td>1</td>
    </tr>
    <tr>
      <td>TMC:cid_58:tabcd_1:LocationCode</td>
      <td>1</td>
      <td>1</td>
      <td>42084</td>
      <td>1</td>
    </tr>
    <tr>
      <td>name</td>
      <td>2</td>
      <td>2</td>
      <td>Federseeried</td>
      <td>1</td>
    </tr>
    <tr>
      <td>wikidata</td>
      <td>1</td>
      <td>1</td>
      <td>Q248234</td>
      <td>1</td>
    </tr>
    <tr>
      <td>wikipedia</td>
      <td>1</td>
      <td>1</td>
      <td>de:Federsee</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We got a ``GeoDataFrame`` with 102 rows and 22 columns.
Each row is a feature of the OSM database and is identifiable by the OSM ID stored in the index.
It contains the elements data of OSM in the *nodes*, *geometry* and *members* columns.

Other columns represent the tag keys (column names) and values (values in the cells of the ``GeoDataFrame``).
The column *natural* is completely filled since we downloaded the footprint type 'natural'.
All the other tag columns contain various degrees of missing values as can be seen from the ``described()`` output. 

Remember that our goal is to prepare the data for a land use and land cover classification with 30m resolution imagery (Part 3 of this post series).
Therefore it makes sense to only keep 'Polygon' and 'MultiPolygon' geometries.


```python
rows_keep_natural = gdf_natural.geometry.type.isin(
    ["Polygon", "MultiPolygon"]
)

print("Counts of removed tag values.")
display(
    pd.crosstab(
        gdf_natural.loc[~rows_keep_natural, :].type,
        gdf_natural.loc[~rows_keep_natural, "natural"],
    )
)

gdf_natural = gdf_natural.loc[rows_keep_natural, :]
```

    Counts of removed tag values.



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
      <th>natural</th>
      <th>tree_row</th>
    </tr>
    <tr>
      <th>row_0</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LineString</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>


We also want to keep only the columns / tags which might contain interesting information with regard to classification task.


```python
keep_tag_columns_natural = [
    "natural",
    "water",
    "wetland",
    "leaf_cycle",
    "leaf_type",
]
for col in keep_tag_columns_natural[1::]:
    display(pd.crosstab(gdf_natural["natural"], gdf_natural[col]))
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
      <th>water</th>
      <th>pond</th>
      <th>reservoir</th>
    </tr>
    <tr>
      <th>natural</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>water</td>
      <td>12</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>wetland</th>
      <th>marsh</th>
      <th>reedbed</th>
      <th>swamp</th>
      <th>wet_meadow</th>
    </tr>
    <tr>
      <th>natural</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>wetland</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



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
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



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
      <th>leaf_type</th>
      <th>broadleaved</th>
    </tr>
    <tr>
      <th>natural</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>wood</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


As we can see *leaf_cycle* column is empty so we remove it from the initial list.
This is probably because there were only values in the removed rows.


```python
# as we can see the leave
keep_tag_columns_natural = [
    "natural",
    "water",
    "wetland",
    "leaf_type",
]

gdf_natural = gdf_natural.loc[
    :, ["geometry"] + keep_tag_columns_natural
]
```

### Download and prepare 'landuse'

We do the same as above... 


```python
gdf_landuse = ox.create_footprints_gdf(
    polygon=gdf_aoi_4326.loc[0, "geometry"],
    footprint_type="landuse",
    retain_invalid=True,
)

rows_keep_landuse = gdf_landuse.geometry.type.isin(
    ["Polygon", "MultiPolygon"]
)
print("Counts of removed tag values.")
display(
    pd.crosstab(
        gdf_landuse.loc[~rows_keep_landuse, :].type,
        gdf_landuse.loc[~rows_keep_landuse, "landuse"],
    )
)
gdf_landuse = gdf_landuse.loc[rows_keep_landuse, :]

print(
    "\nNumber of (rows, columns) of the Geodataframe:",
    gdf_natural.shape,
)

print("\nFirst two rows of the GeoDataFrame:")
display(gdf_landuse.head(2))

print(
    "\nTransposed GeoDataFrame description without geometry, nodes:"
)
# describe() does not work on the three columns we drop here first
gdf_landuse.drop(
    ["geometry", "nodes", "members"], axis=1
).describe().transpose()

keep_tag_columns_landuse = ["landuse", "leaf_type"]
for col in keep_tag_columns_landuse[1::]:
    display(pd.crosstab(gdf_landuse["landuse"], gdf_landuse[col]))

# keep what we need for the classification
gdf_landuse = gdf_landuse.loc[
    :, ["geometry"] + keep_tag_columns_landuse
]
```

    Counts of removed tag values.



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
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    
    Number of (rows, columns) of the Geodataframe: (85, 5)
    
    First two rows of the GeoDataFrame:



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
      <th>nodes</th>
      <th>landuse</th>
      <th>geometry</th>
      <th>name</th>
      <th>toilets:wheelchair</th>
      <th>tourism</th>
      <th>wheelchair</th>
      <th>source</th>
      <th>leaf_type</th>
      <th>religion</th>
      <th>basin</th>
      <th>members</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>8030302</td>
      <td>[60020682, 1369161162, 1369161159, 60020683, 1...</td>
      <td>forest</td>
      <td>POLYGON ((9.67498 48.09177, 9.67556 48.09173, ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>8030303</td>
      <td>[60020695, 4878237619, 4878237618, 1369161194,...</td>
      <td>forest</td>
      <td>POLYGON ((9.67105 48.09603, 9.67099 48.09655, ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    
    Transposed GeoDataFrame description without geometry, nodes:



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
      <th>leaf_type</th>
      <th>broadleaved</th>
      <th>mixed</th>
      <th>needleleaved</th>
    </tr>
    <tr>
      <th>landuse</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>forest</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


### Remove spatial overlaps

Before we combine the data we want to check if there are overlaps between 

* the (multi)-polygons of each ``GeoDataFrame``
* between the (multi)-polygons of the two ``GeoDataFrame``s.

For this tutorial we keep it simple and remove these (multi-)polygons as suspicious or ambiguous cases.
But of course in a real world scenario it might be worth looking into that cases in more detail to find out if these polygons are useful. 

To get overlaps between (multi-)polygons in one ``Geodataframe`` we define the following function.
It takes a dataframe and returns ``None`` if there are no overlapping polygons and the subset of overlapping polygons otherwise.


```python
def get_overlapping_polygons(gdf):
    """Get the overlap between polygons in the same dataframe."""
    # This is not a performant solution!
    overlaps = []
    for i, row_i in gdf.iterrows():
        for j, row_j in gdf.iterrows():
            if i < j:
                if row_i.geometry.overlaps(row_j.geometry):
                    overlaps.append(i)
                    overlaps.append(j)

    if overlaps:
        return gdf.loc[np.unique(overlaps), :]
    else:
        return None
```


```python
gdf_natural_overlaps = get_overlapping_polygons(gdf_natural)
gdf_natural_overlaps
```

There are no overlaping polygons in ``gdf_natural``.


```python
gdf_landuse_overlaps = get_overlapping_polygons(gdf_landuse)
gdf_landuse_overlaps
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
      <th>geometry</th>
      <th>landuse</th>
      <th>leaf_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5747104</td>
      <td>POLYGON ((9.57793 48.06533, 9.57764 48.06582, ...</td>
      <td>forest</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>5747254</td>
      <td>POLYGON ((9.59307 48.06887, 9.59314 48.06881, ...</td>
      <td>meadow</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>6043203</td>
      <td>POLYGON ((9.72595 48.11495, 9.72592 48.11505, ...</td>
      <td>forest</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>9945268</td>
      <td>POLYGON ((9.65546 48.09192, 9.65490 48.09161, ...</td>
      <td>residential</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>601113438</td>
      <td>POLYGON ((9.65429 48.08923, 9.65427 48.08917, ...</td>
      <td>farmyard</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>601113454</td>
      <td>POLYGON ((9.66902 48.10880, 9.67136 48.10888, ...</td>
      <td>farmland</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



There are 6 overlaping polygons in ``gdf_landuse``.

Let us inspect what we loose when we just remove these polygons spatially and in terms of polygon counts per class.


```python
ax = gdf_aoi_4326.plot(color="#d95f02", figsize=(16, 10))
ax = gdf_landuse_overlaps.plot(color="#7570b3", ax=ax)
txt = ax.set_title(
    "AOI (orange) and overlapping landuse polygons (violet)."
)
```


![output_21_0.png](/img/blog/2019-10-26-1_federsee-blog-series_part-2_osm/output_21_0.png)


For the polygon counts comparison we define a function since we will need the same later.


```python
def compare_value_counts(df1, df2, col):
    """Count the values of the same column (col) and create a dataframe with counts and count differences."""
    value_counts_1 = df1[col].value_counts()
    value_counts_2 = df2[col].value_counts()
    df_comparison = pd.concat(
        [
            value_counts_1.rename(f"{col}_1"),
            value_counts_2.rename(f"{col}_2"),
            (value_counts_1 - value_counts_2).rename("1 - 2"),
        ],
        axis=1,
        sort=False,
    )
    return df_comparison


compare_value_counts(
    gdf_landuse,
    gdf_landuse.drop(gdf_landuse_overlaps.index),
    col="landuse",
)
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
      <th>landuse_1</th>
      <th>landuse_2</th>
      <th>1 - 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>meadow</td>
      <td>71</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <td>farmland</td>
      <td>48</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <td>forest</td>
      <td>21</td>
      <td>19</td>
      <td>2</td>
    </tr>
    <tr>
      <td>farmyard</td>
      <td>14</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <td>residential</td>
      <td>13</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <td>orchard</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <td>allotments</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <td>commercial</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>plant_nursery</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>industrial</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>cemetery</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>grass</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>basin</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Let us decide that we can get over it when we remove these polygons and do it. 


```python
gdf_landuse = gdf_landuse.drop(gdf_landuse_overlaps.index)
```

To get overlaping (multi-)polygons between two ``Geodataframe`` we can use the [``overlay`` functionality of **geopandas**](https://geopandas.readthedocs.io/en/latest/set_operations.html).



```python
gdf_intersection = gpd.overlay(
    gdf_landuse.reset_index(),
    gdf_natural.reset_index(),
    how="intersection",
)

display(gdf_intersection)

ax = gdf_aoi_4326.plot(color="#d95f02", figsize=(16, 10))
# gdf_intersection.plot(color="#7570b3", ax=ax)
ax = gdf_landuse.loc[gdf_intersection["index_1"]].plot(
    color="#7570b3", ax=ax
)
ax = gdf_natural.loc[gdf_intersection["index_2"]].plot(
    color="#1b9e77", ax=ax
)
txt = ax.set_title(
    "AOI (orange), overlapping landuse (violet) and natural (green)"
    "polygons."
)
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
      <th>index_1</th>
      <th>landuse</th>
      <th>leaf_type_1</th>
      <th>index_2</th>
      <th>natural</th>
      <th>water</th>
      <th>wetland</th>
      <th>leaf_type_2</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>129014138</td>
      <td>commercial</td>
      <td>NaN</td>
      <td>216506723</td>
      <td>water</td>
      <td>pond</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POLYGON ((9.64559 48.06889, 9.64563 48.06883, ...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>129014140</td>
      <td>residential</td>
      <td>NaN</td>
      <td>600407995</td>
      <td>water</td>
      <td>pond</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POLYGON ((9.64581 48.07079, 9.64586 48.07075, ...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>601767693</td>
      <td>meadow</td>
      <td>NaN</td>
      <td>601767688</td>
      <td>grassland</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POLYGON ((9.66622 48.07509, 9.66598 48.07507, ...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>715419648</td>
      <td>allotments</td>
      <td>NaN</td>
      <td>715419645</td>
      <td>wood</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POLYGON ((9.62574 48.10542, 9.62576 48.10541, ...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>715706685</td>
      <td>meadow</td>
      <td>NaN</td>
      <td>715706686</td>
      <td>wetland</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POLYGON ((9.59961 48.09108, 9.60090 48.09078, ...</td>
    </tr>
    <tr>
      <td>5</td>
      <td>715706685</td>
      <td>meadow</td>
      <td>NaN</td>
      <td>715706687</td>
      <td>wetland</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POLYGON ((9.59995 48.09326, 9.60021 48.09306, ...</td>
    </tr>
    <tr>
      <td>6</td>
      <td>7283742</td>
      <td>meadow</td>
      <td>NaN</td>
      <td>496089851</td>
      <td>scrub</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POLYGON ((9.66512 48.09539, 9.66490 48.09536, ...</td>
    </tr>
  </tbody>
</table>
</div>



![output_27_1.png](/img/blog/2019-10-26-1_federsee-blog-series_part-2_osm/output_27_1.png)



```python
compare_value_counts(
    gdf_landuse,
    gdf_landuse.drop(gdf_intersection["index_1"]),
    col="landuse",
)
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
      <th>landuse_1</th>
      <th>landuse_2</th>
      <th>1 - 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>meadow</td>
      <td>70</td>
      <td>67</td>
      <td>3</td>
    </tr>
    <tr>
      <td>farmland</td>
      <td>47</td>
      <td>47</td>
      <td>0</td>
    </tr>
    <tr>
      <td>forest</td>
      <td>19</td>
      <td>19</td>
      <td>0</td>
    </tr>
    <tr>
      <td>farmyard</td>
      <td>13</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <td>residential</td>
      <td>12</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <td>orchard</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <td>allotments</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <td>commercial</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <td>plant_nursery</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>industrial</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>cemetery</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>grass</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>basin</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
compare_value_counts(
    gdf_natural,
    gdf_natural.drop(gdf_intersection["index_2"]),
    col="natural",
)
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
      <th>natural_1</th>
      <th>natural_2</th>
      <th>1 - 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>wetland</td>
      <td>27</td>
      <td>25</td>
      <td>2</td>
    </tr>
    <tr>
      <td>water</td>
      <td>22</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <td>wood</td>
      <td>19</td>
      <td>18</td>
      <td>1</td>
    </tr>
    <tr>
      <td>grassland</td>
      <td>11</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <td>scrub</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let us also decide that we can get over that lost. 


```python
gdf_landuse = gdf_landuse.drop(gdf_intersection["index_1"])
gdf_natural = gdf_natural.drop(gdf_intersection["index_2"])
```

### Combine 'landuse' and 'natural'

We combine the two ``GeoDataFrame``s with a simple non-spatial concatenate operation.
We are confident that we do not get any overlapping polygons since we removed them already.
Still we add an assertion statement here to make sure that this is true.

Also we make the assertion that there we do not have a value for 'landuse' *and* 'natural' in any row since we want to combine this information in one column.


```python
gdf_lun = pd.concat([gdf_landuse, gdf_natural], sort=False)

assert get_overlapping_polygons(gdf_lun) is None
assert (gdf_lun[["landuse", "natural"]].isna().sum(axis=1) == 1).all()
```

Since we did not get an ``AssertionError`` our assertions are valid and we can create the label column for the classification task.
Let us call it 'lun' for 'landuse' and 'natural'.


```python
gdf_lun["lun"] = gdf_lun["landuse"]
gdf_lun.loc[gdf_lun["landuse"].isna(), "lun"] = gdf_lun["natural"]
```

### Final cleansing

In a final step we 

* merge 'commercial', 'industrial', 'residential', 'farmyard' into a 'buildup' class (knowning that they might include vegetated areas),
* merge 'meadow', 'grassland' into one 'grassland' class,
* merge 'forest', 'wood' into one 'forest' class (knowing that we stretch the meanding of forest here),
* remove polygons of very small size, i.e. smaller than 5 pixels,
* remove label categories with less than 3 polygons (after the steps above).


```python
gdf_lun["lun_l1"] = gdf_lun["lun"]

gdf_lun.loc[
    gdf_lun["lun_l1"].isin(
        ["commercial", "industrial", "residential", "farmyard"]
    ),
    "lun_l1",
] = "buildup"
gdf_lun.loc[
    gdf_lun["lun_l1"].isin(["meadow", "grassland"]), "lun_l1"
] = "grassland"
gdf_lun.loc[
    gdf_lun["lun_l1"].isin(["forest", "wood"]), "lun_l1"
] = "forest"

gdf_lun["area_m2"] = gdf_lun.to_crs(crs={"init": "epsg:32632"}).area
gdf_lun = gdf_lun[gdf_lun["area_m2"] > 5 * 30 * 30]

n_polygons_per_class = gdf_lun["lun_l1"].value_counts()
gdf_lun = gdf_lun[
    gdf_lun["lun_l1"].isin(
        n_polygons_per_class[n_polygons_per_class >= 3].index
    )
]
```

## Visualize and store the cleansed data

Usually we want to use a specific color map for land use / land cover maps.
With ``geopandas.GeoDataFrame.plot`` we cannot specify such custom color maps.
Therefore we need create a column containing the colors and create a legend by making use of the lower-level ``matplotlib`` plotting functions.

Here is what our labeled data for the classification task in the next post of this series looks like.


```python
# define color code for lun_l1 and add as colors as column
class_color_map = {
    "buildup": "#e31a1c",
    "farmland": "#ff7f00",
    "forest": "#33a02c",
    "grassland": "#b2df8a",
    "orchard": "#b15928",
    "water": "#1f78b4",
    "wetland": "#a6cee3",
}
gdf_lun["color"] = gdf_lun["lun_l1"].map(class_color_map)

# create legend elements 
n_polygons_per_class = gdf_lun["lun_l1"].value_counts()
legend_elements = [
    Patch(
        facecolor=value,
        edgecolor="black",
        label=f"{key} ({n_polygons_per_class[key]})",
    )
    for key, value in class_color_map.items()
]


# plot the aoi and the lun_l1 column 
ax = gdf_aoi_4326.plot(
    color="black", edgecolor="black", figsize=(16, 10)
)
txt = ax.set_title("Land use and land cover classes.")
ax = gdf_lun.plot(
    color=gdf_lun["color"],
    edgecolor="black",
    ax=ax,
)
# add legend to plot
ax = ax.legend(
    handles=legend_elements,
    title="Class (Polygons Count)",
    loc="upper right",
    ncol=2,
)
```

![output_39_0.png](/img/blog/2019-10-26-1_federsee-blog-series_part-2_osm/output_39_0.png)

Let us store the data such that it is available in the next post.


```python
gdf_lun.to_file(
    "./data_federsee/osm_feedersee_cleansed_4326.geojson",
    driver="GeoJSON",
)
```

# The end

OpenStreetMap is a great open project. 
The number of people involved and data collected is impressive.

According to the [OpenStreetMap stats report](https://www.openstreetmap.org/stats/data_stats.html) run at 2019-10-24 22:00:06 +0000 the numbers look as follows:
```
Number of users                      5,769,940
Number of uploaded GPS points    7,547,182,003
Number of nodes                  5,549,201,409
Number of ways                     615,216,915
Number of relations                  7,201,811
```

In this post we downloaded a small part of the OSM data with **osmnx** and manipulated the data with **geopandas** and **pandas**.
Together with the satellite data downloaded in the [first part of the series]({% post_url 2019-09-29-1_federsee-blog-series_part-1_cog %}) we do now have the two incredients that we need for creating a land use and land cover map with a supervised classification algorithm in the next and final post of this post series.

Please feel free to leave a comment or contact me if you have any feedback.
