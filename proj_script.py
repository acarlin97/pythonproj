# === 1. Imports ===
# load libraries for file handling, arrays, raster/vector data and map plotting
import os
import numpy as np
import rasterio as rio
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.patches as mpatches

# === 2. Function Definitions ===
# first four are helper functions used later in the script for visual styling and spatial analysis
# these functions help to create legends, scale bars, raster display and plot the HER sites
def generate_handles(labels, colors, edge='k', alpha=1):
    """
    Generate matplotlib patch handles to create a legend of each of the features in the map.

    Parameters
    ----------
    labels : list(str)
        the text labels of the features to add to the legend
    colors : list(matplotlib color)
        the colors used for each of the features included in the map.
    edge : matplotlib color (default: 'k')
        the color to use for the edge of the legend patches.
    alpha : float (default: 1.0)
        the alpha value to use for the legend patches.

    Returns
    -------
    handles : list(matplotlib.patches.Rectangle)
        the list of legend patches to pass to ax.legend()
    """
    lc = len(colors)  # get the length of the colour list
    handles = [] # create an empty list to store legend patches
    for ii in range(len(labels)):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[ii % lc], edgecolor=edge, alpha=alpha))
    return handles

def scale_bar(ax, length=20, location=(0.92, 0.95)):
    """
    Create a scale bar in a cartopy GeoAxes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        the cartopy GeoAxes to add the scalebar to.
    length : int, float (default 20)
        the length of the scalebar, in km
    location : tuple(float, float) (default (0.92, 0.95))
        the location of the center right corner of the scalebar, in fractions of the axis.

    Returns
    -------
    ax : cartopy.mpl.geoaxes.GeoAxes
        the cartopy GeoAxes object

    """
    x0, x1, y0, y1 = ax.get_extent() # get the current extent of the axis
    sbx = x0 + (x1 - x0) * location[0] # get the right x coordinate of the scale bar
    sby = y0 + (y1 - y0) * location[1] # get the right y coordinate of the scale bar

    ax.plot([sbx, sbx-length*1000], [sby, sby], color='k', linewidth=4, transform=ax.projection) # plot a thick black line
    ax.plot([sbx-(length/2)*1000, sbx-length*1000], [sby, sby], color='w', linewidth=2, transform=ax.projection) # plot a white line from 0 to halfway

    ax.text(sbx, sby-(length/4)*1000, f'{length} km', ha='center', transform=ax.projection, fontsize=6) # add a label on the right side
    ax.text(sbx-(length/2)*1000, sby-(length/4)*1000, f'{int(length/2)} km', ha='center', transform=ax.projection, fontsize=6) # add a label in the center
    ax.text(sbx-length*1000, sby-(length/4)*1000, '0 km', ha='center', transform=ax.projection, fontsize=6) # add a label on the left side

    return ax

def percentile_stretch(img, pmin=0., pmax=100.):
    '''
    Applies a linear contrast stretch to a 2D image using percentile values.

    Parameters
    ----------
    img : numpy.ndarray
        A 2D array (grayscale image) to be stretched.
    pmin : float, optional (default: 0.0)
        Lower percentile to define the minimum value of the stretch.
    pmax : float, optional (default: 100.0)
        Upper percentile to define the maximum value of the stretch.

    Returns
    -------
    stretched : numpy.ndarray
        A 2D array with values scaled between 0 and 1.

    Raises
    ------
    ValueError
        If pmin or pmax are out of range (0–100), or if pmin >= pmax.
        If input image is not 2D.
    '''
    # here, we make sure that pmin < pmax, and that they are between 0, 100
    if not 0 <= pmin < pmax <= 100:
        raise ValueError('0 <= pmin < pmax <= 100')
    # here, we make sure that the image is only 2-dimensional
    if not img.ndim == 2:
        raise ValueError('Image can only have two dimensions (row, column)')

    minval = np.percentile(img, pmin)
    maxval = np.percentile(img, pmax)

    stretched = (img - minval) / (maxval - minval)  # stretch the image to 0, 1.
    stretched[img < minval] = 0  # set anything less than minval to the new minimum, 0.
    stretched[img > maxval] = 1  # set anything greater than maxval to the new maximum, 1.

    return stretched

def img_display(img, ax, bands, stretch_args=None, **imshow_args):
    '''
    Displays a multi-band image (e.g. RGB) on a given matplotlib axis with optional contrast stretching.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D array representing a multi-band image in the format (bands, rows, columns).
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axis on which to display the image.
    bands : tuple or list of int
        The band indices to display (e.g., (0, 1, 2) for RGB).
    stretch_args : dict, optional
        Dictionary of arguments to pass to `percentile_stretch()`, such as {'pmin': 2, 'pmax': 98}.
        If None, default stretch between 0th and 100th percentiles is applied.
    **imshow_args : dict
        Additional keyword arguments passed to `ax.imshow()` for customizing display (e.g., cmap, alpha).

    Returns
    -------
    handle : matplotlib.image.AxesImage
        The image object returned by `imshow()`.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis on which the image is plotted.
    '''
    dispimg = img.copy().astype(np.float32)  # make a copy of the original image

    for b in range(img.shape[0]):  # loop over each band, stretching using percentile_stretch()
        if stretch_args is None:  # if stretch_args is None, use the default values for percentile_stretch
            dispimg[b] = percentile_stretch(img[b])
        else:
            dispimg[b] = percentile_stretch(img[b], **stretch_args)

    dispimg = dispimg.transpose([1, 2, 0]) # transpose the image to re-order the indices

    handle = ax.imshow(dispimg[:, :, bands], **imshow_args) # display the image

    return handle, ax

# the following four functions are new functions
def count_sites_by_county(sites_gdf, counties_gdf, label, distance_label='150m'):
    """
    Perform a spatial join to count how many HER sites fall within each county.

    Parameters
    ----------
    sites_gdf : GeoDataFrame
        Clipped HER site GeoDataFrame (e.g. smr_near_coast)
    counties_gdf : GeoDataFrame
        County boundaries GeoDataFrame (e.g. counties_utm)
    label : str
        Name of HER dataset (e.g. 'SMR')
    distance_label : str (default: '150m')
        Label for buffer distance (e.g. '150m', '15m')

    Returns
    -------
    GeoDataFrame:
        Joined dataset with county info
    """
    joined = gpd.sjoin(sites_gdf, counties_gdf, how='inner', predicate='within')
    print(f'{label} sites per county that are within {distance_label} of the coast:')
    print(joined['CountyName'].value_counts())
    print('\n')

    return joined

def plot_her_sites(ax, smr, ihr, dhr, transform, size=4):
    """
    Plots HER sites (SMR, IHR, DHR) on the given axis.

    Parameters
    ----------
    ax : Matplotlib axis
        Axis to plot on.
    smr, ihr, dhr : GeoDataFrame
        GeoDataFrames of HER sites to plot.
    transform : cartopy CRS
        CRS transform to apply to the site coordinates.
    size : int (default: 4)
        Marker size (ms) for plotting.

    Returns
    -------
    list:
        List of Matplotlib plot handles [smr_handle, ihr_handle, dhr_handle].
    """
    smr_handle = ax.plot(smr.geometry.x, smr.geometry.y, 's',
                         color='red', mec='black', mew=0.5, ms=size, transform=transform)
    ihr_handle = ax.plot(ihr.geometry.x, ihr.geometry.y, '^',
                         color='blue', mec='black', mew=0.5, ms=size, transform=transform)
    dhr_handle = ax.plot(dhr.geometry.x, dhr.geometry.y, 'o',
                         color='green', mec='black', mew=0.5, ms=size, transform=transform)

    return smr_handle + ihr_handle + dhr_handle

def add_raster_backdrop(ax, raster_img, bounds, transform, extent_crs, mask_geom, stretch_args=None):
    """
    Displays a raster image with contrast stretching and adds a white mask outside the defined boundary.

    Parameters
    ----------
    ax : map axis
        Axis to plot on.
    raster_img : numpy.ndarray
        Multi-band raster image to display.
    bounds : bounding box
        Bounding box of the raster (e.g. from rasterio).
    transform : transform
        Transform of the raster (e.g. from rio.open().transform).
    extent_crs : map projection
        CRS to use (e.g. UTM).
    mask_geom : shapely.geometry
        Geometry used to mask areas outside the region of interest.
    stretch_args : dict, optional
        Parameters for contrast stretching, e.g. {'pmin': 2, 'pmax': 98}.

    Returns
    -------
    ax : map axis
        Updated map axis with raster and mask.
    """
    # display the raster image using the existing img_display() function
    img_display(raster_img, ax, bands=[2, 1, 0],
            stretch_args=stretch_args,
            transform=extent_crs,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            alpha=1.0)

    # add a semi-transparent white mask over areas outside NI
    mask_feature = ShapelyFeature([mask_geom], extent_crs, facecolor='white', alpha=0.6, edgecolor='none')
    ax.add_feature(mask_feature)

    return ax

def buffer_and_clip(coastline_gdf, her_dict, distance):
    """
    Buffers the coastline and clips each HER dataset to the buffer.

    Parameters
    ----------
    coastline_gdf : GeoDataFrame
        Coastline geometry for buffering.
    her_dict : dict[str, GeoDataFrame]
        Dictionary of HER datasets (e.g. {'SMR': gdf, 'IHR': gdf, ...}).
    distance : float
        Buffer distance in meters.

    Returns
    -------
    clipped_dataset : dict[str, GeoDataFrame]
        Clipped HER datasets by type.
    buffer_gdf : GeoDataFrame
        Buffer geometry used for clipping.
    """
    buffer = coastline_gdf.buffer(distance)
    buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs=coastline_gdf.crs)
    clipped_dataset = {name: gpd.clip(gdf, buffer_gdf) for name, gdf in her_dict.items()}

    return clipped_dataset, buffer_gdf

# === 3. Load and Reproject Data ===
# open HER data shapefiles
smr_data = gpd.read_file('data_files/Sites_and_Monuments_Record_13Mar2025.shp')
ihr_data = gpd.read_file('data_files/Industrial_Heritage_Record_13Mar2025.shp')
dhr_data = gpd.read_file('data_files/Defence_Heritage_Record_13Mar2025.shp')

# open NI Outline, Counties, Water and Coastline shapefiles
ni_outline = gpd.read_file(os.path.abspath('data_files/NI_outline.shp'))
counties = gpd.read_file(os.path.abspath('data_files/Counties.shp'))
water = gpd.read_file(os.path.abspath('data_files/Water.shp'))
coastline = gpd.read_file(os.path.abspath('data_files/2021_NI_Coastal_Survey.shp'))

# open NI_Mosaic raster image to use as a visual backdrop
with rio.open('data_files/NI_Mosaic.tif') as mosaic:
    ni_img = mosaic.read()
    ni_bounds = mosaic.bounds
    ni_transform = mosaic.transform

# check the CRS of each dataset for consistency
print(f'SMR: {smr_data.crs}\nIHR: {ihr_data.crs}\nDHR: {dhr_data.crs}')
print(f'NI Outline: {ni_outline.crs}\nCounties: {counties.crs}\nWater: {water.crs}\nCoastline: {coastline.crs}\nNI Mosaic: {mosaic.crs}\n')

# create a Universal Transverse Mercator (UTM) reference system to transform our data
ni_utm = ccrs.UTM(29)

# reproject datasets to a common CRS (UTM Zone 29)
smr_utm = smr_data.to_crs(epsg=32629)
ihr_utm = ihr_data.to_crs(epsg=32629)
dhr_utm = dhr_data.to_crs(epsg=32629)
counties_utm = counties.to_crs(epsg=32629)
water_utm = water.to_crs(epsg=32629)
coastline_utm = coastline.to_crs(epsg=32629)

# recheck the CRS of each dataset after reprojecting
print(f'SMR: {smr_utm.crs}\nIHR: {ihr_utm.crs}\nDHR: {dhr_utm.crs}')
print(f'NI Outline: {ni_outline.crs}\nCounties: {counties_utm.crs}\nWater: {water_utm.crs}\nCoastline: {coastline_utm.crs}\nNI Mosaic: {mosaic.crs}\n')

# count the number of features in each HER dataset
print('Number of SMR features: {}'.format(len(smr_data)))
print('Number of IHR features: {}'.format(len(ihr_data)))
print('Number of DHR features: {}'.format(len(dhr_data)))
print('\n')

# === 4. Mapping All HER Sites ===
# create the map of HER sites across Northern Ireland (figure 1)
fig = plt.figure(figsize=(8, 8))  # create a figure of size 8x8 (representing the page size in inches)
ax = plt.axes(projection=ni_utm)  # create axes object in figure using UTM projection

# add NI outline boundary
outline_feature = ShapelyFeature(ni_outline['geometry'], ni_utm, edgecolor='black', facecolor='white')
ax.add_feature(outline_feature)

# set the map extent
xmin, ymin, xmax, ymax = ni_outline.total_bounds
ax.set_extent([xmin-5000, xmax+5000, ymin-5000, ymax+5000], crs=ni_utm)

# assign colours to each county for map display
county_colors = ['orchid', 'yellow', 'seagreen', 'firebrick', 'darkorange', 'lightblue']

# get a list of unique names for the county boundaries
county_names = list(counties_utm.CountyName.unique())
county_names.sort() # sort the counties alphabetically by name

# create legend handle for county data and labels for reuse in maps
# update county_names to take it out of uppercase text
county_handle = generate_handles(counties_utm.CountyName.unique(), county_colors, alpha=0.25)
lc_names = [name.title() for name in county_names]

# plot each county with a unique colour fill and black edge
for ii, name in enumerate(county_names):
    feat = ShapelyFeature(counties_utm.loc[counties_utm['CountyName'] == name, 'geometry'],
                          ni_utm,
                          edgecolor='black',
                          facecolor=county_colors[ii],
                          linewidth=1, # set the outline width to be 1 pt
                          alpha=0.25) # set the transparency to be 0.25 (out of 1)
    ax.add_feature(feat)

# add water features to the map
water_feat = ShapelyFeature(water['geometry'], # first argument is the geometry
                            ccrs.CRS(water.crs), # second argument is the CRS
                            edgecolor='black', # set the edgecolor to be black
                            facecolor='white', # set the facecolor to be white
                            linewidth=1) # set the outline width to be 1 pt
ax.add_feature(water_feat)

# plot HER sites (SMR, IHR, DHR) using consistent symbology
# transform ni_utm is stated to ensure the coordinates match that of the map's CRS
# mec and mew are shortened from markeredgecolor and markeredgewidth respectively: https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html
site_handles = plot_her_sites(ax, smr_utm, ihr_utm, dhr_utm, transform=ni_utm, size=3)

# add handles and labels information to be added to the legend
handles = county_handle + site_handles
labels = lc_names + ['SMR Sites', 'IHR Sites', 'DHR Sites']

# add figure title
ax.set_title('HER Sites Across Northern Ireland', fontsize=12, fontweight='bold')

# add legend
leg = ax.legend(handles,labels, title='Legend', title_fontsize=12,
                 fontsize=10, loc='upper left', frameon=True, framealpha=1)
leg.get_title().set_fontweight('bold') # make the legend title bold

# add gridlines
gridlines = ax.gridlines(draw_labels=True, # draw  labels for the gridlines
                         xlocs=[-8, -7.5, -7, -6.5, -6, -5.5], # add longitude lines at 0.5 deg intervals
                         ylocs=[54, 54.5, 55, 55.5]) # add latitude lines at 0.5 deg intervals
gridlines.top_labels = False # turn off the top labels
gridlines.right_labels = False # turn off the right-side labels

# add scale bar to the upper right-hand corner of the map window
scale_bar(ax)

# save figure
fig.savefig('Figure1.png', bbox_inches='tight', dpi=300)

# === 5. Spatial Analysis: Buffer and Clip HER Sites (150m) ===
# define HER dictionary
her_dict = {'SMR': smr_utm, 'IHR': ihr_utm, 'DHR': dhr_utm}

# apply buffer and clip for 150m
clipped_150m, buffer_gdf = buffer_and_clip(coastline_utm, her_dict, 150)

# extract HER datasets from 150m clipped results
smr_near_coast = clipped_150m['SMR']
ihr_near_coast = clipped_150m['IHR']
dhr_near_coast = clipped_150m['DHR']

print(f'Number of SMR sites (within 150m of coast): {len(smr_near_coast)} features')
print(f'Number of IHR sites (within 150m of coast): {len(ihr_near_coast)} features')
print(f'Number of DHR sites (within 150m of coast): {len(dhr_near_coast)} features')
print('\n')

# use new count_sites_by_county function to perform spatial join and count HER sites per county
smr_county = count_sites_by_county(smr_near_coast, counties_utm, 'SMR', '150m')
ihr_county = count_sites_by_county(ihr_near_coast, counties_utm, 'IHR', '150m')
dhr_county = count_sites_by_county(dhr_near_coast, counties_utm, 'DHR', '150m')

# === 6. Mapping HER Sites Within 150m of the NI Coastline ===
# save clipped 150m coastal HER datasets to shapefiles
smr_near_coast.to_file('data_files/SMR_near_coast.shp')
ihr_near_coast.to_file('data_files/IHR_near_coast.shp')
dhr_near_coast.to_file('data_files/DHR_near_coast.shp')

# figure 2 - neutral basemap on map
# create the map of HER sites across Northern Ireland
fig2 = plt.figure(figsize=(8, 8))
ax2 = plt.axes(projection=ni_utm)

# add NI outline boundary
outline_feature = ShapelyFeature(ni_outline['geometry'], ni_utm, edgecolor='black', facecolor='white')
ax2.add_feature(outline_feature)

# set the map extent
xmin, ymin, xmax, ymax = ni_outline.total_bounds
ax2.set_extent([xmin-5000, xmax+5000, ymin-5000, ymax+5000], crs=ni_utm)

# plot each county with a unique colour fill and black edge
for ii, name in enumerate(county_names):
    feat = ShapelyFeature(counties_utm.loc[counties_utm['CountyName'] == name, 'geometry'],
                          ni_utm,
                          edgecolor='black',
                          facecolor=county_colors[ii],
                          linewidth=1,
                          alpha=0.25)
    ax2.add_feature(feat)

# add water features to the map
water_feat = ShapelyFeature(water['geometry'],
                            ccrs.CRS(water.crs),
                            edgecolor='black',
                            facecolor='white',
                            linewidth=1)
ax2.add_feature(water_feat)

# add the 150m buffer around coastline (as dark orange fill)
buffer_feat = ShapelyFeature(buffer_gdf['geometry'], ni_utm, edgecolor='darkorange', facecolor='none', linewidth=0.75)
ax2.add_feature(buffer_feat)

# plot HER sites (SMR, IHR, DHR) using consistent symbology
site_handles = plot_her_sites(ax2, smr_near_coast, ihr_near_coast, dhr_near_coast, transform=ni_utm, size=4)

# add a custom patch for coastal buffer (orange line)
buffer_patch = mpatches.Patch(facecolor='orange', edgecolor='darkorange', label='150m Coastal Buffer', linewidth=0.75)

# create and add legend
handles = county_handle + [buffer_patch] + site_handles
labels = lc_names + ['150m Coastal Buffer', 'SMR Sites (150m)', 'IHR Sites (150m)', 'DHR Sites (150m)']
leg2 = ax2.legend(handles,labels, title='Legend', title_fontsize=12,
                 fontsize=10, loc='upper left', frameon=True, framealpha=1)
leg2.get_title().set_fontweight('bold')

# add figure title
ax2.set_title('HER Sites Within 150m of NI Coastline', fontsize=12, fontweight='bold')

# add gridlines
gridlines2 = ax2.gridlines(draw_labels=True,
                         xlocs=[-8, -7.5, -7, -6.5, -6, -5.5],
                         ylocs=[54, 54.5, 55, 55.5])
gridlines2.top_labels = False
gridlines2.right_labels = False

# add scale bar
scale_bar(ax2)

# save figure
fig2.savefig('Figure2.png', bbox_inches='tight', dpi=300)

# figure 2.5 - NI_Mosaic raster backdrop on map
# create the figure and axes with UTM projection
fig2_5 = plt.figure(figsize=(8, 8))
ax2_5 = plt.axes(projection=ni_utm)

# set the map extent
xmin, ymin, xmax, ymax = ni_outline.total_bounds
ax2_5.set_extent([ni_bounds.left, ni_bounds.right, ni_bounds.bottom, ni_bounds.top], crs=ni_utm)

# create a rectangle that covers the map extent
map_bounds = box(xmin - 5000, ymin - 5000, xmax + 5000, ymax + 5000)

# merge all NI geometries into one shape
ni_outline['geometry'] = ni_outline['geometry'].buffer(0) # fix invalid geometries before union
ni_union = unary_union(ni_outline.geometry)

# subtract NI from the full map extent to get outer mask
mask_geom = map_bounds.difference(ni_union)

# define contrast stretch
stretch_args = {'pmin': 0.1, 'pmax': 99.9}

# call custom raster backdrop function
add_raster_backdrop(ax2_5, ni_img, ni_bounds, ni_transform, ni_utm, mask_geom, stretch_args)

# add county boundaries as red outlines (no fill)
for _, row in counties_utm.iterrows():
    county_feat = ShapelyFeature([row['geometry']], ni_utm,
                                 edgecolor='red', facecolor='none', linewidth=1)
    ax2_5.add_feature(county_feat)

# add the 150m buffer around coastline (as dark orange fill)
buffer_feat = ShapelyFeature(buffer_gdf['geometry'], ni_utm, edgecolor='darkorange', facecolor='none', linewidth=0.75)
ax2_5.add_feature(buffer_feat)

# plot 150m coastal HER sites (SMR, IHR, DHR) using consistent symbology
site_handles = plot_her_sites(ax2_5, smr_near_coast, ihr_near_coast, dhr_near_coast, transform=ni_utm, size=4)

# add a custom patch for county boundary (red) and coastal buffer (orange line)
county_outline_patch = mpatches.Patch(facecolor='none', edgecolor='red', label='County Boundaries')
buffer_patch = mpatches.Patch(facecolor='orange', edgecolor='darkorange', label='150m Coastal Buffer', linewidth=0.75)
handles = [county_outline_patch, buffer_patch] + site_handles
labels = ['County Boundaries', '150m Coastal Buffer', 'SMR Sites (150m)', 'IHR Sites (150m)', 'DHR Sites (150m)']

# add title
ax2_5.set_title('HER Sites Within 150m of NI Coastline', fontsize=12, fontweight='bold')

# add legend
leg2_5 = ax2_5.legend(handles, labels, title='Legend', title_fontsize=12,
                  fontsize=10, loc='upper left', frameon=True, framealpha=1)
leg2_5.get_title().set_fontweight('bold')

# add gridlines
gridlines2_5 = ax2_5.gridlines(draw_labels=True,
                           xlocs=[-8, -7.5, -7, -6.5, -6, -5.5],
                           ylocs=[54, 54.5, 55, 55.5])
gridlines2_5.top_labels = False
gridlines2_5.right_labels = False

# add scale bar
scale_bar(ax2_5)

# save the figure
fig2_5.savefig('Figure2_5.png', bbox_inches='tight', dpi=300)

# === 7. Spatial Analysis: Buffer and Clip HER Sites (15m) ===
# apply buffer and clip for 15m
clipped_15m, buffer_gdf_15m = buffer_and_clip(coastline_utm, her_dict, 15)

# extract HER datasets from 15m clipped results
smr_near_coast_15m = clipped_15m['SMR']
ihr_near_coast_15m = clipped_15m['IHR']
dhr_near_coast_15m = clipped_15m['DHR']

print(f'Number of SMR sites (within 15m of coast): {len(smr_near_coast_15m)} features')
print(f'Number of IHR sites (within 15m of coast): {len(ihr_near_coast_15m)} features')
print(f'Number of DHR sites (within 15m of coast): {len(dhr_near_coast_15m)} features')
print('\n')

# use new count_sites_by_county function to perform spatial join and count HER sites per county
smr_15m_county = count_sites_by_county(smr_near_coast_15m, counties_utm, 'SMR', '15m')
ihr_15m_county = count_sites_by_county(ihr_near_coast_15m, counties_utm, 'IHR', '15m')
dhr_15m_county = count_sites_by_county(dhr_near_coast_15m, counties_utm, 'DHR', '15m')

# === 8. Mapping HER Sites Within 15m of the NI Coastline ===
# save clipped 15m coastal HER datasets to shapefiles
smr_near_coast_15m.to_file('data_files/SMR_near_coast_15m.shp')
ihr_near_coast_15m.to_file('data_files/IHR_near_coast_15m.shp')
dhr_near_coast_15m.to_file('data_files/DHR_near_coast_15m.shp')

# figure 3 - neutral basemap on map
# create the map of HER sites across Northern Ireland
fig3 = plt.figure(figsize=(8, 8))
ax3 = plt.axes(projection=ni_utm)

# add NI outline boundary
outline_feature = ShapelyFeature(ni_outline['geometry'], ni_utm, edgecolor='black', facecolor='white')
ax3.add_feature(outline_feature)

# set the map extent
xmin, ymin, xmax, ymax = ni_outline.total_bounds
ax3.set_extent([xmin-5000, xmax+5000, ymin-5000, ymax+5000], crs=ni_utm)

# plot each county with a unique colour fill and black edge
for ii, name in enumerate(county_names):
    feat = ShapelyFeature(counties_utm.loc[counties_utm['CountyName'] == name, 'geometry'],
                          ni_utm,
                          edgecolor='black',
                          facecolor=county_colors[ii],
                          linewidth=1,
                          alpha=0.25)
    ax3.add_feature(feat)

# add water features to the map
water_feat = ShapelyFeature(water['geometry'],
                            ccrs.CRS(water.crs),
                            edgecolor='black',
                            facecolor='white',
                            linewidth=1)
ax3.add_feature(water_feat)

# add the 150m buffer around coastline (as dark orange fill)
buffer_feat = ShapelyFeature(buffer_gdf['geometry'], ni_utm, edgecolor='darkorange', facecolor='none', linewidth=0.75)
ax3.add_feature(buffer_feat)

# plot HER sites (SMR, IHR, DHR) using consistent symbology
site_handles = plot_her_sites(ax3, smr_near_coast_15m, ihr_near_coast_15m, dhr_near_coast_15m, transform=ni_utm, size=4)

# add a custom patch for coastal buffer (orange line)
buffer_patch = mpatches.Patch(facecolor='orange', edgecolor='darkorange', label='15m Coastal Buffer', linewidth=0.75)

# create and add legend
handles = county_handle + [buffer_patch] + site_handles
labels = lc_names + ['15m Coastal Buffer', 'SMR Sites (15m)', 'IHR Sites (15m)', 'DHR Sites (15m)']
leg3 = ax3.legend(handles,labels, title='Legend', title_fontsize=12,
                 fontsize=10, loc='upper left', frameon=True, framealpha=1)
leg3.get_title().set_fontweight('bold')

# add figure title
ax3.set_title('HER Sites Within 15m of NI Coastline', fontsize=12, fontweight='bold')

# add gridlines
gridlines3 = ax3.gridlines(draw_labels=True,
                         xlocs=[-8, -7.5, -7, -6.5, -6, -5.5],
                         ylocs=[54, 54.5, 55, 55.5])
gridlines3.top_labels = False
gridlines3.right_labels = False

# add scale bar to the upper right-hand corner of the map window
scale_bar(ax3)

# save figure
fig3.savefig('Figure3.png', bbox_inches='tight', dpi=300)

# figure 3.5 - NI_Mosaic raster backdrop on map
# create the figure and axes with UTM projection
fig3_5 = plt.figure(figsize=(8, 8))
ax3_5 = plt.axes(projection=ni_utm)

# set the map extent
xmin, ymin, xmax, ymax = ni_outline.total_bounds
ax3_5.set_extent([ni_bounds.left, ni_bounds.right, ni_bounds.bottom, ni_bounds.top], crs=ni_utm)

# create a rectangle that covers the map extent
map_bounds = box(xmin - 5000, ymin - 5000, xmax + 5000, ymax + 5000)

# merge all NI geometries into one shape
ni_outline['geometry'] = ni_outline['geometry'].buffer(0) # fix invalid geometries before union
ni_union = unary_union(ni_outline.geometry)

# subtract NI from the full map extent to get outer mask
mask_geom = map_bounds.difference(ni_union)

# define contrast stretch
stretch_args = {'pmin': 0.1, 'pmax': 99.9}

# call custom raster backdrop function
add_raster_backdrop(ax3_5, ni_img, ni_bounds, ni_transform, ni_utm, mask_geom, stretch_args)

# add county boundaries as red outlines (no fill)
for _, row in counties_utm.iterrows():
    county_feat = ShapelyFeature([row['geometry']], ni_utm,
                                 edgecolor='red', facecolor='none', linewidth=1)
    ax3_5.add_feature(county_feat)

# add the 15m buffer around coastline (as dark orange fill)
buffer_feat = ShapelyFeature(buffer_gdf_15m['geometry'], ni_utm, edgecolor='darkorange', facecolor='none', linewidth=0.75)
ax3_5.add_feature(buffer_feat)

# plot 15m coastal HER sites (SMR, IHR, DHR) using consistent symbology
site_handles = plot_her_sites(ax3_5, smr_near_coast_15m, ihr_near_coast_15m, dhr_near_coast_15m, transform=ni_utm, size=4)

# add a custom patch for county boundary (red) and coastal buffer (orange line)
county_outline_patch = mpatches.Patch(facecolor='none', edgecolor='red', label='County Boundaries')
buffer_patch = mpatches.Patch(facecolor='orange', edgecolor='darkorange', label='15m Coastal Buffer', linewidth=0.75)
handles = [county_outline_patch, buffer_patch] + site_handles
labels = ['County Boundaries', '15m Coastal Buffer', 'SMR Sites (15m)', 'IHR Sites (15m)', 'DHR Sites (15m)']

# add title
ax3_5.set_title('HER Sites Within 15m of NI Coastline', fontsize=12, fontweight='bold')

# add legend
leg3_5 = ax3_5.legend(handles, labels, title='Legend', title_fontsize=12,
                  fontsize=10, loc='upper left', frameon=True, framealpha=1)
leg3_5.get_title().set_fontweight('bold')

# add gridlines
gridlines3_5 = ax3_5.gridlines(draw_labels=True,
                           xlocs=[-8, -7.5, -7, -6.5, -6, -5.5],
                           ylocs=[54, 54.5, 55, 55.5])
gridlines3_5.top_labels = False
gridlines3_5.right_labels = False

# add scale bar
scale_bar(ax3_5)

# save the figure
fig3_5.savefig('Figure3_5.png', bbox_inches='tight', dpi=300)

# === 9. Summary Tables ===
# 9.1. create a summary table of HER site counts (using a dictionary)
summary_data = {
    'Dataset': ['SMR', 'IHR', 'DHR'],
    'Total': [len(smr_data), len(ihr_data), len(dhr_data)],
    '150m': [len(smr_near_coast), len(ihr_near_coast), len(dhr_near_coast)],
    '15m': [len(smr_near_coast_15m), len(ihr_near_coast_15m), len(dhr_near_coast_15m)]
}

# create the dataframe and print it
summary_df = pd.DataFrame(summary_data)

print('\n=== Summary: HER Sites Found Near Coastline ===')
print(summary_df.to_string(index=False))

# save to csv file
summary_df.to_csv('data_files/her_summary_table.csv', index=False)

# 9.2. count the number of HER sites within 150m buffer, grouped by county
smr_150m_count = smr_county['CountyName'].value_counts().rename('SMR (150m)')
ihr_150m_count = ihr_county['CountyName'].value_counts().rename('IHR (150m)')
dhr_150m_count = dhr_county['CountyName'].value_counts().rename('DHR (150m)')

# combine HER counts into a single dataset
# .fillna(0) handles missing values/combinations (e.g. counties with no SMR/IHR/DHR sites) and .astype(int) ensures that the values are integers
county_summary_150m = pd.concat([smr_150m_count, ihr_150m_count, dhr_150m_count], axis=1).fillna(0).astype(int)

# add row totals
county_summary_150m['Total (150m)'] = county_summary_150m.sum(axis=1)

# add percentages of HER types by row (per county)
county_summary_150m['% of SMR'] = round((county_summary_150m['SMR (150m)'] / county_summary_150m['Total (150m)']) * 100, 2) # 2 saves to 2dp (like .2f)
county_summary_150m['% of IHR'] = round((county_summary_150m['IHR (150m)'] / county_summary_150m['Total (150m)']) * 100, 2)
county_summary_150m['% of DHR'] = round((county_summary_150m['DHR (150m)'] / county_summary_150m['Total (150m)']) * 100, 2)

# reset index to include CountyName as a column
county_summary_150m = county_summary_150m.reset_index().rename(columns={'index': 'CountyName'})

# print data and save to csv file
print("\n=== County Summary: HER Sites Within 150m of Coastline ===")
print(county_summary_150m)
county_summary_150m.to_csv('data_files/county_summary_150m.csv', index=False)

# 9.3. count the number of HER sites within 15m buffer, grouped by county
smr_15m_count = smr_15m_county['CountyName'].value_counts().rename('SMR (15m)')
ihr_15m_count = ihr_15m_county['CountyName'].value_counts().rename('IHR (15m)')
dhr_15m_count = dhr_15m_county['CountyName'].value_counts().rename('DHR (15m)')

# combine HER counts into a single dataset
county_summary_15m = pd.concat([smr_15m_count, ihr_15m_count, dhr_15m_count], axis=1).fillna(0).astype(int)

# add row totals
county_summary_15m['Total (15m)'] = county_summary_15m.sum(axis=1)

# add percentages of HER types by row (per county)
county_summary_15m['% of SMR'] = round((county_summary_15m['SMR (15m)'] / county_summary_15m['Total (15m)']) * 100, 2)
county_summary_15m['% of IHR'] = round((county_summary_15m['IHR (15m)'] / county_summary_15m['Total (15m)']) * 100, 2)
county_summary_15m['% of DHR'] = round((county_summary_15m['DHR (15m)'] / county_summary_15m['Total (15m)']) * 100, 2)

# reset index to turn index into a CountyName column
county_summary_15m = county_summary_15m.reset_index().rename(columns={'index': 'CountyName'})

# print data and save to csv file
print("\n=== County Summary: HER Sites Within 15m of Coastline ===")
print(county_summary_15m)
county_summary_15m.to_csv('data_files/county_summary_15m.csv', index=False)

# === 10. Bar Chart/Plots for 150m/15m HER sites ===
# sort counties by total number of 150m coastal sites
county_data_150m = county_summary_150m.sort_values('Total (150m)', ascending=False)
county_data_150m['CountyName'] = county_data_150m['CountyName'].str.title()

# figure 4 - grouped bar chart for 150m sites
# create a new figure and axes object
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 8))

# plot the bars for each HER category, keeping consistent with HER sites colours
x = np.arange(len(county_data_150m))  # gives each county a numeric position
bar_width = 0.25 # set width of individual bars in the grouped plot
ax4.bar(x - bar_width, county_data_150m['SMR (150m)'], width=bar_width, label='SMR', color='red') # x - bar_width shifts the bar to the left of centre point (x) to avoid overlap
ax4.bar(x, county_data_150m['IHR (150m)'], width=bar_width, label='IHR', color='blue')
ax4.bar(x + bar_width, county_data_150m['DHR (150m)'], width=bar_width, label='DHR', color='green') # x + bar_width shifts the bar to the right of centre point to avoid overlap

# format the axis
ax4.set_xlabel('County', fontweight='bold')
ax4.set_ylabel('Number of HER Sites', fontweight='bold')
ax4.set_title('HER Sites Within 150m of Coastline by County', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(county_data_150m['CountyName'], rotation=45) # rotates labels by 45 degrees to avoid overlap
ax4.legend(title='HER Types').get_title().set_fontweight('bold') # makes the legend title bold

# save the figure
fig4.savefig('Figure4.png', bbox_inches='tight', dpi=300)

# sort counties by total number of 15m coastal sites
county_data_15m = county_summary_15m.sort_values('Total (15m)', ascending=False)
county_data_15m['CountyName'] = county_data_15m['CountyName'].str.title()

# figure 5 - grouped bar chart for 15m sites
# create a new figure and axes object
fig5, ax5 = plt.subplots(1, 1, figsize=(8, 8))

# plot the bars for each HER category
x = np.arange(len(county_data_15m))
bar_width = 0.25
ax5.bar(x - bar_width, county_data_15m['SMR (15m)'], width=bar_width, label='SMR', color='red')
ax5.bar(x, county_data_15m['IHR (15m)'], width=bar_width, label='IHR', color='blue')
ax5.bar(x + bar_width, county_data_15m['DHR (15m)'], width=bar_width, label='DHR', color='green')

# format the axis
ax5.set_xlabel('County', fontweight='bold')
ax5.set_ylabel('Number of HER Sites', fontweight='bold')
ax5.set_title('HER Sites Within 15m of Coastline by County', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(county_data_15m['CountyName'], rotation=45)
ax5.legend(title='HER Types').get_title().set_fontweight('bold')

# save the figure
fig5.savefig('Figure5.png', bbox_inches='tight', dpi=300)