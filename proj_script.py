# === 1. Imports ===
import os
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import matplotlib.patches as mpatches

# === 2. Function Definitions ===
# first two are helper functions - used later in script for visual styling and spatial analysis
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
    lc = len(colors)  # get the length of the color list
    handles = [] # create an empty list
    for ii in range(len(labels)): # for each label and color pair that we're given, make an empty box to pass to our legend
        handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[ii % lc], edgecolor=edge, alpha=alpha))
    return handles

# adapted this question: https://stackoverflow.com/q/32333870
# answered by SO user Siyh: https://stackoverflow.com/a/35705477
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

# new function to count the HER sites per county
def count_sites_by_county(sites_gdf, counties_gdf, label, distance_label='150m'):
    """
    Perform a spatial join to count how many HER sites fall within each county.

    Parameters
    ----------
    sites_gdf : GeoDataFrame
        The clipped HER site GeoDataFrame (e.g. smr_near_coast)

    counties_gdf : GeoDataFrame
        The county boundaries GeoDataFrame (e.g. counties_utm)

    label : str
        The name of the HER dataset for printing output (e.g. 'SMR')

    distance_label : str (default: '150m')
        A label for the buffer distance (e.g. '150m', '15m') to include in the print statement

    Returns
    -------
    GeoDataFrame
        The joined dataset with county info
    """
    joined = gpd.sjoin(sites_gdf, counties_gdf, how='inner', predicate='within')
    print(f'{label} sites per county that are within {distance_label} of the coast:')
    print(joined['CountyName'].value_counts())
    print('\n')
    return joined

# === 3. Load and Reproject Data ===
# open HER data shapefiles
smr_data = gpd.read_file('data_files/Sites_and_Monuments_Record_13Mar2025.shp')
ihr_data = gpd.read_file('data_files/Industrial_Heritage_Record_13Mar2025.shp')
dhr_data = gpd.read_file('data_files/Defence_Heritage_Record_13Mar2025.shp')

# open NI Outline, Counties and Water shapefiles
ni_outline = gpd.read_file(os.path.abspath('data_files/NI_outline.shp'))
counties = gpd.read_file(os.path.abspath('data_files/Counties.shp'))
water = gpd.read_file(os.path.abspath('data_files/Water.shp'))
coastline = gpd.read_file(os.path.abspath('data_files/2021_NI_Coastal_Survey.shp'))

# check the CRS of each dataset
print(f'SMR: {smr_data.crs}\nIHR: {ihr_data.crs}\nDHR: {dhr_data.crs}')
print(f'NI Outline: {ni_outline.crs}\nCounties: {counties.crs}\nWater: {water.crs}\nCoastline: {coastline.crs}\n')
print('\n')

# create a Universal Transverse Mercator reference system to transform our data
ni_utm = ccrs.UTM(29)

# reproject datasets to UTM Zone 29 - SMR/IHR/DHR and Counties
smr_utm = smr_data.to_crs(epsg=32629)
ihr_utm = ihr_data.to_crs(epsg=32629)
dhr_utm = dhr_data.to_crs(epsg=32629)
counties_utm = counties.to_crs(epsg=32629)
water_utm = water.to_crs(epsg=32629)
coastline_utm = coastline.to_crs(epsg=32629)

# recheck the CRS of each dataset
print(f'SMR: {smr_utm.crs}\nIHR: {ihr_utm.crs}\nDHR: {dhr_utm.crs}')
print(f'NI Outline: {ni_outline.crs}\nCounties: {counties_utm.crs}\nWater: {water_utm.crs}\nCoastline: {coastline_utm.crs}\n')
print('\n')

# counts the number of rows within each table
print('Number of SMR features: {}'.format(len(smr_data)))
print('Number of IHR features: {}'.format(len(ihr_data)))
print('Number of DHR features: {}'.format(len(dhr_data)))
print('\n')

# === 4. Mapping All HER Sites ===
# create the map
fig = plt.figure(figsize=(8, 8))  # create a figure of size 8x8 (representing the page size in inches)
ax = plt.axes(projection=ni_utm)  # create axes object in figure using UTM projection

# first, we just add the outline of Northern Ireland using cartopy's ShapelyFeature
outline_feature = ShapelyFeature(ni_outline['geometry'], ni_utm, edgecolor='black', facecolor='white')
ax.add_feature(outline_feature)

# set the extent of the map to be fixed on our shapefile feature boundary
xmin, ymin, xmax, ymax = ni_outline.total_bounds
ax.set_extent([xmin-5000, xmax+5000, ymin-5000, ymax+5000], crs=ni_utm)

# choose appropriate colours for the individual county boundaries
county_colors = ['orchid', 'gold', 'seagreen', 'firebrick', 'darkorange', 'y']

# get a list of unique names for county boundaries
county_names = list(counties_utm.CountyName.unique())
county_names.sort() # sort the counties alphabetically by name

# next, add the counties to the map using the colors that we've picked
for ii, name in enumerate(county_names):
    feat = ShapelyFeature(counties_utm.loc[counties_utm['CountyName'] == name, 'geometry'], # first argument is the geometry
                          ni_utm, # second argument is the CRS
                          edgecolor='black', # outline the feature in red
                          facecolor=county_colors[ii], # set the face color to the corresponding color from the list
                          linewidth=1, # set the outline width to be 1 pt
                          alpha=0.25) # set the alpha (transparency) to be 0.25 (out of 1)
    ax.add_feature(feat) # once we have created the feature, we have to add it to the map using ax.add_feature()

# add water features to the map
water_feat = ShapelyFeature(water['geometry'], # first argument is the geometry
                            ccrs.CRS(water.crs), # second argument is the CRS
                            edgecolor='black', # set the edgecolor to be black
                            facecolor='white', # set the facecolor to be white
                            linewidth=1) # set the outline width to be 1 pt
ax.add_feature(water_feat)

# plot SMR (red squares), IHR (blue triangles) and DHR (green circles) site points differently on map
# transform ni_utm is stated to ensure the coordinates match that of the map's CRS
# mec and mew are shortened from markeredgecolor and markeredgewidth respectively: https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html
smr_handle = ax.plot(smr_utm.geometry.x, smr_utm.geometry.y, 's', color='red', mec='black', mew=0.5, ms=3, transform=ni_utm)
ihr_handle = ax.plot(ihr_utm.geometry.x, ihr_utm.geometry.y, '^', color='blue', mec='black', mew=0.5, ms=3, transform=ni_utm)
dhr_handle = ax.plot(dhr_utm.geometry.x, dhr_utm.geometry.y, 'o', color='green', mec='black', mew=0.5, ms=3, transform=ni_utm)

# generate a list of handles for the county datasets
# first, we add the list of names, then the list of colors, and finally we set the transparency
# (since we set it in the map)
county_handle = generate_handles(counties_utm.CountyName.unique(), county_colors, alpha=0.25)

# update county_names to take it out of uppercase text
lc_names = [name.title() for name in county_names]

# add handles and labels information to be passed to ax.legend()
handles = county_handle + smr_handle + ihr_handle + dhr_handle
labels = lc_names + ['SMR Sites', 'IHR Sites', 'DHR Sites']

# add title to figure
ax.set_title('HER Sites Across Northern Ireland', fontsize=12, fontweight='bold')

# add legend
leg = ax.legend(handles,labels, title='Legend', title_fontsize=12,
                 fontsize=10, loc='upper left', frameon=True, framealpha=1)
leg.get_title().set_fontweight('bold') # make the legend title bold

#add gridlines
gridlines = ax.gridlines(draw_labels=True, # draw  labels for the grid lines
                         xlocs=[-8, -7.5, -7, -6.5, -6, -5.5], # add longitude lines at 0.5 deg intervals
                         ylocs=[54, 54.5, 55, 55.5]) # add latitude lines at 0.5 deg intervals
gridlines.top_labels = False # turn off the top labels
gridlines.right_labels = False # turn off the right-side labels

# add a scale bar to the upper right-hand corner of the map window
scale_bar(ax) # scale bar will be created with 10km divisions

# save figure
fig.savefig('Figure1.png', bbox_inches='tight', dpi=300)

# === 5. Spatial Analysis: Coastal Buffer and Clipping ===
# create a 150 m buffer around the NI coastline
ni_buffer = coastline_utm.buffer(150)
buffer_gdf = gpd.GeoDataFrame(geometry=ni_buffer)
buffer_gdf.set_crs(coastline_utm.crs, inplace=True)

# clip HER datasets (SMR, IHR, DHR) to the buffer
smr_near_coast = gpd.clip(smr_utm, buffer_gdf)
ihr_near_coast = gpd.clip(ihr_utm, buffer_gdf)
dhr_near_coast = gpd.clip(dhr_utm, buffer_gdf)

print(f'Number of SMR sites (within 150m of coast): {len(smr_near_coast)} features')
print(f'Number of IHR sites (within 150m of coast): {len(ihr_near_coast)} features')
print(f'Number of DHR sites (within 150m of coast): {len(dhr_near_coast)} features')
print('\n')

# use new count_sites_by_county function to perform spatial join and count HER sites per county
smr_county = count_sites_by_county(smr_near_coast, counties_utm, 'SMR', '150m')
ihr_county = count_sites_by_county(ihr_near_coast, counties_utm, 'IHR', '150m')
dhr_county = count_sites_by_county(dhr_near_coast, counties_utm, 'DHR', '150m')

# === 6. Mapping HER Sites Within 150m of the NI Coastline ===
# save clipped HER datasets to shapefiles
smr_near_coast.to_file('data_files/SMR_near_coast.shp')
ihr_near_coast.to_file('data_files/IHR_near_coast.shp')
dhr_near_coast.to_file('data_files/DHR_near_coast.shp')

# create the figure and axes with UTM projection ---
fig2 = plt.figure(figsize=(8, 8))
ax2 = plt.axes(projection=ni_utm)

# add the NI outline
outline_feature2 = ShapelyFeature(ni_outline['geometry'], ni_utm, edgecolor='black', facecolor='none')
ax2.add_feature(outline_feature2)

# set the map extent
xmin, ymin, xmax, ymax = ni_outline.total_bounds
ax2.set_extent([xmin-5000, xmax+5000, ymin-5000, ymax+5000], crs=ni_utm)

# add county polygons with the same color scheme
for ii, name in enumerate(county_names):
    county_feat = ShapelyFeature(counties_utm.loc[counties_utm['CountyName'] == name, 'geometry'],
                                 ni_utm,
                                 edgecolor='black',
                                 facecolor=county_colors[ii],
                                 linewidth=1,
                                 alpha=0.25)
    ax2.add_feature(county_feat)

# add water features
water_feat2 = ShapelyFeature(water_utm['geometry'], ni_utm, edgecolor='black', facecolor='white', linewidth=1)
ax2.add_feature(water_feat2)

# add the 150m buffer around coastline (as transparent dark orange fill)
buffer_feat = ShapelyFeature(buffer_gdf['geometry'], ni_utm, edgecolor='darkorange', facecolor='none', linewidth=1)
ax2.add_feature(buffer_feat)

# plot the clipped HER sites (coastal only)
smr_handle2 = ax2.plot(smr_near_coast.geometry.x, smr_near_coast.geometry.y, 's',
                       color='red', mec='black', mew=0.5, ms=3, transform=ni_utm)
ihr_handle2 = ax2.plot(ihr_near_coast.geometry.x, ihr_near_coast.geometry.y, '^',
                       color='blue', mec='black', mew=0.5, ms=3, transform=ni_utm)
dhr_handle2 = ax2.plot(dhr_near_coast.geometry.x, dhr_near_coast.geometry.y, 'o',
                       color='green', mec='black', mew=0.5, ms=3, transform=ni_utm)

# generate handles for counties (using same colors)
county_handles2 = generate_handles(counties_utm.CountyName.unique(), county_colors, alpha=0.25)
lc_names2 = [name.title() for name in county_names]

# add legend
handles2 = county_handles2 + smr_handle2 + ihr_handle2 + dhr_handle2
labels2 = lc_names2 + ['SMR Sites (150m from coast)', 'IHR Sites (150m from coast)', 'DHR Sites (150m from coast)']

leg2 = ax2.legend(handles2, labels2, title='Legend', title_fontsize=12,
                  fontsize=10, loc='upper left', frameon=True, framealpha=1)
leg2.get_title().set_fontweight('bold')

# add title
ax2.set_title('HER Sites Within 150m of NI Coastline', fontsize=12, fontweight='bold')

# add gridlines
gridlines2 = ax2.gridlines(draw_labels=True,
                           xlocs=[-8, -7.5, -7, -6.5, -6, -5.5],
                           ylocs=[54, 54.5, 55, 55.5])
gridlines2.top_labels = False
gridlines2.right_labels = False

# add scale bar
scale_bar(ax2)

# save the figure
fig2.savefig('Figure2.png', bbox_inches='tight', dpi=300)

# === 7. Spatial Analysis: Coastal Buffer and Clipping ===
# create a 15 m buffer around the NI coastline
ni_buffer_15m = coastline_utm.buffer(15)
buffer_gdf_15m = gpd.GeoDataFrame(geometry=ni_buffer_15m)
buffer_gdf_15m.set_crs(coastline_utm.crs, inplace=True)

# clip HER datasets (SMR, IHR, DHR) to the buffer
smr_near_coast_15m = gpd.clip(smr_utm, buffer_gdf_15m)
ihr_near_coast_15m = gpd.clip(ihr_utm, buffer_gdf_15m)
dhr_near_coast_15m = gpd.clip(dhr_utm, buffer_gdf_15m)

print(f'Number of SMR sites (within 15m of coast): {len(smr_near_coast_15m)} features')
print(f'Number of IHR sites (within 15m of coast): {len(ihr_near_coast_15m)} features')
print(f'Number of DHR sites (within 15m of coast): {len(dhr_near_coast_15m)} features')
print('\n')

# use new count_sites_by_county function to perform spatial join and count HER sites per county
smr_15m_county = count_sites_by_county(smr_near_coast_15m, counties_utm, 'SMR', '15m')
ihr_15m_county = count_sites_by_county(ihr_near_coast_15m, counties_utm, 'IHR', '15m')
dhr_15m_county = count_sites_by_county(dhr_near_coast_15m, counties_utm, 'DHR', '15m')

# === 8. Mapping HER Sites Within 15m of the NI Coastline ===
# save clipped HER datasets to shapefiles
smr_near_coast_15m.to_file('data_files/SMR_near_coast_15m.shp')
ihr_near_coast_15m.to_file('data_files/IHR_near_coast_15m.shp')
dhr_near_coast_15m.to_file('data_files/DHR_near_coast_15m.shp')

# create the figure and axes with UTM projection ---
fig3 = plt.figure(figsize=(8, 8))
ax3 = plt.axes(projection=ni_utm)

# add the NI outline
outline_feature3 = ShapelyFeature(ni_outline['geometry'], ni_utm, edgecolor='black', facecolor='none')
ax3.add_feature(outline_feature3)

# set the map extent
xmin, ymin, xmax, ymax = ni_outline.total_bounds
ax3.set_extent([xmin-5000, xmax+5000, ymin-5000, ymax+5000], crs=ni_utm)

# add county polygons with the same color scheme
for ii, name in enumerate(county_names):
    county_feat = ShapelyFeature(counties_utm.loc[counties_utm['CountyName'] == name, 'geometry'],
                                 ni_utm,
                                 edgecolor='black',
                                 facecolor=county_colors[ii],
                                 linewidth=1,
                                 alpha=0.25)
    ax3.add_feature(county_feat)

# add water features
water_feat3 = ShapelyFeature(water_utm['geometry'], ni_utm, edgecolor='black', facecolor='white', linewidth=1)
ax3.add_feature(water_feat3)

# add the 15m buffer around coastline (as transparent dark orange fill)
buffer_feat = ShapelyFeature(buffer_gdf_15m['geometry'], ni_utm, edgecolor='darkorange', facecolor='none', linewidth=1)
ax3.add_feature(buffer_feat)

# plot the clipped HER sites (15m coastal only)
smr_handle3 = ax3.plot(smr_near_coast_15m.geometry.x, smr_near_coast_15m.geometry.y, 's',
                       color='red', mec='black', mew=0.5, ms=3, transform=ni_utm)
ihr_handle3 = ax3.plot(ihr_near_coast_15m.geometry.x, ihr_near_coast_15m.geometry.y, '^',
                       color='blue', mec='black', mew=0.5, ms=3, transform=ni_utm)
dhr_handle3 = ax3.plot(dhr_near_coast_15m.geometry.x, dhr_near_coast_15m.geometry.y, 'o',
                       color='green', mec='black', mew=0.5, ms=3, transform=ni_utm)

# generate handles for counties (using same colors)
county_handles3 = generate_handles(counties_utm.CountyName.unique(), county_colors, alpha=0.25)
lc_names3 = [name.title() for name in county_names]

# add legend
handles3 = county_handles3 + smr_handle3 + ihr_handle3 + dhr_handle3
labels3 = lc_names3 + ['SMR Sites (15m from coast)', 'IHR Sites (15m from coast)', 'DHR Sites (15m from coast)']

leg3 = ax3.legend(handles3, labels3, title='Legend', title_fontsize=12,
                  fontsize=10, loc='upper left', frameon=True, framealpha=1)
leg3.get_title().set_fontweight('bold')

# add title
ax3.set_title('HER Sites Within 15m of NI Coastline', fontsize=12, fontweight='bold')

# add gridlines
gridlines3 = ax3.gridlines(draw_labels=True,
                           xlocs=[-8, -7.5, -7, -6.5, -6, -5.5],
                           ylocs=[54, 54.5, 55, 55.5])
gridlines3.top_labels = False
gridlines3.right_labels = False

# add scale bar
scale_bar(ax3)

# save the figure
fig3.savefig('Figure3.png', bbox_inches='tight', dpi=300)

# === 9. Summary Table ===
# create a summary table using a dictionary
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