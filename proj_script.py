import os
import geopandas as gpd
import matplotlib.pyplot as plt
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import matplotlib.patches as mpatches

# open HER data shapefiles
smr_data = gpd.read_file('data_files/Sites_and_Monuments_Record_13Mar2025.shp')
ihr_data = gpd.read_file('data_files/Industrial_Heritage_Record_13Mar2025.shp')
dhr_data = gpd.read_file('data_files/Defence_Heritage_Record_13Mar2025.shp')

# open NI Outline, Counties and Water shapefiles
ni_outline = gpd.read_file(os.path.abspath('data_files/NI_outline.shp'))
counties = gpd.read_file(os.path.abspath('data_files/Counties.shp'))
water = gpd.read_file(os.path.abspath('data_files/Water.shp'))

# check the CRS of each dataset
print(smr_data.crs)
print(ihr_data.crs)
print(dhr_data.crs)
print(ni_outline.crs)
print(counties.crs)
print(water.crs)

# create a Universal Transverse Mercator reference system to transform our data
ni_utm = ccrs.UTM(29)

# reproject datasets to UTM Zone 29 - SMR/IHR/DHR and Counties
smr_utm = smr_data.to_crs(epsg=32629)
ihr_utm = ihr_data.to_crs(epsg=32629)
dhr_utm = dhr_data.to_crs(epsg=32629)
counties_utm = counties.to_crs(epsg=32629)
water_utm = water.to_crs(epsg=32629)

# recheck the CRS of each dataset
print(smr_utm.crs)
print(ihr_utm.crs)
print(dhr_utm.crs)
print(ni_outline.crs)
print(counties_utm.crs)
print(water_utm.crs)

# counts the number of rows within each table
print('Number of SMR features: {}'.format(len(smr_data)))
print('Number of IHR features: {}'.format(len(ihr_data)))
print('Number of DHR features: {}'.format(len(dhr_data)))

# helper functions from mapping with cartopy practical are used here
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

    ax.text(sbx, sby-(length/4)*1000, f"{length} km", ha='center', transform=ax.projection, fontsize=6) # add a label on the right side
    ax.text(sbx-(length/2)*1000, sby-(length/4)*1000, f"{int(length/2)} km", ha='center', transform=ax.projection, fontsize=6) # add a label in the center
    ax.text(sbx-length*1000, sby-(length/4)*1000, '0 km', ha='center', transform=ax.projection, fontsize=6) # add a label on the left side

    return ax

# create the map
fig = plt.figure(figsize=(8, 8))  # create a figure of size 8x8 (representing the page size in inches)
ax = plt.axes(projection=ni_utm)  # create axes object in figure using UTM projection

# first, we just add the outline of Northern Ireland using cartopy's ShapelyFeature
outline_feature = ShapelyFeature(ni_outline['geometry'], ni_utm, edgecolor='k', facecolor='w')
ax.add_feature(outline_feature)

# set the extent of the map to be fixed on our shapefile feature boundary
xmin, ymin, xmax, ymax = ni_outline.total_bounds
ax.set_extent([xmin-5000, xmax+5000, ymin-5000, ymax+5000], crs=ni_utm)

# choose appropriate colours for the individual county boundaries
county_colors = ['orchid', 'gold', 'seagreen', 'firebrick', 'darkorange', 'y']

# get a list of unique names for county boundaries
county_names = list(counties_utm.CountyName.unique())
county_names.sort() # sort the counties alphabetically by name

# next, add the municipal outlines to the map using the colors that we've picked.
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
                            facecolor='none', # set the facecolor to have no fill
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

plt.show()

# save figure
fig.savefig('Figure1.png', bbox_inches='tight', dpi=300)