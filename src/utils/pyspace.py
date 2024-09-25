import pysal as ps
from matplotlib import colors
import esda
import libpysal as lps
import numpy as np
from esda.moran import Moran_Local
from esda.getisord import G_Local
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from splot.esda import lisa_cluster
from libpysal.weights.distance import get_points_array
from scipy.spatial import cKDTree
from splot.esda import plot_local_autocorrelation
from pointpats import PointPattern, PoissonPointProcess
from matplotlib_scalebar.scalebar import ScaleBar
from esda import Moran
from esda import Moran_Rate
from splot.esda import moran_scatterplot
from splot.esda import plot_moran
from esda import Moran_Local_Rate
from splot.esda import plot_local_autocorrelation
import seaborn as sns
from geofeather import to_geofeather, from_geofeather
from matplotlib import patches, colors
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.offline as pyo
from matplotlib.patches import Circle
np.random.seed(42)

def make_gdf(df, crs, x, y):
    """Takes a DataFrame as input and returns a GeoDataFrame
     df: Pandas DataFrame
     crs: Projected Coordinate Reference Systems
     x : Projected coordinates or longitude
     y : Projected coordinates or latitude"""
    geometry = [Point(xy) for xy in zip(df[x], df[y])]
    return gpd.GeoDataFrame(df, crs=crs, geometry=geometry)


def get_distanceBandW(df, distance, lon, lat, transform='r'):
    """One liner to get fast distance band weights calculation.
    `get_points_array` function: This function extracts the coordinates of all vertices 
    for a variety of geometry packages in Python and returns a `numpy` array.
    
    Then, we must build the `KDTree` using `scipy`. For nearly any application, the `cKDTree` will be faster. 
    `KDTree` is an implementation of the datastructure in pure Python, whereas the `cKDTree` is 
    an implementation in Cython."""
    pp = PointPattern(df[[lon, lat]])
    df['nnd'] = pp.nnd
    df = df[df.nnd <= distance]
    weight = lps.weights.DistanceBand(cKDTree(get_points_array(df.geometry)), distance)
    weight.transform = transform
    return df.copy(), weight


def add_random_noise(db,lon,lat):
    db[lon] = db[lon] + np.random.randint(0, 100000.0, db.shape[0]) / 100000.0
    db[lat] = db[lat] - np.random.randint(0, 100000.0, db.shape[0]) / 100000.0
    geometry = [Point(xy) for xy in zip(db[lon], db[lat])]
    crs = 'epsg:2056'
    db = gpd.GeoDataFrame(db, crs=crs, geometry=geometry)
    return db.copy()


def compute_lisa(df, col, w, n_perm, p_value):
    li = Moran_Local(df[col], w, n_jobs = -1, seed = 42, permutations=n_perm)
    sig = li.p_sim < p_value
    hotspot, lowhigh, coldspot, highlow = 1 * (sig * li.q == 1), 2 * (sig * li.q == 2), 3 * (sig * li.q == 3), 4 * (
                sig * li.q == 4)
    spots = hotspot + coldspot + lowhigh + highlow
    spot_labels = ['0 Non-Significant', '1 High-High', '2 Low-High', '3 Low-Low', '4 High-Low']
    labels = [spot_labels[i] for i in spots]
    # hmap = colors.ListedColormap(['white', 'red', 'lightblue', 'blue', 'pink'])
    classes = '{}_cl'.format(col)
    Is = '{}_Is'.format(col)
    z_sim = '{}_Zs'.format(col)
    p_sim = '{}_psim'.format(col)
    cat = '{}_cat'.format(col)
    df[classes] = labels
    df[cat] = li.q
    df[Is] = li.Is
    df[z_sim] = li.z_sim
    df[p_sim] = li.p_sim
    print('*' * 30, " Done ", '*' * 30)
    return li


def compute_lisa_rate(df, col, e, b, w, n_permut, p_value, adjusted=True, transform_type='r'):
    li = Moran_Local_Rate(e, b, w, adjusted=adjusted, transformation=transform_type, permutations=n_permut, n_jobs=-1, seed = 42)
    sig = li.p_sim < p_value
    hotspot, lowhigh, coldspot, highlow = 1 * (sig * li.q == 1), 2 * (sig * li.q == 2), 3 * (sig * li.q == 3), 4 * (
                sig * li.q == 4)
    spots = hotspot + coldspot + lowhigh + highlow
    spot_labels = ['0 Non-Significant', '1 High-High', '2 Low-High', '3 Low-Low', '4 High-Low']
    labels = [spot_labels[i] for i in spots]
    # hmap = colors.ListedColormap(['white', 'red', 'lightblue', 'blue', 'pink'])
    classes = '{}_cl'.format(col)
    Is = '{}_Is'.format(col)
    z_sim = '{}_Zs'.format(col)
    p_sim = '{}_psim'.format(col)
    cat = '{}_cat'.format(col)
    df[col] = li.y
    df[classes] = labels
    df[cat] = li.q
    df[Is] = li.Is
    df[z_sim] = li.z_sim
    df[p_sim] = li.p_sim
    print('*' * 30, " Done ", '*' * 30)
    return li


def compute_getis(df, col, w, n_permut,transform_type = 'R', star=True, p_001=False):
    '''
        df: a DataFrame containing data on which the Getis Ord Gi analysis will be performed.
        col: a string representing the column in the DataFrame df that will be used for the analysis.
        w: a weight matrix or weight list for the spatial weights. It is used to assign weights to each observation, which can affect the outcome of the analysis.
        n_permut: an integer representing the number of permutations to be used when calculating the p-value.
        transform_type: a string, representing the type of transformation that will be applied to the data. It has a default value of 'R', meaning that the data will be row-standardized. If the value is 'B', the data will be transformed to binary.
        star: a Boolean, representing whether to use the "star" option in the analysis. Default value is True.
        p_001: a Boolean, indicating whether the p-value threshold should be set to 0.001. Default value is False.
        The function first prints out a message indicating that the Getis Ord Gi analysis is running for the specified variable and number of permutations. Then it prints out the type of transformation that will be applied to the data.

        Next, it creates an instance of G_Local class from esda package, passing the appropriate parameters, then it creates several new columns in the dataframe and assigns the results of the analysis to the new columns.

        The function creates the following new columns:

        {col}_G_cl: a column containing the classification of each observation as a hot spot, cold spot, or not significant.
        {col}_G_Zs: a column containing the Z-scores for each observation.
        {col}_G_psim: a column containing the p-values for each observation.
        The function then uses boolean indexing to classify each observation according to the p-value and Z-score. The classification is based on the following thresholds:

        Hot Spot - p < 0.1: Z-score > 0 and p-value < 0.1
        Hot Spot - p < 0.05: Z-score > 0 and p-value < 0.05
        Hot Spot - p < 0.01: Z-score > 0 and p-value < 0.01
        Hot Spot - p < 0.001: Z-score > 0 and p-value < 0.001 (if the p_001 parameter is True)
        Cold Spot - p < 0.1: Z-score < 0 and p-value < 0.1
        Cold Spot - p < 0.05: Z-score < 0 and p-value < 0.05
        Cold Spot - p < 0.01: Z-score < 0 and p-value < 0.01
        Cold Spot - p < 0.001: Z-score < 0 and p-value < 0.001 (if the p_001 parameter is True)
        Not significant: p-value >= 0.1
        Lastly the function prints out a message indicating that the analysis is done, and it returns the instance of G_Local class created earlier.
    '''
    print("Computing Getis Ord Gi statistic")
    g = G_Local(df[col], w, transform = transform_type, n_jobs=-1, star = star, seed=42, permutations = n_permut)

    classes = f"{col}_G_cl"
    z_sim = f"{col}_G_Zs"
    p_sim = f"{col}_G_psim"
    df[z_sim] = g.Zs
    df[p_sim] = g.p_sim

    significance_levels = [0.1, 0.05, 0.01]
    if p_001:
        significance_levels.append(0.001)

    for level in significance_levels:
        df.loc[(df[p_sim] < level) & (df[z_sim] > 0), classes] = f"Hot Spot - p < {level}"
        df.loc[(df[p_sim] < level) & (df[z_sim] < 0), classes] = f"Cold Spot - p < {level}"
    df.loc[(df[p_sim] >= 0.1), classes] = 'Not significant'

    return g


def plot_lisa_by_class(df, x, y, y_label,showfliers = False):
    """
    This function is used to plot a boxplot of a specified column (y) of a pandas dataframe (df) grouped by the values in another specified column (x) using the seaborn library.

    The plot_lisa_by_class function takes four required arguments:

    df: A pandas dataframe that contains the data to be plotted.
    x: The name of the column in the dataframe to use for grouping the data. This column should contain categorical values that correspond to the different classes of a LISA (Local Indicator of Spatial Association) analysis.
    y: The name of the column in the dataframe to use for plotting the values. This column should contain numerical values.
    y_label: The label to use for the y-axis of the plot.
    It also has an optional argument showfliers, which is a boolean value that controls whether or not outliers are shown on the plot. By default, this argument is set to False which means outliers will not be shown.

    Returns the figure and axis objects, which can be used for further customization of the plot if desired.

    """
    sns.set_context("paper")
    fig, ax = plt.subplots(figsize=(8, 8))
    x_label = 'Classes LISA'
    chart = sns.boxplot(x=x, y=y, order=['0 Non-Significant', '1 High-High', '2 Low-High', '3 Low-Low', '4 High-Low'],
                        showfliers=showfliers, palette=colors_cl, data=df, ax=ax)
    chart.set_xticklabels(chart.get_xticklabels(), size=12, rotation=45, horizontalalignment='right')
    chart.set_xlabel(x_label, size=14)
    chart.set_ylabel(y_label, size=14)
    return fig, ax


def plot_getis_by_class(df, x, y, label, xtick_size, title_size, xlabel_size, ylabel_size, p_001=False,showfliers = False, boxplot=False):
    """
    The function plot_getis_by_class creates a box plot using the Seaborn library to visualize the distribution of values in a column of a DataFrame. The plot will categorize the data into Getis Classes based on another column of the DataFrame.

    Parameters:
    df (pandas.DataFrame) : The dataframe to be used for the plot.
    x (str) : The column in the DataFrame to be used for categorizing the data into Getis Classes.
    y (str) : The column in the DataFrame to be used for plotting the data.
    label (str) : The label to be used for the y-axis in the plot.
    xtick_size (int) : The font size to be used for the x-tick labels in the plot.
    title_size (int) : The font size to be used for the title in the plot.
    xlabel_size (int) : The font size to be used for the x-axis label in the plot.
    ylabel_size (int) : The font size to be used for the y-axis label in the plot.
    p_001 (bool, optional) : If True, additional colors will be used for plotting the data. Defaults to False.
    showfliers (bool, optional) : If True, outliers will be plotted in the plot. Defaults to False.

    Returns:
    fig, ax : A tuple of the figure and axis objects for the plot.
    """
    sns.set_context("paper")
    fig, ax = plt.subplots(figsize=(5, 5))
    # colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#b2182b', '#ef8a62', '#fddbc7', '#f7f7f7']
    colors_cl_getis = {
        'Cold Spot - p < 0.1': '#d1e5f0',
        'Cold Spot - p < 0.05': '#67a9cf',
        'Cold Spot - p < 0.01': '#2166ac',
        'Hot Spot - p < 0.1': '#fddbc7',
        'Hot Spot - p < 0.05': '#ef8a62',
        'Hot Spot - p < 0.01': '#c70820',
        'Not significant': '#bdbdbd'
    }
    order = ['Cold Spot - p < 0.01', 'Cold Spot - p < 0.05', 'Cold Spot - p < 0.1', 'Hot Spot - p < 0.01',
    'Hot Spot - p < 0.05', 'Hot Spot - p < 0.1', 'Not significant']
    # Add p < 0.001 colors if required
    if p_001:
        colors_cl_getis['Cold Spot - p < 0.001'] = '#1c4978'
        colors_cl_getis['Hot Spot - p < 0.001'] = '#991d2c'

        # Ordering colors according to the unique sorted values in the column
        order = ['Cold Spot - p < 0.001', 'Cold Spot - p < 0.01', 'Cold Spot - p < 0.05', 'Cold Spot - p < 0.1', 'Hot Spot - p < 0.001', 'Hot Spot - p < 0.01',
 'Hot Spot - p < 0.05', 'Hot Spot - p < 0.1', 'Not significant']
    colors_plot = [colors_cl_getis[i] for i in df[x].sort_values().unique()]

    ylim_max = df[y].quantile(q=0.99)
    ylim_max *= 1.25
    x_label = 'Getis Classes'
    if boxplot :
        chart = sns.boxplot(x=x, y=y, order=order,
                    palette=colors_plot, data=df, showfliers=False,zorder=1, ax=ax)
    else :
        chart = sns.barplot(x=x, y=y, order=order,
                            palette=colors_plot, data=df,zorder=1, ax=ax)
    chart.set_xticklabels(chart.get_xticklabels(), size=xtick_size, rotation=45, horizontalalignment='right')
    chart.set_title('{} by Getis Class'.format(label, size=title_size))
    chart.set_xlabel(x_label, size=xlabel_size)
    chart.set_ylabel(label, size=ylabel_size)
    # plt.grid(axis = 'y',zorder=0)

    # chart.set_ylim([0, ylim_max])
    return fig, ax



def get_n_neighbors(df, w):
    for key, neighbors in w:
        df.at[key, 'neighbors'] = len(neighbors)


def fdr_correction(df, pcol, alpha):
    """
    Perform FDR correction on a dataframe's p-value column.
    Parameters:
    df (pandas DataFrame): DataFrame containing the p-value column.
    pcol (str): Name of the p-value column.
    alpha (float): Significance level.

    Returns:
    df (pandas DataFrame): DataFrame with an additional column containing the FDR-corrected p-values.
    """
    df = df.sort_values(pcol)
    df = df.reset_index(drop=True)
    df.index += 1
    df['I'] = df.index
    fdr_col = 'fdr_' + pcol
    df[fdr_col] = df['I'] * alpha / df.shape[0]
    return df.copy()

def plotGetisMap_ge(db, col, p_001=False,markersize_s=1, markersize_l=5, commune_name=False):
    """
    This function creates a Getis map and plots it using Matplotlib and GeoPandas.

    Inputs:

    db: a GeoDataFrame with the locations to plot.
    col: the column name of the class column that maps the observations to their class.
    p_001: a boolean indicating whether to include 'p < 0.001' classes (default False).
    commune_name: a boolean indicating whether to include the name of the communes in the plot (default False).
    Output:
    A plot of the Getis map with each observation colored according to its class.

    The function reads in shapefiles for the cantons and communes of Switzerland and converts them to the Swiss coordinate system (2056).
    It then uses the 'intersects' predicate to join the input data with the cantons and communes shapefiles.
    The function also reads in a shapefile for Lake Geneva.

    The function uses Matplotlib and GeoPandas to plot the map. The class of each observation is indicated by a color, which is specified in a dictionary. The size of the marker for each observation can be modified, with significant observations having a larger marker size.

    Finally, if 'commune_name' is True, the names of the communes will be annotated on the plot.
    """
    print('Plot Getis Map')
    fig, ax = plt.subplots(1, figsize=(10, 10))
    y = db[col] + ' [' + db[col].map(db[col].value_counts()).astype(str) + ']'
    # y = db[col] + ' (' + db[col].map(db.groupby(col)[col.split('_G_cl')[0]].mean().mul(100).round(1)).astype(str) + '%)'


    colors_cl_getis = {
        'Cold Spot - p < 0.1': '#d1e5f0',
        'Cold Spot - p < 0.05': '#67a9cf',
        'Cold Spot - p < 0.01': '#2166ac',
        'Hot Spot - p < 0.1': '#fddbc7',
        'Hot Spot - p < 0.05': '#ef8a62',
        'Hot Spot - p < 0.01': '#b2182b',
        'Not significant': '#bdbdbd'
    }

    # Add p < 0.001 colors if required
    if p_001:
        colors_cl_getis['Cold Spot - p < 0.001'] = '#1c4978'
        colors_cl_getis['Hot Spot - p < 0.001'] = '#991d2c'

    # Ordering colors according to the unique sorted values in the column
    hmap = colors.ListedColormap([colors_cl_getis[i] for i in db[col].sort_values().unique()])


    cantons = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.shp', engine='pyogrio')

    communes = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp', engine='pyogrio')

    cantons, communes = cantons.to_crs(2056), communes.to_crs(2056)
    cantons_to_map = cantons[cantons.NAME == 'Genève']
    communes = communes[communes.KANTONSNUM.isin(cantons_to_map.KANTONSNUM)]
    lake = gpd.read_file('/Users/david/Dropbox/PhD/GitHub/COVID19/Data/Mapping/lake.geojson', engine='pyogrio')
    lake.NOM = ['Lake Geneva', '', '']
    lake = lake.to_crs(2056)

    db['markersize'] = markersize_s
    db.loc[db[col] != 'Not significant', 'markersize'] = markersize_l
    ## Plotting
    cantons_to_map.geometry.boundary.plot(ax=ax, edgecolor='k', color=None, linewidth=0.3)
    communes.plot(ax=ax, label='Communes', alpha=1, color=None, edgecolor='#636363', facecolor='white', linewidth=0.1)
    if commune_name:
        communes[communes.NAME.isin(['Genève','Bâle','Zürich','Basel','Lausanne','Bern'])].apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', size=8),
                       axis=1)
    lake.plot(ax=ax, label='Lake', alpha=1, color='lightgrey', zorder = 5)
    db.plot(y, cmap=hmap, markersize='markersize', linewidth = 0.01, ax=ax, legend=True, categorical=True, zorder=6)
    legend = ax.get_legend()
    legend.set_bbox_to_anchor((0.03, 0.8, 0.2, 0.2))
    ax.set_axis_off()
    ax.set_facecolor('grey')

    lake.apply(lambda x: ax.annotate(text=x.NOM, xy=x.geometry.centroid.coords[0], ha='center', size=12, zorder = 6), axis=1)
    # add scale bar
    scalebar = ScaleBar(1, units="m", location="lower right")
    ax.add_artist(scalebar)
    ax.set_axis_off()
    x, y, arrow_length = 0.90, 0.15, 0.08
    ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20,
                xycoords=ax.transAxes)
    return fig, ax

def plotGetisMap(db, col, p_001=False,markersize_s=1, markersize_l=5, commune_name=False):
    """
    This function creates a Getis map and plots it using Matplotlib and GeoPandas.

    Inputs:

    db: a GeoDataFrame with the locations to plot.
    col: the column name of the class column that maps the observations to their class.
    p_001: a boolean indicating whether to include 'p < 0.001' classes (default False).
    commune_name: a boolean indicating whether to include the name of the communes in the plot (default False).
    Output:
    A plot of the Getis map with each observation colored according to its class.

    The function reads in shapefiles for the cantons and communes of Switzerland and converts them to the Swiss coordinate system (2056).
    It then uses the 'intersects' predicate to join the input data with the cantons and communes shapefiles.
    The function also reads in a shapefile for Lake Geneva.

    The function uses Matplotlib and GeoPandas to plot the map. The class of each observation is indicated by a color, which is specified in a dictionary. The size of the marker for each observation can be modified, with significant observations having a larger marker size.

    Finally, if 'commune_name' is True, the names of the communes will be annotated on the plot.
    """
    print('Plot Getis Map')
    fig, ax = plt.subplots(1, figsize=(10, 10))
    y = db[col] + ' [' + db[col].map(db[col].value_counts()).astype(str) + ']'
    # y = db[col] + ' (' + db[col].map(db.groupby(col)[col.split('_G_cl')[0]].mean().mul(100).round(1)).astype(str) + '%)'


    colors_cl_getis = {
        'Cold Spot - p < 0.1': '#d1e5f0',
        'Cold Spot - p < 0.05': '#67a9cf',
        'Cold Spot - p < 0.01': '#2166ac',
        'Hot Spot - p < 0.1': '#fddbc7',
        'Hot Spot - p < 0.05': '#ef8a62',
        'Hot Spot - p < 0.01': '#b2182b',
        'Not significant': '#bdbdbd'
    }

    # Add p < 0.001 colors if required
    if p_001:
        colors_cl_getis['Cold Spot - p < 0.001'] = '#1c4978'
        colors_cl_getis['Hot Spot - p < 0.001'] = '#991d2c'

    # Ordering colors according to the unique sorted values in the column
    hmap = colors.ListedColormap([colors_cl_getis[i] for i in db[col].sort_values().unique()])



    cantons = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.shp', engine='pyogrio')
    communes = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp', engine='pyogrio')

    cantons, communes = cantons.to_crs(2056), communes.to_crs(2056)
    cantons_to_map = gpd.sjoin(cantons[['UUID','NAME','geometry']], db[['geometry']], predicate='contains', how='left')
    cantons_to_map = cantons_to_map[~cantons_to_map.index_right.isnull()].drop_duplicates(subset='NAME').drop('index_right', axis=1)
    # cantons_to_map = cantons_to_map[cantons_to_map.NAME == 'Genève']
    # communes = communes[communes.KANTONSNUM.isin(cantons_to_map.KANTONSNUM)]
    communes = gpd.sjoin(communes, cantons_to_map[['geometry']], predicate='intersects').drop('index_right', axis=1)
    lakes = gpd.read_file('/Users/david/Dropbox/PhD/GitHub/COVID19/input/g2s15.shp', engine='pyogrio')
    # lake = gpd.read_file('/Users/david/Dropbox/PhD/GitHub/COVID19/Data/Mapping/lake.geojson', engine='pyogrio')
    # lake.NOM = ['Lake Geneva', '', '']
    lakes = lakes.to_crs(2056)
    # lake = lake.to_crs(2056)

    db['markersize'] = markersize_s
    db.loc[db[col] != 'Not significant', 'markersize'] = markersize_l
    ## Plotting
    cantons_to_map.geometry.boundary.plot(ax=ax, edgecolor='k', color=None, linewidth=0.3)
    communes.plot(ax=ax, label='Communes', alpha=1, color=None, edgecolor='#636363', facecolor='white', linewidth=0.1)
    if commune_name:

        communes[communes.NAME.isin(['Genève','Bâle','Zürich','Basel','Lausanne','Bern'])].apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', size=8),
                       axis=1)
    lakes.plot(ax=ax, label='Lake', alpha=1, color='lightgrey', zorder = 5)
    db.plot(y, cmap=hmap, markersize='markersize', linewidth = 0.01, ax=ax, legend=True, categorical=True, zorder=6)
    legend = ax.get_legend()
    legend.set_bbox_to_anchor((0.03, 0.8, 0.2, 0.2))
    ax.set_axis_off()
    ax.set_facecolor('grey')

    # lakes.apply(lambda x: ax.annotate(text=x.GMDNAME, xy=x.geometry.centroid.coords[0], ha='center', size=12, zorder = 6), axis=1)
    # add scale bar
    scalebar = ScaleBar(1, units="m", location="lower right")
    ax.add_artist(scalebar)
    ax.set_axis_off()
    x, y, arrow_length = 0.90, 0.15, 0.08
    ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20,
                xycoords=ax.transAxes)
    return fig, ax


# colors_cl = {'0 Non-Significant':'#f0f0f0', '2 Low-High':'#d1e5f0','4 High-Low':'#ef8a62', '1 High-High':'#b2182b','3 Low-Low':'#2166ac'}
colors_cl = {'0 Non-Significant': 'lightgrey',
             '1 High-High': '#d7191c',
             '2 Low-High': '#abd9e9',
             '3 Low-Low': '#2c7bb6',
             '4 High-Low': '#fdae61'}


def plotLISAMap_polygons(db, col, letter, commune_name=False, add_lakes = False):
    """
    Visualize Getis results on a map
    ...
    Arguments
    -------------
    db : GeodataFrame
    y : Series containing Getis classes"""

    lake = gpd.read_file('/Users/david/Dropbox/PhD/GitHub/COVID19/Data/Mapping/lake.geojson', engine='pyogrio')
    lake.NOM = ['Lake Geneva', '', '']
    if add_lakes:
        lakes = gpd.read_file('/Users/david/Dropbox/PhD/GitHub/COVID19/input/g2s15.shp', engine='pyogrio')
        lakes = lakes[lakes.GMDNAME.isin(['Lac de Joux','Lac Léman','Lac de la Gruyère','Lac de Neuchâtel'])]
        lakes = lakes.to_crs(2056)
    communes = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp', engine='pyogrio')
    # communes = communes[communes.KANTONSNUM == 25]
    communes = communes.to_crs(2056)
    communes_to_map = gpd.sjoin(db, communes, predicate='intersects')
    communes = communes[communes.NAME.isin(communes_to_map.NAME)]
    communes = communes[communes.NAME.str.contains('Lac') == False]
    if add_lakes:
        lakes_to_map = gpd.sjoin(db, lakes, predicate='intersects')
        lakes = lakes[lakes.GMDNAME.isin(lakes_to_map.GMDNAME)]
    fig, ax = plt.subplots(1, figsize=(12, 12))
    hfont = {'fontname': 'Palatino', 'size': 12}
    # hmap = colors.ListedColormap(['lightgrey', '#d7191c', '#abd9e9', '#2c7bb6', '#fdae61'])
    hmap = colors.ListedColormap([colors_cl[i] for i in db[col].sort_values().unique()])

    communes.plot(ax = ax, label='Communes',alpha = 0.8,color=None,edgecolor='white',linewidth = 0.1,facecolor='grey')
    # db['markersize'] = 3
    # db.loc[db[col] != '0 Non-Significant', 'markersize'] = 10
    if commune_name:
        communes.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', size=8),
                       axis=1)
    # y = db[col] + ' (' + db[col].map(db.groupby(col)[col.split('_cl')[0]].mean().round(1)).astype(str) + '%)'
    y = db[col] + ' [' + db[col].map(db.groupby(col)[col.split('_cl')[0]].size().astype(int)).astype(str) + ']'
    if add_lakes:
        lakes.plot(ax=ax, label='Lake', alpha=1, color='lightblue', zorder=2)
        lakes.apply(lambda x: ax.annotate(text=x.GMDNAME, xy=x.geometry.centroid.coords[0], ha='center', size=12), axis=1)
    lake.plot(ax=ax, label='Lake', alpha=1, color='lightblue', zorder=2)
    lake.apply(lambda x: ax.annotate(text=x.NOM, xy=x.geometry.centroid.coords[0], ha='center', size=12), axis=1)

    db.plot(y, cmap=hmap, ax=ax, linewidth=0.01, legend=True, alpha=1, edgecolor='white', categorical=True,
            legend_kwds={'loc': 'upper left', 'fontsize': 12, 'title': 'LISA Classes', 'title_fontsize': 14,
                              }, zorder=1)

    ax.set_facecolor('white')
    #     ax.legend(handles=[ns_patch, lh_patch, hl_patch,hh_patch, ll_patch])

    ax.text(0.9, 0.93, letter, transform=ax.transAxes, weight='bold', fontsize=32)

    # add scale bar
    scalebar = ScaleBar(1, units="m", location="lower right")
    ax.add_artist(scalebar)
    ax.set_axis_off()
    x, y, arrow_length = 0.91, 0.12, 0.05
    ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                arrowprops=dict(facecolor='black', width=3, headwidth=8),
                ha='center', va='center', fontsize=14,
                xycoords=ax.transAxes)
    #     plt.title(filename.split('.')[0])
    #     plt.savefig(localautocorr_spatial_result_folder/filename, dpi = 200,bbox_inches = 'tight')
    return fig, ax


def plotLISAMap_points(db, col, letter, commune_name=False):
    """
    Visualize Getis results on a map
    ...
    Arguments
    -------------
    db : GeodataFrame
    y : Series containing Getis classes"""

    lake = gpd.read_file('/Users/david/Dropbox/PhD/GitHub/COVID19/Data/Mapping/lake.geojson', engine='pyogrio')
    lake.NOM = ['Lake Geneva', '', '']
    # communes = gpd.read_file('../GeoSalt/Data/Raw Data/communes_ge.geojson',driver = 'GeoJSON', engine='pyogrio')
    cantons = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.shp', engine='pyogrio')
    communes = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp', engine='pyogrio')

    cantons, communes = cantons.to_crs(2056), communes.to_crs(2056)
    kantonnum_to_name = cantons.set_index('KANTONSNUM')['NAME'].to_dict()
    cantons_to_map = gpd.sjoin(cantons[['UUID','NAME','geometry']], db[['geometry']], predicate='contains', how='left')
    cantons_to_map = cantons_to_map[~cantons_to_map.index_right.isnull()].drop_duplicates(subset='NAME').drop('index_right', axis=1)
    cantons_to_map = cantons_to_map[cantons_to_map.NAME == 'Genève']
    communes = gpd.sjoin(communes, cantons_to_map, predicate='intersects').drop('index_right', axis=1)

    fig, ax = plt.subplots(1, figsize=(12, 12))
    hfont = {'fontname': 'Palatino', 'size': 12}
    hmap = colors.ListedColormap([colors_cl[i] for i in db[col].sort_values().unique()])
    communes.plot(ax=ax, label='Communes', alpha=0.6, color=None, facecolor ='white', edgecolor='white', linewidth=0.3, zorder=2)
    cantons_to_map.geometry.boundary.plot(ax=ax, edgecolor='k', color=None, linewidth=0.3, zorder=2)

    db['markersize'] = 3
    db.loc[db[col] != '0 Non-Significant', 'markersize'] = 6
    if commune_name:
        communes.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', size=6),
                       axis=1)
    y = db[col] + ' [' + db[col].map(db[col].value_counts()).astype(str) + ']'

    lake.plot(ax=ax, label='Lake', alpha=1, color='lightblue', zorder=2)

    db.plot(y, cmap=hmap, ax=ax, markersize='markersize', linewidth=0.4, legend=True, alpha=1, categorical=True,
            legend_kwds={'fontsize': '14', 'loc': 'upper left'}, zorder=3)

    # ax.set_facecolor('white')
    #     ax.legend(handles=[ns_patch, lh_patch, hl_patch,hh_patch, ll_patch])

    ax.text(0.9, 0.93, letter, transform=ax.transAxes, weight='bold', fontsize=32)
    lake.apply(lambda x: ax.annotate(text=x.NOM, xy=x.geometry.centroid.coords[0], ha='center', size=12), axis=1)
    # add scale bar
    scalebar = ScaleBar(1, units="m", location="lower right")
    ax.add_artist(scalebar)
    ax.set_axis_off()
    x, y, arrow_length = 0.91, 0.15, 0.06
    ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                arrowprops=dict(facecolor='black', width=3, headwidth=8),
                ha='center', va='center', fontsize=14,
                xycoords=ax.transAxes)
    return fig, ax

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica']})
plt.rcParams['pdf.fonttype'] = 42

def plotLISAMap_points_ordered(db, col, ylabel, letter, binary=False):
    """
    Visualize LISA results on a map
    ...
    Arguments
    -------------
    db : GeodataFrame
    y : Series containing LISA classes
    distance : Distance based threshold used for the weight matrix
    """

    fig, ax = plt.subplots(1, figsize=(12, 12))
    hmap = colors.ListedColormap(['#bdbdbd', '#b2182b', '#d1e5f0', '#2166ac', '#ef8a62'])

    lake = gpd.read_file('/Users/david/Dropbox/PhD/GitHub/COVID19/Data/Mapping/lake.geojson', engine='pyogrio')
    lake.NOM = ['Lake Geneva', '', '']
    # communes = gpd.read_file('../GeoSalt/Data/Raw Data/communes_ge.geojson',driver = 'GeoJSON', engine='pyogrio')
    cantons = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.shp', engine='pyogrio')
    communes = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp', engine='pyogrio')

    cantons, communes = cantons.to_crs(2056), communes.to_crs(2056)
    kantonnum_to_name = cantons.set_index('KANTONSNUM')['NAME'].to_dict()
    cantons_to_map = gpd.sjoin(cantons[['UUID','NAME','geometry']], db[['geometry']], predicate='contains', how='left')
    cantons_to_map = cantons_to_map[~cantons_to_map.index_right.isnull()].drop_duplicates(subset='NAME').drop('index_right', axis=1)
    cantons_to_map = cantons_to_map[cantons_to_map.NAME == 'Genève']
    communes = gpd.sjoin(communes, cantons_to_map, predicate='intersects').drop('index_right', axis=1)

    ## Plotting background elements
    communes.plot(ax=ax, label='Communes', alpha=1, color=None, edgecolor='#636363', facecolor='white', linewidth=0.1, zorder = 1)
    # communes.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', size=8), axis=1);
    cantons_to_map.geometry.boundary.plot(ax=ax, edgecolor='k', color=None, linewidth=0.5, zorder = 1)

    lake.plot(ax=ax, label='Lake', alpha=1, color='lightblue', zorder=2)
    lake.apply(lambda x: ax.annotate(text=x.NOM, xy=x.geometry.centroid.coords[0], ha='center', size=12), axis=1)
    ## Plotting LISA
    
    ns = db[db[col] == '0 Non-Significant']
    lh = db[db[col] == '2 Low-High']
    hl = db[db[col] == '4 High-Low']
    hh = db[db[col] == '1 High-High']
    ll = db[db[col] == '3 Low-Low']
    percentage_ns = round((ns[col].value_counts().values[0] / db.shape[0])*100, 1)
    percentage_hh = round((hh[col].value_counts().values[0] / db.shape[0])*100, 1)
    percentage_lh = round((lh[col].value_counts().values[0] / db.shape[0])*100, 1)
    percentage_ll = round((ll[col].value_counts().values[0] / db.shape[0])*100, 1)
    percentage_hl = round((hl[col].value_counts().values[0] / db.shape[0])*100, 1)

    ns_label = '0 Non-Significant' + ' - n = ' + ns[col].value_counts().values[0].astype(str)+ ' ('+ str(percentage_ns) + '%)'
    hh_label = '1 High-High' + ' - n = ' + hh[col].value_counts().values[0].astype(str)+ ' ('+ str(percentage_hh) + '%)'
    lh_label = '2 Low-High' + ' - n = ' + lh[col].value_counts().values[0].astype(str)+' ('+ str(percentage_lh) + '%)'
    ll_label = '3 Low-Low' + ' - n = ' + ll[col].value_counts().values[0].astype(str)+ ' ('+ str(percentage_ll) + '%)'
    hl_label = '4 High-Low' + ' - n = ' + hl[col].value_counts().values[0].astype(str)+ ' ('+ str(percentage_hl) + '%)'


    ns_patch = patches.Patch(color="#bdbdbd", label=ns_label)
    hh_patch = patches.Patch(color="#d7191c", label=hh_label)
    lh_patch = patches.Patch(color="#abd9e9", label=lh_label)
    ll_patch = patches.Patch(color="#2c7bb6", label=ll_label)
    hl_patch = patches.Patch(color="#fdae61", label=hl_label)
    
    ns.plot(color='#bdbdbd', markersize=8, linewidth = 0, ax=ax, alpha=0.8, zorder=3)
    lh.plot(color='#abd9e9', markersize=12, linewidth = 0, ax=ax, alpha=0.4, zorder=4)
    hl.plot(color='#fdae61', markersize=12, linewidth = 0, ax=ax, alpha=0.4, zorder=5)
    hh.plot(color='#d7191c', markersize=12, linewidth = 0, ax=ax, alpha=0.6, zorder=7)
    ll.plot(color='#2c7bb6', markersize=12, linewidth = 0, ax=ax, alpha=0.6, zorder=6)
    ax.legend(handles=[ns_patch, hh_patch, lh_patch,ll_patch, hl_patch], loc = 'upper left', fontsize = 12, title = 'Classes', title_fontsize = 14) 
    
    scalebar = ScaleBar(1, units="m", location="lower right")
    ax.add_artist(scalebar)
    ax.set_axis_off()
    x, y, arrow_length = 0.91, 0.12, 0.05
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=3, headwidth=8),
            ha='center', va='center', fontsize=14,
            xycoords=ax.transAxes)
    ax.text(0.9, 0.93, letter, transform=ax.transAxes, weight='bold', fontsize=24)
    ax.set_axis_off()
    ax.set_facecolor('grey')

    left, bottom, width, height = [0.204, 0.55, 0.15, 0.15]
    ax2 = fig.add_axes([left, bottom, width, height])
    sns.set_context("paper")
    x_label = 'LISA Categories'
    y_label = ylabel
    lagged_col = col.replace('_cl', '_lag')
    if binary:
        chart = sns.boxplot(x=col, y=lagged_col, order=['0 Non-Significant', '1 High-High', '2 Low-High', '3 Low-Low', '4 High-Low'],
                    palette=colors_cl, showfliers=False, data=db)
    else:
        chart = sns.boxplot(x=col, y=col.replace('_cl','_lag'), order=['0 Non-Significant', '1 High-High', '2 Low-High', '3 Low-Low', '4 High-Low'],
                    palette=colors_cl, showfliers=False, data=db)
    chart.set_xticklabels(chart.get_xticklabels(), size=11, rotation=45, horizontalalignment='right')
    chart.set_xlabel('', size=11)
    chart.set_ylabel(y_label, size=11)
    return fig, ax


def masked_moran_scatterplot(moran_loc, p, quadrant, attribute_name, zstandard=False, aspect_equal=False):
    fig, ax = plt.subplots(figsize=(9, 9))
    # Moran Scatterplot
    moran_scatterplot(moran_loc, p=p, zstandard=zstandard, aspect_equal=aspect_equal, ax=ax)
    ax.set_xlabel(attribute_name, size=14)
    ax.set_ylabel('Spatial Lag of %s' % attribute_name, size=14)
    if aspect_equal is True:
        ax.set_aspect('equal', 'datalim')
    else:
        ax.set_aspect('auto')

    if quadrant is not None:
        # Quadrant masking in Scatterplot
        mask_angles = {1: 0, 2: 90, 3: 180, 4: 270}  # rectangle angles
        # We don't want to change the axis data limits, so use the current ones
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        # We are rotating, so we start from 0 degrees and
        # figured out the right dimensions for the rectangles for other angles
        mask_width = {1: abs(xmax),
                      2: abs(ymax),
                      3: abs(xmin),
                      4: abs(ymin)}
        mask_height = {1: abs(ymax),
                       2: abs(xmin),
                       3: abs(ymin),
                       4: abs(xmax)}
        ax.add_patch(patches.Rectangle((0, 0), width=mask_width[quadrant],
                                       height=mask_height[quadrant],
                                       angle=mask_angles[quadrant],
                                       color='#E5E5E5', zorder=-1, alpha=0.8))
    return fig, ax


def masked_lisa_cluster_polygons(gdf, moran_loc, attribute, quadrant, p, scheme,commune = False, commune_name = False, lake = False, legend=True):
    fig, ax = plt.subplots(figsize=(9, 9))
    lisa_cluster(moran_loc, gdf, p=p, ax=ax, legend=legend,
                 legend_kwds={'loc': 'upper left', 'fontsize': 12, 'title': 'LISA Classes', 'title_fontsize': 14,
                              'bbox_to_anchor': (0.82, 0.95)})

    df_lake = gpd.read_file('/Users/david/Dropbox/PhD/GitHub/COVID19/Data/Mapping/lake.geojson', engine='pyogrio')
    df_lake.NOM = ['Lake Geneva', '', '']
    # communes = gpd.read_file('../GeoSalt/Data/Raw Data/communes_ge.geojson',driver = 'GeoJSON', engine='pyogrio')
    cantons = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.shp', engine='pyogrio')
    df_communes = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp', engine='pyogrio')

    cantons, df_communes = cantons.to_crs(2056), df_communes.to_crs(2056)
    kantonnum_to_name = cantons.set_index('KANTONSNUM')['NAME'].to_dict()
    cantons_to_map = gpd.sjoin(gdf, cantons, predicate='intersects')
    df_communes = df_communes[df_communes.KANTONSNUM.isin(cantons_to_map.KANTONSNUM)]
    df_communes = df_communes[df_communes.KANTONSNUM == 25]

    # quadrant selection in maps

    non_quadrant = ~(moran_loc.q == quadrant)
    mask_quadrant = gdf[non_quadrant]
    df_quadrant = gdf.iloc[~non_quadrant]
    union2 = df_quadrant.unary_union.boundary
    if commune:
        df_communes.plot(ax=ax, label='Communes', alpha=0.3, color=None, edgecolor='black', linewidth=0.4, facecolor='grey')
    if commune_name:
        df_communes.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', size=8), axis=1);

    if lake:
        df_lake.plot(ax=ax, label='Lake', alpha=1, color='lightblue')
        df_lake.apply(lambda x: ax.annotate(text=x.NOM, xy=x.geometry.centroid.coords[0], ha='center', size=12), axis=1)

    # LISA Cluster mask and cluster boundary
    with warnings.catch_warnings():  # temorarily surpress geopandas warning
        warnings.filterwarnings('ignore', category=UserWarning)
        mask_quadrant.plot(column=attribute, scheme=scheme, color='white',
                           ax=ax, alpha=0.8, zorder=1)

    gpd.GeoSeries([union2]).plot(linewidth=0.01, ax=ax, color='#E5E5E5', zorder = 2)
    return fig, ax


def masked_lisa_cluster_points(gdf, moran_loc, attribute, quadrant, p, scheme, legend=True):
    '''TO DO: Revise the function to have a better result, because points can spatially overlap (>< polygons) the map
    is messy. Need to change the function so that the masked points don't appear over the non-masked.'''
    fig, ax = plt.subplots(figsize=(9, 9))

    lisa_cluster(moran_loc, gdf, p=p, ax=ax, legend=legend,
                 legend_kwds={'loc': 'upper left', 'fontsize': 12, 'title': 'LISA Classes', 'title_fontsize': 14,
                              'bbox_to_anchor': (0.82, 0.95)}, zorder=3)

    lake = gpd.read_file('/Users/david/Dropbox/PhD/GitHub/COVID19/Data/Mapping/lake.geojson', engine='pyogrio')
    lake.NOM = ['Lake Geneva', '', '']
    # communes = gpd.read_file('../GeoSalt/Data/Raw Data/communes_ge.geojson',driver = 'GeoJSON', engine='pyogrio')
    cantons = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.shp', engine='pyogrio')
    communes = gpd.read_file(
        '/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp', engine='pyogrio')

    cantons, communes = cantons.to_crs(2056), communes.to_crs(2056)
    kantonnum_to_name = cantons.set_index('KANTONSNUM')['NAME'].to_dict()
    cantons_to_map = gpd.sjoin(gdf, cantons, predicate='intersects')
    communes = communes[communes.KANTONSNUM.isin(cantons_to_map.KANTONSNUM)]


    # quadrant selection in maps

    non_quadrant = ~(moran_loc.q == quadrant)
    mask_quadrant = gdf[non_quadrant]
    df_quadrant = gdf.iloc[~non_quadrant]
    union2 = df_quadrant.unary_union.boundary
    communes.plot(ax=ax, label='Communes', alpha=0.3, color=None, edgecolor='black', linewidth=0.4, facecolor='grey',
                  zorder=1)
    # communes.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', size=8), axis=1);
    lake.plot(ax=ax, label='Lake', alpha=1, color='lightblue', zorder=2)
    lake.apply(lambda x: ax.annotate(text=x.NOM, xy=x.geometry.centroid.coords[0], ha='center', size=12), axis=1)

    # LISA Cluster mask and cluster boundary
    with warnings.catch_warnings():  # temorarily surpress geopandas warning
        warnings.filterwarnings('ignore', category=UserWarning)
        mask_quadrant.plot(column=attribute, scheme=scheme, color='white',
                           ax=ax, alpha=0.3, zorder=4)

    df_quadrant.plot(linewidth=1, ax=ax, color='#E5E5E5', zorder = 5)
    return fig, ax

def masked_choropleth_lisa(gdf, moran_loc, attribute, quadrant, scheme, cmap, legend_title, legend=True):
    fig, ax = plt.subplots(figsize=(9, 9))

    lake = gpd.read_file('/Users/david/Dropbox/PhD/GitHub/COVID19/Data/Mapping/lake.geojson', engine='pyogrio')
    lake.NOM = ['Lake Geneva', '', '']

    gdf.plot(column=attribute, scheme=scheme, cmap=cmap,
             legend=legend, zorder=1,
             legend_kwds={'loc': 'upper left', 'title': legend_title, 'fontsize': 12, 'title_fontsize': 14},
             ax=ax, alpha=1)
    # lake.plot(ax=ax, label='Lake', alpha=1, color='#5487ba', zorder=2)

    ax.set_axis_off()
    # CHOROPLETH MASK
    if quadrant is not None:
        # quadrant selection in maps
        non_quadrant = ~(moran_loc.q == quadrant)
        mask_quadrant = gdf[non_quadrant]
        df_quadrant = gdf.iloc[~non_quadrant]
        union2 = df_quadrant.unary_union.boundary
        with warnings.catch_warnings():  # temorarily surpress geopandas warning
            warnings.filterwarnings('ignore', category=UserWarning)
            mask_quadrant.plot(column=attribute, scheme=scheme, color='white',
                               ax=ax, alpha=0.95, zorder=1)
        gpd.GeoSeries([union2]).plot(linewidth=1, ax=ax, color='#E5E5E5')
    return fig, ax


def bar_plot_cat(df, y, n, h, w, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(h, w))
    df.groupby(y).size().sort_values(ascending=False).head(n).plot.bar(ax=ax)
    ax.bar_label(ax.containers[0], size=12)
    ax.set_xlabel(xlabel, labelpad=10, size=16)
    ax.set_ylabel(ylabel, labelpad=10, size=16)
    return fig, ax


def radar_plot_3cat(df):
    figure = go.Figure()

    cluster_cls = ['Cold Spot', 'Hot Spot', 'Not Significant']

    cluster_color = {'Cold Spot': '#2166ac',
                     'Hot Spot': '#b2182b',
                     'Not Significant': '#bdbdbd'}
    for cluster in cluster_cls:
        color = cluster_color.get(cluster, 'lightslategrey')
        highlight = color != 'lightslategrey'
        data_filtered = df[df.cluster_cl == cluster]
        plot_data = data_filtered.groupby(['cluster_cl'], as_index=False).mean().round(2)
        plot_data = plot_data.T.drop('cluster_cl', axis=0)
        plot_data = plot_data.reset_index()
        plot_data.columns = ['Covariates', 'Mean']
        #         plot_data['Covariates'] = plot_data['Covariates'].map(labels)
        axis = plot_data.Covariates.tolist()
        axis.append(axis[0])
        plot_data = plot_data.Mean.tolist()
        plot_data.append(plot_data[0])
        figure.add_trace(
            go.Scatterpolar(
                r=plot_data,
                theta=axis,
                showlegend=highlight,
                name=cluster,
                hoverinfo='name+r',
                hovertemplate='%{r:0.0}',
                mode='lines',
                line_color=color,
                opacity=0.8 if highlight else 0.25,
                line_shape='spline',
                line_smoothing=0.6,
                line_width=2 if highlight else 0.6
            )
        )
    title = 'Getis Gi* clusters characteristics (Z-scores)' \
            '<br><span style="font-size:10px"><i>Bus Santé participants</i></span>'

    figure.update_layout(
        template=None,  # Make plot a bit less minimalistic
        title_text=title,
        title_font_color='#333333',
        title_font_size=14,
        polar_bgcolor='white',
        polar_radialaxis_visible=True,
        polar_radialaxis_showticklabels=True,
        polar_radialaxis_tickfont_color='darkgrey',
        polar_angularaxis_color='grey',
        polar_angularaxis_showline=True,
        polar_radialaxis_color='grey',
        polar_radialaxis_showline=True,
        polar_radialaxis_layer='below traces',
        polar_radialaxis_gridcolor='#F2F2F2',
        polar_radialaxis_range=(-2, 2),
        polar_radialaxis_tickvals=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
        polar_radialaxis_ticktext=['-2', '-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5', '2'],
        polar_radialaxis_tickmode='array',
        legend_font_color='grey',  # We don't want to draw attention to the legend
        legend_itemclick='toggleothers',  # Change the default behaviour, when click select only that trace
        legend_itemdoubleclick='toggle',  # Change the default behaviour, when double click ommit that trace
        width=1200,  # chart size
        height=900  # chart size
    )
    return figure

def radarPrep(df, y, cols, analysis):
    radar_db = df[cols]
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    scaled_radar_db = pd.DataFrame(scaler.fit_transform(radar_db))
    scaled_radar_db.columns = radar_db.columns
    if analysis == 'Getis':
        y_col = y + '_G_cl'
        df.loc[df[y_col].str.contains('Not'), 'cluster_cl_str'] = 'Not Significant'
        df.loc[df[y_col].str.contains('Cold Spot - p < 0.0'), 'cluster_cl_str'] = 'Cold Spot'
        df.loc[df[y_col].str.contains('Hot Spot - p < 0.0'), 'cluster_cl_str'] = 'Hot Spot'
        scaled_radar_db = scaled_radar_db.assign(cluster_cl=df['cluster_cl_str'])
    if analysis == 'LISA':
        y_col = y + '_cl'
        scaled_radar_db = scaled_radar_db.assign(cluster_cl=df[y_col])
    return scaled_radar_db


def radar_plot_3cat_lisa(df, three_classes=True):
    figure = go.Figure()
    if three_classes:

        cluster_cls = ['1 High-High', '3 Low-Low', '0 Non-Significant']

        cluster_color = {'3 Low-Low': '#2166ac',
                         '1 High-High': '#b2182b',
                         '0 Non-Significant': '#bdbdbd'}
    else:
        cluster_cls = ['1 High-High', '2 Low-High', '3 Low-Low', '4 High-Low', '0 Non-Significant']

        cluster_color = {'4 High-Low': '#ef8a62',
                         '3 Low-Low': '#2166ac',
                         '2 Low-High': '#d1e5f0',
                         '1 High-High': '#b2182b',
                         '0 Non-Significant': '#bdbdbd'}
    for cluster in cluster_cls:
        color = cluster_color.get(cluster, 'lightslategrey')
        highlight = color != 'lightslategrey'
        data_filtered = df[df.cluster_cl == cluster]
        plot_data = data_filtered.groupby(['cluster_cl'], as_index=False).mean().round(2)
        plot_data = plot_data.T.drop('cluster_cl', axis=0)
        plot_data = plot_data.reset_index()
        plot_data.columns = ['Covariates', 'Mean']
        #         plot_data['Covariates'] = plot_data['Covariates'].map(labels)
        axis = plot_data.Covariates.tolist()
        axis.append(axis[0])
        plot_data = plot_data.Mean.tolist()
        plot_data.append(plot_data[0])
        figure.add_trace(
            go.Scatterpolar(
                r=plot_data,
                theta=axis,
                showlegend=highlight,
                name=cluster,
                hoverinfo='name+r',
                hovertemplate='%{r:0.0}',
                mode='lines',
                line_color=color,
                opacity=0.8 if highlight else 0.25,
                line_shape='spline',
                line_smoothing=0.6,
                line_width=2 if highlight else 0.6
            )
        )
    title = 'Spatial clusters characteristics' \
            '<br><span style="font-size:10px"><i>1993-2018 Bus Santé participants</i></span>'

    figure.update_layout(
        template=None,  # Make plot a bit less minimalistic
        #         title_text = title,
        #         title_font_color = '#333333',
        #         title_font_size = 14,
        polar_bgcolor='white',
        polar_radialaxis_visible=True,
        polar_radialaxis_showticklabels=True,
        polar_radialaxis_tickfont_color='darkgrey',
        polar_angularaxis_color='grey',
        polar_angularaxis_showline=True,
        polar_radialaxis_color='grey',
        polar_radialaxis_showline=True,
        polar_radialaxis_layer='below traces',
        polar_radialaxis_gridcolor='#F2F2F2',
        polar_radialaxis_range=(-1, 1),
        polar_radialaxis_tickvals=[-1, -0.5, 0, 0.5, 1],
        polar_radialaxis_ticktext=['-1', '-0.5', '0', '0.5', '1'],
        polar_radialaxis_tickmode='array',
        # Make labels bold ... but can't make only certain labels bold
        polar=dict(
            radialaxis_tickfont_size=12,
            angularaxis=dict(
                tickfont_size=12,  # start position of angular axis
            ), ),
        legend_font_color='grey',  # We don't want to draw attention to the legend
        legend_itemclick='toggleothers',  # Change the default behaviour, when click select only that trace
        legend_itemdoubleclick='toggle',  # Change the default behaviour, when double click ommit that trace
        width=1200,  # chart size
        height=900  # chart size
    )
    return figure
