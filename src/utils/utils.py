import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from typing import Dict, List, Tuple
import numpy as np
import rasterio
from rasterstats import zonal_stats, point_query
import concurrent.futures
from pathlib import Path
import os



def min_dist(point: Point, distance: int, statpop_gdf_point: gpd.GeoDataFrame) -> str:
    """
    Calculate the minimum distance to the nearest polygon from a point,
    adjusting the distance if no polygons are found within the initial buffer.

    Args:
    point (Point): The point to check.
    distance (int): The initial buffer distance.
    statpop_gdf_point (GeoDataFrame): GeoDataFrame containing the RELI and geometry columns.

    Returns:
    str: The RELI of the closest polygon.
    """
    buffer200 = gpd.GeoDataFrame(gpd.GeoSeries(point), crs=2056, columns=['geometry'])
    buffer200['geometry'] = buffer200.geometry.buffer(distance)
    polygons = gpd.sjoin(statpop_gdf_point[['RELI', 'geometry']], buffer200, predicate='intersects')
    while polygons.shape[0] == 0:
        distance += 100
        buffer200['geometry'] = buffer200.geometry.buffer(distance)
        polygons = gpd.sjoin(statpop_gdf_point[['RELI', 'geometry']], buffer200, predicate='intersects')
    polygons = polygons.set_index('RELI')
    polygon_index = polygons.distance(point).sort_values().index[0]
    return polygon_index

def apply_min_dist(row: pd.Series, statpop_gdf_point: gpd.GeoDataFrame) -> str:
    """
    Apply the minimum distance function to a DataFrame row.

    Args:
    row (Series): A pandas Series object with 'RELI' and 'geometry'.
    statpop_gdf_point (GeoDataFrame): GeoDataFrame containing the RELI and geometry columns.

    Returns:
    str: The RELI of the closest polygon or existing RELI.
    """
    if pd.isna(row['RELI']):
        point = row['geometry']
        return min_dist(point, 200, statpop_gdf_point)
    else:
        return row['RELI']
def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]



def import_raster_and_get_min_max(statpop_gdf_point, data_dir, raster_path, i):
    statpop = pd.read_csv('/Users/david/Dropbox/PhD/Data/Databases/OFS/ag-b-00.03-vz2020statpop/STATPOP2020.csv',sep = ';')
    geometry = [Point(xy) for xy in zip(statpop['E_KOORD'], statpop['N_KOORD'])]
    fp = os.path.join(data_dir, raster_path)
    # Open the file:
    raster = rasterio.open(fp)
    # Check type of the variable 'raster'
    type(raster)
    # Read the raster band as separate variable
    band1 = raster.read(1)

    # Check type of the variable 'band'
#     print(type(band1))

    # Data type of the values
#     print(band1.dtype)
    # Read all bands
    array = raster.read()

    # Get the affine
    affine = raster.transform
    # Calculate statistics for each band
    stats = []
    for band in array:
        stats.append({
            'min': band.min(),
            'mean': band.mean(),
            'median': np.median(band),
            'max': band.max()})
    no2_satellite_pt = point_query(statpop_gdf_point.to_crs(4326), band1, affine=affine)
    statpop_gdf_ha[['NO2_tropospheric_{}'.format(i)]] = pd.DataFrame(no2_satellite_pt)
    # Show stats for each channel
    return stats[0]


def process_raster_data(raster_path: str, index: int, data_dir: str, statpop_points_gdf: gpd.GeoDataFrame, statpop_ha_gdf: gpd.GeoDataFrame) -> Tuple[Dict[str, float], gpd.GeoDataFrame]:
    """
    Import a raster file, calculate statistics for each band, and add NO2 data to a GeoDataFrame.

    Args:
        raster_path (str): Path to the raster file relative to the data directory.
        index (int): Index used to name the new column in the GeoDataFrame.
        data_dir (str): Path to the data directory.
        statpop_points_gdf (gpd.GeoDataFrame): GeoDataFrame containing population points.
        statpop_ha_gdf (gpd.GeoDataFrame): GeoDataFrame to which NO2 data will be added.

    Returns:
        Tuple[Dict[str, float], gpd.GeoDataFrame]: A tuple containing:
            - A dictionary with the minimum, mean, median, and maximum values of the first band.
            - The updated statpop_ha_gdf GeoDataFrame.

    Raises:
        FileNotFoundError: If the raster file is not found.
        ValueError: If the raster file is empty or corrupted.
    """
    file_path = os.path.join(data_dir, raster_path)
    
    try:
        with rasterio.open(file_path) as raster:
            band_array = raster.read()
            
            if band_array.size == 0:
                raise ValueError("The raster file is empty or corrupted.")
            
            affine_transform = raster.transform
            band_stats = calculate_band_statistics(band_array)
            
            first_band = band_array[0]
            no2_data = query_no2_data(statpop_points_gdf, first_band, affine_transform)
            
            statpop_ha_gdf[f'NO2_tropospheric_{index}'] = pd.Series(no2_data)
    
    except rasterio.errors.RasterioIOError:
        raise FileNotFoundError(f"Raster file not found: {file_path}")
    
    return band_stats[0], statpop_ha_gdf



def calculate_band_statistics(band_array: np.ndarray) -> List[Dict[str, float]]:
    """Calculate statistics for each band in the array."""
    return [{
        'min': float(band.min()),
        'mean': float(band.mean()),
        'median': float(np.median(band)),
        'max': float(band.max())
    } for band in band_array]

def query_no2_data(points_gdf, band_data: np.ndarray, affine_transform) -> np.ndarray:
    """Query NO2 data at the points of the given GeoDataFrame."""
    return point_query(points_gdf.to_crs(4326), band_data, affine=affine_transform)
