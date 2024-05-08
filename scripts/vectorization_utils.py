import numpy as np
import rasterio
import rasterio.features
import shapely
#import numba as nb
import geopandas as gpd

# MIN_SEGMENT_SIZE = 277
MAX_ID = 2**16-1


#@nb.njit()
def _reindex(res_img, too_high_ids, free_ids):
    j = 0
    for id in too_high_ids:
        res_img[res_img == id] = free_ids[j]
        j += 1
        if j == len(free_ids):
            break
    return res_img


def _filter_and_reindex(img):
    img = img.copy()
    # ids, counts = np.unique(img, return_counts=True)
    # img[np.isin(img, ids[counts < MIN_SEGMENT_SIZE])] = 0

    ids = np.unique(img)

    res_img = np.ravel(img)
    free_ids = np.arange(MAX_ID)
    free_ids = free_ids[~np.isin(free_ids, ids)]
    too_high_ids = ids[ids > MAX_ID]
    # print("\nReindexing {} segments".format(len(too_high_ids)))
    _reindex(res_img, too_high_ids, free_ids)
    return res_img.reshape(img.shape)


# Create a generator of polygons from the raster segments
def raster_to_polygons(raster):
    for geom, value in rasterio.features.shapes(raster):
        yield shapely.geometry.shape(geom), value


def raster_to_gdf(raster):
    #    print("1: start reindexing")
    #    raster = _filter_and_reindex(raster)
    # Convert raster segments to vector polygons
    print("\n1: start polygonizing")
    polygonized = raster_to_polygons(raster.astype(rasterio.int32))
    print("2: start creating features")
    polygons = []
    for polygon, value in polygonized:
        if value != 0:  # Skip polygons where value is 0
            polygons.append(
                {'geometry': polygon, 'properties': {'value': int(value)}})
    print("3: start creating dataframe")
    # Convert polygons to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(polygons)
    return gdf

def get_transformation(tile_geometry):
    tile_coords = np.array(tile_geometry.exterior.coords)[:-1, ::-1]

    original_corners = [(0, 4096), (4096, 4096), (4096, 0), (0, 0)]
     # Calculate transformation matrices
    A = np.column_stack((tile_coords[:, ::-1], np.ones(len(tile_coords))))
    B = np.column_stack((original_corners, np.ones(len(original_corners))))

    T = np.linalg.lstsq(B, A, rcond=None)[0].T
    T = (T[0, 0], T[0, 1], T[1, 0], T[1, 1], T[0, 2], T[1, 2])

    return T

