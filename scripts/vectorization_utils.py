import numpy as np
import rasterio
import shapely
import numpy as nb
import geopandas as gpd

# MIN_SEGMENT_SIZE = 277
MAX_ID = 2**16-1


@nb.njit()
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
    print("1: start reindexing")
    raster = _filter_and_reindex(raster)
    # Convert raster segments to vector polygons
    print("2: start polygonizing")
    polygonized = raster_to_polygons(raster.astype(np.uint16))
    print("3: start creating features")
    polygons = []
    for polygon, value in polygonized:
        if value != 0:  # Skip polygons where value is 0
            polygons.append(
                {'geometry': polygon, 'properties': {'value': int(value)}})
    print("4: start creating dataframe")
    # Convert polygons to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(polygons)
    return gdf


def _sort_quad(polygon):
    # Create a Shapely polygon from the coordinates
    polygon_coords = np.array(polygon.exterior.coords)[:-1, :]

    # Calculate the centroid of the polygon
    centroid = polygon.centroid

    # Get the centroid coordinates
    centroid_x, centroid_y = centroid.x, centroid.y

    # Create a list to store the vertices categorized by their positions
    # {'lower_left': [], 'lower_right': [], 'upper_left': [], 'upper_right': []}
    sorted_vertices = []

    # Iterate through each vertex and categorize them based on their positions relative to the centroid
    for vertex in polygon_coords:
        x, y = vertex
        if x < centroid_x and y < centroid_y:
            sorted_vertices.append(vertex)
        elif x > centroid_x and y < centroid_y:
            sorted_vertices.append(vertex)
        elif x > centroid_x and y > centroid_y:
            sorted_vertices.append(vertex)
        elif x < centroid_x and y > centroid_y:
            sorted_vertices.append(vertex)

    return np.array(sorted_vertices)


def map_points(geoseries, transformed_corners):
    transformed_corners = _sort_quad(transformed_corners)
    original_corners = [(0, 4096), (4096, 4096), (4096, 0), (0, 0)]

    # Calculate transformation matrices
    A = np.column_stack(
        (transformed_corners[:, ::-1], np.ones(len(transformed_corners))))
    B = np.column_stack((original_corners, np.ones(len(original_corners))))
    T = np.linalg.lstsq(B, A, rcond=None)[0]

    # Apply transformation to each polygon in the GeoSeries
    transformed_geometries = []
    for polygon in geoseries.geometry:
        transformed_exterior = []
        for point in polygon.exterior.coords:
            # Extend point to include homogeneous coordinate
            extended_point = [point[0], point[1], 1]
            transformed_point = np.dot(extended_point, T)
            transformed_exterior.append(
                (transformed_point[1], transformed_point[0]))
        transformed_polygon = shapely.geometry.Polygon(transformed_exterior)
        transformed_geometries.append(transformed_polygon)

    return gpd.GeoSeries(transformed_geometries)
