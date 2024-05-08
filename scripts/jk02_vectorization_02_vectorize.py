from tqdm import tqdm
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import cv2 as cv
import sys
import shapely.affinity

from vectorization_utils import raster_to_gdf, get_transformation

batch = int(sys.argv[1])
data_dir = "/home/johannes/data/"

meta_df = gpd.read_file(os.path.join(data_dir, "tiles_karnataka.gpkg"))

gdf_batch = pd.DataFrame()

meta_df_batch = meta_df.loc[meta_df["batch"] == batch]
for i, row in tqdm(meta_df_batch.iterrows()):
    tile_id = row["tile_id"]
    batch = row["batch"]

    img_path = os.path.join(
        data_dir, "watershed/india_{:05d}.npy".format(tile_id))
    if os.path.exists(img_path):
        watershed_img = np.load(img_path)
    else:
        print("Segmentation for image {:05d} not found!".format(tile_id))
        continue
    img_path = os.path.join(
        data_dir, "instance_uncertainty/india_{:05d}.png".format(tile_id))
    if os.path.exists(img_path):
        instance_u_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    else:
        print("Instance u for image {:05d} not found!".format(tile_id))
        continue
    img_path = os.path.join(
        data_dir, "semantic_uncertainty/india_{:05d}.png".format(tile_id))
    if os.path.exists(img_path):
        semantic_u_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    else:
        print("Semantic u for image {:05d} not found!".format(tile_id))
        continue

    tile_geometry = row.geometry
    transformation = get_transformation(tile_geometry)

    gdf = raster_to_gdf(watershed_img)

    print("3: add metainfo")
    centroid_coords = np.array([(point.x, point.y)
                                for point in gdf.geometry.centroid])
    centroid_coords = centroid_coords.astype(int)
    gdf["instance_uncertainty"] = instance_u_img[centroid_coords[:, 0],
                                                 centroid_coords[:, 1]] / 255.0
    gdf["semantic_uncertainty"] = semantic_u_img[centroid_coords[:, 0],
                                                 centroid_coords[:, 1]] / 255.0
    gdf["tile_id"] = tile_id
    gdf["batch"] = batch

    print("4: transform")
    gdf.geometry = gdf.geometry.apply(lambda geom: shapely.affinity.affine_transform(geom, np.ravel(transformation)))

    print("5: add to batch df")
    gdf_batch = pd.concat([gdf_batch, gdf], ignore_index=True)

gdf_batch = gpd.GeoDataFrame(gdf_batch, crs=gdf_transformed.crs)
gdf_batch.set_geometry(gdf_batch.geometry, inplace=True)

res_file = os.path.join(data_dir, '/res/india_{:02d}.gpkg'.format(batch))
gdf_batch.to_file(res_file, driver='GPKG', layer='segments')
dest = "gs://india_field_delineation/predictions/vectorized/"
os.system(
    f"gsutil cp {res_file} {dest}"
)
