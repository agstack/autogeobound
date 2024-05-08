import os
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

data_dir = "/home/johannes/data/"

#source ="gs://india_field_delineation/metadata/vectorization/tiles_karnataka.gpkg"
#dest = data_dir
#os.system(
#    f"gsutil cp {source} {dest}"
#)

meta_df = gpd.read_file(os.path.join(data_dir, "tiles_karnataka.gpkg"))

for folder in ["watershed", "instance_uncertainty", "semantic_uncertainty", "res"]:
    os.makedirs(os.path.join(data_dir, folder), exist_ok=True)

batches = meta_df.batch.unique()

for batch in tqdm(batches):
    source = "gs://india_field_delineation/predictions/watershed_instances_4096p_tileable/res/india_{:02d}/*.npy".format(batch)
    dest = os.path.join(data_dir, "watershed")
    os.system(
        f"gsutil cp {source} {dest}"
    )
    print("downloaded watershed")

    source = "gs://india_field_delineation/predictions/instance_uncertainty_4096px/india_{:02d}/*.png".format(batch)
    dest = os.path.join(data_dir, "instance_uncertainty")
    os.system(
        f"gsutil cp {source} {dest}"
    )
    print("downloaded instance uncertainty")

    source = "gs://india_field_delineation/predictions/semantic_uncertainty_4096px/india_{:02d}/*.png".format(batch)
    dest = os.path.join(data_dir, "semantic_uncertainty")
    os.system(
        f"gsutil cp {source} {dest}"
    )
    print("downloaded semantic uncertainty")
