from common import VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS, DINO_COCO_ROOT, VLM_ANNOTATION_ROOT
from common import load_stats_fwf
import pandas as pd


numeric_cols = ["number", "annotated_number", "ckpt_data_number"]

df_bbox = load_stats_fwf(f"{DINO_COCO_ROOT}/dino_coco_data_stats.txt", numeric_cols)
df_deprecated = df_bbox[df_bbox["notes"] == "deprecated"]
df_bbox_uploaded = df_bbox[(df_bbox["is_uploaded"] == True) & (df_bbox["notes"] == "deprecated")]
df_deprecated_done = df_deprecated[df_deprecated["annotated_number"]!=0]
df_deprecated_done_uploaded = df_deprecated_done[df_deprecated_done["is_uploaded"]]
df_new = df_bbox[df_bbox["notes"] == "no"]

df_deprecated_done.groupby("depart")["annotated_number"].sum()
df_new.groupby("depart")["number"].sum()

vlm_loaded = load_stats_fwf(f"{VLM_ANNOTATION_ROOT}/vlm_data_stats.txt", numeric_cols)
ckpt1_2 = vlm_loaded[vlm_loaded["ckpt_status"].isin(["ckpt1", "ckpt2"])]

import ipdb; ipdb.set_trace()



