from common import VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS, DINO_COCO_ROOT, VLM_ANNOTATION_ROOT
import pandas as pd


bbox = pd.read_csv(f"{DINO_COCO_ROOT}/dino_coco_stats.csv")
bbox_depr = bbox[bbox["notes"] == "deprecated"]
bbox_uploaded = bbox[(bbox["uploaded"] == True) & (bbox["notes"] == "deprecated")]
bbox_depr_done = bbox_depr[bbox_depr["annotated"]!=0]
bbox_depr_done_uploaded = bbox_depr_done[bbox_depr_done["uploaded"]]
bbox_new = bbox[bbox["notes"] == "no"]

bbox_depr_done.groupby("depart")["annotated"].sum()
bbox_new.groupby("depart")["number"].sum()

print("current done bbox data stats")
print(bbox_depr_done.groupby("depart")["annotated"].sum().apply(lambda x: f"{x:,}"))
print("current new bbox data stats")
print(bbox_new.groupby("depart")["number"].sum().apply(lambda x: f"{x:,}"))
pc_new = bbox_new[bbox_new["depart"]=="Ports_Corporation"]
sd_new = bbox_new[bbox_new["depart"]=="Sports_Development"]
mrt_new = bbox_new[bbox_new["depart"]=="Mass_Rapid_Transit"]


vlm_loaded = pd.read_csv(f"{VLM_ANNOTATION_ROOT}/vlm_data_stats.csv")
ckpt1_2 = vlm_loaded[vlm_loaded["ckpt"].isin(["ckpt1", "ckpt2"])]

import ipdb; ipdb.set_trace()



