
import pathlib
from common import get_depart, DEPARTS_EN, DATA_CURATION_ROOT, DINO_COCO_SPLITS_0418, DINO_COCO_FOLDERS_0418, DINO_COCO_SPLITS_0508,  DINO_COCO_SPLITS_0617, AUGMENTED_CURATED_EXCLUDED_JSONS, AUGMENTED_CURATED_JSONS_0617
import pandas as pd
import json


root = "/mnt/lighthouseACD/QAed-data/bbox/upload0527-iou38"
column_names = ["uploaded", "remaining"]
row_names = DEPARTS_EN + ["total"]
df = pd.DataFrame(0, index=row_names, columns=column_names)

uploaded_list = list(pathlib.Path(root).rglob("*split*"))
for uploaded in uploaded_list:
    depart = get_depart(uploaded)
    df.loc[depart, "uploaded"] += len(list(uploaded.glob("images/*")))

for json_file in AUGMENTED_CURATED_JSONS_0617:
    with open(json_file, 'r') as f:
        data = json.load(f)
    depart = get_depart(json_file)
    df.loc[depart, "remaining"] += len(data)
df.loc["total"] = df.sum(axis=0)
df_formatted = df.map(lambda x: f"{x:,}")
print(df_formatted)

import ipdb; ipdb.set_trace()