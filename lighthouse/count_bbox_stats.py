import pathlib
import pandas as pd
from common import get_depart, DEPARTS_EN


data_root_map = {
    "/mnt/data-home/mobility-multimodal/revised_bbox/deduplicated": "*/*",
    "/mnt/data-home/mobility-multimodal/revised_bbox/datasets": "*/*",
    "/mnt/lighthouseACD/ACD-gdino-COCO/Transportation_20250115_image_list_keep_0.95_rededuplicate": "*",
}

data_name_map = {
    "/mnt/data-home/mobility-multimodal/revised_bbox/deduplicated": "deduplicated",
    "/mnt/data-home/mobility-multimodal/revised_bbox/datasets": "datasets",
    "/mnt/lighthouseACD/ACD-gdino-COCO/Transportation_20250115_image_list_keep_0.95_rededuplicate": "trans_250115",
}

column_names = ["deduplicated", "datasets", "trans_250115", "bbox all"]
row_names = DEPARTS_EN + ["total"]
df = pd.DataFrame(0, index=row_names, columns=column_names)

for data_root, pattern in data_root_map.items():
    folder_list = list(pathlib.Path(data_root).glob(pattern))
    data_name = data_name_map[data_root]
    for folder in folder_list:
        image_list = list((folder/"images").glob("*"))
        df.loc[get_depart(folder), data_name] += len(image_list)

df["bbox all"] = df["deduplicated"] + df["datasets"] + df["trans_250115"]
df.loc["total"] = df.sum(axis=0)
df_formatted = df.map(lambda x: f"{x:,}")
print(df_formatted)
import ipdb; ipdb.set_trace()
