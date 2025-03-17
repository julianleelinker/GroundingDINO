import tqdm
from common import VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS, DEPARTS_EN, DINO_COCO_FOLDERS
from common import get_depart
import pandas as pd


column_names = ["VLM CKPT1", "VLM CKPT2", "VLM ALL", "DINO COCO"]
row_names = DEPARTS_EN + ["total"]
df = pd.DataFrame(0, index=row_names, columns=column_names)

for folder_list, name in [
        (VLM_CKPT1_FOLDERS, "VLM CKPT1"),
        (VLM_CKPT2_FOLDERS, "VLM CKPT2"),
    ]:
    print(f"counting {name} data ...")
    for folder in tqdm.tqdm(folder_list):
        image_list = list((folder/"images").glob("*"))
        df.loc[get_depart(folder), name] += len(image_list)
df["VLM ALL"] = df["VLM CKPT1"] + df["VLM CKPT2"]
print(df.map(lambda x: f"{x:,}"))

print("counting DINO COCO data...")
for folder in tqdm.tqdm(DINO_COCO_FOLDERS):
    splits = list(folder.glob("split*"))
    for split in splits:
        image_list = list((split/"images").glob("*"))
        df.loc[get_depart(folder), "DINO COCO"] += len(image_list)
df.loc["total"] = df.sum(axis=0)
df_formatted = df.map(lambda x: f"{x:,}")
print(df.map(lambda x: f"{x:,}"))