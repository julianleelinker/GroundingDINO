import pathlib
import pandas as pd

excel_path = '/mnt/data-home/julian/lighthouse/misc/第一期交付_文本與圖像的多模態數據標註10萬組.xlsx'
df=pd.read_excel(excel_path)
print(df.columns)

depart_map = dict()
root_path = "/mnt/data-home/mobility-multimodal/vlm-annotations"
depart_path = {
   "運發局_20241223": "Sports_Development/20241223/Sports_Development_20241223_curated_t7",
   "運發局_20241223_part2 ": "Sports_Development/20241223/Sports_Development_20241223_curated_t5_VLA_makeup",
   "運發局_20250109": "Sports_Development/20250109/Sports_Development_20250109_llava-onevision-0.5b-full",
   "水利局_20250106 ": "Water_Resources/20250106/Water_Resources_20250106_curated_t5",
   "交通局_20250109 ": "Transportation/20250109/Transportation_20250109_curated_t7",
   "捷運局_20250109 ": "Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95",
   "交通局_20241230 ": "Transportation/20241230/Transportation_20241230_curated_t7",
   "交通局_20241230": "Transportation/20241230/Transportation_20241230_curated_t7",
   "台電_20250106 ": "Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95",
   "工務局_20250106 ": "Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95",
   "工務局_20241230 ": "Public_Works/20241230/Public_Works_20241230_curated_t5_part",
}
linker_vision_path = "Kaohsiung-full-dataset/Kaoshsiung_76152_retrieval_curated_t220_part"
for i in range(1, 6):
    for j in range(1, 5):
        depart_path[f"Linker_Vision_part{i}_{j}"] = f"{linker_vision_path}{i}_{j}"
i = 6
for j in range(1, 4):
    depart_path[f"Linker_Vision_part{i}_{j}"] = f"{linker_vision_path}{i}_{j}"

for depart, local_path in depart_path.items():
    images_path = pathlib.Path(root_path) / local_path / 'images'
    if not images_path.exists():
        print(f"{depart} images not found")
    anno_path = pathlib.Path(root_path) / local_path / 'annotations'
    if not anno_path.exists():
        print(f"{depart} annotations not found")

import ipdb; ipdb.set_trace()
