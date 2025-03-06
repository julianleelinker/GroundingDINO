import pathlib
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import tqdm


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
linker_vision_path = "Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part"
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


excel_path = '/mnt/data-home/julian/lighthouse/misc/第一期交付_文本與圖像的多模態數據標註10萬組.xlsx'
df = pd.read_excel(excel_path)
print(df.columns)

# iterate over the rows with iterrows
dataset_field = '(B) 資料標註審查'
image_field = 'Unnamed: 5'
invalid_dataset = {'DataSet 資料夾(局處＋Linker)', np.nan}
problematic_images = defaultdict(set)
for index, row in df.iterrows():
    if row[dataset_field] not in depart_path:
        invalid_dataset.add(row[dataset_field])
        continue
    dataset_path = pathlib.Path(root_path) / depart_path[row[dataset_field]]
    image_name = row[image_field]
    if image_name is np.nan:
        continue
    problematic_images[dataset_path].add(image_name)

removed = pathlib.Path('/mnt/data-home/mobility-multimodal/vlm-annotations/Water_Resources/20250106/Water_Resources_20250106_curated_t5')
problematic_images.pop(removed)
bad_data = pathlib.Path('/mnt/data-home/mobility-multimodal/vlm-annotations/Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95')

# problematic_images[bad_data].pop('DJI_20240903101645_0088_D.jpg') # this name will not be removed, its ok to keep in problematic_images, just add another correct one
problematic_images[bad_data].add('DJI_20240903101645_0088_D.JPG')



count = 0
found = defaultdict(set)
for dataset_path, images in problematic_images.items():
    anno_path_list = list((dataset_path / 'annotations').rglob('*.json'))
    annos = []
    for anno_path in anno_path_list:
        with anno_path.open() as f:
            anno_data = json.load(f)
        annos.extend(anno_data)

    new_anno = []
    for anno in tqdm.tqdm(annos):
        if anno['image'] in images:
            found[dataset_path].add(anno['image'])
            del_image = dataset_path / 'images' / anno['image']
            count += 1
            if del_image.exists():
                # deltete del_image
                del_image.unlink()
                print(f"{del_image} deleted")
            else:
                print(f"{del_image} does not exist")
        else:
            new_anno.append(anno)

    new_anno_path = dataset_path / 'annotations' / 'new_anno_0306.json'
    with new_anno_path.open('w') as f:
        json.dump(new_anno, f, ensure_ascii=False, indent=4)
print(f'{count=}')
diff = dict()
for dataset_path, images in problematic_images.items():
    diff[dataset_path] = images - found[dataset_path]
import ipdb; ipdb.set_trace()
