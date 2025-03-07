import pathlib
import json

root_path = "/mnt/data-home/mobility-multimodal/vlm-annotations"
folder_list = [
#    "Sports_Development/20241223/Sports_Development_20241223_curated_t7-revised",
#    "Sports_Development/20241223/Sports_Development_20241223_curated_t5_VLA_makeup-revised",
#    "Sports_Development/20250109/Sports_Development_20250109_llava-onevision-0.5b-full-revised",
#    "Water_Resources/20250106/Water_Resources_20250106_curated_t5-revised",
#    "Transportation/20250109/Transportation_20250109_curated_t7-revised",
#    "Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95-revised",
#    "Transportation/20241230/Transportation_20241230_curated_t7-revised",
#    "Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95-revised",
#    "Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95-revised",
#    "Public_Works/20241230/Public_Works_20241230_curated_t5_part-revised",
]
linker_vision_path = "Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part"
for i in range(1, 6):
    for j in range(1, 5):
        folder_list.append(f"{linker_vision_path}{i}_{j}")
i = 6
for j in range(1, 4):
    folder_list.append(f"{linker_vision_path}{i}_{j}")

folder_list = [root_path + '/' + folder for folder in folder_list]


total_data_num = 0
for depart_path in folder_list:
    images_path = pathlib.Path(depart_path) / 'images'
    if not images_path.exists():
        print(f"{depart_path} images not found")
    # find number of files under images_path
    # print(f"{depart_path=}")
    image_num = len(list(images_path.glob('*')))
    print(f"{image_num=}")
    anno_path = pathlib.Path(depart_path) / 'annotations' / 'revised_anno.json'
    if not anno_path.exists():
        print(f"{depart_path} annotations not found")
    with anno_path.open() as f:
        data = json.load(f)
    anno_num = len(data)
    print(f"{anno_num=}") 
    assert image_num == anno_num, f"number of images and annotations are not equal for {depart_path}"
    total_data_num += anno_num
print(len(folder_list))
print(f"{total_data_num=}")
