# %%
import pathlib


src_root = "/mnt/lighthouseACD/QAed-data/bbox/"
dst_root = "/mnt/lighthouseACD/image_text/"
root_name = "hand0428"
root_path = pathlib.Path(f"{src_root}/{root_name}")
split_dirs = list(root_path.rglob("*split*"))
non_exist_dirs = []
exsit_dirs = []
image_count = {"Public_Works":0, "Sports_Development":0, "Transportation":0, "Water_Resources":0, "Taiwan_power":0, "China_steel":0, "Mass_rapid_transit":0, "Ports_corporation":0}
for split_dir in split_dirs:
    dst_dir = str(split_dir).replace(src_root, dst_root) + "_s2.5_mt0.26"
    dst_dir = pathlib.Path(dst_dir)
    if dst_dir.exists():
        exsit_dirs.append(split_dir)
    else:
        non_exist_dirs.append(split_dir)
        for depart in image_count.keys():
            if depart in str(split_dir):
                depart_name = depart
                break
        image_count[depart_name] += len(list((split_dir / 'images').glob("*.jpg")))
        print(f"Found non exist split dir: {split_dir} with {len(list((split_dir / 'images').glob('*.jpg')))} images")

print(f"{split_dirs=}")
print(f"{len(non_exist_dirs)=} non-exist split dirs")
print(f"{len(exsit_dirs)=} exist split dirs")
print(f"{image_count=}")
# %%
for x in non_exist_dirs:
    print(x)

# %%
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split19_0.30_0.35 with 3729 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split20_0.30_0.35 with 3879 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split21_0.30_0.35 with 3941 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split23_0.30_0.35 with 3905 images

/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split25_0.30_0.35 with 3615 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split26_0.30_0.35 with 3689 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split27_0.30_0.35 with 3669 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split28_0.30_0.35 with 4008 images

/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split1_0.30_0.35 with 425 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split29_0.30_0.35 with 4279 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split2_0.30_0.35 with 716 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split31_0.30_0.35 with 2770 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split3_0.30_0.35 with 1434 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split4_0.30_0.35 with 458 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split5_0.30_0.35 with 733 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split6_0.30_0.35 with 927 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split7_0.30_0.35 with 352 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split8_0.30_0.35 with 1855 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20241230_image_list_keep_0.95/split9_0.30_0.35 with 1596 images
/mnt/lighthouseACD/QAed-data/bbox/hand0428/Public_Works_20250106_image_list_keep_0.95/split1_0.30_0.35 with 179 images
