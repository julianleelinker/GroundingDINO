# %%
import pathlib
import tqdm

old_root = pathlib.Path(f"/mnt/lighthouseACD/QAed-data/bbox/hand0526")
new_root = pathlib.Path(f"/mnt/lighthouseACD/QAed-data/bbox/hand0526-merged")
# new_root = pathlib.Path(f"/mnt/lighthouseACD/QAed-data/bbox/hand0526-merge-updated")

old_split_list = list(old_root.glob("*/*"))
new_split_list = list(new_root.glob("*/*"))
# %%

# tmp = old_split_list[0]
# find the corresponding new split, using replace name
differ_numbers = 0
for old_path in tqdm.tqdm(old_split_list):
    new_path = pathlib.Path(str(old_path).replace(str(old_root), str(new_root)).replace("_0.30_0.35", "_0.30_0.35_s2.5_mt0.26"))
    assert new_path.exists()
    old_images = list(old_path.glob("images/*"))
    new_images = list(new_path.glob("images/*"))
    if len(old_images) != len(new_images):
        print(f"{old_path} and {new_path} have different number of images: {len(old_images)} vs {len(new_images)}")
        differ_numbers += 1
print(f"Total different number of images: {differ_numbers}")

# %%
