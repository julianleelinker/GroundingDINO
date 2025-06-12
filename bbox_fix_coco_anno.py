import pathlib
import tqdm
import json


def main():
    # data_root = "/mnt/lighthouseACD/QAed-data/bbox/hand0526-merge-updated/"
    data_root = "/mnt/lighthouseACD/QAed-data/bbox/upload0527-merged/"
    split_root_list = list(pathlib.Path(f"{data_root}").glob("*/*"))
    print(f"Number of splits: {len(split_root_list)}")

    # split_root_list = DINO_COCO_SPLITS_0508
    # output_root = pathlib.Path(f"/mnt/lighthouseACD/QAed-data/bbox/upload0527-merged")

    # for split in split_root:
    #     image_path_list = list(split.rglob("images/*"))
    #     print(f"Split: {split}, Number of images: {len(image_path_list)}")

    # split_root = pathlib.Path("/mnt/lighthouseACD/QAed-data/bbox/hand0526/Transportation_20250109_image_list_keep_0.95/split16_0.30_0.35")
    print(len(split_root_list))
    # import ipdb; ipdb.set_trace()
    for split_root in tqdm.tqdm(split_root_list):
        json_path = split_root / "annotations" / "labels.json"
        with open(json_path, "r") as f:
            anno_data = json.load(f)
        anno_data["categories"] = [{"id": 1, "name": "object"}]
        with open(json_path, "w") as f:
            json.dump(anno_data, f, indent=4)

if __name__ == "__main__":
    main()