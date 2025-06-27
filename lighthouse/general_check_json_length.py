import json
import fire
import pathlib


def main(json_path):
    # json_path = "/mnt/data-home/mobility-multimodal/data-curation/Taiwan_Power/20250526/Taiwan_Power_20250526_image_list_keep_0.95.json"
    json_root = pathlib.Path("/mnt/data-home/mobility-multimodal/data-curation/Bus")
    json_list = list(json_root.glob("**/*.json"))
    json_list = [str(x) for x in json_list if "image_list_keep_0.95" in str(x)]

    for json_path in json_list:
        with open(json_path, 'r') as f:
            image_list = json.load(f)
        print(f"{json_path=}")
        print(f"{len(image_list)=}")


if __name__ == "__main__":
    fire.Fire(main)