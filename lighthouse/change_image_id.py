import pathlib
import json
import tqdm
import argparse


def make_parser():
    parser = argparse.ArgumentParser("ask chatgpt to describe image")
    parser.add_argument(
        "-f",
        "--folder",
        required=True,
        type=str,
        help="the data folder path of to change image id",
    )
    parser.add_argument(
        "-i",
        "--start_id",
        required=True,
        type=int,
        help="the start id of image",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = make_parser()
    annotation_root = pathlib.Path(args.folder) / 'annotations'
    json_path_list = sorted(annotation_root.rglob('*.json'))
    i = args.start_id
    for json_path in json_path_list:
        with json_path.open('r') as f:
            json_data = json.load(f)
        for data in tqdm.tqdm(json_data):
            data['id'] = i
            i += 1
        with json_path.open('w') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f'number of annotations: {i - args.start_id}')
    image_root = pathlib.Path(args.folder) / 'images'
    file_list = list(image_root.rglob('*'))
    print(f'number of files in images: {len(file_list)}')
    print(f'next start id: {i}')

