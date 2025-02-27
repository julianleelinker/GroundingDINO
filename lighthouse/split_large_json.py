import json
import argparse
import pathlib


def make_parser():
    parser = argparse.ArgumentParser("ask chatgpt to describe image")
    parser.add_argument(
        "-j",
        "--json_path",
        required=True,
        type=str,
        help="the json file path of image list",
    )
    parser.add_argument(
        "-s",
        "--split_size",
        type=int,
        help="the size of each split json file",
        default=10000,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = make_parser()
    json_path = args.json_path
    file_name = pathlib.Path(json_path).stem
    file_root = pathlib.Path(json_path).parent
    print(file_name)
    
    with open(args.json_path, 'r') as f:
        json_data = json.load(f)

    print(f'{len(json_data)=}')
    for i in range(0, len(json_data), args.split_size):
        save_json = f"{file_root}/{file_name}_part{(i//args.split_size+1)}.json"
        with open(save_json, 'w') as f:
            json.dump(json_data[i:i+args.split_size], f, indent=4, ensure_ascii=False)
        print(f"json split dump to {save_json}")