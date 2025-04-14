import pathlib
from common import get_depart, DATA_CURATION_ROOT, AUGMENTED_CURATED_EXCLUDED_JSONS


def main():
    json_list = [path for path in pathlib.Path(DATA_CURATION_ROOT).rglob('*image_list_keep*') if path.is_file()]
    print(len(json_list))
    json_list = [x for x in json_list if x not in AUGMENTED_CURATED_EXCLUDED_JSONS]
    print(len(json_list))

    for json_path in json_list:
        print(f'"{json_path}",')


if __name__ == "__main__":
    main()