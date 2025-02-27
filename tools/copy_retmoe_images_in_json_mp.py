import json
import pathlib
import subprocess
import multiprocessing
from tqdm import tqdm

SOURCES_BY_DEPART = {
    'Public_Works': 0,
    'Water_Resources': 0,
    'Transportation': 0,
    'Mass_Rapid_Transit': 0,
    'Sports_Development': 0,
    'Taiwan_Power': 0,
    'Ports_Corporation': 0,
    'China_Steel': 0,
    'Linker_Vision_Data_V2': 0,
    'Kaohsiung_Data_V2': 0,
}

JSON_LIST_JAN = [
    '/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250109/Transportation_20250109_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95.json',
]

ROOT_LIST = {
    'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'
}

REMOTE_HOST = 'julian@10.1.24.90'
OUTPUT_ROOT = '/mnt/data-home/julian/lighthouse/augmented-curated-data'
FILES_PER_FOLDER = 5000
NUM_PROCESSES = 32


def get_concat_file_name(file_path, root_list):
    """Get concatenated filename based on root folders"""
    root = file_path.parent
    while root.name not in root_list:
        root = root.parent
    return str(file_path.relative_to(root)).replace('/', '-')


def rsync_file(remote_path):
    """Copy file using rsync with multiprocessing"""
    global duplicated_count

    image_name = get_concat_file_name(remote_path, ROOT_LIST)
    folder_index = (image_path_list.index(remote_path) // FILES_PER_FOLDER) + 1
    local_path = output_folder / f'split{folder_index}' / image_name

    if not local_path.exists():
        command = ["rsync", "-avz", f"{REMOTE_HOST}:{remote_path}", str(local_path)]
        subprocess.run(command, check=False)
    else:
        return remote_path  # File already exists


if __name__ == '__main__':
    for json_path in JSON_LIST_JAN:
        print(f'Processing JSON: {json_path}')

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        image_path_list = [pathlib.Path(data) for data in json_data]
        output_folder = pathlib.Path(OUTPUT_ROOT) / pathlib.Path(json_path).stem
        output_folder.mkdir(parents=True, exist_ok=True)

        # Create split folders
        for i in range(1, (len(image_path_list) - 1) // FILES_PER_FOLDER + 2):
            (output_folder / f'split{i}').mkdir(parents=True, exist_ok=True)

        # Use multiprocessing to parallelize rsync
        with multiprocessing.Pool(NUM_PROCESSES) as pool:
            duplicated_files = list(tqdm(pool.imap(rsync_file, image_path_list), total=len(image_path_list)))

        # Count duplicates
        duplicated_count = sum(1 for item in duplicated_files if item is not None)

        print(f'{duplicated_count} files were already present and skipped.')
        print(f'Files saved to {str(output_folder)}')
