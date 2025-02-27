import os
import pathlib
import shutil
import tqdm


def distribute_files_into_folders(
    source_dir: str,
    dest_dir: str,
    files_per_folder: int = 5000
):
    """
    Distribute all files from source_dir evenly into `num_folders` subfolders
    in dest_dir.
    """
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    # files = sorted(files)

    for i in range((len(files)-1)//files_per_folder + 1):
        (pathlib.Path(dest_dir) / f'split{i}').mkdir(parents=True, exist_ok=True)
        # os.makedirs(os.path.join(dest_dir, f"split_{i}"), exist_ok=True)
    
    for index, filename in tqdm.tqdm(enumerate(files), total=len(files)):
        folder_index = index // files_per_folder
        
        src_path = os.path.join(source_dir, filename)
        dest_path = pathlib.Path(dest_dir) / f'split{folder_index}' / filename
        # dest_path = os.path.join(dest_dir, f"split_{folder_index}", filename)
        
        if os.path.exists(dest_path):
            continue
        shutil.move(src_path, dest_path)

    print(f"Done! {len(files)} files have been distributed to {folder_index} splits, with {files_per_folder} per folder.")

if __name__ == "__main__":
    # Example usage:
    # SOURCE_DIR = "/path/to/your/source"
    # SOURCE_DIR = '/mnt/data-home/mobility-multimodal/data-curation/Kaohsiung-full-dataset/100K_images'
    # SOURCE_DIR = '/mnt/data-home/julian/lighthouse/augmented-curated-data/Sports_Development_20241223_image_list_keep_0.95'
    SOURCE_DIR = '/mnt/data-home/julian/lighthouse/augmented-curated-data/Public_Works_20241230_image_list_keep_0.95/'
    DEST_DIR = SOURCE_DIR
    FILES_PER_FOLDER = 5000

    distribute_files_into_folders(SOURCE_DIR, DEST_DIR, FILES_PER_FOLDER)