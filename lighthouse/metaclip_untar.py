import os
import tarfile

def untar_all_under_split_dirs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath).startswith("split"):
            print(f"\n[INFO] Processing: {dirpath}")
            for fname in filenames:
                if fname.endswith(".tar"):
                    tar_path = os.path.join(dirpath, fname)
                    try:
                        with tarfile.open(tar_path, "r") as tar:
                            tar.extractall(path=dirpath)
                            print(f"  ✅ Extracted: {tar_path}")
                    except Exception as e:
                        print(f"  ❌ Failed to extract {tar_path}: {e}")

root_dir = "/mnt/lighthouseACD/image_text/hand0428"
untar_all_under_split_dirs(root_dir)

# Example usage: scan the folders you already extracted
# untar_all_under_split_dirs("extracted_000000")
# untar_all_under_split_dirs("extracted_000001")
