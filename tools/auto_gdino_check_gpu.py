import subprocess
import time
import pathlib


def get_gpu_memory():
    try:
        command = "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits"
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")  # Split output into lines
            result = {}
            for idx, line in enumerate(lines):
                total_memory, used_memory = map(int, line.split(", "))  # Convert to integers
                print(f"GPU {idx}: {used_memory}MB / {total_memory}MB")
                result[idx] = total_memory - used_memory
            return result
        else:
            print("Error running nvidia-smi:", result.stderr)
            return None
    except Exception as e:
        print("Error:", e)
        return None


CONDA_ENV = "lighthouse"
DATASET = 'Public_Works_20241230_image_list_keep_0.95'
DATA_ROOT = '/mnt/lighthouseACD/augmented-curated-data/'
THRESHOLD = 4000
WAIT_TIME = 300

base_command = (
    "CUDA_VISIBLE_DEVICES={gpu_id} nohup conda run -n {conda_env} python demo/inference_gpt_on_images_in_folder.py "
    "-c groundingdino/config/GroundingDINO_SwinT_OGC.py "
    "-p weights/groundingdino_swint_ogc.pth "
    "-o /mnt/lighthouseACD/ACD-gdino-COCO/{dataset} "
    "--box_threshold 0.4 "
    "--text_threshold 0.3 "
    "--high_threshold 0.35 "
    "-t 'all' "
    "--enlarge_scale 1.4 "
    "--ios_threshold 0.3 "
    "-i {dataroot}/{dataset}/split{i} "
    "> {dataset}-{i}.log 2>&1 &"
)

while True:
    result = get_gpu_memory()
    for gpu_id, memory in result.items():
        if memory > THRESHOLD:
            data_path_list = pathlib.Path(f'{DATA_ROOT}/{DATASET}').glob('split*')
            data_path_list = sorted(data_path_list, key=lambda x: int(str(x).split('split')[-1]))
            for data_path in data_path_list:
                if not (data_path / 'runned').exists():
                    break
            split_index = int(str(data_path).split('split')[-1])
            command = base_command.format(i=split_index, conda_env=CONDA_ENV, dataroot =DATA_ROOT, dataset=DATASET, gpu_id=gpu_id)
            print(f"GPU {gpu_id} has enough memory, running inference on {data_path}")
            print(f"Executing: {command}")
            subprocess.run(command, shell=True, executable="/bin/bash")
            (data_path / 'runned').touch()
        else:
            print(f"GPU {gpu_id} doesn't have enough memory, check later")
    print(f'Waiting for {WAIT_TIME} seconds...\n')
    time.sleep(WAIT_TIME)
