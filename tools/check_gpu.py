import subprocess
import time

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

threshold = 4000
wait_time = 6
while True:
    result = get_gpu_memory()
    for gpu, memory in result.items():
        if memory > threshold:
            print(f"run command on GPU {gpu}")
        else:
            print(f"GPU {gpu} has no available memory, check later")
    print(f'Waiting for {wait_time} seconds...')
    time.sleep(wait_time)
    print(f'')
