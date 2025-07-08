import json
import pathlib
import tqdm


# data_root = "/mnt/lighthouseACD/image_text-back/"
data_root = "/mnt/lighthouseACD/image_text"
image_list = list(pathlib.Path(data_root).rglob("*.jpg"))

# data for metaclip
# a json store the fowllowing information:
# a list of dict, each dict contains:
# "image_path": "/mnt/Jan/Sports_Development/20250109/NO.22_H車道/20250109_NO.22_H車道_2024_10_3 上午 (UTC+08_00) 09_59_59_s0.jpg",
# "text":  "Provide a one-sentence caption​ for the provided image.", 

anno_list = []
for image_path in tqdm.tqdm(image_list):
    text_path = image_path.with_suffix(".txt")
    text_content = text_path.read_text(encoding="utf-8")
    anno = {
        "image_path": str(image_path),
        "text": text_content.strip(),
    }
    anno_list.append(anno)

with open(f"{data_root}/metaclip_annos.json", "w") as f:
    json.dump(anno_list, f, indent=4, ensure_ascii=False)
