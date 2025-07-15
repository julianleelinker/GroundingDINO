import json
import pathlib

# DONE
# Sports_Development_20241223_image_list_keep_0.95
# Sports_Development_20250109_image_list_keep_0.95 # Sports_Development_20250213_image_list_keep_0.95
# Sports_Development_20250226_image_list_keep_0.95
# Sports_Development_20250319_image_list_keep_0.95
# Sports_Development_20250526_image_list_keep_0.95
# Sports_Development_20250610_image_list_keep_0.95
# Water_Resources_20250106_image_list_keep_0.95
# Water_Resources_20250213_image_list_keep_0.95
# Water_Resources_20250324_image_list_keep_0.95
# Mass_Rapid_Transit_20250109_image_list_keep_0.95_0.30_0.35
# Mass_Rapid_Transit_20250213_image_list_keep_0.95
# Mass_Rapid_Transit_20250402_image_list_keep_0.95
# Mass_Rapid_Transit_20250515_image_list_keep_0.95
# Ports_Corporation_20250226_image_list_keep_0.95


json_path = "/mnt/data-home/mobility-multimodal/data-curation/Ports_Corporation/20250124/Ports_Corporation_20250124_image_list_keep_0.95.json"

with open(json_path, "r") as f:
    data = json.load(f)
print(len(data))