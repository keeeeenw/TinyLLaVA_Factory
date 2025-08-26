# If you know the individual files you want to remove.
# import json

# # Path to the input/output file
# file_path = 'data/text_files/llava_v1_5_mix665k_cleaned_data_w_ocr_vqa.json'

# # List of image keys to remove
# images_to_remove = {
#     "ocr_vqa/images/898620996.jpg",
#     "ocr_vqa/images/898796008.jpg",
#     # Add more image paths to remove here
# }

# # Read the original data
# with open(file_path, 'r') as f:
#     data = json.load(f)

# # Filter the data
# filtered_data = [entry for entry in data if 'image' not in entry or entry['image'] not in images_to_remove]

# # Write back to the same file
# with open(file_path, 'w') as f:
#     json.dump(filtered_data, f, indent=2)

# print(f"Filtered {len(data) - len(filtered_data)} entries. File updated: {file_path}")

# If you don't know the files
import json
import os

# File path to the JSON data
# base finetune
json_path = 'data/text_files/llava_v1_5_mix665k_cleaned_data_w_ocr_vqa.json'
# base sharegpt4v
# json_path = 'data/text_files/cleaned_sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k_ocr_reduced.json'
base_image_dir = '/home/ken/workspace/TinyLLaVA_Factory/data/'

# Load the JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

# Filter the data
filtered_data = []
removed_count = 0

for entry in data:
    image_path = entry.get('image')
    if image_path and image_path.startswith('ocr_vqa/images/'):
        full_path = os.path.join(base_image_dir, image_path)
        if not os.path.exists(full_path):
            removed_count += 1
            continue
    filtered_data.append(entry)

# Overwrite the original file
with open(json_path, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"Removed {removed_count} entries with missing images. Updated file: {json_path}")

