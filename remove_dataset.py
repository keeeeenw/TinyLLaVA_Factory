import json

# Input and output file paths
input_file_path = "data/text_files/llava_v1_5_mix665k.json"
# output_file_path = "data/text_files/llava_v1_5_mix665k_cleaned_data.json"
output_file_path = "data/text_files/llava_v1_5_mix665k_cleaned_data_w_ocr_vqa.json"

# Load JSON data from file
with open(input_file_path, "r", encoding="utf-8") as input_file:
    dataset = json.load(input_file)

# Filter out entries where the image path contains "ocr_vqa/images/"
# filtered_dataset = [
#     entry for entry in dataset
#     if "ocr_vqa/images/" not in entry.get("image", "")
# ]

# List of nonexisting files
nonexisting_files = {
    "ocr_vqa/images/1421539896.jpg",
    "ocr_vqa/images/141393394.jpg",
    "ocr_vqa/images/316881791.jpg",
    "ocr_vqa/images/140445692.jpg",
    "ocr_vqa/images/142153990X.jpg",
    "ocr_vqa/images/689852649.jpg",
}

# Filter out entries whose image path exists in existing_files
filtered_dataset = [
    entry for entry in dataset
    if entry.get("image", "") not in nonexisting_files
]

# Save the cleaned data back to a file
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(filtered_dataset, output_file, ensure_ascii=False, indent=2)

print(f"Removed {len(dataset) - len(filtered_dataset)} entries.")
print(f"Cleaned dataset saved to {output_file_path}")
