import json
import os

# --- Configuration ---
FILES_TO_SPLIT = [
    "yelp_business.json",
    "yelp_data.json"
]
NUM_PARTS = 4 # Split each file into 4 parts
# --- End Configuration ---

def split_file(file_path, num_parts):
    print(f"Starting split for: {file_path}")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Assuming the JSON file is structured as one JSON object per line (common for Yelp data)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    total_lines = len(lines)
    chunk_size = total_lines // num_parts
    
    base_name, ext = os.path.splitext(file_path)
    
    for i in range(num_parts):
        start_index = i * chunk_size
        # The last chunk takes all remaining lines to handle uneven division
        end_index = (i + 1) * chunk_size if i < num_parts - 1 else total_lines
        
        chunk_lines = lines[start_index:end_index]
        
        # Create the new filename: e.g., yelp_business_part_1.json
        new_file_path = f"{base_name}_part_{i+1}{ext}"
        
        with open(new_file_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(chunk_lines)
            
        print(f"  Created {new_file_path} with {len(chunk_lines)} lines.")

# Run the split for both files
for file in FILES_TO_SPLIT:
    split_file(file, NUM_PARTS)