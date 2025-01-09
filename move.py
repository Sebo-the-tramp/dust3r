import os
import json
import shutil

def move_folders(json_file, source_dir, dest_dir):
    """
    Moves folders listed in a JSON file from the source directory to the destination directory.

    Parameters:
    - json_file (str): Path to the JSON file containing a list of folder names.
    - source_dir (str): Path to the source directory (e.g., 'Training').
    - dest_dir (str): Path to the destination directory (e.g., 'Test').
    """
    
    # Check if JSON file exists
    if not os.path.isfile(json_file):
        print(f"Error: JSON file '{json_file}' does not exist.")
        return

    # Load folder names from JSON
    with open(json_file, 'r') as f:
        try:
            folder_list = json.load(f)
            if not isinstance(folder_list, list):
                print("Error: JSON file does not contain a list.")
                return
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON file. {e}")
            return

    # Check if source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # Create destination directory if it doesn't exist
    if not os.path.isdir(dest_dir):
        print(f"Destination directory '{dest_dir}' does not exist. Creating it.")
        try:
            os.makedirs(dest_dir)
        except Exception as e:
            print(f"Error: Failed to create destination directory. {e}")
            return

    # Iterate over folder names and move them
    for folder in folder_list:
        src_path = os.path.join(source_dir, folder)
        dest_path = os.path.join(dest_dir, folder)

        if not os.path.exists(src_path):
            print(f"Warning: Source folder '{src_path}' does not exist. Skipping.")
            continue

        if os.path.exists(dest_path):
            print(f"Warning: Destination folder '{dest_path}' already exists. Skipping.")
            continue

        try:
            shutil.move(src_path, dest_path)
            print(f"Moved '{src_path}' to '{dest_path}'.")
        except Exception as e:
            print(f"Error: Failed to move '{src_path}' to '{dest_path}'. {e}")

if __name__ == "__main__":
    # Define paths
    json_file = '/arkitscenes_pairs/Test/scene_list.json'  # Path to your JSON file
    source_dir = 'Training'        # Source directory
    dest_dir = 'Test'              # Destination directory

    # Call the function to move folders
    move_folders(json_file, source_dir, dest_dir)
