"""
Preprocess the BIOSCAN-5M dataset by grouping images by species into species-specific folders. Because of the paper, it is of interest to us to
have access to the images organized by species. To do so we make use of symbolic links (i.e. copying the path of the images from the original
dataset directory, in order to save on memory...)
"""

import csv
import os
from collections import defaultdict
from tqdm import tqdm


# CONFIGURATION, CHANGE PATHS AS NEEDED
IMAGE_DIR = "/Users/danielpanariti/Documents/universite/m2_2025-2026_MIND/s3/deep_l/projet/BIOSCAN-5M/bioscan5m/images/cropped_256"
CSV_FILE_PATH = "/Users/danielpanariti/Documents/universite/m2_2025-2026_MIND/s3/deep_l/projet/BIOSCAN-5M/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"
OUTPUT_DIR = "./preprocessed_images"


def get_all_images_path(root_dir): 
    """
    Walk through the image directory and store all image file paths in a dictionary, 
    indexed by their filename without extension.
    """
    image_paths = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames: 
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')): 
                continue
            image_id = os.path.splitext(filename)[0]
            image_paths[image_id] = os.path.join(dirpath, filename)
    return image_paths


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Scanning image directory and getting all image paths...")
    image_paths = get_all_images_path(IMAGE_DIR)
    print(f"Total images found: {len(image_paths)}")

    # Build a dictionary mapping each species to its list of sample IDs
    species_dict = defaultdict(list)
    with open(CSV_FILE_PATH, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header row

        for row in tqdm(reader, desc="Processing CSV"):
            # Only consider rows with species-level annotation
            if len(row[9]) > 0:
                # Join species name as a single string with underscore
                species_name = '_'.join(row[9].split())
                species_dict[species_name].append(row[0])

    print(f"Total unique species with species-level annotation: {len(species_dict)}")

    # Create a folder for each species and create symlinks to their images
    # (using symlinks is faster than copying the actual image files)
    total_linked = 0
    species_with_images = 0

    for species, ids in tqdm(species_dict.items(), desc="Creating symlinks"):
        valid_image_ids  = [image_id for image_id in ids if image_id in image_paths]

        if not valid_image_ids :
            continue

        species_with_images += 1
        species_dir = os.path.join(OUTPUT_DIR, species)
        os.makedirs(species_dir, exist_ok=True)

        for image_id in valid_image_ids :
            src = image_paths[image_id]
            dst = os.path.join(species_dir, f"{image_id}.jpg")
            if not os.path.exists(dst):
                os.symlink(src, dst)

        total_linked += len(valid_image_ids)

    print("\n--- Preprocessing Results ---")
    print(f"Species with images: {species_with_images}")
    print(f"Total images linked: {total_linked}")


if __name__ == "__main__":
    main()