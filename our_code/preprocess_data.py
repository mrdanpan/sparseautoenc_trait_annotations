"""
Preprocess the BIOSCAN-5M dataset by grouping images by species into species-specific folders.
"""

import csv
import os
import shutil
from collections import defaultdict
from tqdm import tqdm
import argparse

def get_species_name(row):
    """Return species name as a single string with underscores."""
    species = row[9]
    return '_'.join(species.split())

def is_species(row):
    """Check if row contains species-level annotation."""
    return len(row[9]) > 0

def find_images(root_dir):
    """Return dict mapping process_id (filename without ext) -> full path."""
    image_map = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            pid = os.path.splitext(fname)[0]
            image_map[pid] = os.path.join(dirpath, fname)
    return image_map

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("Building image map...")
    image_map = find_images(args.image_dir)
    print(f"Found {len(image_map)} images.")

    # species -> list of process_ids
    species_dict = defaultdict(list)

    # Read CSV
    with open(args.csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # skip header

        for row in tqdm(reader, desc="Processing CSV"):
            if args.split and row[20] != args.split:
                continue
            if is_species(row):
                species_name = get_species_name(row)
                species_dict[species_name].append(row[0])

    print(f"Found {len(species_dict)} species with species-level annotation.")

    # Create folders and copy images
    total_linked = 0
    total_missing = 0
    species_with_images = 0
    species_without_images = 0

    for species, ids in tqdm(species_dict.items(), desc="Copying images"):
        available_ids = [pid for pid in ids if pid in image_map]
        if not available_ids:
            species_without_images += 1
            continue

        species_with_images += 1
        species_dir = os.path.join(args.out_dir, species)
        os.makedirs(species_dir, exist_ok=True)

        for pid in available_ids:
            src = image_map[pid]
            dst = os.path.join(species_dir, f"{pid}.jpg")
            if not os.path.exists(dst):
                shutil.copy(src, dst)

        total_linked += len(available_ids)
        total_missing += len(ids) - len(available_ids)

    # Summary
    print("\n=== Summary ===")
    print(f"Species with images: {species_with_images}")
    print(f"Species without images: {species_without_images}")
    print(f"Total images copied: {total_linked}")
    print(f"Total missing images: {total_missing}")
    if total_linked + total_missing > 0:
        print(f"Match rate: {total_linked / (total_linked + total_missing) * 100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", type=str, required=True, help="Path to BIOSCAN CSV metadata")
    parser.add_argument("--image-dir", type=str, required=True, help="Path to BIOSCAN images")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory to store images by species")
    parser.add_argument("--split", type=str, default=None, help="Optional: train, val, test, or None for all")
    args = parser.parse_args()
    main(args)
