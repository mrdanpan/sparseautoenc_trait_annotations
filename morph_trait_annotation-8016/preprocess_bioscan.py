import csv
from tqdm import tqdm
from collections import defaultdict
import argparse
import numpy as np
import random
from copy import deepcopy
import sys, os

def get_scientific_name(row):
    phylum, class1, order, family, subfamily, genus, species = row[3], row[4], row[5], row[6], row[7], row[8], row[9]
    name = '_'.join(species.split())
    return name

def is_species(row):
    return len(row[9]) > 0

def find_images(root_dir):
    image_map = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Skip non-image files
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            name_without_ext = os.path.splitext(filename)[0]
            full_path = os.path.join(dirpath, filename)
            image_map[name_without_ext] = full_path
    return image_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', type=str, default='/fs/ess/user/bioscan-5m/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv')
    parser.add_argument('--seed', type=int, default=3875671)
    parser.add_argument('--image-dir', type=str, default='/fs/ess/user/bioscan-5m/bioscan5m/images/')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--split', type=str, default=None, help='Filter to specific split: train, val, test, or None for all')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    total = 0
    total_lines = 5150851
    species_dna_dict = defaultdict(list)

    # Create a mapping from process id to filename
    print("Building image map...")
    process_id_to_imgfile_dict = find_images(args.image_dir)
    print(f"Found {len(process_id_to_imgfile_dict)} images")

    with open(args.csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)

        for row_id, row in enumerate(tqdm(reader, total=total_lines)):
            if row_id == 0: 
                continue  # Skip header
            
            process_id = row[0]  # Unique sample identifier
            split = row[20]      # Split column

            # Filter by split if specified
            if args.split is not None:
                if split != args.split:
                    continue  # Skip if not the requested split

            # Only include samples with species-level annotation
            if is_species(row):
                species_dna_dict[get_scientific_name(row)].append(process_id)
            
            total += 1

        print(f'Total rows processed: {total}')
        print(f'Number of unique species: {len(species_dna_dict)}')

        # Count total samples with species annotation
        total_samples = sum(len(v) for v in species_dna_dict.values())
        print(f'Total samples with species annotation: {total_samples}')

        all_species = list(species_dna_dict.keys())
        selected_species = all_species

        # Statistics
        total_found = 0
        total_missing = 0
        species_with_images = 0
        species_without_images = 0

        for species in tqdm(selected_species, desc="Creating symlinks"):
            all_samples_species = deepcopy(species_dna_dict[species])

            # Filter to only samples with available images
            available_samples = [pid for pid in all_samples_species if pid in process_id_to_imgfile_dict]
            missing_samples = len(all_samples_species) - len(available_samples)

            total_found += len(available_samples)
            total_missing += missing_samples

            if len(available_samples) == 0:
                species_without_images += 1
                continue  # Skip species with no images

            species_with_images += 1

            # Create folder for species
            species_dir = os.path.join(args.out_dir, species)
            os.makedirs(species_dir, exist_ok=True)

            # Create symlinks
            for process_id in available_samples:
                train_file_src = process_id_to_imgfile_dict[process_id]
                train_file_dst = os.path.join(species_dir, '{}.jpg'.format(process_id))
                if not os.path.lexists(train_file_dst):
                    os.symlink(train_file_src, train_file_dst)

        print(f"\n=== Summary ===")
        print(f"Species with at least 1 image: {species_with_images}")
        print(f"Species with 0 images (skipped): {species_without_images}")
        print(f"Total images linked: {total_found}")
        print(f"Total images missing: {total_missing}")
        print(f"Match rate: {total_found/(total_found+total_missing)*100:.1f}%")