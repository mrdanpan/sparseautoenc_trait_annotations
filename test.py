import os

image_dir = os.path.expanduser("~/Datasets/bioscan-5m/bioscan5m/images/cropped_256/train")

# Get sample image filenames from ALL subdirectories
sample_images = []
for root, dirs, files in os.walk(image_dir):
    for f in files:
        if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.'):
            name_without_ext = os.path.splitext(f)[0]
            sample_images.append(name_without_ext)
            if len(sample_images) >= 10:
                break
    if len(sample_images) >= 10:
        break

print("=== Sample IMAGE filenames (without extension) ===")
for img in sample_images:
    print(f"  '{img}'")

# Also show full path of one image
for root, dirs, files in os.walk(image_dir):
    for f in files:
        if f.endswith(('.jpg', '.jpeg', '.png')):
            print(f"\n=== Example full path ===")
            print(f"  {os.path.join(root, f)}")
            break
    break