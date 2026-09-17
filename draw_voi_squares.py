import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os
import glob
import re
from collections import defaultdict

USE_FULL_DATASET = False

if USE_FULL_DATASET:
    csv_path = '/home/pausanch/Documents/DualEnergyTomolab_private/samples_VOIs_centers_full.csv'
else:
    csv_path = '/home/pausanch/Documents/DualEnergyTomolab_private/samples_VOIs_centers.csv'

centers_by_sample = defaultdict(list)

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sample = int(row['Sample'])
        center_x = int(row['Center_x'])
        center_y = int(row['Center_y'])
        centers_by_sample[sample].append((center_x, center_y))

img_dir = '/home/pausanch/Documents/DualEnergyTomolab_private/scale_bar_images'

plt.style.use('bmh')
orange_color = '#f07e22'
print(f"Using orange color: {orange_color}")

SQUARE_SIZE = 50

image_files = [
    f for f in glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))
    if '_with_vois' not in os.path.basename(f)
]

for sample_num, centers in centers_by_sample.items():
    matching_images = []
    for img_path in image_files:
        basename = os.path.basename(img_path)
        match = re.match(r'^(\d+)', basename)
        if match and int(match.group(1)) == sample_num:
            matching_images.append(img_path)

    if not matching_images:
        print(f"No images found for sample {sample_num}")
        continue

    for img_path in matching_images:
        img = mpimg.imread(img_path)

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)

        for center_x, center_y in centers:
            bottom_left_x = center_x - SQUARE_SIZE / 2
            bottom_left_y = center_y - SQUARE_SIZE / 2

            rect = patches.Rectangle(
                (bottom_left_x, bottom_left_y),
                SQUARE_SIZE,
                SQUARE_SIZE,
                linewidth=2,
                edgecolor=orange_color,
                facecolor='none'
            )
            ax.add_patch(rect)

        ax.axis('off')

        basename = os.path.basename(img_path)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(img_dir, f"{name}_with_vois{ext}")

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)

        print(f"Saved: {output_path}")

print("Done!")
