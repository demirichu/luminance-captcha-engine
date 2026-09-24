import numpy as np
from config import GOLDEN_DATASET, NEW_GOLDEN_DATASET

# 1. Main Golden Dataset from the first triage pass
try:
    old_golden = np.load(GOLDEN_DATASET)
except FileNotFoundError:
    print(f"Error: {GOLDEN_DATASET} not found.")
    exit()

# 2. New Golden Dataset (e.g., rescued from a secondary pass or manual review)
try:
    new_golden = np.load(NEW_GOLDEN_DATASET)
    print("Merging datasets...")

    all_images = np.concatenate([old_golden['images'], new_golden['images']], axis=0)
    all_labels = np.concatenate([old_golden['labels'], new_golden['labels']], axis=0)

    # Overwrite the main golden dataset with the merged data
    np.savez_compressed(GOLDEN_DATASET, images=all_images, labels=all_labels)

    print(f"READY TO GO! A total of {len(all_images)} Pure Golden data saved to '{GOLDEN_DATASET}'.")
except FileNotFoundError:
    print(f"Notice: '{NEW_GOLDEN_DATASET}' not found. Nothing to merge.")