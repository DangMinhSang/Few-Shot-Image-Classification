#!/usr/bin/env python3
"""
Download Oxford Flowers 102 Dataset via torchvision.

Creates the following structure:
    data/
    ├── train/   (1020 images, 10 per class)
    ├── val/     (1020 images, 10 per class)
    └── test/    (6149 images)

Oxford Flowers 102:
- 102 flower categories
- Challenging dataset with high intra-class variation
- Standard few-shot benchmark

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --data_dir my_data/
"""
import argparse
import shutil
from pathlib import Path

import torchvision.datasets as dsets

# Flower class names (102 classes)
FLOWER_NAMES = [
    'pink_primrose', 'hard-leaved_pocket_orchid', 'canterbury_bells',
    'sweet_pea', 'english_marigold', 'tiger_lily', 'moon_orchid',
    'bird_of_paradise', 'monkshood', 'globe_thistle', 'snapdragon',
    "colts_foot", 'king_protea', 'spear_thistle', 'yellow_iris',
    'globe-flower', 'purple_coneflower', 'peruvian_lily', 'balloon_flower',
    'giant_white_arum_lily', 'fire_lily', 'pincushion_flower', 'fritillary',
    'red_ginger', 'grape_hyacinth', 'corn_poppy', 'prince_of_wales_feathers',
    'stemless_gentian', 'artichoke', 'sweet_william', 'carnation',
    'garden_phlox', 'love_in_the_mist', 'mexican_aster', 'alpine_sea_holly',
    'ruby-lipped_cattleya', 'cape_flower', 'great_masterwort', 'siam_tulip',
    'lenten_rose', 'barbeton_daisy', 'daffodil', 'sword_lily', 'poinsettia',
    'bolero_deep_blue', 'wallflower', 'marigold', 'buttercup', 'oxeye_daisy',
    'common_dandelion', 'petunia', 'wild_pansy', 'primula', 'sunflower',
    'pelargonium', 'bishop_of_llandaff', 'gaura', 'geranium', 'orange_dahlia',
    'pink-yellow_dahlia', 'cautleya_spicata', 'japanese_anemone', 'black-eyed_susan',
    'silverbush', 'californian_poppy', 'osteospermum', 'spring_crocus',
    'bearded_iris', 'windflower', 'tree_poppy', 'gazania', 'azalea',
    'water_lily', 'rose', 'thorn_apple', 'morning_glory', 'passion_flower',
    'lotus', 'toad_lily', 'anthurium', 'frangipani', 'clematis',
    'hibiscus', 'columbine', 'desert-rose', 'tree_mallow', 'magnolia',
    'cyclamen', 'watercress', 'canna_lily', 'hippeastrum', 'bee_balm',
    'pink_quill', 'foxglove', 'bougainvillea', 'camellia', 'mallow',
    'mexican_petunia', 'bromelia', 'blanket_flower', 'trumpet_creeper', 'blackberry_lily'
]


def download_and_organize(data_dir: str = 'data/'):
    """Download Oxford Flowers 102 and organize into class folders."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Oxford Flowers 102...")
    print("This may take a few minutes (download ~328MB)...")
    
    # Download all splits
    for split in ['train', 'val', 'test']:
        print(f"\nDownloading {split} split...")
        dataset = dsets.Flowers102(
            root=str(data_path),
            split=split,
            download=True,
        )
        
        # Organize into class folders
        split_dir = data_path / split
        split_dir.mkdir(exist_ok=True)
        
        print(f"Organizing {split} images into class folders...")
        for idx, (img_path, label) in enumerate(zip(dataset._image_files, dataset._labels)):
            # Class folder name
            class_name = FLOWER_NAMES[label] if label < len(FLOWER_NAMES) else f'class_{label:03d}'
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Copy image
            dest = class_dir / img_path.name
            if not dest.exists():
                shutil.copy2(str(img_path), str(dest))
            
            if (idx + 1) % 200 == 0:
                print(f"  Progress: {idx+1}/{len(dataset._image_files)}")
        
        # Count
        n_images = sum(1 for _ in split_dir.rglob('*.jpg'))
        n_classes = len(list(split_dir.iterdir()))
        print(f"  {split}: {n_images} images, {n_classes} classes")
    
    print(f"\nDataset ready at: {data_path.absolute()}")
    print("\nDirectory structure:")
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        if split_dir.exists():
            classes = list(split_dir.iterdir())
            total = sum(len(list(c.glob('*.jpg'))) for c in classes)
            print(f"  {split}/: {len(classes)} classes, {total} images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    args = parser.parse_args()
    download_and_organize(args.data_dir)
