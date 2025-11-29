from PIL import Image, ImageEnhance
import numpy as np
import os
from pathlib import Path
import random

def augment_dataset(base_path, multiplier=3):
    """
    Augments YOLO dataset maintaining train/valid/test structure
    base_path: root containing train/valid/test folders
    multiplier: total images = original * multiplier
    """
    augmentations = ['grayscale', 'noise', 'rotate_5', 'hue', 'saturation']
    
    for split in ['train', 'valid', 'test']:
        img_dir = Path(base_path) / split / 'images'
        lbl_dir = Path(base_path) / split / 'labels'
        
        if not img_dir.exists():
            continue
            
        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        
        for img_path in images:
            # Get corresponding label
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
                
            # Generate (multiplier - 1) augmented versions
            augs_needed = multiplier - 1
            selected_augs = random.choices(augmentations, k=augs_needed)
            
            for i, aug_type in enumerate(selected_augs):
                img = Image.open(img_path)
                
                # Apply augmentation
                if aug_type == 'grayscale':
                    img = ImageEnhance.Color(img).enhance(0)
                elif aug_type == 'noise':
                    arr = np.array(img)
                    noise = np.random.randint(-25, 25, arr.shape, dtype=np.int16)
                    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(arr)
                elif aug_type == 'rotate_5':
                    img = img.rotate(random.choice([-5, 5]), expand=False)
                elif aug_type == 'hue':
                    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
                elif aug_type == 'saturation':
                    img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))
                
                # Save augmented image
                new_name = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
                img.save(img_dir / new_name)
                
                # Copy label file (YOLO labels remain same for these augmentations)
                new_lbl = lbl_dir / f"{img_path.stem}_aug{i+1}.txt"
                with open(lbl_path, 'r') as f:
                    with open(new_lbl, 'w') as nf:
                        nf.write(f.read())

# Usage
augment_dataset('path/to/dataset', multiplier=3)