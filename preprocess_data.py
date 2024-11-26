import os
import random
import shutil
from PIL import Image
from tqdm import tqdm

def cleanup_processed_dataset():
    """Remove existing processed dataset directory"""
    processed_dir = './processed_dataset'
    if os.path.exists(processed_dir):
        print(f"Removing existing processed dataset at {processed_dir}")
        shutil.rmtree(processed_dir)
        print("Cleanup complete!")

def create_dataset_structure():
    base_dir = './processed_dataset'
    splits = ['train', 'val', 'test']
    classes = ['chair', 'cupboard', 'fridge', 'table', 'tv', 'bed', 'sofa']
    
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")

def process_images(source_dir):
    target_dir = './processed_dataset'

    # Maximum images per class
    MAX_IMAGES_PER_CLASS = 200

    class_info = {
        'chair': {'ext': '.jpg', 'pattern': r'\d{8}\.jpg', 'total': 3000},
        'cupboard': {'ext': '.jpeg', 'pattern': r'image_\d+\.jpeg', 'total': 3000},
        'fridge': {'ext': '.jpeg', 'pattern': r'image_\d+\.jpeg', 'total': 3000},
        'table': {'ext': '.jpeg', 'pattern': r'image_\d+\.jpeg', 'total': 3000},
        'tv': {'ext': '.jpeg', 'pattern': r'image_\d+\.jpeg', 'total': 3000},
        'bed': {'ext': '.jpg', 'pattern': r'\d{8}\.jpg', 'total': 900},
        'sofa': {'ext': '.jpg', 'pattern': r'\d{8}\.jpg', 'total': 1000}
    }
    
    for cls, info in class_info.items():
        print(f"\nProcessing {cls} images...")
        class_dir = os.path.join(source_dir, cls)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Source directory {class_dir} not found!")
            continue
            
        ext = info['ext']
        images = [f for f in os.listdir(class_dir) if f.endswith(ext)]
        
        if len(images) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        # Randomly select images
        if len(images) > MAX_IMAGES_PER_CLASS:
            images = random.sample(images, MAX_IMAGES_PER_CLASS)
        
        total_images = len(images)
        
        # Calculate split indices - 60/30/10 split
        train_idx = int(0.6 * total_images) 
        val_idx = int(0.9 * total_images)   
        
        print(f"Processing {cls}:")
        print(f"- Original dataset size: {info['total']}")
        print(f"- Selected for processing: {total_images}")
        print(f"- Training: {train_idx} images")
        print(f"- Validation: {val_idx - train_idx} images")
        print(f"- Testing: {total_images - val_idx} images")
        
        # Process images for each split
        for idx, img_name in enumerate(tqdm(images)):
            if idx < train_idx:
                split = 'train'
            elif idx < val_idx:
                split = 'val'
            else:
                split = 'test'
                
            try:
                # Read image
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to maintain aspect ratio
                target_size = 177
                ratio = max(target_size/img.size[0], target_size/img.size[1])
                new_size = tuple([int(dim * ratio) for dim in img.size])
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Center crop
                left = (img.size[0] - target_size)/2
                top = (img.size[1] - target_size)/2
                right = (img.size[0] + target_size)/2
                bottom = (img.size[1] + target_size)/2
                img = img.crop((left, top, right, bottom))
                
                # Generate standardized filename for saving
                # Use class prefix to avoid naming conflicts
                std_name = f"{cls}_{idx+1:03d}.jpg"
                
                # Save processed image
                target_path = os.path.join(target_dir, split, cls, std_name)
                img.save(target_path, 'JPEG', quality=95)
                
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                print(f"Full path: {img_path}")

def verify_dataset():
    """Verify the processed dataset structure and count"""
    base_dir = './processed_dataset'
    splits = ['train', 'val', 'test']
    classes = ['chair', 'cupboard', 'fridge', 'table', 'tv', 'bed', 'sofa']
    
    print("\nVerifying processed dataset:")
    for split in splits:
        print(f"\n{split.capitalize()} split:")
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                print(f"- {cls}: {count} images")
            else:
                print(f"- {cls}: directory not found!")

if __name__ == "__main__":
    print("Starting dataset preprocessing...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Cleanup existing processed dataset
    cleanup_processed_dataset()
    
    # Create the directory structure
    print("\nCreating directory structure...")
    create_dataset_structure()
    
    print("\nProcessing images...")
    source_dir = './dataset' 
    process_images(source_dir)
    
    # Verify the processed dataset
    verify_dataset()
    
    print("\nDataset preprocessing complete!")
    print("\nFinal dataset structure (per class):")
    print("- Total images: 200")
    print("- Training:     120 images (60%)")
    print("- Validation:   60 images (30%)")
    print("- Testing:      20 images (10%)")