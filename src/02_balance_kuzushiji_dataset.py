import os
import shutil
from images import *
from PIL import Image
from tqdm import tqdm
import random


def balance_class(input_dir, output_dir, target_count=10000):
    # List all images in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    # If we have more than target_count images, randomly sample target_count
    if len(image_files) >= target_count:
        selected_files = random.sample(image_files, target_count)
        for i, file in enumerate(selected_files):
            shutil.copy(str(os.path.join(input_dir, file)), str(os.path.join(output_dir, f"{i:05d}.png")))
    else:
        # Copy all existing images
        for i, file in enumerate(image_files):
            shutil.copy(os.path.join(input_dir, file), os.path.join(output_dir, f"{i:05d}.png"))

        # Augment images until we reach target_count
        i = len(image_files)
        pbar = tqdm(total=target_count - len(image_files), desc=f"Augmenting {input_dir}")
        while i < target_count:
            # Randomly select an image to augment
            original_file = random.choice(image_files)
            original_image = Image.open(os.path.join(input_dir, original_file)).convert('L')
            original_array = np.array(original_image) / 255.0

            # Augment the image
            augmented_array = augment_image(original_array)
            augmented_image = Image.fromarray((augmented_array * 255).astype(np.uint8))

            # Save the augmented image
            augmented_image.save(os.path.join(output_dir, f"{i:05d}.png"))
            i += 1
            pbar.update(1)
        pbar.close()


def main():
    input_base_dir = "../data/kuzushiji"
    output_base_dir = "../data/kuzushiji-balanced"

    # Create the output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Process each class directory
    for class_dir in tqdm(os.listdir(input_base_dir), desc="Processing classes"):
        input_dir = os.path.join(input_base_dir, class_dir)
        output_dir = os.path.join(output_base_dir, class_dir)

        # Skip if not a directory
        if not os.path.isdir(input_dir):
            continue

        # Create the output directory for this class
        os.makedirs(output_dir, exist_ok=True)

        # Balance the class
        balance_class(input_dir, output_dir)

    print("Dataset balancing complete!")


if __name__ == "__main__":
    main()
