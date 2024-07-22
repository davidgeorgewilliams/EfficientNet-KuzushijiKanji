from images import *

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
