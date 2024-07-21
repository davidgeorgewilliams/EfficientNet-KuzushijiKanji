import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_index_json(input_dir):
    codepoint_to_id = {}
    id_counter = 0

    for codepoint in sorted(os.listdir(input_dir)):
        if os.path.isdir(os.path.join(input_dir, codepoint)):
            # Assuming the codepoint is in the format "U+XXXX"
            char = chr(int(codepoint[2:], 16))
            codepoint_to_id[f"{char} ({codepoint})"] = id_counter
            id_counter += 1

    index_data = {"codepoint_to_id": codepoint_to_id}

    return index_data


def prepare_array_data(input_dir, index_data):
    codepoint_to_id = index_data["codepoint_to_id"]
    num_images = len(codepoint_to_id) * 10000  # 10000 images per codepoint

    images = np.zeros((num_images, 64, 64), dtype=np.uint8)
    labels = np.zeros(num_images, dtype=np.uint32)

    image_index = 0
    for codepoint_char, id_value in tqdm(codepoint_to_id.items()):
        codepoint = codepoint_char.split()[-1][1:-1]  # Extract codepoint from the string
        codepoint_dir = os.path.join(input_dir, codepoint)

        for filename in os.listdir(codepoint_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(codepoint_dir, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                images[image_index] = np.array(img)
                labels[image_index] = id_value
                image_index += 1

    return images, labels


def main():
    input_dir = "../data/kuzushiji-balanced"
    output_dir = "../data/kuzushiji-arrays"
    os.makedirs(output_dir, exist_ok=True)

    # Create index.json
    index_data = create_index_json(input_dir)
    with open(os.path.join(output_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=4)

    # Prepare numpy arrays
    images, labels = prepare_array_data(input_dir, index_data)

    # Save numpy arrays
    np.save(os.path.join(output_dir, "images.npy"), images)
    np.save(os.path.join(output_dir, "labels.npy"), labels)

    print(f"Processed {len(images)} images with {len(index_data['codepoint_to_id'])} unique codepoints.")
    print(f"Output saved in {output_dir}")


if __name__ == "__main__":
    main()