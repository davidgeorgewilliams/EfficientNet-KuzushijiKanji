from arrays import *
import os
import json
import numpy as np


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
