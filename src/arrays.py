import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def create_index_json(input_dir):
    """
    Create an index mapping Unicode codepoints to unique integer IDs.

    This function scans the input directory for subdirectories named with Unicode codepoints,
    and creates a mapping between these codepoints (along with their corresponding characters)
    and unique integer IDs. This mapping is useful for creating a label encoding for
    machine learning models.

    Args:
        input_dir (str): Path to the directory containing subdirectories named with
                         Unicode codepoints (e.g., "U+3042").

    Returns:
        dict: A dictionary with a single key 'codepoint_to_id', whose value is another
              dictionary mapping codepoint strings to integer IDs. The codepoint strings
              are in the format "character (U+XXXX)", where 'character' is the actual
              Unicode character and 'U+XXXX' is the codepoint.

    Example:
        >>> index_data = create_index_json("/path/to/kuzushiji/data")
        >>> print(index_data)
        {
            'codepoint_to_id': {
                'あ (U+3042)': 0,
                'い (U+3044)': 1,
                ...
            }
        }

    Note:
        - The function assumes that the subdirectory names in input_dir are valid
          Unicode codepoints in the format "U+XXXX".
        - The resulting ID mapping starts from 0 and increments for each unique codepoint.
        - The codepoints are sorted alphabetically before assigning IDs.
    """
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
    """
    Prepare image and label arrays from the Kuzushiji dataset.

    This function reads image files from the specified directory structure and creates
    two NumPy arrays: one for the images and one for their corresponding labels.
    It uses the provided index data to map codepoints to label IDs.

    Args:
        input_dir (str): Path to the root directory containing subdirectories of Kuzushiji images,
                         where each subdirectory is named with a Unicode codepoint (e.g., "U+3042").
        index_data (dict): A dictionary containing the 'codepoint_to_id' mapping, as created by
                           the create_index_json function.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - images (np.ndarray): A 3D array of shape (num_images, 64, 64) containing grayscale
                                   images as uint8 values.
            - labels (np.ndarray): A 1D array of shape (num_images,) containing the corresponding
                                   label IDs as uint32 values.

    Note:
        - The function dynamically determines the number of images for each codepoint.
        - Images are converted to grayscale and assumed to be 64x64 pixels.
        - The function uses tqdm to display a progress bar during processing.
        - The order of images in the returned arrays corresponds to the order of codepoints in
          the index_data dictionary and the alphabetical order of image filenames within each
          codepoint directory.

    Example:
        >>> index_data = create_index_json("/path/to/kuzushiji/data")
        >>> images, labels = prepare_array_data("/path/to/kuzushiji/data", index_data)
        >>> print(images.shape, labels.shape)
    """
    codepoint_to_id = index_data["codepoint_to_id"]

    # Dynamically determine the total number of images
    total_images = sum(len([f for f in os.listdir(os.path.join(input_dir, codepoint.split()[-1][1:-1]))
                            if f.endswith('.png') and not f.startswith("._")])
                       for codepoint in codepoint_to_id)

    images = np.zeros((total_images, 64, 64), dtype=np.uint8)
    labels = np.zeros(total_images, dtype=np.int32)

    image_index = 0
    for codepoint_char, id_value in tqdm(codepoint_to_id.items()):
        codepoint = codepoint_char.split()[-1][1:-1]  # Extract codepoint from the string
        codepoint_dir = os.path.join(input_dir, codepoint)

        for filename in os.listdir(codepoint_dir):
            if filename.endswith('.png') and not filename.startswith("._"):  # MacOS fix for ._ file names
                img_path = os.path.join(codepoint_dir, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                images[image_index] = np.array(img)
                labels[image_index] = id_value
                image_index += 1

    return images[:image_index], labels[:image_index]  # Trim any unused allocated space
