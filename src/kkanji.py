import hashlib
import os
import shutil
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from images import resize_and_save


def load_npz(file_path):
    """
    Load and return the first array from a NumPy .npz file.

    This function opens a .npz file at the specified path and returns the array
    stored under the key 'arr_0'. It's designed to work with .npz files that
    contain a single array or where the desired array is stored under 'arr_0'.

    Args:
        file_path (str): The path to the .npz file to be loaded.

    Returns:
        numpy.ndarray: The array stored in the .npz file under the key 'arr_0'.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If the .npz file does not contain an array with the key 'arr_0'.
        ValueError: If the file is not a valid .npz file.

    Example:
        >>> array = load_npz('path/to/your/file.npz')
        >>> print(array.shape)
        (1000, 28, 28)  # Example output, actual shape depends on the content of the file

    Note:
        - This function assumes that the .npz file contains at least one array
          stored under the key 'arr_0'.
        - If the .npz file contains multiple arrays, only the one under 'arr_0'
          will be returned. Other arrays in the file will be ignored.
        - The function uses a context manager (with statement) to ensure that
          the file is properly closed after reading.
    """
    with np.load(file_path) as data:
        return data['arr_0']


def load_label_mapping(csv_path):
    """
    Load a label mapping from a CSV file into a dictionary.

    This function reads a CSV file containing label mappings and returns a dictionary
    where the keys are the 'index' values and the values are the corresponding 'codepoint' values.

    Args:
        csv_path (str): The file path to the CSV containing the label mappings.

    Returns:
        dict: A dictionary mapping indices to codepoints.
            Key: int or str (depending on the 'index' column type in the CSV)
            Value: str (the 'codepoint' value)

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        KeyError: If the CSV doesn't contain 'index' or 'codepoint' columns.
        ValueError: If the CSV file is malformed or cannot be properly read.

    Example:
        Assuming a CSV file 'label_map.csv' with contents:
        index,codepoint
        0,U+3042
        1,U+3044
        2,U+3046

        >>> mapping = load_label_mapping('label_map.csv')
        >>> print(mapping)
        {0: 'U+3042', 1: 'U+3044', 2: 'U+3046'}

    Note:
        - The CSV file must have 'index' and 'codepoint' columns.
        - The function assumes that the 'index' values are unique.
        - If there are duplicate 'index' values, only the last occurrence will be kept in the dictionary.
        - The function uses pandas to read the CSV, so it can handle various CSV formats and options.
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df['index'], df['codepoint']))


def generate_unique_filename():
    """
    Generate a unique filename for image storage.

    This function creates a unique filename by combining the current timestamp
    and a random integer, then hashing this combination using MD5. The resulting
    filename is highly unlikely to collide with existing filenames.

    Returns:
        str: A unique filename string in the format "[32 character MD5 hash].png"

    Example:
        >>> filename = generate_unique_filename()
        >>> print(filename)
        '8f0d3f7ae71810fb0c8b6cd9e89d3a77.png'

    Note:
        - The function uses the current time (to microsecond precision) and a
          random integer to create a unique string.
        - The unique string is hashed using MD5 to produce a fixed-length filename.
        - While collisions are theoretically possible, they are extremely unlikely
          due to the use of both time and random number generation.
        - The function always returns a filename with a .png extension, assuming
          the file will be a PNG image.
        - This function does not actually create a file; it only generates a filename.

    Warning:
        - MD5 is used here for filename generation, not for cryptographic purposes.
        - If used in a multi-threaded or distributed environment, there's a minute
          possibility of collision if called at exactly the same microsecond.
    """
    unique_string = f"{time.time()}\t{np.random.randint(int(1e10))}"
    return hashlib.md5(unique_string.encode()).hexdigest() + ".png"


def process_k49(imgs, labels, output_dir, prefix, label_mapping):
    """
    Process and save images from the Kuzushiji-49 dataset.

    This function takes arrays of images and their corresponding labels from the
    Kuzushiji-49 dataset, resizes them, and saves them to an organized directory
    structure based on their Unicode codepoints.

    Args:
        imgs (numpy.ndarray): Array of images from the Kuzushiji-49 dataset.
        labels (numpy.ndarray): Array of corresponding labels for the images.
        output_dir (str): Path to the directory where processed images will be saved.
        prefix (str): A prefix string for the progress bar description (e.g., 'train' or 'test').
        label_mapping (dict): A dictionary mapping label indices to Unicode codepoints.

    Returns:
        None

    Side effects:
        - Creates directories for each unique codepoint in the output_dir.
        - Saves resized images as PNG files in their respective codepoint directories.

    Notes:
        - The function uses tqdm to display a progress bar during processing.
        - Each image is resized using the `resize_and_save` function (not shown here).
        - Unique filenames are generated for each image using `generate_unique_filename`.
        - The directory structure will be: output_dir/codepoint/image.png

    Raises:
        KeyError: If a label in `labels` is not found in `label_mapping`.
        IOError: If there are issues creating directories or saving images.

    Example:
        >>> imgs = np.random.rand(1000, 28, 28)  # Example image array
        >>> labels = np.random.randint(0, 49, 1000)  # Example label array
        >>> label_mapping = {i: f"U+{3042+i:04X}" for i in range(49)}  # Example mapping
        >>> process_k49(imgs, labels, '/path/to/output', 'train', label_mapping)

    This will process 1000 images, resize them, and save them in the appropriate
    subdirectories of '/path/to/output' based on their codepoints.
    """
    for img, label in tqdm(zip(imgs, labels), total=len(imgs), desc=f"Processing {prefix}"):
        codepoint = label_mapping[label]
        class_dir = os.path.join(output_dir, codepoint)
        os.makedirs(class_dir, exist_ok=True)

        img_name = generate_unique_filename()
        save_path = os.path.join(str(class_dir), img_name)
        resize_and_save(img, save_path)


def process_kkanji(kkanji_dir, output_dir):
    """
    Process and organize images from the Kuzushiji-Kanji (KKanji) dataset.

    This function walks through the KKanji dataset directory, copying PNG images
    to a new directory structure while preserving their class (character) organization.

    Args:
        kkanji_dir (str): Path to the root directory of the KKanji dataset.
        output_dir (str): Path to the directory where processed images will be saved.

    Returns:
        None

    Side effects:
        - Creates directories in output_dir mirroring the class structure of kkanji_dir.
        - Copies PNG images from kkanji_dir to their corresponding directories in output_dir.

    Notes:
        - The function uses tqdm to display a progress bar during processing.
        - Only files with '.png' extension are processed.
        - The original file names are preserved in the copying process.
        - The directory structure will be: output_dir/class_name/image.png
        - Class names are assumed to be the names of the immediate parent directories of the PNG files.

    Raises:
        FileNotFoundError: If kkanji_dir does not exist or is not accessible.
        IOError: If there are issues creating directories or copying files.

    Example:
        >>> process_kkanji('/path/to/kkanji_dataset', '/path/to/output')

    This will walk through the '/path/to/kkanji_dataset' directory, creating a
    mirrored structure in '/path/to/output' and copying all PNG files to their
    respective class directories.

    Warning:
        This function will overwrite files in the output directory if they have
        the same names as files in the input directory.
    """
    for root, _, files in tqdm(os.walk(kkanji_dir), desc="Processing KKanji"):
        for file in files:
            if file.endswith('.png'):
                src_path = os.path.join(root, file)
                class_name = os.path.basename(root)
                dst_dir = os.path.join(output_dir, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(str(dst_dir), file)
                shutil.copy2(str(src_path), str(dst_path))
