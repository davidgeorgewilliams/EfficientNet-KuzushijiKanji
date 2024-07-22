import os
import random
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import transform
from tqdm import tqdm


def resize_and_save(img, save_path):
    """
    Resize an image to 64x64 pixels and save it to the specified path.

    This function takes a numpy array representing an image, converts it to a PIL Image,
    resizes it to 64x64 pixels using the Lanczos resampling filter, and then saves it
    to the specified file path.

    Args:
        img (numpy.ndarray): A 2D or 3D numpy array representing a grayscale image.
                             If 3D, the function will squeeze it to 2D.
        save_path (str): The file path where the resized image should be saved.

    Returns:
        None

    Notes:
        - The input image is assumed to be grayscale.
        - The function uses the Lanczos resampling filter, which generally provides
          high-quality results for downscaling.
        - The output image is always saved as a grayscale PNG file.

    Raises:
        ValueError: If the input is not a 2D or 3D numpy array.
        IOError: If there's an issue saving the file to the specified path.

    Example:
        >>> import numpy as np
        >>> img = np.random.rand(100, 100) * 255  # Create a random 100x100 grayscale image
        >>> resize_and_save(img, 'resized_image.png')
        This will create a 64x64 pixel grayscale PNG image named 'resized_image.png'.
    """
    img_pil = Image.fromarray(img.squeeze(), mode='L')
    img_resized = img_pil.resize((64, 64), Image.LANCZOS)
    img_resized.save(save_path)


def balance_class(input_dir, output_dir, target_count=10000):
    """
    Balance a class of images to a target count through sampling or augmentation.

    This function ensures that the output directory contains exactly `target_count` images
    for a given class. If the input directory contains more images than the target count,
    it randomly samples from the existing images. If it contains fewer, it copies all
    existing images and then augments them to reach the target count.

    Args:
        input_dir (str): Path to the directory containing the original images for a single class.
        output_dir (str): Path to the directory where balanced images will be saved.
        target_count (int, optional): The desired number of images for the class. Defaults to 10000.

    Returns:
        None

    Notes:
        - Images in the input directory should be in PNG format.
        - Output images are renamed numerically (e.g., '00000.png', '00001.png', etc.).
        - If augmentation is needed, it uses the `augment_image` function (not shown here).
        - Augmented images are converted to grayscale before processing.
        - A progress bar (tqdm) is displayed during augmentation.

    Behavior:
        1. If input_dir has >= target_count images:
           - Randomly sample target_count images and copy them to output_dir.
        2. If input_dir has < target_count images:
           - Copy all existing images to output_dir.
           - Augment randomly chosen images until target_count is reached.

    Example:
        >>> balance_class('/path/to/original/あ', '/path/to/balanced/あ', target_count=10000)
        This will ensure the '/path/to/balanced/あ' directory contains exactly 10000 images,
        either by sampling or augmenting the images from '/path/to/original/あ'.

    Warning:
        This function will overwrite existing files in the output directory if they have
        the same names as the generated files.
    """
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


def elastic_transform(image, alpha=1000, sigma=30, random_state=None):
    """
    Apply elastic deformation to an image.

    This function performs an elastic transformation on the input image, which can be
    used for data augmentation in machine learning tasks, especially for handwritten
    character recognition.

    Args:
        image (numpy.ndarray): Input image as a numpy array. Should be a 2D array
                               representing a grayscale image.
        alpha (float, optional): Scaling factor for the deformation field. Higher values
                                 result in more pronounced deformations. Default is 1000.
        sigma (float, optional): Standard deviation of the Gaussian filter used to smooth
                                 the deformation field. Higher values result in smoother,
                                 more global deformations. Default is 30.
        random_state (numpy.random.RandomState, optional): Random number generator state
                                                           used for reproducibility. If None,
                                                           a new RandomState is created.

    Returns:
        numpy.ndarray: The elastically transformed image as a numpy array with the same
                       shape as the input image.

    Note:
        This function uses Gaussian filters to generate smooth deformation fields and
        applies them to the image using interpolation. The deformation is applied
        separately in the x and y directions.

    Example:
        >>> import numpy as np
        >>> from scipy.misc import face
        >>> image = face(gray=True)
        >>> transformed = elastic_transform(image)
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def affine_transform(image):
    """
    Apply a random affine transformation to an image.

    This function performs a combination of random rotation, scaling, and translation
    on the input image. It's useful for data augmentation in image processing and
    machine learning tasks, particularly for tasks like handwritten character recognition
    where these transformations can simulate natural variations in writing.

    Args:
        image (numpy.ndarray): Input image as a numpy array. Should be a 2D array
                               representing a grayscale image or a 3D array for RGB.

    Returns:
        numpy.ndarray: The affine-transformed image as a numpy array with the same
                       shape as the input image.

    Transformations applied:
        - Rotation: Random rotation between -15 and 15 degrees.
        - Scaling: Random uniform scaling between 0.9 and 1.1.
        - Translation: Random translation between -5 and 5 pixels in both x and y directions.

    Note:
        The function uses skimage.transform.AffineTransform to create the transformation
        matrix and skimage.transform.warp to apply the transformation. The 'reflect' mode
        is used for handling pixels outside the image boundaries.

    Example:
        >>> import numpy as np
        >>> from skimage import data
        >>> image = data.camera()
        >>> transformed = affine_transform(image)
    """
    # Random rotation between -15 and 15 degrees
    angle = np.random.uniform(-15, 15)
    # Random scaling between 0.9 and 1.1
    scale = np.random.uniform(0.9, 1.1)
    # Random translation
    translation = (np.random.uniform(-5, 5), np.random.uniform(-5, 5))

    tform = transform.AffineTransform(scale=(scale, scale), rotation=np.deg2rad(angle), translation=translation)
    return transform.warp(image, tform.inverse, mode='reflect')


def add_noise(image, noise_factor=0.05):
    """
    Add Gaussian noise to an image.

    This function adds random Gaussian noise to the input image. It's useful for data
    augmentation in image processing and machine learning tasks, simulating variations
    in image quality or sensor noise.

    Args:
        image (numpy.ndarray): Input image as a numpy array. Should be normalized
                               to the range [0, 1].
        noise_factor (float, optional): Scaling factor for the noise. Higher values
                                        result in more pronounced noise. Default is 0.05.

    Returns:
        numpy.ndarray: The noisy image as a numpy array with the same shape as the
                       input image, clipped to the range [0, 1].

    Note:
        The function generates Gaussian noise with mean 0 and standard deviation 1,
        scales it by the noise_factor, adds it to the image, and then clips the
        result to ensure pixel values remain in the valid range [0, 1].

    Example:
        >>> import numpy as np
        >>> image = np.random.rand(100, 100)  # Create a random image
        >>> noisy_image = add_noise(image, noise_factor=0.1)
    """
    noise = np.random.normal(loc=0, scale=1, size=image.shape)
    noisy_image = image + noise_factor * noise
    return np.clip(noisy_image, 0, 1)


def augment_image(image):
    """
    Apply a series of augmentations to an image.

    This function applies multiple image augmentation techniques in sequence:
    elastic transformation, affine transformation, and noise addition. It's designed
    to create varied versions of the input image for data augmentation purposes.

    Args:
        image (numpy.ndarray): Input image as a numpy array. Should be normalized
                               to the range [0, 1].

    Returns:
        numpy.ndarray: The augmented image as a numpy array with the same shape
                       as the input image.

    Note:
        The augmentations are applied in the following order:
        1. Elastic transform: Applies random elastic deformations.
        2. Affine transform: Applies random rotation, scaling, and translation.
        3. Noise addition: Adds random Gaussian noise.

        Each transformation uses its default parameters. Adjust the individual
        functions for more control over the augmentation process.

    Example:
        >>> import numpy as np
        >>> from skimage import data
        >>> image = data.camera() / 255.0  # Normalize to [0, 1]
        >>> augmented = augment_image(image)
    """
    augmented = elastic_transform(image)
    augmented = affine_transform(augmented)
    augmented = add_noise(augmented)
    return augmented
