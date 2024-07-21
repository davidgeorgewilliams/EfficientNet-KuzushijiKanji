import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import transform


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
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

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
