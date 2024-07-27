import os
import shutil


def ensure_directory(directory, remove_existing=False):
    """
    Ensure that a directory exists, optionally removing it if it already exists.

    This function checks if the specified directory exists and creates it if it doesn't.
    If the `remove_existing` flag is set to True, it will remove the directory and its
    contents before recreating it.

    Args:
        directory (str): The path to the directory to ensure exists.
        remove_existing (bool, optional): If True, remove the directory if it already
                                          exists before recreating it. Defaults to False.

    Returns:
        None

    Raises:
        OSError: If there are permission issues or other OS-level errors when
                 trying to remove or create the directory.

    Warning:
        Use the `remove_existing` option with caution, as it will delete all contents
        of the existing directory without any further prompts.

    Example:
        >>> ensure_directory('/path/to/my/directory')
        # Creates '/path/to/my/directory' if it doesn't exist

        >>> ensure_directory('/path/to/my/directory', remove_existing=True)
        # Removes '/path/to/my/directory' if it exists, then recreates it

    Note:
        This function uses `shutil.rmtree()` for directory removal and `os.makedirs()`
        for directory creation, which will create all necessary parent directories.
    """
    if remove_existing and os.path.isdir(directory):
        shutil.rmtree(directory)

    if not os.path.isdir(directory):
        os.makedirs(directory)
