import glob
import os

from PIL import Image
from typing import List, Tuple


class Controller:
    def __init__(self):
        self.image_directory_path = None
        self.image_paths = []
        self.selected_image_paths = []
        self.loaded_images = []

    def set_image_directory_path(self, path_to_directory: str) -> Tuple[bool, str]:
        # Try to parse the provided path
        try:
            # Reset the previously stored stuff
            self.image_directory_path = None
            self.image_paths = []
            self.selected_image_paths = []
            self.loaded_images = []

            folder_path = os.path.normpath(path_to_directory)
            if os.path.isdir(folder_path):
                self.image_directory_path = folder_path

                # Use glob to get all files with extension .jpg or .png from the provided path
                jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
                jpeg_files = glob.glob(os.path.join(folder_path, "*.jpeg"))
                png_files = glob.glob(os.path.join(folder_path, "*.png"))
                all_image_files = jpg_files + jpeg_files + png_files
                self.image_paths = all_image_files

                # Sanity check
                if len(all_image_files) == 0:
                    return False, "The provided directory does not contain any images, try again!"

                print(self.image_paths)
                return True, f"{len(all_image_files)} images were found under the provided directory."

        except Exception as e:
            print(e)
        return False, "The provided directory is not a proper path, try again!"

    def load_selected_images(self, selected_image_paths: List[str]) -> Tuple[bool, str]:
        # Reset the previously stored stuff
        self.selected_image_paths = []
        self.loaded_images = []

        # Sanity check
        if self.image_directory_path is None or len(self.image_paths) == 0:
            return False, "No proper image directory path was set, no images could be loaded!"
        if len(selected_image_paths) == 0:
            return False, "No images were selected from the list of images, no images could be loaded!"

        # Load in images using PIL
        self.selected_image_paths = selected_image_paths
        self.loaded_images = [Image.open(p) for p in selected_image_paths]
        return True, f"{len(selected_image_paths)} images were successfully loaded."

    def invoke_for_image_with_index(self, index: int) -> Tuple[bool, str, str]:
        # Sanity checks
        if self.image_directory_path is None or len(self.image_paths) == 0:
            return False, "No proper image directory path was set, no prediction can be computed yet!", ""
        if len(self.selected_image_paths) == 0:
            return False, "No images were selected from the list of images, no prediction can be computed yet!", ""
        if len(self.loaded_images) == 0:
            return False, "No images were loaded, no prediction can be computed yet!", ""
        if index < 0 or index >= len(self.loaded_images):
            return False, "The provided index is out of bounds for the loaded images!", ""

        # TODO: Actually call model here
        return True, "", "Prediction"
