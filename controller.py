import cv2
import joblib
import glob
import os
import timm
import torch
import torchvision.transforms as T

from baseline import extract_h_features
from PIL import Image
from typing import List, Tuple


MODEL_DIR = os.path.join("checkpoints", "20_45_04.2845554", "model_params")
BASELINE_DIR = "random_forest_baseline.pkl"


class Controller:
    def __init__(self):
        self.image_directory_path = None
        self.image_paths = []
        self.selected_image_paths = []
        self.loaded_images = []

        # Load in the baseline
        print("Load in the baseline...")
        self.baseline = joblib.load(BASELINE_DIR)
        print("Loaded in the baseline.")

        # Load in the checkpoint state dict of the model to proper device
        print("Loading in model...")
        if torch.cuda.is_available():
            checkpoint = torch.load(MODEL_DIR)
        else:
            checkpoint = torch.load(MODEL_DIR, map_location=torch.device('cpu'))

        # Create empty model, replace necessary layers
        model = timm.create_model(model_name="resnet101", pretrained=False)
        if hasattr(model, "fc"):
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
        elif hasattr(model, "classifier"):
            model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
        model.load_state_dict(checkpoint, strict=True)

        # Do not update model anymore
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self.model = model
        print("Loaded in the model.")

    def baseline_prediction(self, img_path):
        res = self.baseline.predict(extract_h_features(cv2.imread(img_path)).reshape(1, -1))[0]
        return "Healthy" if res <= 0.5 else "Bleeding"

    def model_prediction(self, img):
        # Transformation pipeline, add batch dimension
        t = T.Compose([T.ToTensor()])
        model_input = t(img).unsqueeze(0)

        with torch.no_grad():
            model_output = self.model(model_input)
            predicted_class = torch.argmax(model_output).int()
        return "Healthy" if predicted_class <= 0.5 else "Bleeding"

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

        # Get model and baseline prediciton and format into string
        baseline_pred = self.baseline_prediction(self.selected_image_paths[index])
        model_pred = self.model_prediction(self.loaded_images[index])
        return True, "", f"Baseline: {baseline_pred} \nModel: {model_pred}"

