import customtkinter as ctk
import os

from controller import Controller
from PIL import ImageTk
from tkinter import filedialog, messagebox
from typing import List


# Constants
PROCESS_NAME = "Bleeding Detecting on Capsule Endoscopy"


class App(ctk.CTk):
    def __init__(self, controller):
        super().__init__()

        # Grid configuration
        self.geometry("1280x720")   # 16:9
        self.title(PROCESS_NAME)
        self.columnconfigure(0, weight=3)  # Image display
        self.columnconfigure(1, weight=1)  # Sidebar
        self.rowconfigure(0, weight=1)

        # Display for the images, will be in the center
        self.image_display = ImageDisplay(self)

        # Sidebar frame: Parent that will hold every widget of the sidebar, will be placed right
        self.sidebar_frame = ctk.CTkFrame(self)
        self.sidebar_frame.grid(row=0, column=1, pady=20, padx=20, sticky="nsew")

        self.sidebar_frame.columnconfigure(0, weight=1) # Adjust grid for the sidebar itself
        # Configure sidebar so that expansion is possible while still sticking buttons to the bottom as well
        self.sidebar_frame.grid_rowconfigure(0, weight=0)  # Row 0 static
        self.sidebar_frame.grid_rowconfigure(1, weight=0)  # Row 1 static
        self.sidebar_frame.grid_rowconfigure(2, weight=0)  # Row 2 static
        self.sidebar_frame.grid_rowconfigure(3, weight=1)  # Row 3: Empty expanding space
        self.sidebar_frame.grid_rowconfigure(4, weight=0)  # Row 4: Bottom space, static

        # Button for browsing the location of the image directory, parent to the sidebar
        self.image_folder_button = ctk.CTkButton(self.sidebar_frame, text="Browse location", command=self.callback_browse_image_folder_location)
        self.image_folder_button.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")

        # Panel to manage the image paths
        self.image_path_panel = ImageListPanel(self.sidebar_frame)

        # Button to load in selected images
        self.load_image_button = ctk.CTkButton(self.sidebar_frame, text="Load selected images", command=self.load_selected_images)
        self.load_image_button.grid(row=2, column=0, pady=10, padx=10, sticky="nsew")

        # Create the button that invokes the model for the currently selected image
        self.invoke_button = ctk.CTkButton(self.sidebar_frame, text="Compute prediction for image", fg_color="green", command=self.invoke_for_current_image)
        self.invoke_button.grid(row=4, column=0, pady=10, padx=10, sticky="s")

        # Controller for communication with backend
        self.controller = controller

    def callback_browse_image_folder_location(self):
        # Let the user select a folder
        folder_path = filedialog.askdirectory(title="Select a Folder")
        if folder_path:
            status, code = self.controller.set_image_directory_path(folder_path)

            # Only create panel if loading worked!
            if status:
                messagebox.showinfo(PROCESS_NAME, code)
                self.image_display.reset()
                self.image_path_panel.create_entries_for(folder_path, self.controller.image_paths)
            else:
                messagebox.showerror(PROCESS_NAME, "Error: " + code)

    def load_selected_images(self):
        # Load the selected images and show a pop-up with the result.
        status, code = self.controller.load_selected_images(self.image_path_panel.get_selected_files())
        if status:
            messagebox.showinfo(PROCESS_NAME, code)
            preproc_img_names = [p.split(os.path.sep)[-1] for p in self.controller.selected_image_paths]
            self.image_display.set_images(self.controller.loaded_images, preproc_img_names)
        else:
            messagebox.showerror(PROCESS_NAME, "Error: " + code)

    def invoke_for_current_image(self):
        # Call the model with the currently shown image.
        status, code, result = self.controller.invoke_for_image_with_index(self.image_display.current_image_index)
        if status:
            # TODO: Don't create pop-up maybe but something prettier
            messagebox.showinfo(PROCESS_NAME, result)
        else:
            messagebox.showerror(PROCESS_NAME, "Error: " + code)


class ImageListPanel:
    def __init__(self, root):
        self.root = root

        # Create a frame that holds the header and the scrollable frame together
        self.frame = ctk.CTkFrame(self.root)
        self.frame.grid(row=1, column=0, pady=10, padx=10, sticky="nsew")
        self.frame.columnconfigure(0, weight=1)

        # Create a header for the scrollable frame
        self.header_label = ctk.CTkLabel(self.frame, text="Select image directory to display images", font=("Arial", 14))
        self.header_label.grid(row=0, column=0, pady=5, padx=10, sticky="nsew")

        # Create the scrollable frame so that it scales with a larger number of imageas
        self.scrollable_frame = ctk.CTkScrollableFrame(self.frame, fg_color="white")
        self.scrollable_frame.grid(row=1, column=0, pady=10, padx=10, sticky="nsew")

        # Store selected files
        self.image_checkboxes = {}

    def create_entries_for(self, root_dir: str, image_paths: List[str]):

        # Delete all previous ones if this is called
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Set the header
        self.header_label.configure(text=f"Images in: {root_dir}")

        # Create new widgets, one for each option and manage in the dict
        for image_file in image_paths:
            checkbox_var = ctk.BooleanVar(value=True)
            checkbox = ctk.CTkCheckBox(self.scrollable_frame, text=image_file.split(os.path.sep)[-1], variable=checkbox_var)
            checkbox.pack(anchor="w", padx=10, pady=2)
            self.image_checkboxes[image_file] = checkbox_var

    def get_selected_files(self) -> List[str]:
        checked_files = []
        for file, var in self.image_checkboxes.items():
            if var.get():
                checked_files.append(file)
        return checked_files


class ImageDisplay:
    def __init__(self, root):
        self.root = root

        # List of images with pointer in it. Empty, at first
        self.images = []
        self.image_names = []
        self.current_image_index = 0
        self.desired_image_size = (500, 500)

        # Create label that will hold the image, give it empty placeholder at first
        self.image_label = ctk.CTkLabel(
            self.root,
            text="<No images loaded yet>",
            fg_color="white",
            width=self.desired_image_size[0],
            height=self.desired_image_size[1],
            compound="bottom",  # Image above, name below
            font=ctk.CTkFont(size=14)
        )
        self.image_label.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")

        # Parent frame that will hold the arrows, created for grid layout
        self.arrow_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.arrow_frame.grid(row=1, column=0, pady=10, sticky="n")

        # Left arrow button to go to previous image, root to the arrow frame
        self.left_button = ctk.CTkButton(self.arrow_frame, text="←", command=self.show_previous_image)
        self.left_button.pack(side="left", padx=10)

        # Right arrow button to go to next image
        self.right_button = ctk.CTkButton(self.arrow_frame, text="→", command=self.show_next_image)
        self.right_button.pack(side="right", padx=10)

    def set_images(self, images, image_names):
        # Convert images into tkinter format and resize to desired size
        self.images = [ImageTk.PhotoImage(img.resize(self.desired_image_size)) for img in images]
        self.image_names = image_names

        # Show the first image and reset index
        self.image_label.configure(image=self.images[0], text=self.image_names[0])
        self.current_image_index = 0

    def show_previous_image(self):
        # Move to previous image and roll around if needed using modulo
        if len(self.images) > 0:
            self.current_image_index = (self.current_image_index - 1) % len(self.images)
            self.image_label.configure(
                image=self.images[self.current_image_index], text=self.image_names[self.current_image_index]
            )

    def show_next_image(self):
        # Move to next image and roll around if needed using modulo
        if len(self.images) > 0:
            self.current_image_index = (self.current_image_index + 1) % len(self.images)
            self.image_label.configure(
                image=self.images[self.current_image_index], text=self.image_names[self.current_image_index]
            )

    def reset(self):
        self.images = []
        self.image_names = []
        self.current_image_index = 0


if __name__ == "__main__":
    controller = Controller()
    app = App(controller)
    app.mainloop()
