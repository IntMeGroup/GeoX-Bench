from PIL import Image
from typing import Union

class CenterDivImageCroper():
    

    def __init__(self):
        "The location of the image center after the crop .e.g. Using left means after the crop the cetner will appear at the left of the image."
        self.factor_dict = {
            "center": (1/6, 1/6, 1/6, 1/6),

            "left": (1/3, 1/6, 0, 1/6),
            "right": (0, 1/6, 1/3, 1/6),
            "top": (1/6, 1/3, 1/6, 0),
            "bottom": (1/6, 0, 1/6, 1/3),

            "top_left": (1/3, 1/3, 0, 0),
            "top_right": (0, 1/3, 1/3, 0),
            "bottom_left": (1/3, 0, 0, 1/3),
            "bottom_right": (0, 0, 1/3, 1/3),
        }

    def __call__(self, 
                 image: Union[Image.Image, str], 
                 crop_location: str) -> Image.Image:
        """
        Crops the given image using the provided crop location.

        Parameters:
        image (Union[Image.Image, str]): The PIL Image object or a path to an image file.
        crop_location (str): The location to crop from. Must be one of:
                    "center", "left", "right", "top", "bottom", 
                    "top_left", "top_right", "bottom_left", "bottom_right"
                    
        Returns:
        Image.Image: A new cropped PIL Image object.
        """
        # Open the image if it's a path
        if isinstance(image, str):
            image = Image.open(image)
            
        width, height = image.size

        # Unpack the factors
        x1, y1, x2, y2 = self.factor_dict[crop_location]
        left = width * abs(x1)
        right = width * (1 - abs(x2))
        top = height * abs(y1)
        bottom = height * (1 - abs(y2))

        # print(f"left: {left}, right: {right}, top: {top}, bottom: {bottom}")
        # print(f"width: {width}, height: {height}")

        # Return the cropped image
        return image.crop((left, top, right, bottom))
