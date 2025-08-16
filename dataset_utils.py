import random
def random_crop(img, crop_size):
    """
    Randomly crops an image to the specified crop_size (width, height)
    and returns the cropped image along with the (left, top) starting coordinates.
    
    Parameters:
      img: PIL.Image object.
      crop_size: tuple (crop_width, crop_height).
      
    Returns:
      cropped_img: The cropped PIL.Image.
      start_pos: tuple (left, top) representing the starting position of the crop.
    """
    crop_w, crop_h = crop_size
    width, height = img.size  # PIL image size is (width, height)
    
    if width < crop_w or height < crop_h:
        raise ValueError("Image is smaller than the crop size.")
    
    left = random.randint(0, width - crop_w)
    top = random.randint(0, height - crop_h)

    right = left + crop_w
    bottom = top + crop_h
    
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img, (left, top)