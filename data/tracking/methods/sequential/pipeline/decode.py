import torch
import cv2
def _decode_image(image_path):
    image = cv2.imread(image_path)
    image = torch.Tensor(image).permute(2, 0, 1)
    image = image.to(torch.float)
    image /= 255.
    return image