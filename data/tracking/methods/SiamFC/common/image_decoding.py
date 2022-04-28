import torch
import cv2


class SiamFCImageDecodingProcessor:
    def __init__(self, post_processor=None):
        self.post_processor = post_processor

    def __call__(self, z_image_path, z_bbox, x_image_path, x_bbox, is_positive):
        event_path = "/single/"

        z_image = cv2.imread(z_image_path)
        z_image = torch.Tensor(z_image).permute(2, 0, 1)
        z_event = cv2.imread(z_image_path.replace('/img/', event_path))
        z_event = torch.Tensor(z_event).permute(2, 0, 1)

        if z_image_path != x_image_path:
            x_image = cv2.imread(x_image_path)
            x_image = torch.Tensor(x_image).permute(2, 0, 1)
            x_event = cv2.imread(x_image_path.replace('/img/', event_path))
            x_event = torch.Tensor(x_event).permute(2, 0, 1)
        else:
            x_image = z_image
            x_event = z_event
        data = (z_image, z_event, z_bbox, x_image, x_event, x_bbox, is_positive)
        if self.post_processor is not None:
            return self.post_processor(*data)
        else:
            return data
