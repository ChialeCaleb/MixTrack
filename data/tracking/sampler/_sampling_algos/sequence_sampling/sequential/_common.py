from data.operator.bbox.validity import bbox_is_valid_xywh
from data.operator.bbox.spatial.utility.half_pixel_offset.image import bounding_box_is_intersect_with_image
import pdb

def _check_bounding_box_validity(bounding_box, bounding_box_validity_flag, image_size):
    if bounding_box_validity_flag is not None and not bounding_box_validity_flag:
        bounding_box = None
    elif bounding_box is not None:
        assert bbox_is_valid_xywh(bounding_box)
    return bounding_box
