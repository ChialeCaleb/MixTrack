import os
from datasets.types.data_split import DataSplit
from datasets.SOT.constructor.base_interface import SingleObjectTrackingDatasetConstructor
from miscellanies.parser.ini import parse_ini_file
from miscellanies.parser.txt import load_numpy_array_from_txt
from miscellanies.numpy.dtype import try_get_int_array
import ast
import numpy as np
import pdb
import cv2

def _construct_VisEvent_public_data(constructor: SingleObjectTrackingDatasetConstructor, sequence_list):
    for sequence_name, sequence_path in sequence_list:
        images = os.listdir(sequence_path + '/event_imgs')
        images = [image for image in images if image.endswith('.bmp')]
        images.sort()
        absence_array = load_numpy_array_from_txt(os.path.join(sequence_path, 'absent_label.txt'), dtype=bool)
        absence_array = (1 - absence_array).astype(np.bool)
        # Values 0~8 in file cover.label correspond to ranges of object visible ratios: 0%, (0%, 15%], (15%~30%], (30%, 45%], (45%, 60%], (60%, 75%], (75%, 90%], (90%, 100%) and 100% respectively.

        bounding_boxes = load_numpy_array_from_txt(os.path.join(sequence_path, 'groundtruthxyxy.txt'), delimiter=',')
        bounding_boxes = try_get_int_array(bounding_boxes)

        assert len(images) == len(absence_array) == len(bounding_boxes)
        frame = cv2.imread(os.path.join(sequence_path, 'event_imgs', images[0]))
        frame_size = frame.shape[:2]
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            for image, bounding_box, absence in zip(images, bounding_boxes, absence_array):
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_path, 'event_imgs', image), frame_size)
                    frame_constructor.set_bounding_box(bounding_box.tolist(), validity=not absence)
                    frame_constructor.set_object_attribute('absence', absence.item())


def _construct_VisEvent_non_public_data(constructor: SingleObjectTrackingDatasetConstructor, sequence_list):
    for sequence_name, sequence_path in sequence_list:
        images = os.listdir(sequence_path + '/event_imgs')
        images = [image for image in images if image.endswith('.bmp')]
        images.sort()

        bounding_box = load_numpy_array_from_txt(os.path.join(sequence_path, 'ground_truth_first_frame.txt'), delimiter=',')
        bounding_box = try_get_int_array(bounding_box)

        if all(bounding_box==[0, 0, 0, 0]):
            continue
        assert bounding_box.ndim == 1 and bounding_box.shape[0] == 4

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            for index_of_image, image in enumerate(images):
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_path, 'event_imgs', image))
                    if index_of_image == 0:
                        frame_constructor.set_bounding_box(bounding_box.tolist())


def construct_VisEvent(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split
    sequence_filter = seed.sequence_filter

    if data_split == DataSplit.Training:
        folder = 'train'
    elif data_split == DataSplit.Validation:
        folder = 'val'
    elif data_split == DataSplit.Testing:
        folder = 'test'
    else:
        raise RuntimeError(f'Unsupported dataset split {data_split}')

#    constructor.set_category_id_name_map({k: v for k, v in enumerate(_category_names)})

    sequence_list = []
    for sequence_name in open(os.path.join(root_path, folder, 'list.txt'), 'r'):
        sequence_name = sequence_name.strip()
        current_sequence_path = os.path.join(root_path, folder, sequence_name)
        sequence_list.append((sequence_name, current_sequence_path))

    if sequence_filter is not None:
        sequence_id_file_path = os.path.join(os.path.dirname(__file__), 'data_specs', f'{sequence_filter}.txt')
        sequence_ids = np.loadtxt(sequence_id_file_path, dtype=np.uint32)
        sequence_list = [sequence_list[id_] for id_ in sequence_ids]

    constructor.set_total_number_of_sequences(len(sequence_list))

    if data_split in (DataSplit.Training, DataSplit.Validation):
        _construct_VisEvent_public_data(constructor, sequence_list)
    else:
        _construct_VisEvent_non_public_data(constructor, sequence_list)