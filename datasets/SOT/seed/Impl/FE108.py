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

def _construct_FE108_public_data(constructor: SingleObjectTrackingDatasetConstructor, sequence_list):
    for sequence_name, sequence_path in sequence_list:
        images = os.listdir(sequence_path + '/img')
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        bounding_boxes = load_numpy_array_from_txt(os.path.join(sequence_path, 'groundtruth_xyxy.txt'), delimiter=',')
        bounding_boxes = try_get_int_array(bounding_boxes)

        assert len(images) == len(bounding_boxes)
        frame = cv2.imread(os.path.join(sequence_path, 'img', images[0]))
        frame_size = frame.shape[:2]
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            for image, bounding_box in zip(images, bounding_boxes):
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_path, 'single', image), frame_size)
                    frame_constructor.set_bounding_box(bounding_box.tolist())

def construct_FE108(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split
    sequence_filter = seed.sequence_filter

    if data_split == DataSplit.Training:
        folder = 'train.txt'
    elif data_split == DataSplit.Validation:
        folder = 'test.txt'
    else:
        raise RuntimeError(f'Unsupported dataset split {data_split}')

#    constructor.set_category_id_name_map({k: v for k, v in enumerate(_category_names)})

    sequence_list = []
    for sequence_name in open(os.path.join(root_path, folder), 'r'):
        sequence_name = sequence_name.strip()
        current_sequence_path = os.path.join(root_path, sequence_name)
        sequence_list.append((sequence_name, current_sequence_path))

    if sequence_filter is not None:
        sequence_id_file_path = os.path.join(os.path.dirname(__file__), 'data_specs', f'{sequence_filter}.txt')
        sequence_ids = np.loadtxt(sequence_id_file_path, dtype=np.uint32)
        sequence_list = [sequence_list[id_] for id_ in sequence_ids]

    constructor.set_total_number_of_sequences(len(sequence_list))

    if data_split in (DataSplit.Training, DataSplit.Validation):
        _construct_FE108_public_data(constructor, sequence_list)
    else:
        raise NotImplementedError