import argparse
from multiprocessing import cpu_count, Pool, Manager
from collections import defaultdict
import os

import pycocotools.mask as mask_util

from utils.constants import AREA_MIN_THRESHOLD
from utils.io import read_serialized, write_serialized, json


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", "--input_folder", help="train, demo or peculiar", type=str, required=True)
    parser.add_argument("-e", '--end_index', help='image index to end', type=int)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    parser.add_argument("-s", "--segm_file", help="segmentation file", type=str, required=True)
    return parser.parse_args()


def main(case_id, case_name, case_folder, segmentation, annotations):
    anns_file = read_serialized(os.path.join(case_folder, "{}_ann.yaml".format(case_name)))
    anns = []
    n_frames = len(anns_file.scene)
    assert n_frames < 500
    for image_index in range(n_frames):
        image = anns_file.scene[image_index]
        image_id = case_id * 500 + image_index

        image.image_filename = os.path.join(case_folder, "imgs", os.path.basename(image.image_path))
        image.image_index = image_id

        image.gt_objects = []
        for obj in image.objects:
            obj.area = int(mask_util.area(obj.mask))
            if obj.area < AREA_MIN_THRESHOLD:
                continue
            #  Tell occluders from cubes
            if obj.scale[0] < .05:
                obj.type = "Occluder"
            else:
                obj.type = "Sphere"
                obj.rotation = [0, 0, 0]
            image.gt_objects.append(obj)

        image.objects = segmentation[image_id]
        anns.append(image)
    annotations[case_id] = anns

    print("{} generated".format(case_name))


if __name__ == '__main__':
    args = parse_args()
    worker_args = []
    manager = Manager()
    annotations = manager.dict()
    with open(args.segm_file, "r") as f:
        s = json.load(f)
        segmentation = defaultdict(list)
        for o in s:
            segmentation[o["image_id"]].append(o)

    case_names = sorted(list(case_name for case_name in os.listdir(args.input_folder)))
    if args.end_index is not None:
        case_names = case_names[:args.end_index]
    for i, case_name in enumerate(case_names):
        case_folder = os.path.join(args.input_folder, case_name)
        assert os.path.exists(os.path.join(case_folder, "{}_ann.yaml".format(case_name)))
        worker_args.append((i, case_name, case_folder, segmentation, annotations))

    with Pool(cpu_count()) as p:
        p.starmap(main, worker_args)

    scenes = dict()
    for i, case_name in enumerate(case_names):
        case_folder = os.path.join(args.input_folder, case_name)
        scene = dict(scene=annotations[i], case_name=case_name, case_folder=case_folder, case_id=i)
        scenes[case_name] = scene

    write_serialized(scenes, args.output_file)
