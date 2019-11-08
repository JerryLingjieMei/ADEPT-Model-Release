import argparse
from multiprocessing import cpu_count, Pool, Manager
import os

from tqdm import tqdm
import pycocotools.mask as mask_util
from utils.misc import mask2contour
from utils.constants import AREA_MIN_THRESHOLD, CATEGORY2ID, WIDTH, HEIGHT
from utils.io import read_serialized, write_serialized


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", "--input_folder", help="train, demo or peculiar", type=str, required=True)
    parser.add_argument("-e", '--end_index', help='image index to end', type=int)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    return parser.parse_args()


def main(case_id, case_name, case_folder, args):
    anns_file = read_serialized(os.path.join(case_folder, "{}_ann.yaml".format(case_name)))
    n_frames = len(anns_file.scene)
    assert n_frames < 500
    images = []
    annotations = []
    for image_index in range(n_frames):
        image = anns_file.scene[image_index]
        image.image_path = os.path.join(case_folder, "imgs", os.path.basename(image.image_path))
        image_id = case_id * 500 + image_index

        objects = []
        drop_image = False

        for obj in image.objects:
            obj.segmentation = mask2contour(mask_util.decode(obj.mask))
            
            # otherwise could be interpreted as a bbox
            if len(obj.segmentation) == 4:
                drop_image = True
                continue
            obj.area = int(mask_util.area(obj.mask))
            if obj.area < AREA_MIN_THRESHOLD:
                continue
            obj.bbox = list(mask_util.toBbox(obj.mask))
            obj.category_id = CATEGORY2ID[obj.type]
            del obj.mask
            del obj.type
            obj.image_id = image_id
            obj.id = image_id * 10 + len(objects)
            obj.iscrowd = 0
            objects.append(obj)

        if len(objects) > 10:
            raise ValueError("More than 10 objects")

        if len(objects) > 0 and not drop_image:
            for obj in objects:
                annotations.append(obj)
            images.append(dict(file_name=image.image_path, width=WIDTH, height=HEIGHT,
                               id=image_id))
    return images, annotations


if __name__ == '__main__':
    args = parse_args()
    worker_args = []
    manager = Manager()

    case_names = sorted(list(case_name for case_name in os.listdir(args.input_folder)))
    if args.end_index is not None:
        case_names = case_names[:args.end_index]

    with Pool(cpu_count()) as p:
        with tqdm(total=len(case_names)) as bar:
            results = p.starmap_async(main, [(i, case_name, os.path.join(args.input_folder, case_name), args) for
                                             i, case_name in enumerate(case_names)],
                                      callback=lambda *a: bar.update()).get()
    categories = [dict(id=1, name="Sphere"), dict(id=2, name="Occluder")]
    images = []
    annotations = []
    for i, a in results:
        images.extend(i)
        annotations.extend(a)
    data = dict(images=images, annotations=annotations, categories=categories)
    write_serialized(data, args.output_file)
