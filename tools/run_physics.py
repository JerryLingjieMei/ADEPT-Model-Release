import argparse
import time
import glob
import os
import json
from multiprocessing import Process, cpu_count, Manager
from multiprocessing.pool import ThreadPool

from PIL import Image

from models.particle_filter import FilterUpdater
from tools.physics_defaults import _C as default_cfg
from visualization.plot_summary import plot_case
from utils.io import read_serialized, mkdir, write_serialized
from utils.logger import setup_logger
from utils.misc import assert_proper_output_dir, get_host_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--only_plot", type=int, default=0)
    parser.add_argument("--start_index", type=int)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def run_updater(cfg, args, case_name):
    output_folder = cfg.OUTPUT_FOLDER
    result_folder = mkdir(os.path.join(output_folder, "results"))

    gt = read_serialized(os.path.join(cfg.ANNOTATION_FOLDER, case_name, case_name + "_ann.yaml"))
    camera = gt["camera"]

    observation_path = os.path.join(cfg.OBSERVATION_FOLDER, case_name + ".json")
    scenes = read_serialized(observation_path)['scene']
    for s in scenes:
        for o in s["objects"]:
            if "color" not in o:
                o["color"] = "green"

    if not args.only_plot:
        mkdir(os.path.join(output_folder, "logs"))
        logger = setup_logger("{}{}".format(cfg.LOG_PREFIX, case_name),
                              os.path.join(cfg.OUTPUT_FOLDER, "logs", "{}.txt".format(case_name)))
        logger.info('{} start running '.format(case_name))
        logger.info(args)
        logger.info("Running with config:\n{}".format(cfg))

        # run updater
        init_belief = scenes[0]['objects']
        filter_updater = FilterUpdater(cfg, init_belief, case_name, camera, args.n_filter)
        filter_updater.run(scenes[1:])

        score = filter_updater.get_score()
        write_serialized({case_name: score}, os.path.join(result_folder, "{}.json".format(case_name)))

        with open(os.path.join(output_folder, "{}.txt".format(case_name)), 'w') as fout:
            fout.write(
                '| negative log likelihood: ' + json.dumps(
                    {key: result for key, result in score.items() if
                     key in ["sum", "mean", "max", "sum_lower", "mean_lower", "max_lower"]}) + '\n')

        logger.info('{} completed running '.format(case_name))
    else:
        results = read_serialized(os.path.join(result_folder, "{}.json".format(case_name)))
        score = results[case_name]

    images_files = [ann["image_path"] for ann in gt["scene"]][6:]
    with ThreadPool(cpu_count() * 4) as p:
        images = p.map(Image.open, images_files)

    plot_case(images, score["all"], score["raw"], score["location"], scenes[1:], [None] * len(images),
              case_name, output_folder)

    # os.system(
    #     '/data/vision/billf/object-properties/local/bin/ffmpeg -nostdin -r %d -pattern_type glob -i \'%s/%s.png\' '
    #     '-pix_fmt yuv420p -vcodec libx264 -crf 0 %s.mp4 -y'
    #     % (15, "{}/imgs".format(output_folder), "{}_???".format(case_name),
    #        "{}/{}_summary".format(output_folder, case_name)))
    # print('ffmpeg -nostdin -r %d -pattern_type glob -i \'%s/%s.png\' '
    # '-pix_fmt yuv420p -vcodec libx264 -crf 0 %s.mp4 -y'
    # % (15, "{}/imgs".format(output_folder), "{}_???".format(case_name),
    #    "{}/{}_summary".format(output_folder, case_name)))
    os.system(
    'ffmpeg -nostdin -r %d -pattern_type glob -i \'%s/%s.png\' '
    '-pix_fmt yuv420p -vcodec libx264 -crf 0 %s.mp4 -y'
    % (15, "{}/imgs".format(output_folder), "{}_???".format(case_name),
       "{}/{}_summary".format(output_folder, case_name)))



def main(args):
    cfg = default_cfg.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_folder = mkdir(cfg.OUTPUT_FOLDER)
    assert_proper_output_dir(args.config_file, output_folder)

    start_time = time.time()

    processes = []
    case_names = cfg.CASE_NAMES
    if len(case_names) == 0:
        case_names = sorted([dir_name for dir_name in os.listdir(cfg.ANNOTATION_FOLDER) if
                             os.path.isdir(os.path.join(cfg.ANNOTATION_FOLDER, dir_name)) and "." not in dir_name])
    if args.start_index is None:
        start_index = get_host_id() % args.stride
    else:
        start_index = args.start_index
    case_names = case_names[start_index::args.stride]

    manager = Manager()
    n_filter = manager.Semaphore(1)
    args.n_filter = n_filter

    for case_name in case_names:
        p = Process(target=run_updater, args=(cfg, args, case_name))
        processes.append(p)
        p.start()
        break  #ERASE
    for p in processes:
        p.join()

    if not args.only_plot:
        for case_name in case_names:
            results = read_serialized(os.path.join(output_folder, "results", "{}.json".format(case_name)))
            print(case_name,
                  {key: result for key, result in results[case_name].items() if key in ["sum", "mean", "max"]})

    print('| finish with time ', time.time() - start_time)


if __name__ == '__main__':
    args = parse_args()
    main(args)
