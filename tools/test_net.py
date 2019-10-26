import argparse
import os
import logging
import time
import datetime

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from tools.net_defaults import _C as cfg
from models import build_model
from dataset import cycle
from utils.metric_logger import MetricLogger
from utils.checkpoint import Checkpointer
from solver.build import make_lr_scheduler, make_optimizer
from utils.logger import setup_logger
from utils.io import mkdir
from utils.misc import to_cuda, assert_proper_output_dir
from tools.paths_catalog import DatasetCatalog
from visualization.summary import build_summary_visualization


def test(cfg, args):
    gpu_ids = [_ for _ in range(torch.cuda.device_count())]
    model = build_model(cfg)
    model.to("cuda")
    model = torch.nn.parallel.DataParallel(model, gpu_ids) if not args.debug else model

    logger = logging.getLogger("test_logger")
    logger.info("Start testing")
    test_metrics = MetricLogger(delimiter="  ")
    output_dir = cfg.OUTPUT_DIR

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    checkpointer = Checkpointer(
        model, optimizer, scheduler, output_dir, logger
    )
    checkpointer.load()

    test_set = DatasetCatalog.get(cfg.DATASETS.TEST, args)
    print(test_set.__getitem__(0))
    test_loader = DataLoader(test_set, cfg.SOLVER.IMS_PER_BATCH,
                             num_workers=cfg.DATALOADER.NUM_WORKERS, shuffle=False)
    summary_writer = SummaryWriter(log_dir=os.path.join(output_dir, "summary"))
    visualizer = test_set.visualizer(cfg.VISUALIZATION)(summary_writer)

    model.eval()
    start_training_time = time.time()
    last_batch_time = time.time()

    with torch.no_grad():
        for iteration, inputs in enumerate(tqdm(test_loader)):
            data_time = time.time() - last_batch_time
            iteration = iteration + 1

            inputs = to_cuda(inputs)
            outputs = model(inputs)
            test_set.process_batch(inputs, outputs)

            batch_time = time.time() - last_batch_time
            last_batch_time = time.time()
            test_metrics.update(time=batch_time, data=data_time)

            if iteration % 20 == 0:
                summary_writer.add_scalars("test", test_metrics.mean, iteration)

            if iteration % 100 == 0:
                visualizer.visualize(inputs, outputs, iteration)

    inference_dir = mkdir(os.path.join(output_dir, "inference"))
    result_dir = mkdir(os.path.join(inference_dir, cfg.DATASETS.TEST))
    test_set.process_results(result_dir)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total testing time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (len(test_loader))
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--debug", default=0, type=int)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = mkdir(cfg.OUTPUT_DIR)
    assert_proper_output_dir(args.config_file, output_dir)

    logger = setup_logger("test_logger", os.path.join(output_dir, "log.txt"))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, args)


if __name__ == "__main__":
    main()
