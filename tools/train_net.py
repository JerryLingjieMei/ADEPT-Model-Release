import argparse
import os
import logging
import time
import datetime

import torch
from torch.utils.data.dataloader import DataLoader

from tensorboardX import SummaryWriter
from tools.net_defaults import _C as cfg
from models import build_model
from dataset import cycle
from utils.metric_logger import MetricLogger
from utils.checkpoint import Checkpointer
from solver.build import make_lr_scheduler, make_optimizer
from utils.logger import setup_logger
from utils.io import mkdir
from utils.misc import to_cuda, assert_proper_output_dir, gather_loss_dict
from tools.paths_catalog import DatasetCatalog


def train(cfg, args):
    train_set = DatasetCatalog.get(cfg.DATASETS.TRAIN, args)
    val_set = DatasetCatalog.get(cfg.DATASETS.VAL, args)
    train_loader = DataLoader(train_set, cfg.SOLVER.IMS_PER_BATCH,
                              num_workers=cfg.DATALOADER.NUM_WORKERS, shuffle=True)
    val_loader = DataLoader(val_set, cfg.SOLVER.IMS_PER_BATCH,
                            num_workers=cfg.DATALOADER.NUM_WORKERS, shuffle=True)

    gpu_ids = [_ for _ in range(torch.cuda.device_count())]
    model = build_model(cfg)
    model.to("cuda")
    model = torch.nn.parallel.DataParallel(model, gpu_ids) if not args.debug else model

    logger = logging.getLogger("train_logger")
    logger.info("Start training")
    train_metrics = MetricLogger(delimiter="  ")
    max_iter = cfg.SOLVER.MAX_ITER
    output_dir = cfg.OUTPUT_DIR

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    checkpointer = Checkpointer(
        model, optimizer, scheduler, output_dir, logger
    )
    start_iteration = checkpointer.load() if not args.debug else 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    validation_period = cfg.SOLVER.VALIDATION_PERIOD
    summary_writer = SummaryWriter(log_dir=os.path.join(output_dir, "summary"))
    visualizer = train_set.visualizer(cfg.VISUALIZATION)(summary_writer)

    model.train()
    start_training_time = time.time()
    last_batch_time = time.time()

    for iteration, inputs in enumerate(cycle(train_loader), start_iteration):
        data_time = time.time() - last_batch_time
        iteration = iteration + 1
        scheduler.step()

        inputs = to_cuda(inputs)
        outputs = model(inputs)

        loss_dict = gather_loss_dict(outputs)
        loss = loss_dict["loss"]
        train_metrics.update(**loss_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - last_batch_time
        last_batch_time = time.time()
        train_metrics.update(time=batch_time, data=data_time)

        eta_seconds = train_metrics.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(train_metrics.delimiter.join(["eta: {eta}",
                                                      "iter: {iter}",
                                                      "{meters}",
                                                      "lr: {lr:.6f}",
                                                      "max mem: {memory:.0f}"]).format(
                eta=eta_string,
                iter=iteration,
                meters=str(train_metrics),
                lr=optimizer.param_groups[0]["lr"],
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
            summary_writer.add_scalars("train", train_metrics.mean, iteration)

        if iteration % 100 == 0:
            visualizer.visualize(inputs, outputs, iteration)

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration))

        if iteration % validation_period == 0:
            with torch.no_grad():
                val_metrics = MetricLogger(delimiter="  ")
                for i, inputs in enumerate(val_loader):
                    data_time = time.time() - last_batch_time

                    inputs = to_cuda(inputs)
                    outputs = model(inputs)

                    loss_dict = gather_loss_dict(outputs)
                    val_metrics.update(**loss_dict)

                    batch_time = time.time() - last_batch_time
                    last_batch_time = time.time()
                    val_metrics.update(time=batch_time, data=data_time)

                    if i % 20 == 0 or i == cfg.SOLVER.VALIDATION_LIMIT:
                        logger.info(val_metrics.delimiter.join(["VALIDATION",
                                                                "eta: {eta}",
                                                                "iter: {iter}",
                                                                "{meters}"]).format(eta=eta_string,
                                                                                    iter=iteration,
                                                                                    meters=str(val_metrics)))

                    if i == cfg.SOLVER.VALIDATION_LIMIT:
                        summary_writer.add_scalars("val", val_metrics.mean, iteration)
                        break
        if iteration == max_iter:
            break

    checkpointer.save("model_{:07d}".format(max_iter))
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
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

    logger = setup_logger("train_logger", os.path.join(output_dir, "log.txt"))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args)


if __name__ == "__main__":
    main()
