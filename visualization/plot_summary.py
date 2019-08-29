import argparse
import os
from multiprocessing import Process, cpu_count
from multiprocessing.pool import ThreadPool

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils.io import mkdir, read_serialized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", type=str, required=True)
    parser.add_argument("--ann_folder", type=str, required=True)
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def plot_case(images, all_scores, raw_scores, locations, derender_objects, gt_objects, case_name,
              output_folder):
    fig, (ax1, ax3, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(4.5, 10))
    line, = ax2.plot([], [], "k")
    mkdir(os.path.join(output_folder, "imgs"))
    for i, (image, raw_score, xs, ys, derender_object, gt_object) in enumerate(
            zip(images, raw_scores, locations[0], locations[1], derender_objects, gt_objects), 1):
        ax1.imshow(image)
        ax1.axis('off')

        ax2.clear()
        line.set_xdata(range(i))
        line.set_ydata(all_scores[:i])
        ax2.plot(range(i), all_scores[:i])
        ax2.axvline(x=i, color="r", linestyle='--')
        plt.draw()

        perturbed_score = []
        for score in raw_score:
            perturbed_score.append(score + np.random.rand() * .001)
        bp = ax2.boxplot(perturbed_score, positions=[i], showfliers=False, showcaps=False, whis=[25, 75])
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color="#1f77b4")

        ax2.set_xlim(0, len(images))
        ax2.set_ylim(0, 12)
        ax2.get_xaxis().set_ticklabels([])
        ax2.axes.get_yaxis().set_ticklabels([])

        ax3.clear()
        ax3.scatter(ys, [-x for x in xs], 40, alpha=.2)

        derender_xs = [obj["location"][1] for obj in derender_object["objects"]]
        derender_ys = [-obj["location"][0] for obj in derender_object["objects"]]
        ax3.scatter(derender_xs, derender_ys, 10)

        if gt_object is not None:
            gt_xs = [obj["location"][1] for obj in gt_object["objects"]]
            gt_ys = [-obj["location"][0] for obj in gt_object["objects"]]
            ax3.scatter(gt_xs, gt_ys, 10)

        ax3.set_xlim(-4, 4)
        ax3.set_ylim(-1., 2.5)

        ax3.get_xaxis().set_ticklabels([])
        ax3.get_yaxis().set_ticklabels([])
        fig.savefig("{}/imgs/{}_{:03d}.png".format(output_folder, case_name, i))
        print("{}/imgs/{}_{:03d}.png generated".format(output_folder, case_name, i))
    fig.savefig("{}/{}_score.png".format(output_folder, case_name))


def main(case_name, summary, output_path):
    anns = read_serialized(os.path.join(args.ann_folder, case_name, "{}_ann.yaml".format(case_name)))
    images_files = [ann["image_path"] for ann in anns["scene"]][1:]
    with ThreadPool(cpu_count() * 4) as p:
        images = p.map(Image.open, images_files)
    plot_case(images, summary["all"], summary["raw"], summary["location"], case_name, output_path)

    print("{} generated".format(case_name))


if __name__ == '__main__':
    args = parse_args()
    category = os.path.basename(os.path.split(args.summary_path)[-1])
    data = read_serialized(os.path.join(args.summary_path, "result.json"))

    if args.output_path:
        output_path = args.output_path
    else:
        output_path = "output/{}".format(category)

    processes = []
    for case_name, summary in data.items():
        p = Process(target=main, args=(case_name, summary, output_path))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
