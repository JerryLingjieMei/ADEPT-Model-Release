# Prerequisites

- GPU (2 tesla p100 16gb)
- Unix based system (Ubuntu 16.04.6 LTS)
- Docker (current version 19.03.1)

# Build the docker container

```sh
sudo docker build -t adept-model:latest .
```

# Run the container

this  will mount the root directory of this repo inside the container and start an instance

```sh
sudo docker run -it -v $(dirname "$(pwd)"):/root/adept-model --gpus all adept-model:latest
```

# Finish Instalation

The GPU is not  visible during the  container building so it's necessary to compile maskRCNN after starting the container.

```sh
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && git checkout 192261db14d596da52905e91dc608bd4315552dc \
 && python setup.py build develop
```
<!-- # && git checkout c5ca36fc644dfc1d3dd4ad15739bf6bb4df72d72  \ #Jerry's one -->

# Run the pipeline  with pre-trained weights on a test set


## 1.1 Download Pretrained Weights

inside  the  containers bash:
```sh
cd /root/adept-model/;
./scripts/download_pretrained_model.sh
```

## 1.2 set the directory  for the different sets, here we use an included sample

```sh
export TEST_SET='/root/adept-model/data_sample/human'
export TRAIN_SET='/root/adept-model/data_sample/train'
```

set `TRAIN_ROOT` and `HUMAN_ROOT` in `tools/paths_catalog.py` to `${TRAIN_SET}` and `${TEST_SET}` respectively

set `ANNOTATION_FOLDER` in `experiments/default_physics.yaml` to `${TEST_SET}`

## 1.3 create a json file with the format with coco annotations

for the test set
```sh
mkdir data; #there should not be a data folder when starting fresh
python -m dataset.makers.make_coco -i ${TEST_SET} -o data/human_ann.json
```

same for the training set:
```sh
python -m dataset.makers.make_coco -i ${TRAIN_SET} -o data/ann.json
```

## 1.4 use the downloaded maskRCNN on the test data to create segmentation maps

```sh
python -m tools.detection_test_net --config_file experiments/default_detection.yaml
```

## 1.5 make object proposals placeholder files on the test set from the segmentation maps

```sh
python -m dataset.makers.make_proposal -i ${TEST_SET} -o data/annotated_human_ann.json -s output/default_detection/inference/physics_human/segm.json
```

## 1.6 use the derenderer to create object proposals (physics engine input) for the test set

```sh
python -m tools.test_net --config_file experiments/default_derender.yaml
```

## 1.7 run the physics reasoning model on the test set

```sh
python -m tools.run_physics --config_file experiments/default_physics.yaml
```
#TODO: include training and multiple gpu optional commands
<!-- 
## 1.3 
# TODO
figure  out how to put the data and set it up elegantly

get annotations for the  test data:

```sh
python -m dataset.makers.make_coco -i /root/adept-model/data_sample/human_sample -o data/human_ann.json
```

train on single GPU
```sh
python -m tools.detection_train_net --config_file experiments/default_detection.yaml
```
test and create segmentation  maps
```sh
python -m tools.detection_test_net --config_file experiments/default_detection.yaml
```

Prepare data to create object proposals
```sh
python -m dataset.makers.make_proposal -i /root/adept-model/data_sample/human_sample -o data/annotated_human_ann.json -s output/default_detection/inference/physics_human/segm.json
```
test the derenderer to create object proposals in the human test set
```sh
python -m tools.test_net --config_file experiments/default_derender.yaml
``

multiple gpu
```sh
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools.detection_train_net --config_file experiments/default_detection.yaml
```

```sh
python -m tools.run_physics --config_file experiments/default_physics.yaml
``` -->