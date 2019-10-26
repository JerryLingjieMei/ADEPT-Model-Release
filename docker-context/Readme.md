# Prerequisites

- GPU
- Unix based system
- Docker

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

```sh
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && git checkout 192261db14d596da52905e91dc608bd4315552dc \
 && python setup.py build develop
```
<!-- # && git checkout c5ca36fc644dfc1d3dd4ad15739bf6bb4df72d72  \ #Jerry's one -->

# Download Pretrained Weights

inside  the  containers bash
```sh
cd /root/adept-model/;
./scripts/download_pretrained_model.sh
```
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
```