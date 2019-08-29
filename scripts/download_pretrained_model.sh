#!/usr/bin/env bash

mkdir -p output/default_detection
mkdir -p output/default_derender

echo "Downloading pre-trained weights for detection"

URL=http://physadept.csail.mit.edu/assets/ADEPT-detection-0050000.tar.gz
echo ${URL}
curl ${URL} | tar -zx -C output/default_detection
echo 'output/default-detection/ADEPT-detection-0050000.pth' > output/default_detection/last_checkpoint.txt

echo "Downloading pre-trained weights for derender"

URL=http://physadept.csail.mit.edu//assets/ADEPT-derender-0050000.tar.gz
echo ${URL}
curl ${URL} | tar -zx -C output/default_derender
echo 'output/default-derender/ADEPT-detection-0050000.pth' > output/default_derender/last_checkpoint.txt

