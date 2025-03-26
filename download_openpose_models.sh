#!/bin/bash

# Directory structure
BASE_DIR="./openpose_models"
MODEL_DIR="$BASE_DIR/pose/coco"

# Create directories
mkdir -p "$MODEL_DIR"

echo "Downloading OpenPose model files..."

# Download the prototxt file (small file)
echo "Downloading pose_deploy_linevec.prototxt..."
wget -c "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt" -O "$MODEL_DIR/pose_deploy_linevec.prototxt"

# Download the model file (large file)
echo "Downloading pose_iter_440000.caffemodel (this might take a while)..."
wget -c "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel" -O "$MODEL_DIR/pose_iter_440000.caffemodel"

echo "Download complete. Model files saved to $MODEL_DIR" 