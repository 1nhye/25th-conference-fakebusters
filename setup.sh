#!/bin/bash

if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    chmod +x ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    echo "Initializing Conda..."
    ~/miniconda3/bin/conda init bash
fi

# Reload shell for Conda initialization
echo "Reloading shell..."
source ~/miniconda3/etc/profile.d/conda.sh

mode="all"
config="config.yaml"

# Parse command-line options
while getopts "m:c:" opt; do
  case $opt in
    m)
      mode=$OPTARG
      ;;
    c)
      config=$OPTARG
      ;;
    *)
      echo "Usage: $0 [-m mode] [-c config]"
      exit 1
      ;;
  esac
done

# Create Conda environment
echo "Activating Conda environment for: $mode"

if [ "$mode" == "lipforensics" ]; then
    LIP_PATH=$(grep 'lipforensics-path:' $config | cut -d ':' -f 2- | tr -d ' ')
    conda env create -f $LIP_PATH -n lipforensics
    conda activate lipforensics
    pip install mediapipe

elif [ "$mode" == "mmdet" ]; then
    MMDET_PATH=$(grep 'mmdet-path:' $config | cut -d ':' -f 2- | tr -d ' ')
    conda env create -f $MMDET_PATH -n mmdet
    conda activate mmdet

    dir_path=$(dirname "$MMDET_PATH")
    cd $dir_path/LLaVA
    pip install -e . 

elif [ "$mode" == "fakecatcher" ]; then
    FAKE_PATH=$(grep 'fakecatcher-path:' $config | cut -d ':' -f 2- | tr -d ' ')
    sudo apt-get install -y libgl1-mesa-glx 
    sudo apt-get install -y libglx-mesa0 
    sudo apt-get install -y libgl1
    conda env create -f $FAKE_PATH -n fakecatcher
    conda activate fakecatcher

elif [ "$mode" == "backend" ]; then
    BACK_PATH=$(grep 'backend-path:' $config | cut -d ':' -f 2- | tr -d ' ')
    conda env create -f $BACK_PATH -n backend
    conda activate backend

elif [ "$mode" == "frontend" ]; then
    FRONT_PATH=$(grep 'frontend-path:' $config | cut -d ':' -f 2- | tr -d ' ')
    sudo apt install -y npm
    cd $FRONT_PATH
    npm install
    npm un dev

elif [ "$mode" == "all" ]; then
    echo "will be implemented"

else
    echo "Invalid mode: $mode"
    exit 1
fi