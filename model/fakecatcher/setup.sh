#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 전달받은 인자 확인
if [ $# -ne 1 ]; then
    echo "Usage: $0 <server_number>"
    exit 1
fi

SERVER_NUMBER=$1
# Step 1: Install Miniconda (if not already installed)
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

# Step 2: Set up Conda environment
echo "Setting up Conda environment..."
conda create -n video python==3.11 -y || true
conda activate video

# Step 3: Install dependencies
echo "Installing dependencies..."
conda install conda-forge::libgl -y
conda install -c conda-forge ffmpeg -y
cd /root/25th-conference-fakebusters/model/fakecatcher
pip install -r requirements.txt
python -m pip install mediapipe
cd

# Step 4: Install zip files
echo "Installing zip files..."
sudo apt-get update
sudo apt-get install -y zip

cd /root/25th-conference-fakebusters/model/fakecatcher/data

# Create directories if they don't exist
mkdir -p manipulated_sequences original_sequences

# Define zip file details
declare -A file_details=(
    ["manipulated_sequences_1.zip"]="13cGlSlgv6-85hpEim0n9LwFHapK5vr9d"
)

# Check and download zip files if not already present
for zip_file in "${!file_details[@]}"; do
    if [ ! -f "$zip_file" ]; then
        echo "Downloading $zip_file..."
        gdown "${file_details[$zip_file]}" -O "$zip_file"
    else
        echo "$zip_file already exists. Skipping download."
    fi
done

# Extract and move files
if [ -f "manipulated_sequences_1.zip" ]; then
    unzip -o manipulated_sequences_1.zip -d manipulated_sequences/
fi
cd

python /root/25th-conference-fakebusters/model/fakecatcher/data/fakeforensics.py -b /root/25th-conference-fakebusters/model/fakecatcher/data -s $SERVER_NUMBER
cd

export PYTHONPATH=/root/25th-conference-fakebusters/model/fakecatcher

cd /root/25th-conference-fakebusters/model/fakecatcher
#python model/fakecatcher/cnn/preprocess_map.py -c model/fakecatcher/utils/config.yaml -l model/fakecatcher/data/ppg_map.log -o model/fakecatcher/data

echo "Setup completed successfully!"
