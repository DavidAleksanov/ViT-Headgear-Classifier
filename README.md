# ViT-Headgear-Classifier

This repository contains code for training a Vision Transformer (ViT) based classifier on the Headgear Image Classification dataset.

## Project Structure

The project is structured as follows:

ViT-Headgear-Classifier/
│
├── src/
│ ├── model.py # Python script defining the Vision Transformer model architecture
│ └── train.py # Python script for training the Vision Transformer model
│
└── data/
├── headgear-image-classification-metadata.json # Metadata describing the dataset
└── example-data.txt # Sample link to the dataset

- `src/`: Contains the source code for the ViT model (`model.py`) and the training script (`train.py`).
- `data/`: Contains metadata (`headgear-image-classification-metadata.json`) and an example file for the data description link (`example-data.txt`).

## Features

- Utilizes Vision Transformer (ViT) for image classification.
- Data set metadata to obtain detailed information about the data set.
- An example file with a link to the data is also included.

## Prerequisites

Before running the training script, ensure you have the following installed:

- Python 3.x
- Scikit-learn
- Pandas
- Pytorch
- Torchvision
- Pandas

## Installation

1. **Clone the Repository**:
   
git clone https://github.com/DavidAleksanov/ViT-Headgear-Classifier.git
cd ViT-Headgear-Classifier

## Usage

Тraining model, run:
python src/model.py

To train the Vision Transformer model, run:
python src/train.py

