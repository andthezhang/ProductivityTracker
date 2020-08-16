# Productivity Tracker

[Productivity Tracker](https://storage.googleapis.com/productivemodel/index.html) is a simple website to help you stay productive in front of a computer. It utilizes machine learning to make real-time productivity level prediction based on web cam input. 

## Getting Started

### Prerequisites

Install `NodeJS` and `Python` in your system.

## Code Structure

`./backend`: Code to train a model offline.

`./static_website`: Code for a simple website demo.

`./data `: Directory to download datasets.

./generate_labels.py: Python script to generate labels from datasets.

## How-To Guide

### Generate label on dataset.

We define productivty by labelling some certain range in [valence and arousal](https://en.wikipedia.org/wiki/Emotional_granularity#Valence_and_arousal) as non-productive plus manual annotation on top. Therefore we utlize public datasets with existing valence and arousal annotations. See readmes in data folder for detail.

To generate non-productive frames, run `python generate_labels.py`.

### Generate npy from dataset by NodeJS.
To make use of existing tfjs models ([MobileNet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet) and face detector [BlazeFace](https://github.com/tensorflow/tfjs-models/tree/master/blazeface). We use tfjs in NodeJS to extract feature from datasets and save them as npy. Then we train a model with regular Keras by taking saved npy as input.
`cd backend`

`npm install`

`node generate_npy.js`

### Train Model in Keras and Convert it to tfjis model

`cd backend` + `python train.py`

Convert to tfjs model: 

`tensorflowjs_converter --input_format=tf_saved_model --output_format tfjs_graph_model backend/saved_models/best_model /saved_models/model_tfjs` 

### Static Website

Static Website is located at `static_website`, deployed in [Productivity Tracker](https://storage.googleapis.com/productivemodel/index.html).



## Authors

[Zhiheng (Andy) Zhang](https://www.linkedin.com/in/andyzzh/)

Nashila Jahan