# DeformableCNN-PlantTraits

Trait measurement is critical for the plant breeding and agricultural production pipeline. Typically, a suite of plant traits is measured using laborious manual measurements and then used to train and/or validate higher throughput trait estimation techniques. Here, we introduce a relatively simple convolutional neural network (CNN) model that accepts multiple sensor inputs and predicts multiple continuous trait outputs – i.e. a multi-input, multi-output CNN (MIMO-CNN). Further, we introduce deformable convolutional layers into this network architecture (MIMO-DCNN) to enable the model to adaptively adjust its receptive field, model complex variable geometric transformations in the data, and fine-tune the continuous trait outputs. We examine how the MIMO-CNN and MIMO-DCNN models perform on a multi-input (i.e. RGB + depth images), multi-trait output lettuce [dataset from the 2021 Autonomous Greenhouse Challenge](https://data.4tu.nl/articles/dataset/3rd_Autonomous_Greenhouse_Challenge_Online_Challenge_Lettuce_Images/15023088#!). Ablation studies were conducted to examine the effect of using single versus multiple inputs, and single versus multiple outputs. The MIMO-DCNN model resulted in a normalized mean squared error (NMSE) of 0.068; a substantial improvement over the top 2021 leaderboard score of 0.081.

See accepted [2022 AAAI AI for Ag and Food Systems workshop](https://aiafs-aaai2022.github.io/) paper here **[Simultaneously Predicting Multiple Plant Traits from Multiple Sensors via Deformable CNN Regression](https://arxiv.org/pdf/2112.03205.pdf)**.

## Training

Either use `train.py` or run the training cells in the [example notebook](https://colab.research.google.com/github/plant-ai-biophysics-lab/DeformableCNN-PlantTraits/blob/main/example.ipynb). Change hyperparameters and training options accordingly.

## Evaluation

Either use `eval.py` or run the evaluation cells in the [example notebook](https://colab.research.google.com/github/plant-ai-biophysics-lab/DeformableCNN-PlantTraits/blob/main/example.ipynb). Keep in mind that these cells require ground-truth values in order to quntify error. For deployment, load the image, load the model, pass the image through the model, and the model will output continuous plant traits values.

## Acknowledgments

This project was partly funded by the [National AI Institute for Food Systems (AIFS)](https://aifs.ucdavis.edu).
