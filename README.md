# DeformableCNNLettuceRegressor

As described in the paper **[Simultaneously Predicting Multiple Plant Traits from Multiple Sensors via Deformable CNN Regression
](https://arxiv.org/pdf/2112.03205.pdf)**, we find that deformable convolution-based regression achieves state-of-the-art performance on the [autonomous greenhouse dataset](https://data.4tu.nl/articles/dataset/3rd_Autonomous_Greenhouse_Challenge_Online_Challenge_Lettuce_Images/15023088#!). The pipeline provided allows users to train their own models on the autonomous greenhouse dataset or a custom dataset.

## Training

Either use `train.py` or run the training cells in the `example.ipynb` notebook. Change hyperparameters and training options accordingly.

## Evaluation

Either use eval.py or run the evaluation cells in the example notebook. Keep in mind these cells require GT in order to quntify error. For deployment, load the image, load the model, pass the image through the model, and the output will be the plant traits.

## Acknowledgments

This project was partly funded by the [National AI Institute for Food Systems (AIFS)](www.aifs.ucdavis.edu).
