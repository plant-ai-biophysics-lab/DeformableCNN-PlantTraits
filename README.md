# DeformableCNNLettuceRegressor

 As described in the paper **[Simultaneously Predicting Multiple Plant Traits from Multiple Sensors via Deformable CNN Regression
](https://arxiv.org/pdf/2112.03205.pdf)**, we find that deformble achieive higher performance on the autonomous greenhouse dataset. The pipeline provided allows users to train their own models on the autonomous greenhouse or a custom data set.

## Training

Either use train.py or run the training cells in the example notebook. Change hyperparamters and training options accordingly

## Evaluation

Either use eval.py or run the evaluation cells in the example notebook. Keep in mind these cells require GT in order to quntify error. For deployment, load up the image, load up the model, pass the image through the model and the output will be the plant traits




