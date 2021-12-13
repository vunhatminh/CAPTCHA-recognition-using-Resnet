# CAPTCHA-recognition-using-Resnet

This is the source code for the project: [**CAPTCHA recognition using neural networks**] *by Minh N. Vu*.

Repo description:

  * captcha_gen.py: To generate the training and testing data for the experiments.
  * captcha_train.py: To train the DNN models for the captcha prediction tasks
  * captcha_resnet.py: The modified resnet models from pytorch library to fit for the captcha prediction task.
  * captcha_loader.py: Utility to load the dataset.
  * captcha_view.ipynb: Notebook to view the results
  
To generate the data, run:

`bash job_generate.sh`

To train the model, run:

`bash job_train.sh`
