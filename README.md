
# Paper Repository
This repository contains the source code for the paper 
**Dynamic and Context Dependent Stock Price Prediction Using Attention Modules and News Sentiment**

Please refer to SETUP.md for instructions for installing a virtual environment for the notebooks. The virtual environment ensures that 
the python package dependencies are consistent with those used by the author. 

# Version 1.0
The current version has been tested on MacOS, Windows 10 and Linux. See SETUP.md for further details. 


# Running the code
For simplicity, this repo only contents the notebooks for the `AMZN` stock, but in each notebook the ticker can be changed to be either `TMO` or `BF_B`. The `_conv_` part of the notebook name means these are the notebooks for the bi-variante evalution, meaning the data contains the news sentiment and the historical prices. Hence `_price_` means this notebook only uses the historical pricing data. 

Moreover, the `n_steps_5-n_ahead_5` means the notebook use as input 5 days and predict 5 days ahead. However, this can be customized to what you want. Just note that in the `crossval-folder` there is only data for `5, 10 and 21 days`. So you will have to train the networks from scratch if you wish to look at other intervals. 

Also, you can select in the notebooks if you want to run just the notebooks or if you want to do the cross validation or even train the networks from scratch. 

The `Compute_params_list_RIM.ipynb` is in the repo for your convenience, so you can even customize the tuning of the alpha_t-RIMs hyperparameter, since using all the hyperparamters of the network would take to long.

__Note: For training the networks you should have access to a GPU.__ This is not required if you want to run the notebooks and use the cross-validation data available to you via this repo. 

***

# License

This code is under a CC-NC license, which means the followin:

## You are free to:
- Share — copy and redistribute the material in any medium or format

- Adapt — remix, transform, and build upon the material

## Under the following terms:

- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

- NonCommercial — You may not use the material for commercial purposes.

- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.


## Note:

No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.


## Cite
Cite as: 
@misc{nicolepcx,
author = {Nicole Königstein},
title = {Dynamic and Context-Dependent Stock Price Prediction Using Attention Modules and News Sentiment},
year = {2021},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/Nicolepcx/alpha t-RIM}}
}
