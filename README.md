# Data Classification using ANNs

This repository implements basic anns like SVMs (using two different kernels) and MLP to classify between living and dead skin.

## Dataset
There are two sets of data used for testing/learning. Dead skin data can range from leather to plastics, whereas live skin data is obtained from readings involving human subjects. The data is obtained for different wavelengths, in the range 400 nm to 1600 nm, for different products/test subjects.

## Workflow
* The data was loaded into python in the form of pandas dataframes.
* After augmenting the data by appending the relevant class labels, the data was setup for extracting valuable features for learning the neural networks.
* Feature extraction was based on SelectKBest from sklearn using f_classif, an ANOVA based F-value between label/feature for classification tasks.
* After the preprocessing step, the data is ready to be learned through the neural networks.

## Dependencies
python 3.5
pandas
numpy
matplotlib (for visualisation)
keras
tensorflow
