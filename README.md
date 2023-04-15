# IBBME_TR300M_Project
A 300 million example transcription regression dataset for supervised learning, and a tensorflow model for it, developed with the support of the Institute of Biomaterials and Biomedical Engineering, and the Faculty of Electrical and Computer Engineering at the University of Toronto. All relevant code included here.

There are two major folders in this Github repo.

The first folder, data, contains all the python scripts which were used to generate the project dataset. These should be considered in detail if one wishes to incorporate additional data or modify the dataset in any way.

The second folder, model, contains the collated, preprocessed dataset along with a single python file called conv_localConnect.py. This file should be considered in detail if one wishes to modify the model architecture, hyperparameters, or anything else non-data related. All the code required to load and train the transcription regression model is contained in this file. 
