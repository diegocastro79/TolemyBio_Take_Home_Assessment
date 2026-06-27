**DESCRIPTION**

`The present project contains the code necessary to compare the performance of a graph-based encoder at generating cell state embeddings, compared with a non-graph baseline.`

**ATTENTION!** 
Python 3.14 may have incompatibilities with the `cellxgene_census` API that can create data loading errors. 
To avoid any issue, it is recommended to install 
Python 3.11.x in the virtual environment of this project.

**REQUIREMENTS**

Run `pip install -r requirements.txt` for installing all required packages.

ISTRUNCTIONS TO RUN THE CODE 

- To load the data, run the script `Data/get_trainable_data.py`. This script loads, processes and saves the relevant data for training as `Data/trainable_data.pkl`.
- OBS! Since the loading and processing of the data takes some time, this separate approach is preferred for NN hyperparameter selection.
- To train the two encoders and compare their performances, run the script `tasks/train_compare_models.py`. 
- Please, note that the saved data.pkl file is retrieved by the relevant functions for the training and comparison, so there is NO need to load the data again for a new training.
- The architectural hyperparameters for both models can be modified within the same script. 
- The training produces validation and training loss plots that are saved in `tasks/plots`.
- The comparison produces displays of confusion matrices and ROC curve plots, both saved in `tasks/plots`.
- Additionally, a statistics summary is printed showing accuracy and F1 scores for both prediction approaches.