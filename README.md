# Multi-resBind
Source code for the paper of "Multi-resBind: a residual network-based multi-label classifier for \emph{in vivo} RNA binding prediction and preference visualization"
# Requirements
The code was developed under Python 3.7.3 in macOS Mojave version 10.14.5 system.  
To install all relevant packages by running the following command：  
`pip install -r requirements.txt`
# Data
The original dataset for performance comparison between Multi-resBind and DeepRiPe can be downloaded from: <br />
>https://ohlerlab.mdc-berlin.de/software/DeepRiPe_140/. <br />
- data_RBPslow.h5: with peaks of <15,000. <br />
- data_RBPsmed.h5: with peaks of >15,000 but <100,000.  <br />
- data_RBPshigh.h5: with peaks of >100,000. <br />

In each category, 70% of the data consisting of RNA sequences and region types were used for training the model, and 20% and 10% of the data were treated as validation and test datasets, respectively.  

Two low datasets consist of RNA secondary structural profiles predicted by computation tools can be downloaded from: <br />
>https://doi.org/10.5281/zenodo.4743499.<br />

- data_RBPslow_RNAplfold.h5: Structural profiles with paired-unpaired annotations were calculated using a modified script of RNAplfold.
- data_RBPslow_CapR.h5: Structural profiles with a stem (S), hairpin (H), bulge (B), internal (I), multibranch (M), or exterior (E) loop were calculated using CapR.

# Usage
## Training
Train and evaluate the model with the commands:  
`python3 main.py`
# Model
The pretrined models to reproduce the Figures and tables of paper can be found in the Model dirctory.
