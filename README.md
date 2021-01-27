# HMMPhysionet
Includes implementation of applying a Gaussian HMM model on Physionet2012 data to predict Length of stay of ICU patients
The paper written based on this implementation can be found here: 
http://manisci.github.io/files/pred_length.pdf

You'll need the following dependenciet to be able to run these scripts, you can install most of these using pip:
Python 2.7, scipy, hmmlearn, numpy, sklearn,statsmodel


It is assumed that you've already obtained the dataset file which can be downloaded here:
https://physionet.org/challenge/2012/

Once you have the data, and all the data in train folder, put outcome file and the files in this repository into a folder called "HMMphys" to be able to run these scripts without much change. 

This repository includes the following files:

sing.py : running the HMM model on all patients(ICU type 5 ) or single ICU units (1,2,3,4)

eighttwelve.py : running the HMM model on the combined model of training ICU types (1,2) together and (3,4) together too. 

apache.txt, mpm.txt, recid.txt,saps.txt, sofa.txt : Output scores used solely for performance comparison with HMM model. 

plt.R : R script to generate the circle plots showing count and average length of stay of various start end state pairs.

ttest.R: R script to output ttest values for a given result file containing RMSE values across the different runs for both the baseline model and the HMM model.

Feel free to use these files in your research , but kindly cite us if you end up using these:
Mani Sotoodeh, Joyce Ho. (2019). "Improving length of stay prediction using a hidden Markov model." AMIA 2019 Summit

Should you have any questions contact me at: mani.sotoodeh--at--gmail.com

## a simple demo 
1. export to los_hmm_cohort.csv and los_hmm_dataset.csv
2. python3 knn-hmm-lasso-on-aggregated-dataset.py
