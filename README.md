# HMMPhysionet
Includes implementation of applying a Gaussian HMM model on Physionet2012 data to predict Length of stay of ICU patients
The paper written based on this implementation can be found here: 
http://manisci.github.io/files/pred_length.pdf

You'll need the following dependenciet to be able to run these scripts, you can install most of these using pip:
Python 2.7, scipy, hmmlearn, numpy, sklearn,statsmodel


It is assumed that you've already obtained the dataset file which can be downloaded here:
https://physionet.org/challenge/2012/

Once you have the data, and all the data in training, put output file and the files in this repository into a folder called "HMMphys" to 
This repositoryincludes the following files:

sing.py : running the HMM model on all patients(ICU type 5 ) or single ICU units (1,2,3,4)
eightwelve.py : running the HMM model on the combined model of training ICU types (1,2) together and (3,4) together too. 
apache.txt, mpm.txt, recid.txt,saps.txt, sofa.txt : Output scores used solely for performance comparison with HMM model. 
plt.R : R script to generate the circle plots showing count and average length of stay of various start end state pairs. 
ttest.R: R script to output ttest values for a given result file containing RMSE values across the different runs for both the baseline model and the HMM model.
