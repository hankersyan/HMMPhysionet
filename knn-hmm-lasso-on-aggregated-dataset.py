import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from collections import Counter
import numpy as np
import math
import random
import sys
import time
from hmmlearn import hmm

COL_ID = 'icustay_id'
COL_TIME = 'chart_time'
FEATURES = ["HR", "WBC", "TEMP", "GCS", "Glucose", "NIDBP" , "Urine"]
FEATURES_FOR_KNN = ['gender','admission_age']
COLUMNS = [COL_ID, COL_TIME] + FEATURES_FOR_KNN + FEATURES

def main():
    start_tm = time.time()
    labvitals = pd.read_csv('./los_hmm_dataset.csv')
    labvitals = labvitals.sort_values(by=[COL_ID, COL_TIME])
    #print(labvitals)
    patients = pd.read_csv('./los_hmm_cohort.csv')
    #print(patients)

    # remove non-intersection
    old_num_pat = patients.shape[0]
    old_num_lv = labvitals.shape[0]
    patients = patients[patients.icustay_id.isin(labvitals.icustay_id)]
    labvitals = labvitals[labvitals.icustay_id.isin(patients.icustay_id)]
    print('Removing non-intersection', old_num_pat-patients.shape[0], patients.shape[0], old_num_lv - labvitals.shape[0], labvitals.shape[0])

    labvitals = makeupFirstRowForEachPatient(patients, labvitals)
    #print(labvitals)

    labvitals = imputeValues(labvitals)
    print(labvitals)
    
    if patients.shape[0] * 6 != labvitals.shape[0]:
        print('ERROR after imputation')
        sys.exit()

    feat_vals = labvitals[FEATURES]
    covtype = 'diag'  # diag tied spherical
    algorithm = 'viterbi' # viterbi map
    num_state = 8
    trainlengths = None
    hmmmodel = hmm.GaussianHMM(n_iter=10,algorithm=algorithm,n_components=int(num_state),covariance_type=covtype).fit(feat_vals)
    score = hmmmodel.score(feat_vals, lengths=trainlengths)
    print(score)

    probs = hmmmodel.predict_proba(feat_vals)
    probs_by_pat = np.reshape(probs, (patients.shape[0], 6, num_state))

    hidden_feats = np.zeros((patients.shape[0], 2 * num_state))
    for i in range(patients.shape[0]):
        hidden_feats[i,0:num_state] = probs_by_pat[i,0,:]
        hidden_feats[i,num_state:2 * num_state] = probs_by_pat[i,-1,:]

    Linearregr(hidden_feats, patients['length_of_stay'])
    print(time.time() - start_tm, 'seconds')

def Linearregr(x, y):
    scaler = sklearn.preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    x = pd.DataFrame(x)

    train_idx = random.sample(range(y.shape[0]), int(y.shape[0] * 0.8))
    train_idx.sort()
    valid_idx = list(set(list(range(y.shape[0]))) - set(train_idx))
    valid_idx.sort()

    #train_idx_x = [n for n in range(y.shape[0]*6) if int(n/6) in train_idx]
    #valid_idx_x = list(set(list(range(y.shape[0]*6))) - set(train_idx_x))

    train_x = x.iloc[train_idx].values
    valid_x = x.iloc[valid_idx].values
    train_y = y.iloc[train_idx].values
    valid_y = y.iloc[valid_idx].values
    
    bestalpha = lrcrossvalidation(train_x, train_y, 'Lasso')
    testmodely = doLRandreport(train_x, train_y, valid_x, valid_y, bestalpha, 'Lasso')

def lrcrossvalidation(trainingx, trainingy, method):
    '''
    finds and returns the best alpha or regularization parameter for the given mode based on the lowest mse score
    '''
    alphs = np.linspace(math.pow(2,-5),math.pow(2,3),num=140) 
    hypscores = []
    for alph in alphs:
        if method == 'Ridge':
            model = linear_model.Ridge(alpha = alph, fit_intercept=True,max_iter =10000)
        if method == 'Lasso':
            model = linear_model.Lasso(alpha = alph, fit_intercept=True,max_iter =10000)
        scores = cross_val_score(model, trainingx, trainingy, scoring="neg_mean_absolute_error", cv=10)
        mse_scores = (-scores)
        hypscores.append(mse_scores.mean())
    bestalpha = alphs[hypscores.index((min(hypscores)))]
    return bestalpha

def doLRandreport(trainingx, trainingy, testx, testy, bestalpha, method):
    '''
    Does the actual regression based ont he best provided parameters and returns predictions
    '''
    if method == 'Ridge':
        realmodel = linear_model.Ridge(alpha = bestalpha, max_iter = 10000,fit_intercept=True)
    if method == 'Lasso':
        realmodel = linear_model.Lasso(alpha = bestalpha, max_iter = 10000,fit_intercept=True)
    # trainingx n*array[16], trainingy n*array[1]
    realmodel.fit(trainingx, trainingy)
    testmodely = realmodel.predict(testx)
    testmodely = list(map(int, testmodely))
    print(list(zip(testy, testmodely)))
    #for i in range(len(testy)):
    #    testmodely[i] = (testmodely[i])
    #    testy[i] = (testy[i])
    diff = []
    for i in range(len(testy)):
        diff.append(abs(testmodely[i] - testy[i]))
        bestpatidx = diff.index(min(diff))
        worstpatidx = diff.index(max(diff))
    coefs = realmodel.coef_
    besty = np.inner(testx[bestpatidx,:],coefs)
    realbesty = testy[bestpatidx]
    worsty = np.inner(testx[worstpatidx,:],coefs)
    realworsty = testy[worstpatidx]
    print('besty',besty,'realbesty',realbesty,'worsty',worsty,'realworsty',realworsty)
    return testmodely

def makeupFirstRowForEachPatient(patients, labvitals):
    selections = []
    prevId = -1
    for i in range(labvitals.shape[0]):
        if prevId != labvitals.iloc[i][COL_ID]:
            selections.append(i)
        prevId = labvitals.iloc[i][COL_ID]

    join_df = pd.merge(patients, labvitals.iloc[selections], on=COL_ID, how='inner')
    join_df = join_df[COLUMNS]

    for i in range(len(FEATURES)):
        feat = FEATURES[i]
        rows_with_values = join_df.loc[join_df[feat].notnull(), COLUMNS]
        rows_without_values = join_df.loc[~join_df[feat].notnull(), COLUMNS]
        KNNtrainX = rows_with_values[FEATURES_FOR_KNN].values
        KNNtrainY = rows_with_values[[feat]].values
        neightrain = KNeighborsRegressor(n_neighbors=5)
        neightrain.fit(KNNtrainX, KNNtrainY)
        rows_without_values[feat] = neightrain.predict(rows_without_values[FEATURES_FOR_KNN].values)
        join_df.loc[~join_df[feat].notnull(), [feat]] = rows_without_values[feat]
        #print(rows_with_values)
        #print(rows_without_values)

    join_df = join_df.sort_values(by=[COL_ID])
    #print(join_df)

    labvitals.set_index([COL_ID, COL_TIME], inplace=True)
    cols = [COL_ID, COL_TIME] + FEATURES
    labvitals.update(join_df[cols].set_index([COL_ID, COL_TIME]))
    labvitals.reset_index(drop=False, inplace=True)
    #print(labvitals.iloc[selections])
    return labvitals

def imputeValues(labvitals):
    cols = [COL_ID, COL_TIME] + FEATURES
    ndf = pd.DataFrame(columns=cols)
    for i in range(labvitals.shape[0]):
        vid = int(labvitals.iloc[i][COL_ID])
        vtime = int(labvitals.iloc[i][COL_TIME])
        if vtime>5:
            print('ERROR on imputing, vtime>5', vid, vtime)
            sys.exit()
        if i<labvitals.shape[0]-1:
            ntime = int(labvitals.iloc[i+1][COL_TIME])
            nid = int(labvitals.iloc[i+1][COL_ID])
        else:
            nid = -1
            ntime = -1
        if vid != nid:
            if vtime < 5:
                for n in range(vtime+1, 5+1):
                    if n>5:
                        print('ERROR on imputing, n>5', vid, n)
                        sys.exit()
                    ndf = ndf.append({COL_ID: vid, COL_TIME: n}, ignore_index=True)
            if nid > 0 and ntime > 0:
                for n in range(ntime):
                    if n>5:
                        print('ERROR- on imputing, n>5', vid, n)
                        sys.exit()
                    ndf = ndf.append({COL_ID: nid, COL_TIME: n}, ignore_index=True)
        else:
            if ntime-vtime>1:
                for n in range(vtime+1, ntime):
                    if n>5:
                        print('ERROR= on imputing, n>5', vid, n)
                        sys.exit()
                    ndf = ndf.append({COL_ID: vid, COL_TIME: n}, ignore_index=True)
    
    merged = pd.concat([labvitals, ndf], ignore_index=True)
    merged = merged.sort_values(by=[COL_ID, COL_TIME])

    labvitals = pushforward(merged)

    for i in range(labvitals.shape[0]):
        if int(labvitals.iloc[i][COL_TIME]) != int(i%6):
            print(i, '\n', labvitals.iloc[i])
            break
    return labvitals

def pushforward(inputhmm):
    for i in range(inputhmm.shape[0]):
        for feat in FEATURES:
            val = inputhmm.iloc[i][feat]
            if val == float(0) or math.isnan(val):
                inputhmm.iloc[i][feat] = inputhmm.iloc[i-1][feat]
    return inputhmm

main()
