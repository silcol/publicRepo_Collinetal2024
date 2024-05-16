#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import csv
import numpy as np
from numpy import array
from scipy.sparse import csr_matrix
from ast import literal_eval
from collections import Counter
from scipy import sparse
import glob
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from scipy.sparse import vstack
from scipy.sparse import hstack
import gc
import matplotlib.pyplot as plt
from scipy import stats
from itertools import chain
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit


# settings

output_dir = '/jukebox/norman/scollin/schema/data/bids/Norman/Silvy/schema/derivatives/rsa/'
onsettimes_dir = '/jukebox/norman/scollin/schema/data/bids/Norman/Silvy/schema/derivatives/extractedWeddingTRs_MNIspace/'

AllSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]

WhichWeddNrs = [1,2,6,17,19,20,22,23,28,29,34,38]

ROI_name = 'REVISEDpath_pVals_thresh_fdr0.05_inverted'


timeseries_dir = '/jukebox/norman/scollin/schema/data/bids/Norman/Silvy/schema/derivatives/extractedWeddingTRs_MNIspace/'
TRlength = 74
TRlimit = 80
NrWedds = 12
Sample_Length = TRlength * NrWedds
EndIntro = 17
EndEvent1 = 23
EndEvent2 = 35
EndEvent3 = 50
EndEvent4 = 66

## load pickle with matrix subj x wedd, which path is each wedding for each subj, ascending order of weddings based on index
directory = '/jukebox/norman/scollin/schema/data/bids/Norman/Silvy/schema/derivatives/resultsHMM/'
dfpathAB_transposed = pd.read_pickle(directory + 'weddOverview_AscendingOrderOfWeddings.pkl')
newpathAB = dfpathAB_transposed.drop(dfpathAB_transposed.index[0])
newpathAB.index = range(len(newpathAB))

# make sure there is ascending order of wedding timeseries to be loaded later on
FileOrderCorrected = [0,3,11,1,2,4,5,6,7,8,9,10]

sameEv_otherSchema_group = pd.DataFrame(columns=[])
otherEv_otherSchema_group = pd.DataFrame(columns=[])
otherEv_sameSchema_group = pd.DataFrame(columns=[])
sameEv_sameSchema_group = pd.DataFrame(columns=[])
        
# loop over subjs
for CurrSub in range(len(AllSubjects)):
    
    TotalDataAllWedds = pd.DataFrame(columns=[0])
    
    sameEv_otherSchema_all = pd.DataFrame(columns=[])
    otherEv_otherSchema_all = pd.DataFrame(columns=[])
    otherEv_sameSchema_all = pd.DataFrame(columns=[])
    sameEv_sameSchema_all = pd.DataFrame(columns=[])
    
    print('run subject',AllSubjects[CurrSub])
    
    # what is A and what is B path for this subj
    currRow = newpathAB.loc[CurrSub]
    listCurrRow = currRow.values.tolist()
    nplistCurrRow = np.array(listCurrRow)
    
    # load files with wedding timeseries
    path = timeseries_dir + str(AllSubjects[CurrSub]) + '/' + ROI_name + '/'
    files = [f for f in glob.glob(path + "*videos1_%s_*.npy" %(ROI_name), recursive=True)]
    
    # sort files into sensible order (same across subjects)
    files = sorted(files)
        
    counter = 0
    counterNA = 0
    counterNB = 0
    counterSA = 0
    counterSB = 0

    # loop over weddings
    for currWedd in FileOrderCorrected:

        timeseries = np.load(files[currWedd])
        print(files[currWedd])

        # make sure to remove TRs where people are answering the 2AFC questions (and only keep timeseries of actual watching of videos)
        if len(timeseries) > TRlimit:
            timeseries = timeseries[list(chain(range(0,34),range(37,52),range(55,len(timeseries))))]
            print(len(timeseries))

        splittedName = np.char.split(files[currWedd], sep = '/')
        splittedName = splittedName.tolist()
        finalName = float(splittedName[-1][4:6])

        fileName = files[currWedd].split('/')

        if nplistCurrRow[counter] == 'NA':
            TotalData = pd.DataFrame(timeseries[:TRlength])
            TotalData['weddIdx'] = 1
            TotalData['subj'] = CurrSub
            TotalData['TRidx'] = list(range(len(TotalData)))
            TotalData['counter'] = counterNA
            TotalData['WhichWedd'] = float(fileName[14][4:7])
            counterNA += 1
            print(nplistCurrRow[counter])
            TotalDataAllWedds = TotalDataAllWedds.append(TotalData,sort=True)
        elif nplistCurrRow[counter] == 'NB':
            TotalData = pd.DataFrame(timeseries[:TRlength])
            TotalData['weddIdx'] = 2
            TotalData['subj'] = CurrSub
            TotalData['TRidx'] = list(range(len(TotalData)))
            TotalData['counter'] = counterNB
            TotalData['WhichWedd'] = float(fileName[14][4:7])
            counterNB += 1
            print(nplistCurrRow[counter])
            TotalDataAllWedds = TotalDataAllWedds.append(TotalData,sort=True)
        elif nplistCurrRow[counter] == 'SA':
            TotalData = pd.DataFrame(timeseries[:TRlength])
            TotalData['weddIdx'] = 3
            TotalData['subj'] = CurrSub
            TotalData['TRidx'] = list(range(len(TotalData)))
            TotalData['counter'] = counterSB
            TotalData['WhichWedd'] = float(fileName[14][4:7])
            counterSB += 1
            print(nplistCurrRow[counter])
            TotalDataAllWedds = TotalDataAllWedds.append(TotalData,sort=True)
        elif nplistCurrRow[counter] == 'SB':
            TotalData = pd.DataFrame(timeseries[:TRlength])
            TotalData['weddIdx'] = 4
            TotalData['subj'] = CurrSub
            TotalData['TRidx'] = list(range(len(TotalData)))
            TotalData['counter'] = counterSA
            TotalData['WhichWedd'] = float(fileName[14][4:7])
            counterSA += 1
            print(nplistCurrRow[counter])
            TotalDataAllWedds = TotalDataAllWedds.append(TotalData,sort=True)
        
        counter += 1

    # RSA
    sameEv_otherSchema_all2 = pd.DataFrame(columns=range(0,74))
    otherEv_otherSchema_all2 = pd.DataFrame(columns=range(0,74))
    otherEv_sameSchema_all2 = pd.DataFrame(columns=range(0,74))
    sameEv_sameSchema_all2 = pd.DataFrame(columns=range(0,74))

    #NORMALIZE
    addsForLater = TotalDataAllWedds.iloc[:,-5:]

    scaler = StandardScaler()
    TotalDataAllWedds_norm = pd.DataFrame(scaler.fit_transform(TotalDataAllWedds.iloc[:,0:-5]))

    TotalDataAllWedds = pd.concat([TotalDataAllWedds_norm,addsForLater.reset_index(drop=True)],axis=1)
    
    # template creation
    for currWhichWedd in WhichWeddNrs:
        
        sameEv_otherSchema_all = pd.DataFrame(columns=[])
        otherEv_otherSchema_all = pd.DataFrame(columns=[])
        otherEv_sameSchema_all = pd.DataFrame(columns=[])
        sameEv_sameSchema_all = pd.DataFrame(columns=[])
        
        # save current wedding to use later for correlation
        currLoop = TotalDataAllWedds.loc[TotalDataAllWedds['WhichWedd'] == currWhichWedd]
        
        # determine whether current wedding is Na, Nb, Sa, or Sb
        WeddIdxCurrLoop = np.array(currLoop['weddIdx'])[0]
        
        # copy of original data
        TotalDataAllWedds_copy = TotalDataAllWedds.copy()
        
        # do not take into account curr wedding into template making
        TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['WhichWedd'] == currWhichWedd] = np.nan
        
        # select E3 TRs only for templates
        TotalDataAllWedds_copy = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['TRidx'] > EndEvent3]
        TotalDataAllWedds_copy = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['TRidx'] < EndEvent4]

        Templ_sameEvSameSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == WeddIdxCurrLoop].mean()
        Templ_sameEvSameSchema = Templ_sameEvSameSchema[0:-5]

        if WeddIdxCurrLoop == 1.0:

            templ_otherEvSameSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 2].mean()
            templ_otherEvSameSchema = templ_otherEvSameSchema[0:-5]

            templ_otherEvOtherSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 4].mean()
            templ_otherEvOtherSchema = templ_otherEvOtherSchema[0:-5]

            templ_sameEvOtherSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 3].mean()
            templ_sameEvOtherSchema = templ_sameEvOtherSchema[0:-5]

        elif WeddIdxCurrLoop == 2.0:

            templ_otherEvSameSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 1].mean()
            templ_otherEvSameSchema = templ_otherEvSameSchema[0:-5]

            templ_otherEvOtherSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 3].mean()
            templ_otherEvOtherSchema = templ_otherEvOtherSchema[0:-5]

            templ_sameEvOtherSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 4].mean()
            templ_sameEvOtherSchema = templ_sameEvOtherSchema[0:-5]

        elif WeddIdxCurrLoop == 3.0:

            templ_otherEvSameSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 4].mean()
            templ_otherEvSameSchema = templ_otherEvSameSchema[0:-5]

            templ_otherEvOtherSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 2].mean()
            templ_otherEvOtherSchema = templ_otherEvOtherSchema[0:-5]

            templ_sameEvOtherSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 1].mean()
            templ_sameEvOtherSchema = templ_sameEvOtherSchema[0:-5]

        elif WeddIdxCurrLoop == 4.0:

            templ_otherEvSameSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 3].mean()
            templ_otherEvSameSchema = templ_otherEvSameSchema[0:-5]

            templ_otherEvOtherSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 1].mean()
            templ_otherEvOtherSchema = templ_otherEvOtherSchema[0:-5]

            templ_sameEvOtherSchema = TotalDataAllWedds_copy.loc[TotalDataAllWedds_copy['weddIdx'] == 2].mean()
            templ_sameEvOtherSchema = templ_sameEvOtherSchema[0:-5]

        for currTr in range(currLoop.shape[0]):
            
            currLoop_copy = currLoop.copy()

            currLoop_copy = np.array(currLoop_copy.loc[currLoop_copy['TRidx'] == currTr])
            currLoop_copy = currLoop_copy[:,0:-5]

            R1_sameSame = np.corrcoef(currLoop_copy,np.array(Templ_sameEvSameSchema))
            R1_sameOther = np.corrcoef(currLoop_copy,np.array(templ_sameEvOtherSchema))
            R1_otherSame = np.corrcoef(currLoop_copy,np.array(templ_otherEvSameSchema))
            R1_otherOther = np.corrcoef(currLoop_copy,np.array(templ_otherEvOtherSchema))
            
            # save TR timeline
            sameEv_otherSchema_all = sameEv_otherSchema_all.append(pd.Series(R1_sameOther[0][1],name=currTr))
            otherEv_otherSchema_all = otherEv_otherSchema_all.append(pd.Series(R1_otherOther[0][1],name=currTr))
            otherEv_sameSchema_all = otherEv_sameSchema_all.append(pd.Series(R1_otherSame[0][1],name=currTr))
            sameEv_sameSchema_all = sameEv_sameSchema_all.append(pd.Series(R1_sameSame[0][1],name=currTr))
            
        sameEv_otherSchema_all2.loc[str(currWhichWedd)] = pd.Series(np.array(sameEv_otherSchema_all).reshape(-1))
        otherEv_otherSchema_all2.loc[str(currWhichWedd)] = pd.Series(np.array(otherEv_otherSchema_all).reshape(-1))
        otherEv_sameSchema_all2.loc[str(currWhichWedd)] = pd.Series(np.array(otherEv_sameSchema_all).reshape(-1))
        sameEv_sameSchema_all2.loc[str(currWhichWedd)] = pd.Series(np.array(sameEv_sameSchema_all).reshape(-1))
        

    sameEv_otherSchema_all2.to_csv(output_dir + 'indivFiles/' + ROI_name + '_' + str(AllSubjects[CurrSub]) + '_rsa_sameEv_otherSchema_norm_event4Templates.csv')
    otherEv_otherSchema_all2.to_csv(output_dir + 'indivFiles/' + ROI_name + '_' + str(AllSubjects[CurrSub]) + '_rsa_otherEv_otherSchema_norm_event4Templates.csv')
    otherEv_sameSchema_all2.to_csv(output_dir + 'indivFiles/' + ROI_name + '_' + str(AllSubjects[CurrSub]) + '_rsa_otherEv_sameSchema_norm_event4Templates.csv')
    sameEv_sameSchema_all2.to_csv(output_dir + 'indivFiles/' + ROI_name + '_' + str(AllSubjects[CurrSub]) + '_rsa_sameEv_sameSchema_norm_event4Templates.csv')

    sameEv_otherSchema_group = sameEv_otherSchema_group.append(pd.Series(sameEv_otherSchema_all2.mean(),name=CurrSub))
    otherEv_otherSchema_group = otherEv_otherSchema_group.append(pd.Series(otherEv_otherSchema_all2.mean(),name=CurrSub))
    otherEv_sameSchema_group = otherEv_sameSchema_group.append(pd.Series(otherEv_sameSchema_all2.mean(),name=CurrSub))
    sameEv_sameSchema_group = sameEv_sameSchema_group.append(pd.Series(sameEv_sameSchema_all2.mean(),name=CurrSub))



sameEv_otherSchema_group.to_csv(output_dir + ROI_name + '_rsa_sameEv_otherSchema_norm_event4Templates.csv')
otherEv_otherSchema_group.to_csv(output_dir + ROI_name + '_rsa_otherEv_otherSchema_norm_event4Templates.csv')
otherEv_sameSchema_group.to_csv(output_dir + ROI_name + '_rsa_otherEv_sameSchema_norm_event4Templates.csv')
sameEv_sameSchema_group.to_csv(output_dir + ROI_name + '_rsa_sameEv_sameSchema_norm_event4Templates.csv')

plt.figure(figsize=(20,12))
plt.title('plot ', fontsize=18)
plt.errorbar(x=range(74),y=sameEv_otherSchema_group.mean(),label='sameEv_otherSchema', color='red')
plt.fill_between(range(74), 
                     sameEv_otherSchema_group.mean()-sameEv_otherSchema_group.sem(), 
                     sameEv_otherSchema_group.mean()+sameEv_otherSchema_group.sem(), 
                     alpha=0.5, color='red')

plt.errorbar(x=range(74),y=otherEv_otherSchema_group.mean(),label='otherEv_otherSchema', color='blue')
plt.fill_between(range(74), 
                     otherEv_otherSchema_group.mean()-otherEv_otherSchema_group.sem(), 
                     otherEv_otherSchema_group.mean()+otherEv_otherSchema_group.sem(), 
                     alpha=0.5, color='blue')

plt.errorbar(x=range(74),y=otherEv_sameSchema_group.mean(),label='otherEv_sameSchema', color='green')
plt.fill_between(range(74), 
                     otherEv_sameSchema_group.mean()-otherEv_sameSchema_group.sem(), 
                     otherEv_sameSchema_group.mean()+otherEv_sameSchema_group.sem(), 
                     alpha=0.5, color='green')

plt.errorbar(x=range(74),y=sameEv_sameSchema_group.mean(),label='sameEv_sameSchema', color='purple')
plt.fill_between(range(74), 
                     sameEv_sameSchema_group.mean()-sameEv_sameSchema_group.sem(), 
                     sameEv_sameSchema_group.mean()+sameEv_sameSchema_group.sem(), 
                     alpha=0.5, color='purple')

plt.legend(fontsize=18)
plt.box(on=None)
plt.ylabel('R', fontsize=18)
plt.xlabel('time (TR)', fontsize=18)
plt.axhline(y=0, color='k')
plt.axvline(x=17, color='k')
plt.axvline(x=23, color='k')
plt.axvline(x=34, color='k')
plt.axvline(x=50, color='k')
plt.axvline(x=66, color='k')

plt.savefig(output_dir + ROI_name + '_rsa_norm_event4Templates.png')
