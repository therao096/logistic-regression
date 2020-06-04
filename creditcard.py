# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:12:59 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cc=pd.read_csv("creditcard.csv")
cc.columns
cc.drop(['Unnamed: 0'],axis=1,inplace=True)
cc.columns
cc.describe()
cc.shape
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cc['cards']=le.fit_transform(cc.card)
cc.cards.unique()
cc['owners']=le.fit_transform(cc.owner)
cc.owners.unique()
cc['selfemps']=le.fit_transform(cc.selfemp)
cc.selfemps.unique()
cc.drop(['card','owner','selfemp'], axis=1,inplace=True)
cc
###corelation based feature selection
correlated_features= set()
correlation_matrix=cc.corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i,j])>0.8 :
            colname=correlation_matrix.columns[i]
            correlated_features.add(colname)
len(correlated_features)
print(correlated_features, end=" ")
###expenditure excluded
cc.drop(labels=correlated_features, axis=1, inplace=True)
import statsmodels.formula.api as sm
cc.columns
cc.corr()
cc.dtypes

logit_model=sm.logit('cards~ reports +age+income+share+dependents+months+majorcards+active+owners+selfemps', data=cc).fit()
logit_model.summary()
### pvalues of age=0.462, months=0.313, majorcards=0.526, owners=0.649. therefore removing these columns
logit_model2= sm.logit('cards~reports +income+share+dependents+active+selfemps', data=cc).fit()
logit_model2.summary()
###pvalue for selfemps=0.535. therefor remove that
logit_model3= sm.logit('cards~reports +income+share+dependents+active', data=cc).fit()
logit_model3.summary()
y_pred=logit_model3.predict(cc)
y_pred
cc['pred_prob']=y_pred
cc['Att_val']=0
cc.loc[y_pred>0.5,'Att_val']=1
cc.Att_val



from sklearn.metrics import classification_report
classification_report(cc.Att_val,cc.cards)


###confusion matrix

confusion_matrix = pd.crosstab(cc['cards'],cc.Att_val)
confusion_matrix
accuracy = (294+1000)/(294+2+23+1000) 
accuracy   ####0.981

###roc curve
 from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(cc.cards, y_pred)
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
roc_auc


#########3train and test data
cc.drop('Att_val', axis=1, inplace=True)
from sklearn.model_selection import train_test_split
train,test=train_test_split(cc,test_size=0.3)
train.isnull().sum();test.isnull().sum()
train_model=sm.logit('cards~reports +income+share+dependents+active', data=train).fit()
train_model.summary()
train_pred = train_model.predict(train.iloc[:,:])
len(train)
train['train_pred']=0 
train.loc[train_pred>0.5,"train_pred"] = 1
### confusion matrix
confusion_matrix=pd.crosstab(train['cards'],train.train_pred)

confusion_matrix
accuracy=(203+703)/(203+2+15+703)
accuracy ###0.98

##prediction on test data
test_pred=train_model.predict(test)
test["test_pred"]=0
test.loc[test_pred>0.5,"test_pred"] = 1
confusion_matrix = pd.crosstab(test['cards'],test.test_pred)

confusion_matrix
accuracy_test = (90+299)/(90+1+6+299) # 0.9823
accuracy_test
