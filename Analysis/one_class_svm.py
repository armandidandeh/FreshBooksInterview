# coding: utf-8

##param_selection
'''
This cell includes the code to find the best parameters 
to build the final model
'''

import pandas as pd
from sklearn import svm
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

train_path = 'data/dataset_2_train.csv'
df = pd.read_csv(train_path)
df = pd.get_dummies(df, columns=["f17", "f18"])
not_churned = df[df['churned_within_30_days'] == 0]
churned = df[df['churned_within_30_days'] == 1]

input_columns = ['f01', u'f02', u'f03', u'f04', u'f05', u'f06', u'f07',
       u'f08', u'f09', u'f10', u'f11', u'f12', u'f13', u'f14', u'f15', u'f16', u'f17_a', u'f17_b', u'f17_c', u'f18_a',
       u'f18_b', u'f18_c']
output_columns = 'churned_within_30_days'

## Select class 0 for train set 
train_size = df.shape[0] * 80 /100
train = shuffle(not_churned)[:train_size]
test = pd.concat([shuffle(not_churned)[train_size+1:],churned])

X_train = train[input_columns]
y_train = train[output_columns]

X_test= test[input_columns]
y_test = test[output_columns]

FPR = []
TPR = []
TPR.append(0)
FPR.append(0)
TPR.append(1)
FPR.append(1)

for i in np.arange(0.01,0.1,0.01):
        clf = svm.OneClassSVM(nu=i, kernel="rbf", gamma='auto')
        clf.fit(X_train)
        y_pred= clf.predict(X_test)
        
        np.place(y_pred,y_pred ==1, 0)
        np.place(y_pred, y_pred ==-1, 1)

        out = pd.DataFrame(columns=['test','pred'])
        out['test'] = y_test.values
        out['pred'] = y_pred

        FP =  float((out[(out['test']==1) & (out['pred'] == 0)]).shape[0])
        TP =  float((out[(out['test']==0) & (out['pred'] == 0)]).shape[0])
        FN =  float((out[(out['test']==0) & (out['pred'] == 1)]).shape[0])
        TN =  float((out[(out['test']==1) & (out['pred'] == 1)]).shape[0])

        FPR.append(FP /(FP + TP))
        TPR.append(TP /(TP +FN))

plt.title('Receiver Operating Characteristic')
output = sorted(zip(TPR, FPR, np.arange(0.01,0.1,0.01)), key=lambda x: x[0])
plt.plot([i[1] for i in output], [i[0] for i in output])
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

##final_model
'''
This cell builds the final model and precit the label
validation set.
'''

train_path = 'data/dataset_2_train.csv'
df = pd.read_csv(train_path)
df = pd.get_dummies(df, columns=["f17", "f18"])
train = df[df['churned_within_30_days'] == 0][input_columns]
clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma='auto')
clf.fit(train)

validation_path = 'data/dataset_2_validation.csv'
validation = pd.read_csv(validation_path)
validation = pd.get_dummies(validation, columns=["f17", "f18"])
y_pred= clf.predict(validation[input_columns])

np.place(y_pred,y_pred ==1, 0)
np.place(y_pred, y_pred ==-1, 1)

validation['churned_within_30_days'] = y_pred
out_path = 'data/Dataset_2_predict_one_class_svm.csv'
validation.to_csv(out_path)


print (y_pred == 0).shape
print sum(y_pred)
print ((y_pred == 0).shape - sum(y_pred))[0]