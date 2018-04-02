# coding: utf-8

##param_selection
'''
This cell includes the code to find the best parameters 
to build the final model
'''

import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import pandas as pd
h2o.init(max_mem_size= '16G', port = 54321)

train_path = 'data/dataset_2_train.csv'
df = pd.read_csv(train_path)

data = h2o.H2OFrame(df)
data['churned_within_30_days'] = data['churned_within_30_days'].asfactor()

training_columns =[u'f01', u'f02', u'f03', u'f04', u'f05', u'f06', u'f07',
       u'f08', u'f09', u'f10', u'f11', u'f12', u'f13', u'f14', u'f15', u'f16',
       u'f17', u'f18']
response_column = 'churned_within_30_days'
train, test = data.split_frame(ratios=[0.8])

m_RF = H2ORandomForestEstimator(ntrees=10, max_depth=9, nfolds=10)
m_RF.train(x=training_columns, y=response_column, training_frame=train, model_id= 'RF_10_9')
per_RF = m_RF.predict(test_data=test)

per_RF = m_RF.predict(test_data=test) 
out = pd.DataFrame(columns=['test','pred'])
out['test'] = test['churned_within_30_days'].as_data_frame()
out['pred'] = per_RF['predict'].as_data_frame()

FP =  float((out[(out['test']==1) & (out['pred'] == 0)]).shape[0])
TP =  float((out[(out['test']==0) & (out['pred'] == 0)]).shape[0])
FN =  float((out[(out['test']==0) & (out['pred'] == 1)]).shape[0])
TN =  float((out[(out['test']==1) & (out['pred'] == 1)]).shape[0])

FPR = FP /(FP + TP)
TPR = TP /(TP +FN)

print (FPR, TPR)


##final_model
'''
This cell builds the final model and precit the label
validation set.
'''

train_path = 'data/dataset_2_train.csv'
df = pd.read_csv(train_path)
train = h2o.H2OFrame(df)
train['churned_within_30_days'] = train['churned_within_30_days'].asfactor()

m_RF = H2ORandomForestEstimator(ntrees=10, max_depth=9, nfolds=10)
m_RF.train(x=training_columns, y=response_column, training_frame=train)

##Validation
validation_path = 'data/dataset_2_validation.csv'
random_forest = pd.read_csv(validation_path)

validation = h2o.H2OFrame(random_forest)
per_RF = m_RF.predict(test_data=validation)

per_RF = m_RF.predict(test_data=validation)
output = per_RF.as_data_frame()
print output.shape
print (output[output['predict'] == 0]).shape
print (output[output['predict'] == 1]).shape

random_forest['churned_within_30_days'] = output['predict']
random_forest['p0'] = output['p0']
random_forest['p1'] = output['p1']

out_path = 'data/Dataset_2_predict_random_forest.csv'
random_forest.to_csv(out_path)

h2o.cluster().shutdown()

