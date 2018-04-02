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

m_GB = H2OGradientBoostingEstimator(ntrees=10, max_depth=9) 
m_GB.train(x=training_columns, y=response_column, training_frame=train, model_id= 'm_GB_10_9')
per_GB = m_GB.predict(test_data=test)

per_GB = m_GB.predict(test_data=test) 
out = pd.DataFrame(columns=['test','pred'])
out['test'] = test['churned_within_30_days'].as_data_frame()
out['pred'] = per_GB['predict'].as_data_frame()

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

m_GB = H2OGradientBoostingEstimator(ntrees=10, max_depth=9, distribution = 'bernoulli') 
m_GB.train(x=training_columns, y=response_column, training_frame=train)

##Validation
validation_path = 'data/dataset_2_validation.csv'
Gradient_Boosting = pd.read_csv(validation_path)

validation = h2o.H2OFrame(Gradient_Boosting)
per_GB = m_GB.predict(test_data=validation)

output = per_GB.as_data_frame()
print output.shape
print (output[output['predict'] == 0]).shape
print (output[output['predict'] == 1]).shape

Gradient_Boosting['churned_within_30_days'] = output['predict']
Gradient_Boosting['p0'] = output['p0']
Gradient_Boosting['p1'] = output['p1']

out_path = 'data/Dataset_2_predict_xgboost.csv'
Gradient_Boosting.to_csv(out_path)

h2o.cluster().shutdown()