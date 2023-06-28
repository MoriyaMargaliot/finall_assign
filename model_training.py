# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:57:01 2023

@author: user1
"""
import pandas as pd
#from madlan_data_prep import prepare_data
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import ElasticNetCV



datafile = "C:/Users/user1/Downloads/output_all_students_Train_v9.xlsx"
df = pd.read_excel(datafile)
data = prepare_data(df)

X = data.drop('price', axis=1)
y = data['price']


num_cols = [col for col in X.columns if X[col].dtypes!='O']
num_cols
cat_cols = [col for col in X.columns if (X[col].dtypes=='O')]
cat_cols

numerical_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='median', add_indicator=False)),
    ('scaling', StandardScaler())
])
    
categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='constant', add_indicator=False, fill_value='missing')),
    ('one_hot_encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])
        
    
column_transformer = ColumnTransformer([
    ('numerical_preprocessing', numerical_pipeline, num_cols),
    ('categorical_preprocessing', categorical_pipeline, cat_cols)
    ], remainder='drop')
    
    
pipe_preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', ElasticNet(alpha=1435.9485453543987, l1_ratio=0.5))
])

model = pipe_preprocessing_model.fit(X, y)




# שמירת המודל לקובץ PKL
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

