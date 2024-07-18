import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import yaml
import os
import joblib
from catboost import CatBoostRegressor 

# код только обучения модели из проекта 1го спринта

def fit_model():
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    data = pd.read_csv('data/initial_data.csv')

    y = data[params['target_col']]
    x = data.drop(['id', 'price', 'building_id', 'target'], axis=1)
    
    binary_cat_features = x[['studio', 'is_apartment', 'has_elevator']]
    other_cat_features = x[['building_type_int']]
    num_features = x.select_dtypes(['float'])

    preprocessor = ColumnTransformer(
        [
        ('binary', OneHotEncoder(drop=params['one_hot_drop']), binary_cat_features.columns.tolist()),
        ('cat', CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
        ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    model = CatBoostRegressor(loss_function='RMSE')
    
    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ],
        verbose=False
    )

    pipeline.fit(x, y) 

    model.save_model('fitted_model')      

    os.makedirs('models', exist_ok=True)
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(pipeline, fd)

if __name__ == '__main__':
	fit_model()