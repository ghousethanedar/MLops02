from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import preprocessing_code as pp
from sklearn.preprocessing import StandardScaler
# from xgboost import XGBRegressor
import config
## defing the steps to include in the pipeline



random_forest = RandomForestClassifier()


train_steps = Pipeline(
    steps=[
        ('DropColumns',pp.Dropcols(variables=config.Drop_features)),
        ('numerical_imputer',pp.Numerical_imputer(variables=config.NUM_FEAUTESS)),
        ('categorical_imputer', pp.Category_imputer(variables=config.CAT_FEATURES)),
        ('RareLabels',pp.RareLabelCategoricalEncoder(variables=config.CAT_FEATURES)),
        ('category_encoder',pp.Categorical_encoding(variables=config.CAT_FEATURES)),
        ('stander Scaler',StandardScaler()),
        ('model',RandomForestClassifier(n_estimators=200,max_features= 8)),
        #('xgmodel',xgb1),
    ]
)


