DATA_FILE_PATH = 'train.csv'
MODEL_FILENAME = 'randomforest01.joblib'

TARGET = 'Surge_Pricing_Type'

## FEATURES

Drop_features = ['Trip_ID']

null_features = ['Type_of_Cab',
                 'Customer_Since_Months',
                 'Life_Style_Index',
                 'Confidence_Life_Style_Index',
                 'Var1']

CAT_FEATURES = ['Type_of_Cab', 'Confidence_Life_Style_Index', 'Destination_Type', 'Gender']

NUM_FEAUTESS = ['Trip_Distance',
                'Customer_Since_Months',
                'Life_Style_Index',
                'Customer_Rating',
                'Cancellation_Last_1Month',
                'Var1',
                'Var2',
                'Var3']
