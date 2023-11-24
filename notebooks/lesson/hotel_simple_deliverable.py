# Executed as:
# python deliverable.py <path to training file> <path_to_test_file>
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer



assert(len(sys.argv)==3)
train_file, test_file = sys.argv[1:]


X = pd.read_csv(train_file)
y = X.pop('is_canceled') # this is the target column
X_test = pd.read_csv(test_file) # will not contain the is_canceled column!
# The following line is only useful to debug the code on the training set:
if "is_canceled" in X_test.columns:
    X_test = X_test.drop(columns=['is_canceled']) 
    
#----- edit below ....

X['arrival_date_month'] = \
    X['arrival_date_month'].map(
        {'January':1, 'February': 2, 'March':3,
         'April':4, 'May':5, 'June':6, 'July':7,
         'August':8, 'September':9, 'October':10,
         'November':11, 'December':12}
    )

numerical_features = X.loc[:, (X.dtypes == int) | (X.dtypes == float)].columns.tolist()
categorical_features = X.loc[:, (X.dtypes != int) & (X.dtypes != float)].columns.tolist()

transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # default fill value=0
    StandardScaler(),
)
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore'),
)

preprocessor = make_column_transformer(
    (transformer_num, numerical_features),
    (transformer_cat, categorical_features),
)

X = preprocessor.fit_transform(X)

# ----

X_test['arrival_date_month'] = \
    X_test['arrival_date_month'].map(
        {'January':1, 'February': 2, 'March':3,
         'April':4, 'May':5, 'June':6, 'July':7,
         'August':8, 'September':9, 'October':10,
         'November':11, 'December':12}
    )


X_test = preprocessor.transform(X_test) # transform() instead of fit_transform()! 


from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss="log_loss", alpha=0.01, penalty=None, random_state=0)
model.fit(X, y)


# --------------------------------------------------------------------
predictions = model.predict(X_test)
print("#12345,23456") # nomi o matricole
for i in predictions:
    print(i)

