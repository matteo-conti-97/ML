# Executed as:
# python deliverable.py <path to training file> <path_to_test_file>
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from imblearn.over_sampling import RandomOverSampler



assert(len(sys.argv)==3)
train_file, test_file = sys.argv[1:]


X = pd.read_csv(train_file)
y = X.pop('is_canceled') # this is the target column
X_test = pd.read_csv(test_file) # will not contain the is_canceled column!
# The following line is only useful to debug the code on the training set:
if "is_canceled" in X_test.columns:
    X_test = X_test.drop(columns=['is_canceled']) 
    
#----- edit below ....

#drop company column cause it has too many NaN values
X = X.drop('company', axis=1) 
X_test = X_test.drop('company', axis=1)

#drop agent column cause it has too many NaN values
X = X.drop('agent', axis=1)
X_test = X_test.drop('agent', axis=1)


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
    OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
)

preprocessor = make_column_transformer(
    (transformer_num, numerical_features),
    (transformer_cat, categorical_features),
)

X = preprocessor.fit_transform(X)

#Oversampling
osmp = RandomOverSampler(sampling_strategy='auto', random_state=2)
X, y = osmp.fit_resample(X, y)

# ----

X_test['arrival_date_month'] = \
    X_test['arrival_date_month'].map(
        {'January':1, 'February': 2, 'March':3,
         'April':4, 'May':5, 'June':6, 'July':7,
         'August':8, 'September':9, 'October':10,
         'November':11, 'December':12}
    )


X_test = preprocessor.transform(X_test) # transform() instead of fit_transform()! 


'''import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

all_scores = []
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
input_shape = [X.shape[1]]

model = keras.Sequential(
    [ 
        layers.Dense(64, activation="relu", input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer='adam',
    loss='binary_focal_crossentropy',
    metrics=['accuracy'],
)

history = model.fit(
    X, y,
    batch_size=512,
    epochs=16,
    verbose=1,
    callbacks=[callback], 
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss']].plot()
print("Min Loss: {:0.4f}".format(history_df['loss'].min()))
print("Max Accuracy: {:0.4f}".format(history_df['accuracy'].max()))'''

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,random_state=1)
print("Training RandomForest with 100 trees it may take a while...")
model.fit(X, y)


# --------------------------------------------------------------------
predictions = model.predict(X_test)
print("Commented in the code there is also a simple FNN that can be used, \
      but at design time it seemed to perform slightly worse than a simple RandomForest (like a 0.04%) in terms of validation accuracy\
      obtain more is probably a thing of feature engineering.")
print("#0323728") # nomi o matricole
for i in predictions:
    print(i)

