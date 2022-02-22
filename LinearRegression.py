import numpy as np
import pandas as pd

df = pd.read_csv("/data/100_companies.csv")
X = df.iloc[:,:-1].values
Y = df.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labels = LabelEncoder()
X[:, 3] = labels.fit_transform(X[:, 3])

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=1234)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)

data = pd.DataFrame()

data['Original'] = Y_test

data['predicted'] = Y_pred

data.to_csv("/output/Result.csv",index=False)
