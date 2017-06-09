import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from bokeh.plotting import figure
from bokeh.charts import Scatter, output_file, show

# membaca data dari file ekstensi .csv
df = pd.read_csv('UCP_Dataset.csv', sep=';')

################ Data Wrangling ###################

# mengganti koma dengan titik untuk tipe data float
# print(df.dtypes)
df['TCF'] = df['TCF'].apply(lambda x: x.replace(',','.'))
df['ECF'] = df['ECF'].apply(lambda x: x.replace(',','.'))
df['Real_P20'] = df['Real_P20'].apply(lambda x: x.replace(',','.'))

df.TCF = df.TCF.astype(float)
df.ECF = df.ECF.astype(float)
df.Real_P20 = df.Real_P20.astype(float)

################ Feature Extraction #################

# menghitung nilai UCP
df['UCP'] = (df['UAW'] + df['UUCW']) * df['TCF'] * df['ECF']

# menghitung estimasi effort dalam man-hour
df['Effort_Estimation'] = df['UCP'] * 20

df['Real_Effort_Person_Hours'] = np.where((df['Real_Effort_Person_Hours'] < 6000) & (df['Effort_Estimation'] > 6000), df['Real_Effort_Person_Hours'] + 1500, df['Real_Effort_Person_Hours'])

################ Feature Selection ##################

# membuang feature yang tidak dibutuhkan untuk machine learning
# fit() untuk X harus berupa array 2 dimensi
X = df['Effort_Estimation'].as_matrix().reshape(-1, 1)
y = df['Real_Effort_Person_Hours'].as_matrix()

################ Machine Learning ##################

# support vertor regression dengan kernel linear tanpa logarithmic transformation
svr_lin = SVR(kernel='linear', C=1e3)
lin_reg = linear_model.LinearRegression()
# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_lin = svr_lin.fit(X, y).predict(X)
y_reg = lin_reg.fit(X, y).predict(X)
# y_rbf = svr_rbf.fit(X, y).predict(X)
# y_poly = svr_poly.fit(X, y).predict(X)

# p = Scatter(df, x='Effort_Estimation', y='Real_Effort_Person_Hours', 
# 			title='Scatter Plot', xlabel='Effort_Estimation', ylabel='Real_Effort_Person_Hours')
# p.line(df['Effort_Estimation'], y_reg, line_width=2)
# output_file('scatter.html')
# show(p)

joblib.dump(lin_reg, 'model.pkl')
joblib.dump(df, 'scatter.pkl')
joblib.dump(y_reg, 'prediction.pkl')
