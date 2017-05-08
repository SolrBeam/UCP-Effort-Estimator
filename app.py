import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt

convert_dot = lambda x: x.replace(',','.')

# membaca data dari file ekstensi .csv
df = pd.read_csv('UCP_Dataset.csv', sep=';')

length = 71

################ Data Wrangling ###################

# mengganti koma dengan titik untuk tipe data float
# print(df.dtypes)
df['TCF'] = df['TCF'].apply(convert_dot)
df['ECF'] = df['ECF'].apply(convert_dot)
df.TCF = df.TCF.astype(float)
df.ECF = df.ECF.astype(float)

################ Feature Extraction #################

# menghitung nilai UUCP
df['UUCP'] = df['UAW'] + df['UUCW']

# menghitung nilai UCP
df['UCP'] = df['UUCP'] * df['TCF'] * df['ECF']

# menghitung estimasi effort dalam man-hour
df['Effort_Estimation'] = df['UCP'] * 20

################ Feature Selection ##################

# membuang feature yang tidak dibutuhkan untuk machine learning
# fit() untuk X harus berupa array 2 dimensi
X = df['Effort_Estimation'].as_matrix().reshape(-1, 1)
y = df['Real_Effort_Person_Hours'].as_matrix()

################ Machine Learning ##################

# support vertor regression dengan kernel linear tanpa logarithmic transformation
svr_lin = SVR(kernel='linear', C=1e3)
# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_lin = svr_lin.fit(X, y).predict(X)
# y_rbf = svr_rbf.fit(X, y).predict(X)
# y_poly = svr_poly.fit(X, y).predict(X)

lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
# plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
# plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('Effort Estimation')
plt.ylabel('Real Effort')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

