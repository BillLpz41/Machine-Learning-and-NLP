import numpy as np
from  sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston
import pandas as pd

np.random.seed(0)
data = np.random.randint(1, 10, size=(1,100))
print (data)

data_pow_2 = data**2
print (data_pow_2)

data_pow_3 = data**3
print (data_pow_3)

lst = [data, data_pow_2, data_pow_3]
df = pd.DataFrame(np.vstack(lst).T, columns = ['size', 'size_pow_2', 'size_pow_3'])
print(df)

#Visualización de los datos
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)
ax4 = fig.add_subplot(1, 4, 4)
ax1.set_title("Datos juntos")
ax1.plot(df)
ax2.set_title("size")
ax2.plot(df['size'], linewidth=0, marker="o", color="blue", markersize=4)
ax3.set_title("size_pow_2")
ax3.plot(df['size_pow_2'], linewidth=0, marker="+", color="orange", markersize=3)
ax4.set_title("size_pow_3")
ax4.plot(df['size_pow_3'], linewidth=0, marker="*", color="green", markersize=2)

#Distribución de los datos
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)
ax1.set_title("size")
ax1.plot(df["size"], linewidth=0, marker="o", color="blue", markersize=4)
ax2.set_title("size_pow_2")
ax2.plot(df["size_pow_2"], linewidth=0, marker="+", color="orange", markersize=3)
ax3.set_title("size_pow_3")
ax3.plot(df["size_pow_3"], linewidth=0, marker="*", color="green", markersize=2)
ax4.set_title("size")
ax4.hist(df["size"], bins=5, color="blue")
ax5.set_title("size_pow_2")
ax5.hist(df["size_pow_2"], bins=5, color="orange")
ax6.set_title("size_pow_3")
ax6.hist(df["size_pow_3"], bins=5, color="green")

####################################
####Escalamiento de los datos####
####################################
data_standard_scaler = preprocessing.StandardScaler().fit_transform(df)
# estandarizado = (X - media) / std
data_robust_scaler = preprocessing.RobustScaler().fit_transform(df)
# estandarizado = (X - rango_intercuartílico) / std
print ('data_standard_scaler {}\n data_robust_scaler {}'.format(data_standard_scaler, data_robust_scaler))
df_standard_scaler = pd.DataFrame(data_standard_scaler, columns=['size', 'size_pow_2', 'size_pow_3'])
df_robust_scaler = pd.DataFrame(data_robust_scaler, columns=['size', 'size_pow_2', 'size_pow_3'])
print (df_standard_scaler)
print (df_robust_scaler)


#Datos escalados
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)
ax1.set_title("size")
ax1.plot(df["size"], linewidth=0, marker="o", color="blue", markersize=4)
ax2.set_title("size_standard_scaler")
ax2.plot(df_standard_scaler["size"], linewidth=0, marker="o", color="blue", markersize=3)
ax3.set_title("size_robust_scaler")
ax3.plot(df_robust_scaler["size"], linewidth=0, marker="o", color="blue", markersize=2)
ax4.set_title("size")
ax4.hist(df["size"], bins=5, color="blue")
ax5.set_title("size_standard_scaler")
ax5.hist(df_standard_scaler["size"], bins=5, color="blue")
ax6.set_title("size_standard_scaler")
ax6.hist(df_robust_scaler["size"], bins=5, color="blue")

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)
ax1.set_title("size_pow_2")
ax1.plot(df["size_pow_2"], linewidth=0, marker="+", color="orange", markersize=4)
ax2.set_title("size_pow_2_standard_scaler")
ax2.plot(df_standard_scaler["size_pow_2"], linewidth=0, marker="+", color="orange", markersize=3)
ax3.set_title("size_pow_2_robust_scaler")
ax3.plot(df_robust_scaler["size_pow_2"], linewidth=0, marker="+", color="orange", markersize=2)
ax4.set_title("size_pow_2")
ax4.hist(df["size_pow_2"], bins=5, color="orange")
ax5.set_title("size_pow_2_standard_scaler")
ax5.hist(df_standard_scaler["size_pow_2"], bins=5, color="orange")
ax6.set_title("size_pow_2_standard_scaler")
ax6.hist(df_robust_scaler["size_pow_2"], bins=5, color="orange")

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)
ax1.set_title("size_pow_3")
ax1.plot(df["size_pow_3"], linewidth=0, marker="*", color="green", markersize=4)
ax2.set_title("size_pow_3_standard_scaler")
ax2.plot(df_standard_scaler["size_pow_3"], linewidth=0, marker="*", color="green", markersize=3)
ax3.set_title("size_pow_3_robust_scaler")
ax3.plot(df_robust_scaler["size_pow_2"], linewidth=0, marker="*", color="green", markersize=2)
ax4.set_title("size_pow_3")
ax4.hist(df["size_pow_3"], bins=5, color="green")
ax5.set_title("size_pow_3_standard_scaler")
ax5.hist(df_standard_scaler["size_pow_3"], bins=5, color="green")
ax6.set_title("size_pow_3_standard_scaler")
ax6.hist(df_robust_scaler["size_pow_3"], bins=5, color="green")



plt.show()


