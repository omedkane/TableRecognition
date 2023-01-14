import tableOCR
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

vars, labels, observations = tableOCR.ocrTable('./testtable.png')

dataframe = pd.DataFrame(columns=vars, index=labels, data=observations)
scaled_data = preprocessing.scale(dataframe)

pca = PCA().fit(scaled_data)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1) #type: ignore


pca_labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]


plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label= pca_labels)

plt.ylabel("Pourcentages de la variance expliqué ")
plt.xlabel("Composant Principal")
plt.title("Éboulis")
plt.show()

pca_data = pca.transform(scaled_data)
pca_df = pd.DataFrame(pca_data, index=labels, columns=pca_labels)
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title("My PCA Graph")
plt.xlabel("PC1 - {0}%".format(per_var[0]))
plt.ylabel("PC2 - {0}%".format(per_var[1]))

plt.show()
