import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor,plot_tree
import matplotlib.pyplot as plt

# Veri dosyasını oku
data = pd.read_csv("C:/Users/ilhan/Downloads/maas1.csv")

# Veriyi kopyala
veri = data.copy()

X = veri["Level"]
y = veri["Salary"]

y = np.array(y).reshape(-1,1)
X = np.array(X).reshape(-1,1)
# Karar ağacı regresyon modelini oluştur
dtr = DecisionTreeRegressor(random_state=0,max_leaf_nodes = 5)

# Modeli eğit
dtr.fit(X, y)

# Tahmin yap
tahmin = dtr.predict(X)


plt.figure(figsize=(20,10),dpi=100)
plot_tree(dtr,feature_names="Level",class_names = "Salary",rounded=True,filled=True)
plt.show()



