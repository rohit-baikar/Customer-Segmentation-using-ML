# import lib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# load data
data = pd.read_csv("new_customers.csv")
print(data)

# check for null data
print(data.isnull().sum())

# features
features = data[["Annual_Income(INR)", "Spending_Score"]]
print(features)

cfeatures = pd.get_dummies(features, drop_first=True)
print(cfeatures)

# feature scaling
mms = MinMaxScaler()
sfeatures = mms.fit_transform(cfeatures)
print(sfeatures)

# model
model = KMeans(n_clusters=5, random_state=0)
res = model.fit_predict(sfeatures)
data["clusters"] = res
print(data)
#data.to_csv("clusters_customers.csv")

# plotting
d0 = data[data.clusters == 0]
d1 = data[data.clusters == 1]
d2 = data[data.clusters == 2]
d3 = data[data.clusters == 3]
d4 = data[data.clusters == 4]

plt.figure()
plt.scatter(d0["Annual_Income(INR)"], d0["Spending_Score"], label="0")		# mi ms
plt.scatter(d1["Annual_Income(INR)"], d1["Spending_Score"], label="1")		# hi ls
plt.scatter(d2["Annual_Income(INR)"], d2["Spending_Score"], label="2")		# hi hs
plt.scatter(d3["Annual_Income(INR)"], d3["Spending_Score"], label="3")		# li ls
plt.scatter(d4["Annual_Income(INR)"], d4["Spending_Score"], label="4")		# li hs
plt.legend()
plt.xlabel("Income")
plt.ylabel("spending")
plt.grid()
plt.show()

# predict
income = float(input("enter income: "))
spending = float(input("enter spending: "))
d = [[income, spending]]
sd = mms.transform(d)
ans = model.predict(sd)

match ans:
	case 0 :	print("mi ms")
	case 1 :	print("hi ls")
	case 2 :	print("hi hs")
	case 3 :	print("li ls")
	case 4 :	print("li hs")
