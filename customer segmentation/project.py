import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
data = pd.read_csv("clusters_customers.csv")
print(data)

print(data.isnull().sum())
print(data.duplicated())

features = data.drop(["clusters","CustomerID"], axis=1)
target = data["clusters"]
print(features)
print(target)

cfeatures = pd.get_dummies(features)
print(cfeatures)

ss = StandardScaler()
mfeatures = ss.fit_transform(cfeatures.values)
print(mfeatures)

x_train, x_test, y_train, y_test = train_test_split(mfeatures, target, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = DecisionTreeClassifier()
mf = model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)

cr = classification_report(y_test, y_pred)
print(cr)

ac = accuracy_score(y_test, y_pred)
print(f'Accuracy: {ac:.2f}')

import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plot_tree(mf,filled=True)
plt.show()

age = int(input("enter age: "))
ai = int(input("enter annual income: "))
sc = int(input("enter spending score: "))
gender = int(input("1 for female 2 for male: "))
if gender == 1:
	d = [[age, ai, sc, 0,1]]
else:
	d = [[age, ai, sc, 1,0]]
nd = ss.transform(d)
res = model.predict(nd)
res = res[0]
print(res)
if res == 0:
	print("C0: min income min spending")
elif res ==1:
	print("C1: high income low spending")
elif res == 2:
	print("C2: high income high spending")
elif res == 3:
	print("C3: low income low spending")
else:
	print("C4: low income high spending")

import pickle
f = open("csmodel.pkl","wb")
pickle.dump(model, f)
f.close()
g = open("ssscaler.pkl","wb")
pickle.dump(ss, g)
g.close()


