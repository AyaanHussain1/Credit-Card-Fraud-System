from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("creditcard.xls")
# print(df.shape)
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df.dropna(inplace=True))
# print(df.drop_duplicates(inplace=True))
# print(df.head())
# print(df.describe())
# print(df.info())

# print(df["Class"].value_counts())
df.rename(columns={"Class":"Detection"},inplace=True)

legit = df[df["Detection"] == 0]
fraud = df[df["Detection"] == 1]

# print(legit.shape)
# print(fraud.shape)

# print(legit["Amount"].describe())
# print(fraud["Amount"].describe())

# print(df.groupby("Detection").mean())

legit_sample = legit.sample(n=401)
new_data_set = pd.concat([legit_sample,fraud],axis=0)

# print(new_data_set.shape)
# print(new_data_set.head())

print(new_data_set["Detection"].value_counts())
print(new_data_set.groupby("Detection").mean())

# fraud.to_csv("fraud.csv")

x = new_data_set.drop(columns="Detection")
y = new_data_set["Detection"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1,stratify=y)

model = LogisticRegression()
model.fit(x_train,y_train)
pred  = model.predict(x_train)
a = accuracy_score(pred,y_train)
# print(f"Train Score : {a}")

pred  = model.predict(x_test)
a = accuracy_score(pred,y_test)
# print(f"Test Score : {a}")

# with open ("model.pkl","wb") as f:
#     pickle.dump(model,f)

input_data = (4462.0,-2.30334956758553,1.759247460267,-0.359744743330052,2.33024305053917,-0.821628328375422,-0.0757875706194599,0.562319782266954,-0.399146578487216,-0.238253367661746,-1.52541162656194,2.03291215755072,-6.56012429505962,0.0229373234890961,-1.47010153611197,-0.698826068579047,-2.28219382856251,-4.78183085597533,-2.61566494476124,-1.33444106667307,-0.430021867171611,-0.294166317554753,-0.932391057274991,0.172726295799422,-0.0873295379700724,-0.156114264651172,-0.542627889040196,0.0395659889264757,-0.153028796529788,239.93)
arr = np.array(input_data)
rs = arr.reshape(1,-1)
pred = model.predict(rs)
if pred[0] == 1:
    print("Fraud!")
else:
    print("Legit...")