import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("Churn_Modelling.csv")
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

model = load_model("model.h5", compile=False)
lb_gender = pickle.load(open("lb_gender.pk1", "rb"))
oh_geo = pickle.load(open("oh_geo.pk2", "rb"))
scalar = pickle.load(open("scalar", "rb"))

df["Gender"] = lb_gender.transform(df["Gender"])
geo = oh_geo.transform(df[["Geography"]])
geo = geo.toarray() if hasattr(geo, "toarray") else geo
geo_df = pd.DataFrame(geo, columns=oh_geo.get_feature_names_out(["Geography"]))
df = pd.concat([df.drop("Geography", axis=1), geo_df], axis=1)

X = df.drop("Exited", axis=1)
y = df["Exited"]

# Use the SAME test_size and random_state as in training
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test_scaled = scalar.transform(X_test)
probs = model.predict(X_test_scaled).ravel()
preds = (probs > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))