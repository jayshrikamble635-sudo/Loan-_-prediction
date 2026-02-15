import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

data = {
    "Gender": ["Male", "Female", "Male", "Female"],
    "Education": ["Graduate", "Not Graduate", "Graduate", "Not Graduate"],
    "ApplicantIncome": [5000, 3000, 6000, 2500],
    "Loan_Status": ["Y", "N", "Y", "N"]
}

df = pd.DataFrame(data)

le_gender = LabelEncoder()
le_edu = LabelEncoder()
le_status = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Education"] = le_edu.fit_transform(df["Education"])
df["Loan_Status"] = le_status.fit_transform(df["Loan_Status"])

X = df[["Gender", "Education", "ApplicantIncome"]]
y = df["Loan_Status"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "loan_prediction_model.pkl")
joblib.dump(le_gender, "gender_encoder.pkl")
joblib.dump(le_edu, "education_encoder.pkl")



