# retrain_pipeline_rf.py

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

# Đọc dữ liệu
data = pd.read_csv("data_for_classification.csv")

# Convert label
data['Recommend?'] = data['Recommend?'].map({'Yes': 1, 'No': 0})

# Khai báo các trường
numerical_cols = [
    "rating_gap", "salary_benefits_gap", "training_learning_gap",
    "management_care_gap", "culture_fun_gap"
]
categorical_cols = [
    "id", "overtime_policy", "working_days", "company_size", "like_topic"
]
text_col = "like_cleaned"

# Khởi tạo ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ("text", CountVectorizer(max_features=5000), text_col)
])

# Khởi tạo pipeline
pipeline_rf = ImbPipeline([
    ("preprocessor", preprocessor),
    ("oversample", RandomOverSampler(random_state=42)),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Fit pipeline
X = data.drop("Recommend?", axis=1)
y = data["Recommend?"]
pipeline_rf.fit(X, y)

# Save
joblib.dump(pipeline_rf, "rf_pipeline_model_v2.joblib")
print("✅ Retrain pipeline_rf thành công và đã lưu vào rf_pipeline_model_v2.joblib")
