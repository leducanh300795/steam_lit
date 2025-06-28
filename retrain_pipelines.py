import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
import joblib


# === Retrain pipeline_like ===
print("👉 Đang retrain pipeline_like...")
data_TC = pd.read_csv("Reviews_cleaned_for_TC_v2.csv")

pipeline_like = Pipeline([
    ("vectorizer", CountVectorizer(max_df=0.95, min_df=5)),
    ("lda", LatentDirichletAllocation(n_components=3, random_state=42)),
    ("kmeans", KMeans(n_clusters=3, random_state=42))
])
pipeline_like.fit(data_TC['like_cleaned'])

joblib.dump(pipeline_like, "pipeline_like.pkl")
print("✅ retrain pipeline_like thành công.")

# === Retrain pipeline_rf ===
print("👉 Đang retrain pipeline_rf...")
data = pd.read_csv("data_for_classification.csv")
data['Recommend?'] = data['Recommend?'].map({'Yes': 1, 'No': 0})

numerical_cols = [
    "rating_gap", "salary_benefits_gap", "training_learning_gap",
    "management_care_gap", "culture_fun_gap"
]
categorical_cols = [
    "id", "overtime_policy", "working_days", "company_size", "like_topic"
]
text_col = "like_cleaned"

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ("text", CountVectorizer(max_features=5000), text_col)
])

pipeline_rf = ImbPipeline([
    ("preprocessor", preprocessor),
    ("oversample", RandomOverSampler(random_state=42)),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

pipeline_rf.fit(data.drop("Recommend?", axis=1), data['Recommend?'])
joblib.dump(pipeline_rf, "rf_pipeline_model_v2.joblib")
print("✅ retrain pipeline_rf thành công.")
