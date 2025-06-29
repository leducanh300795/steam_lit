import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import html
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============ PAGE SETUP ============
st.set_page_config(page_title="ITviec Data App", layout="wide")

# ============ SIDEBAR ============
with st.sidebar:
    st.title("Menu")
    page = st.radio(
        "Chọn trang",
        [
            "Giới thiệu",
            "Phân tích & Kết quả",
            "Content-Based Similarity",
            "Recommend Classification",
            "Retrain Pipeline"
        ]
    )
    st.markdown("---")
    st.markdown("**Thành viên nhóm:**")
    st.markdown("- Mr. Lê Đức Anh")
    st.markdown("Mail: leducanh300795@gmail.com")
    st.markdown("- Mr. Trần Anh Tú")
    st.markdown("**GVHD:**")
    st.markdown("- Ms. Khuất Thùy Phương")
    st.markdown(
        "<div style='font-size: 11px; color: gray; text-align: center;'>"
        "Dự án tốt nghiệp<br>Data Science & Machine Learning<br>TTTN - ĐH KHTN"
        "</div>",
        unsafe_allow_html=True
    )

# ============ PAGE 1: GIỚI THIỆU ============
if page == "Giới thiệu":
    st.image("banner_itviec_2.jpg", caption="Nguồn: ITviec")
    st.title("Giới thiệu")
    st.markdown("""
## Về ITviec
ITViec là nền tảng chuyên cung cấp cơ hội việc làm IT hàng đầu Việt Nam. Ứng viên dễ dàng tìm kiếm việc làm theo kỹ năng, chức danh, công ty; đồng thời tham khảo đánh giá, blog chuyên ngành, báo cáo lương.

## Bộ dữ liệu
Hơn 8.000 đánh giá từ nhân viên và cựu nhân viên ngành IT tại Việt Nam.

Các trường chính:
- `Company Name`
- `Company overview`
- `Our key skills`
- `Why you'll love working here`
- `like_cleaned` (review đã clean)
- `rating_gap`
- `company_size`
- `Recommend?`

## Mục tiêu
- **Content-Based Similarity**: gợi ý các công ty tương tự
- **Recommend Classification**: dự đoán khả năng recommend công ty
    """)

# ============ PAGE 2: PHÂN TÍCH & KẾT QUẢ ============
elif page == "Phân tích & Kết quả":
    st.image("banner_itviec_2.jpg", caption="Nguồn: ITviec")
    st.title("📊 Phân tích & Kết quả")
    st.markdown("""
## 1️⃣ Tiền xử lý dữ liệu
- Xóa giá trị thiếu
- Chuẩn hóa lowercase, bỏ HTML, bỏ emoji, bỏ dấu câu, ký tự đặc biệt
- Tokenization, loại stopwords mở rộng
- Unique token
- Vector hóa bằng TF-IDF (gensim)
- Xây dựng RandomForest với pipeline
- Xử lý mất cân bằng nhãn bằng RandomOverSampler

## 2️⃣ Content-Based Similarity
- Ghép text 3 trường mô tả
- Làm sạch như trên
- TF-IDF
- MatrixSimilarity
- Input: tên công ty hoặc mô tả bất kỳ
- Top 5 công ty tương tự

## 3️⃣ Recommend Classification
- Ghép các đặc trưng: gap features, categorical, text clean
- Pipeline:
  - CountVectorizer
  - StandardScaler
  - OneHotEncoder
- RandomForest + RandomOverSampler
- Cross-validation Accuracy ~88%
- Classification report hiển thị chi tiết

## 4️⃣ Kết quả
- Hệ thống chạy tốt, phản hồi nhanh
- Triển khai thực tế tiềm năng
    """)

# ============ PAGE 3: CONTENT-BASED SIMILARITY ============
elif page == "Content-Based Similarity":
    st.image("banner_itviec_2.jpg", caption="Nguồn: ITviec")
    st.title("🔍 Content-Based Company Similarity")

    @st.cache_data
    def load_overview_data():
        df = pd.read_excel("Data/Overview_Companies.xlsx")
        df = df.dropna(subset=[
            "Company Name", "Company overview", "Our key skills", "Why you'll love working here"
        ])
        return df.reset_index(drop=True)

    df = load_overview_data()

    custom_stopwords = ENGLISH_STOP_WORDS.union({
        "https", "www", "com", "company", "job", "team"
    })

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = html.unescape(text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = [w for w in text.split() if w not in custom_stopwords]
        return ' '.join(dict.fromkeys(words))

    df["Company overview clean"] = df["Company overview"].apply(clean_text)
    df["Our key skills clean"] = df["Our key skills"].apply(clean_text)
    df["Why you'll love working here clean"] = df["Why you'll love working here"].apply(clean_text)

    df["full_text"] = df[
        ["Company overview clean", "Our key skills clean", "Why you'll love working here clean"]
    ].agg(" ".join, axis=1)

    df["tokens"] = df["full_text"].apply(lambda x: list(dict.fromkeys(x.split())))

    dictionary = corpora.Dictionary(df["tokens"])
    corpus = [dictionary.doc2bow(tokens) for tokens in df["tokens"]]
    tfidf_model = models.TfidfModel(corpus)
    index_tfidf = similarities.MatrixSimilarity(tfidf_model[corpus], num_features=len(dictionary))

    company_options = sorted(df["Company Name"].unique().tolist())
    company_options.insert(0, "Nhập mô tả bất kỳ")

    selected_company = st.selectbox("Chọn công ty:", company_options)

    if selected_company == "Nhập mô tả bất kỳ":
        query_text = st.text_area("Nhập mô tả công ty bạn muốn tìm tương tự:")
    else:
        query_text = df[df["Company Name"] == selected_company]["full_text"].values[0]

    if st.button("Tìm công ty tương tự"):
        cleaned_query = clean_text(query_text)
        query_tokens = list(dict.fromkeys(cleaned_query.split()))
        query_bow = dictionary.doc2bow(query_tokens)
        query_tfidf = tfidf_model[query_bow]
        sims = index_tfidf[query_tfidf]
        top_idx = sorted(enumerate(sims), key=lambda x: -x[1])[1:6]

        st.markdown("### 🏢 Công ty tương tự:")
        for i, sim in top_idx:
            row = df.iloc[i]
            company = row["Company Name"]
            overview = row["Company overview"]
            skills = row["Our key skills"]
            why_love = row["Why you'll love working here"]

            st.subheader(f"{company}  (similarity={sims[i]:.2f})")
            st.markdown(f"**📄 Overview:** {overview}")
            st.markdown(f"**🛠️ Skills:** {skills}")
            st.markdown(f"**💖 Why you'll love working here:** {why_love}")
            st.markdown("---")

# ======================== PAGE 4: RECOMMEND CLASSIFICATION ========================
elif page == "Recommend Classification":
    st.image("banner_itviec_2.jpg", caption="Nguồn: ITviec")
    st.header("Recommend Classification")
    try:
        pipeline_rf = joblib.load("rf_pipeline_model_v2.joblib")
        pipeline_like = joblib.load("pipeline_like.pkl")
        df = pd.read_csv("data_for_classification.csv")
        overview = pd.read_excel("Data/Overview_Companies.xlsx")

        # Tạo danh sách công ty hiển thị kiểu: "ID - Company Name"
        company_options = overview[['id', 'Company Name']].drop_duplicates()
        company_options['display'] = company_options.apply(
            lambda x: f"{x['id']} - {x['Company Name']}", axis=1
        )
        selected_company_display = st.selectbox(
            "Chọn công ty:",
            company_options['display'].tolist()
        )
        # tách ra ID từ chuỗi "id - name"
        company_id = int(selected_company_display.split(" - ")[0])

        st.subheader("Điền các thông tin kỳ vọng")
        salary_benefits_exp = st.slider("Salary & Benefits", 1.0, 5.0, 3.0)
        training_learning_exp = st.slider("Training & Learning", 1.0, 5.0, 3.0)
        management_care_exp = st.slider("Management Care", 1.0, 5.0, 3.0)
        culture_fun_exp = st.slider("Culture & Fun", 1.0, 5.0, 3.0)

        rating_exp = np.mean([
            salary_benefits_exp,
            training_learning_exp,
            management_care_exp,
            culture_fun_exp
        ])
        rating_gap = 3.5 - rating_exp
        salary_benefits_gap = 3.5 - salary_benefits_exp
        training_learning_gap = 3.5 - training_learning_exp
        management_care_gap = 3.5 - management_care_exp
        culture_fun_gap = 3.5 - culture_fun_exp

        overtime_policy = st.selectbox("Overtime Policy", [
            "No OT", "Extra salary for OT", "Extra days off for OT"
        ])
        working_days = st.selectbox("Working Days", [
            "Monday - Friday", "Monday - Saturday"
        ])
        company_size = st.selectbox("Company Size", [
            "1-150 employees", "151-500 employees", ">500 employees"
        ])

        like_text = st.text_area("Mong đợi ưu điểm của công ty:")
        if like_text.strip():
            like_topic = pipeline_like.predict([like_text])[0]
        else:
            like_topic = 0

        input_df = pd.DataFrame([{
            "id": company_id,
            "rating_gap": rating_gap,
            "salary_benefits_gap": salary_benefits_gap,
            "training_learning_gap": training_learning_gap,
            "management_care_gap": management_care_gap,
            "culture_fun_gap": culture_fun_gap,
            "overtime_policy": overtime_policy,
            "working_days": working_days,
            "company_size": company_size,
            "like_cleaned": like_text,
            "like_topic": like_topic
        }])

        if st.button("Dự đoán"):
            y_pred = pipeline_rf.predict(input_df)
            y_proba = pipeline_rf.predict_proba(input_df)
            st.success(f"✅ Kết quả: {'Recommend' if y_pred[0]==1 else 'Not Recommend'}")
            st.info(f"Xác suất Recommend: {y_proba[0][1]:.2%}")

    except Exception as e:
        st.error(f"Lỗi: {e}")


# ======================== PAGE 5: RETRAIN PIPELINE ========================
elif page == "Retrain Pipeline":
    st.header("Retrain Pipelines")

    if st.button("🔁 Retrain pipeline_like"):
        data_TC = pd.read_csv("Reviews_cleaned_for_TC_v2.csv")
        pipeline_like = Pipeline([
            ("vectorizer", CountVectorizer(max_df=0.95, min_df=5)),
            ("lda", LatentDirichletAllocation(n_components=3, random_state=42)),
            ("kmeans", KMeans(n_clusters=3, random_state=42))
        ])
        pipeline_like.fit(data_TC['like_cleaned'])
        joblib.dump(pipeline_like, "pipeline_like.pkl")
        st.success("✅ retrain pipeline_like thành công")

    if st.button("🔁 Retrain pipeline_rf"):
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
        st.success("✅ retrain pipeline_rf thành công")