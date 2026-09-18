

---

# Roadmap Machine Learning Engineer (Tabular & MLOps) — Chuyên Risk & Credit Scoring (24 tuần)

Lược bỏ hoàn toàn các phần ngoài lề (LLM, NLP, Computer Vision thuần), tập trung 100% vào thế mạnh **Dữ liệu bảng (Tabular/Structured Data)** và **Pipeline đưa mô hình ra thực tế (MLOps)** — đúng chính xác những gì JD đòi hỏi. 

Domain bài toán xuyên suốt: **Hệ thống đánh giá rủi ro tín dụng (Credit Risk Scoring) và Tự động phê duyệt khoản vay (Loan Approval)** trong ngành Fintech/Ngân hàng.

---

### Giai đoạn 0 — Tuần 1: Python chuyên sâu & Xử lý số học (NumPy & Vectorization)
**Từ khóa:** Python OOP nâng cao (class, decorator, magic methods, type hints), NumPy broadcasting & vectorization, ma trận số học, xác suất thống kê cơ bản cho ML (mean, median, variance, normal distribution, correlation, covariance).  
**Tài liệu:**
- NumPy Official Quickstart: https://numpy.org/doc/stable/user/quickstart.html
- Python Type Hints & Pydantic Basics: https://docs.python.org/3/library/typing.html

---

### Giai đoạn 1 — Tuần 2-3: Khai phá dữ liệu (EDA) & Data Cleaning thực chiến
**Từ khóa:** Pandas DataFrame indexing/slicing, group-by & aggregation, merge/join, xử lý Missing Values (imputation: mean, median, KNN, iterative), Outlier Detection (IQR method, Z-score, Isolation Forest), trực quan hóa dữ liệu với Matplotlib & Seaborn (histogram, boxplot, heatmap tương quan, pairplot), phân tích phân phối lệch (skewness, log-transform).  
**Tài liệu:**
- Pandas Official Tutorials: https://pandas.pydata.org/docs/user_guide/10min.html
- Kaggle Course — Data Cleaning: https://www.kaggle.com/learn/data-cleaning
- Seaborn Tutorial Gallery: https://seaborn.pydata.org/tutorial.html

---

### Giai đoạn 2 — Tuần 4-5: Feature Engineering & Preprocessing chuẩn công nghiệp
Phần quyết định 70-80% độ chính xác của mô hình Tabular trong thực tế.  
**Từ khóa:** Data Leakage (nguyên tắc vàng: fit trên Train, transform trên Val/Test), Categorical Encoding (One-Hot, Ordinal, Target Encoding, CatBoost Encoding), Feature Scaling (StandardScaler, MinMaxScaler, RobustScaler), Numerical Transformation (Log, Box-Cox, PowerTransformer), Domain-specific features cho Risk (Debt-to-Income ratio, Loan-to-Value, Payment-to-Income), WoE (Weight of Evidence) & IV (Information Value) chuyên dùng cho Credit Scoring, Feature Selection (Correlation threshold, Recursive Feature Elimination - RFE).  
**Tài liệu:**
- Feature-engine Docs (thư viện chuyên sâu về Feature Engineering cho Tabular): https://feature-engine.trainindata.com/
- Scikit-learn — Preprocessing Data: https://scikit-learn.org/stable/modules/preprocessing.html
- Weight of Evidence (WoE) & Information Value (IV) Guide for Credit Scoring: https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

---

### Giai đoạn 3 — Tuần 6-8: Thuật toán ML cốt lõi cho Dữ liệu bảng (Tree-based & Boosting)
**Từ khóa:** 
- **Linear/Baseline:** Logistic Regression, Ridge, Lasso (L1/L2 Regularization, Odds Ratio).
- **Decision Trees & Ensembles:** Bagging vs Boosting, Random Forest (Bootstrap, Out-of-Bag error).
- **Gradient Boosting SOTA:** XGBoost (histogram-based, regularization), LightGBM (Leaf-wise vs Level-wise, GOSS, EFB — cực nhanh cho bảng lớn), CatBoost (xử lý trực tiếp categorical features).
- **So sánh:** Khi nào dùng Logistic Regression (khi cần khả năng giải thích tuyệt đối cho ngân hàng nhà nước) vs Gradient Boosting (khi cần tối đa mAP/AUC).  
**Tài liệu:**
- Scikit-learn Supervised Learning Docs: https://scikit-learn.org/stable/supervised_learning.html
- XGBoost Official Documentation: https://xgboost.readthedocs.io/
- LightGBM Documentation: https://lightgbm.readthedocs.io/
- StatQuest — Gradient Boost / XGBoost playlist: https://www.youtube.com/@statquest

---

### Giai đoạn 4 — Tuần 9-10: Đánh giá mô hình, Imbalanced Data & Metrics trong Risk
Lỗi bề mặt hoặc rủi ro vỡ nợ luôn có tỷ lệ cực thấp (chỉ 1% - 5% nợ xấu).  
**Từ khóa:** Mất cân bằng dữ liệu (Class Imbalance), SMOTE, Random Under/Over-sampling, `class_weight='balanced'`, Precision, Recall, F1-Score, PR-AUC vs ROC-AUC (tại sao PR-AUC tốt hơn ROC-AUC khi dữ liệu siêu lệch), Threshold Tuning (chọn ngưỡng cut-off tối ưu), Cost Matrix (thiệt hại khi cho vay nhầm người bùng nợ vs bỏ sót khách hàng tốt), chỉ số KS (Kolmogorov-Smirnov Statistic) và hệ số Gini chuyên dùng trong thẩm định rủi ro.  
**Tài liệu:**
- Imbalanced-learn Docs (thư viện xử lý mất cân bằng dữ liệu): https://imbalanced-learn.org/stable/
- Google Crash Course — Classification: ROC and AUC: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

---

### Giai đoạn 5 — Tuần 11-12: Tự động hóa Pipeline & Hyperparameter Tuning
**Từ khóa:** `sklearn.pipeline.Pipeline`, `ColumnTransformer` (tách nhánh tiền xử lý số và chữ tự động), viết Custom Transformer (kế thừa `BaseEstimator`, `TransformerMixin`), K-Fold Cross-Validation, Stratified K-Fold (giữ nguyên tỷ lệ nợ xấu), Hyperparameter Optimization với **Optuna** (Bayesian Optimization thay thế hoàn toàn GridSearchCV/RandomizedSearchCV chậm chạp).  
**Tài liệu:**
- Scikit-learn — Pipeline & Composite Estimators: https://scikit-learn.org/stable/modules/compose.html
- Optuna Documentation (tối ưu siêu tham số SOTA): https://optuna.readthedocs.io/

---

### Giai đoạn 6 — Tuần 13-14: Model Explainability (XAI) — Giải thích mô hình
Trong lĩnh vực ngân hàng, cho vay hay từ chối vay đều phải giải trình lý do rõ ràng trước kiểm toán.  
**Từ khóa:** Black-box vs White-box models, Global vs Local Explainability, Feature Importance (MDI vs Permutation Importance), **SHAP (SHapley Additive exPlanations)** (TreeSHAP, Waterfall plot, Force plot, Beeswarm summary plot), LIME (Local Interpretable Model-agnostic Explanations), Partial Dependence Plots (PDP).  
**Tài liệu:**
- SHAP Official Docs & Tutorials: https://shap.readthedocs.io/
- Christoph Molnar — "Interpretable Machine Learning" (Sách kinh điển về XAI, đọc miễn phí): https://christophm.github.io/interpretable-ml-book/

---

### Giai đoạn 7 — Tuần 15-16: Đóng gói Model & Viết API Serving (FastAPI / Microservices)
Biến file model `.pkl` / `.joblib` thành dịch vụ phục vụ người dùng thật.  
**Từ khóa:** Serialization (Joblib, Pickle, ONNX cho tabular), REST API kiến trúc, **FastAPI** (tốc độ cao, tự sinh Swagger UI), Pydantic Model (Data Validation & Schema Type-checking tại ngõ vào), xử lý request đơn lẻ (Real-time scoring) vs Batch Inference, xử lý ngoại lệ (Exception Handling) và HTTP Status Codes.  
**Tài liệu:**
- FastAPI Official Tutorial: https://fastapi.tiangolo.com/tutorial/
- Pydantic Docs: https://docs.pydantic.dev/

---

### Giai đoạn 8 — Tuần 17-18: MLOps: Experiment Tracking & Containerization (MLflow & Docker)
Đúng yêu cầu JD: Docker, MLflow, theo dõi thử nghiệm và quản lý vòng đời model.  
**Từ khóa:** 
- **MLflow:** Tracking metrics, parameters, artifacts, Model Registry, staging/production versioning.
- **Docker:** Dockerfile, image layers, caching, `.dockerignore`, containerizing FastAPI service, port forwarding, chạy ứng dụng độc lập môi trường, Docker Compose cơ bản kết hợp API + MLflow server.  
**Tài liệu:**
- MLflow Official Docs (Quickstart): https://mlflow.org/docs/latest/index.html
- Docker Documentation — Get Started: https://docs.docker.com/get-started/

---

### Giai đoạn 9 — Tuần 19-20: Giám sát Mô hình & Data Drift trong Production
Khi mô hình chạy thực tế, phân phối dữ liệu khách hàng thay đổi theo lạm phát, khủng hoảng kinh tế (Drift).  
**Từ khóa:** Data Drift (Covariate Shift) vs Concept Drift, Data Quality Tests, **Evidently AI** (công cụ chuyên đo lường drift cho tabular data, xuất report HTML tự động), cảnh báo trôi ngưỡng (threshold alert), kế hoạch Retrain model định kỳ, viết Unit Test với `pytest` cho pipeline dữ liệu, CI/CD cơ bản với GitHub Actions (tự động test code khi push).  
**Tài liệu:**
- Evidently AI Docs (giám sát model drift cho tabular): https://docs.evidentlyai.com/
- Pytest Official Documentation: https://docs.pytest.org/

---

### Giai đoạn 10 — Tuần 21-23: Capstone Project Thực Chiến
**Tên dự án đề xuất:** **"End-to-End Loan Default Prediction & Credit Scoring Platform"**  
- **Dataset gợi ý:** *Home Credit Default Risk* (Kaggle) hoặc *Lending Club Loan Data*.
- **Kiến trúc hoàn chỉnh:**
  $$\text{Raw Tabular Data} \longrightarrow \text{Automated Scikit-learn Pipeline (Clean + Feature-engine)}$$
  $$\longrightarrow \text{XGBoost / LightGBM Model (được tune bằng Optuna + Track bằng MLflow)}$$
  $$\longrightarrow \text{Đóng gói API FastAPI (Validating input với Pydantic, xuất xác suất nợ xấu + lý do từ chối bằng SHAP)}$$
  $$\longrightarrow \text{Đóng gói thành Docker Container sẵn sàng deploy microservices}$$
  $$\longrightarrow \text{Giám sát Data Drift bằng Evidently AI dashboard}$$

---

### Giai đoạn 11 — Tuần 24: Portfolio, CV & Phỏng vấn Mock Interview
**Bộ câu hỏi phỏng vấn sát sườn theo JD này:**
1. *"Data Leakage là gì? Bạn làm thế nào để đảm bảo không bị rò rỉ dữ liệu khi làm Target Encoding hoặc Imputation trong Cross-Validation?"*  
   $\rightarrow$ Dùng `sklearn.pipeline.Pipeline` và fit bộ transformer riêng biệt trong từng fold của CV.
2. *"Tại sao trong bài toán nợ xấu/gian lận tín dụng, Accuracy là vô nghĩa? Bạn sẽ dùng metric nào?"*  
   $\rightarrow$ Imbalanced classes, cần dùng PR-AUC, Recall, Cost-matrix hoặc KS-statistic/Gini.
3. *"Sự khác biệt cốt lõi giữa Random Forest và Gradient Boosting (XGBoost/LightGBM) là gì?"*  
   $\rightarrow$ Bagging (độc lập, giảm variance) vs Boosting (tuần tự, học từ sai số của cây trước, giảm bias).
4. *"Làm thế nào để giải thích cho khách hàng hiểu lý do tại sao hồ sơ vay vốn của họ bị từ chối?"*  
   $\rightarrow$ Áp dụng SHAP Waterfall Plot / Force Plot để chỉ ra các feature đóng góp âm lớn nhất (ví dụ: DTI quá cao, nợ thẻ tín dụng quá hạn).
5. *"Khi deploy mô hình ra production, làm sao biết được khi nào cần retrain lại model?"*  
   $\rightarrow$ Theo dõi Data Drift / Concept Drift bằng Evidently AI, đo lường sự dịch chuyển phân phối đầu vào (PSI - Population Stability Index, Wasserstein Distance).

---

## 📅 Bảng tổng hợp theo tuần

| Tuần | Nội dung trọng tâm | Deliverable / Sản phẩm cần đạt |
|---|---|---|
| **1** | Python OOP, NumPy vectorization, Thống kê cơ bản | Script xử lý ma trận và tính toán thống kê thuần NumPy |
| **2-3** | Pandas, Data Cleaning, Outlier, EDA & Visuals | File Jupyter Notebook EDA hoàn chỉnh cho dataset tài chính |
| **4-5** | Feature Engineering, Encoders, WoE/IV, Data Leakage | Bộ custom transformers xử lý feature chuyên sâu |
| **6-8** | Logistic Regression, Random Forest, XGBoost, LightGBM | Notebook benchmark so sánh hiệu năng 4 thuật toán |
| **9-10** | Imbalanced Data, SMOTE, PR-AUC, KS, Cost-sensitive | Module đánh giá metric và threshold tuning theo chi phí rủi ro |
| **11-12** | Scikit-learn Pipeline, ColumnTransformer, Optuna | Pipeline tự động hóa từ đầu vào thô đến kết quả dự đoán |
| **13-14** | XAI: SHAP values, LIME, Feature Importance | Dashboard/biểu đồ giải thích lý do từ chối/phê duyệt hồ sơ |
| **15-16** | FastAPI, Pydantic Schema, Real-time Scoring REST API | Service API trả kết quả dự đoán JSON dưới 50ms |
| **17-18** | Dockerizing API, MLflow tracking & registry | Container chạy mượt mà, MLflow log đầy đủ tham số/metric |
| **19-20** | Evidently AI drift monitoring, Pytest, GitHub Actions | Bộ test tự động và báo cáo data drift HTML |
| **21-23** | **Capstone Project: End-to-End Credit Scoring System** | Toàn bộ mã nguồn trên GitHub + README chuẩn kỹ thuật |
| **24** | Hoàn thiện CV, Portfolio GitHub, Luyện phỏng vấn | Sẵn sàng nộp đơn ứng tuyển |

---

> 💡 **Ghi chú chiến lược:** Điểm cộng lớn nhất của bạn khi nộp JD này là **tư duy End-to-End**. Phần lớn ứng viên chỉ biết mở file `.ipynb` chạy `model.fit()` và `predict()`. Nếu bạn có một repo GitHub thể hiện được: Code được module hóa `.py`, đóng gói thành **Pipeline**, bọc bằng **FastAPI**, container hóa bằng **Docker** và có **MLflow** quản lý, bạn sẽ vượt trội hơn 90% ứng viên cùng phân khúc intern/fresher!