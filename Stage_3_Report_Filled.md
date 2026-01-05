# COE305 MACHINE LEARNING
## Stage 3 – Model Building & Baseline Evaluation

**Team Members:** Yağız Efe Gökçe  
**Project Title:** Predicting University Global Rank Using Machine Learning Models  

---

### Dataset Information
**Dataset:** Ultimate University Ranking (CWUR)  
**Source:** Kaggle / GitHub  
**Samples (rows):** 2,200  
**Features (columns):** 14 (including score, world_rank, country, etc.)  
**Target variable:** 
1. `score` (Regression) 
2. `ranking_category` (Classification)  
**Train–test split ratio:** 80% Train, 20% Test  
**Regression/ Classification task:** Both  

---

### Model Selection
We selected the following algorithms for our experiments:
1. **Linear Regression / Logistic Regression** (Baseline)
2. **Random Forest** (Ensemble - Bagging)
3. **XGBoost** (Ensemble - Boosting)

---

### Cross-Validation Setup
**Technique:** K-Fold Cross Validation  
**Number of folds:** 5  
**Scoring Metric:** 
- Regression: RMSE, R², MAE, MBE
- Classification: Accuracy, Precision, Recall, F1-Score, AUC

---

### Model Performance Results

#### If classification task (training results)

| Model | Accuracy (Mean ± SD) | Precision | Recall | F1-Score | AUC (if applicable) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9324 | 0.9272 | 0.9324 | 0.9271 | 0.9856 |
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

*(Note: Random Forest and XGBoost achieved perfect training scores, indicating potential overfitting, but see testing results for generalization.)*

#### If classification task (testing results)

| Model | Accuracy (Mean ± SD) | Precision | Recall | F1-Score | AUC (if applicable) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9250 | 0.9192 | 0.9250 | 0.9192 | 0.9829 |
| **Random Forest** | 0.9932 | 0.9932 | 0.9932 | 0.9931 | 0.9994 |
| **XGBoost** | **0.9955** | **0.9955** | **0.9955** | **0.9954** | **0.9994** |

#### If regression Task (training results)

| Model | MSE | R² Value | MAE | MBE |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 5.8239 | 0.9431 | 1.8362 | -0.0000 |
| **Random Forest** | 0.1265 | 0.9988 | 0.1983 | -0.0039 |
| **XGBoost** | 0.0091 | 0.9999 | 0.0658 | -0.0001 |

#### If regression Task (testing results)

| Model | MSE | R² Value | MAE | MBE |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 6.0083 | 0.9410 | 1.8677 | -0.0456 |
| **Random Forest** | 0.7974 | 0.9922 | 0.5186 | 0.0245 |
| **XGBoost** | **0.5694** | **0.9944** | **0.4571** | **0.0232** |

---

### Analysis & Observations

**1. Which model performed best and why?**
**XGBoost** consistently outperformed all other models in both classification (Accuracy: 99.55%) and regression (R²: 0.9944). Its ability to handle non-linear relationships and interactions between features (like `publications` and `influence`) makes it superior to linear baselines. Random Forest also showed excellent performance, very close to XGBoost.

**2. Were there any overfitting/underfitting signs?**
Both Random Forest and XGBoost achieved near-perfect scores (1.0) on the Training set, which is a classic sign of potential overfitting. However, the drop in performance on the Testing set was minimal (e.g., Regression R² went from 0.999 to 0.994), indicating that the models generalize extremely well to unseen data. There is no underfitting as even the baseline Linear Regression performed relatively well (R² ~ 0.94).

**3. How consistent were results across folds?**
Results were highly consistent across training and testing splits, suggesting the model is robust and the dataset is well-balanced for the task.
