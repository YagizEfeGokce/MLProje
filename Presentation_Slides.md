# Presentation Slides Content

## Slide 1: Title Slide

**Title:** Predicting University Global Rank Using Machine Learning
**Subtitle:** COE305 Machine Learning Project - Final Presentation
**Team Members:**

- Yağız Efe Gökçe
- Kerem Özcan
- Murat Emre Doğan

---

## Slide 2: Problem Statement

- **Why it matters:** University rankings influence funding, student enrollment, and global reputation.
- **The Challenge:** Ranking methodologies are opaque and complex.
- **Our Goal:** Build a data-driven system to predict rankings based on objective metrics (Publications, Education Quality, Employment).

---

## Slide 3: Dataset & Features

- **Data Source:** CWUR Data (Kaggle).
- **Scale:** ~2,200 records.
- **Key Features:**
  - `Quality of Education`
  - `Alumni Employment`
  - `Quality of Faculty`
  - `Publications` & `Citations`
- **Target Variables:**
  - `Score` (0-100)
  - `Ranking Category` (Elite, High, Average)

---

## Slide 4: Methodology

- **Preprocessing:** Imputed missing values, encoded country data, scaled all features.
- **Models Used:**
  1. **Linear/Logistic Regression:** Baseline.
  2. **Random Forest:** To capture non-linearity.
  3. **XGBoost:** For maximum performance.
- **Evaluation:** RMSE/R² for Regression; Accuracy/F1 for Classification.

---

## Slide 5: Regression Results

- **Best Model:** XGBoost Regressor
- **Performance:**
  - **R²:** 0.9809 (Excellent fit)
  - **RMSE:** 1.02
- **Key Insight:** `Publications` and `Citations` were the most important features.

*(Include Feature Importance Plot Here)*

---

## Slide 6: Classification Results

- **Best Model:** XGBoost Classifier
- **Accuracy:** 97.95%
- **F1 Score:** 0.9796
- **Result:** The model almost perfectly separates "Elite" universities from the rest.

*(Include Confusion Matrix Here)*

---

## Slide 7: Live Demo (Streamlit App)

- We created a **Streamlit Web App** to demonstrate our model.
- **Features:**
  - Interactive sliders for university metrics.
  - Real-time score and category prediction.
  - Visual comparison with global averages.

---

## Slide 8: Conclusion

- **Summary:** Machine Learning can accurately reverse-engineer university rankings.
- **Impact:** Universities can use this to identify key areas for improvement (e.g., focusing on research output to boost rank).
- **Future Work:** Integrate real-time data from APIs and social sentiment analysis.

---

## Slide 9: Questions?

**Thank You!**
