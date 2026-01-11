# Speaking Script: Predicting University Global Rank

**Presentation Title:** Prediciting University Global Rank Category Using Machine Learning
**Team:** Yağız Efe Gökçe, Kerem Özcan, Murat Emre Doğan
**Target Duration:** ~10 Minutes

---

## 0:00 - 1:00 | Slide 1: Introduction (Title Slide)

**Speaker: Murat**
"Good afternoon Professor.
We are your students from section 2, and we are excited to present our final project for the Machine Learning course.

Our project is titled **'Predicting University Global Rank Category Using Machine Learning'**.

I am **Murat**, and I am joined today by my colleagues **Kerem** and **Yağız**. Over the past semester, we have been working hard to understand the complex world of university rankings. We wanted to see if we could strip away the prestige and the marketing, and use pure data to predict where a university stands on the global stage.

We have built a comprehensive machine learning pipeline that not only predicts these rankings with high accuracy but also explains *why* a university is ranked the way it is."

---

## 1:00 - 2:30 | Slide 2: Problem Statement & Motivation

**Speaker: Murat**
"Let's dive into the problem we are addressing.
Every year, universities around the world hold their breath waiting for the 'Big Three' rankings: QS, Times Higher Education, and ARWU.

**Why? Why does this matter?**
A university's rank is not just a vanity metric. It has real-world consequences.

1. **Student Enrollment:** Top-ranked universities attract the best students from around the globe.
2. **Funding:** Governments and private investors often allocate billions of dollars in grants based on these tiers.
3. **Reputation:** Being in the 'Top 100' is a badge of honor that drives alumni donations and industry partnerships.

**The Challenge:**
The problem is that the methodologies used to calculate these rankings can be incredibly opaque. They often use complex, proprietary formulas, or subjective measures like 'reputation surveys'. This leaves many universities guessing at what they actually need to improve.

**Our Goal:**
Our goal was to demystify this process. We wanted to build a **data-driven system** that takes objective, hard metrics—like how many papers a faculty publishes or how employable the graduates are—and predicts the university's ranking category.
Specifically, we treat this as a classification problem: Can we predict if a university is **Elite (Top 100)**, **High (Rank 101-500)**, or **Average (Rank > 500)**?"

---

## 2:30 - 3:30 | Slide 3: Dataset Overview

**Speaker: Murat**
"To build this system, we needed high-quality data. We chose the **Center for World University Rankings (CWUR)** dataset.

Unlike some other datasets that rely heavily on surveys, CWUR is known for relying on **objective indicators**.
Our dataset contains data on approximately **2,200 universities** worldwide.

It includes **13 distinct features**, which act as our independent variables. These include:

* **Quality of Education:** Measured by the number of alumni who have won major international awards (like Nobel Prizes).
* **Alumni Employment:** The number of alumni who have held CEO positions at the world's top companies.
* **Quality of Faculty:** Based on distinctions and medals held by faculty members.
* And crucial research metrics like **Publications**, **Influence**, and **Citations**.

As I mentioned, our Target Variable is the 'Ranking Category', which groups these specific ranks into broad, actionable tiers."

---

## 3:30 - 5:00 | Slide 4: Data Preprocessing & EDA

**Speaker: Murat**
"Raw data is rarely ready for Machine Learning, and ours was no different. We applied a rigorous preprocessing pipeline.

First, **Missing Values:** Real-world data often has gaps. We found missing values in the 'broad_impact' column. Instead of dropping these valuable rows, we imputed them using the **median value**. We chose the median because it is robust to outliers—and in university data, you definitely have outliers like Harvard or MIT that act as massive spikes.

Second, **Encoding:** We had categorical data like 'Country'. Machines can't understand 'USA' or 'Turkey', so we encoded these into numerical values.

Third, **Scaling:** This was critical. Some features, like 'Number of Publications', might be in the thousands, while 'Score' is between 0 and 100. To prevent the model from being biased toward larger numbers, we scaled all features using **StandardScaler**, bringing them to a mean of 0 and variance of 1.

**Exploratory Data Analysis (EDA):**
When we visualized the data, we found something fascinating. There is an incredibly strong correlation between **Publications**, **Citations**, and **Influence**. They move together almost perfectly. This gave us our first hint: Research output is likely the dominant factor in this ranking system.

Now, I will hand it over to **Kerem** to explain how we modeled this data."

---

## 5:00 - 6:30 | Slide 5: Ensemble Learning Models (Overview)

**Speaker: Kerem**
"Thank you, Murat.

For this project, we decided to focus on **Ensemble Learning** techniques.
In machine learning, a single model can sometimes be 'weak'—it might overfit to the training data or miss complex patterns. Ensemble learning solves this by combining multiple models to separate the signal from the noise. It’s like asking a committee of experts for a decision instead of just one person.

We implemented three specific ensemble algorithms:

1. **Random Forest:** This builds hundreds of decision trees and averages their outputs. It’s excellent for handling non-linear data and rarely overfits.
2. **Gradient Boosting:** This builds trees sequentially, where each new tree tries to correct the errors of the previous one.
3. **XGBoost (Extreme Gradient Boosting):** This is the gold standard in competitions today. It’s a highly optimized version of gradient boosting designed for speed and performance."

---

## 6:30 - 7:30 | Slide 6: Baseline Results (Before Tuning)

**Speaker: Kerem**
"We started by training these models with their default settings to establish a baseline. We used an 80/20 train-test split to ensure we were evaluating on unseen data.

The results were impressive right out of the gate:

* **Random Forest** achieved an accuracy of **97.50%**.
* **Gradient Boosting** was very similar at **97.27%**.
* **XGBoost** took the lead, achieving **97.95%** accuracy.

To put that in perspective: Out of every 100 universities we classified, XGBoost correctly identified the tier for 98 of them. This is an incredibly high level of precision for social science data."

---

## 7:30 - 8:30 | Slide 7: Hyperparameter Tuning & Slide 8: After Tuning

**Speaker: Kerem**
"We didn't want to stop there. We wondered: Can we get to 99%?
We treated this as an optimization problem. We used **RandomizedSearchCV** to tune the hyperparameters of our Random Forest model.

We experimented with:

* **n_estimators:** How many trees to build?
* **max_depth:** How deep (and complex) should each tree be?
* **min_samples_split:** How much data do we need to justify a new decision branch?

**The Result:**
Surprisingly, after extensive tuning, our accuracy actually dropped slightly to **96.82%**.
Why? It turns out the default parameters in the Scikit-Learn library (which allow trees to grow as deep as needed) were already near-optimal for this dataset. By forcing constraints (like limiting depth to avoiding overfitting), we may have actually underfitted the complex 'Elite' universities slightly.

Sometimes, simple usage is best.

I'll now pass the mic to **Yağız** to wrap up our findings."

---

## 8:30 - 9:30 | Slide 9: Final Results & Comparison

**Speaker: Yağız**
"Thanks, Kerem.

So, bringing it all together. We compared Logistic Regression, Decision Trees, KNN, and our three Ensemble models.

The winner was clear: **XGBoost Classifier**.

* It had the Highest **Accuracy**.
* The Highest **F1-Score** (which matters because our classes were slightly imbalanced).
* And an **AUC Score of 0.999**—which is essentially perfect distinguishability.

This confirms that for tabular, university performance data, tree-boosting algorithms are the state-of-the-art solution."

---

## 9:30 - 10:15 | Slide 10: Feature Importance

**Speaker: Yağız**
"One of the best things about these models is that they are interpretable. We asked the model: 'Which features actually mattered?'

The results were definitive:

1. **Publications**
2. **Citations**
3. **Influence**

These three features dominated the decision-making process. Factors like 'Alumni Employment' or 'Quality of Education' (based on awards) mattered, but far less.
**The Insight:** If a university wants to improve its global rank quickly, it shouldn't just focus on teaching—it needs to publish. Research output is the currency of global rankings."

---

## 10:15 - 11:30 | Slide 11: Conclusion, Future Work & Thank You

**Speaker: Yağız**
"To conclude:
We successfully demonstrated that University Rankings are not magic; they are mathematical. We can reverse-engineer them with 98% accuracy.
We showed that **Ensemble methods** are far superior to simple linear models for this task.
And we provided a tool that Universities could essentially use to 'audit' themselves.

**Future Work:**
If we had more time, we would:

1. **Expand the Dataset:** Include data from 2024 and 2025 to track trends over time.
2. **Add New Features:** Incorporate financial data—like university endowments or research budgets—to see the return on investment.
3. **Deep Learning:** Experiment with Neural Networks to see if they can find even more subtle patterns.

That brings us to the end of our presentation. We hope you found our analysis insightful.
Thank you for your time and attention—especially to our Professor for guiding us through this course.

**We are now happy to answer any questions you may have.**"
