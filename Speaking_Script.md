# Speaking Script: Predicting University Global Rank

**Total Estimated Time:** ~10 Minutes
**Speakers:** 3 (Assigned as Speaker 1, 2, and 3)

---

## 0:00 - 0:30 | Slide 1: Title Slide

**Murat:**
"Good morning Professor Femilda. We are your students from Section 2, and today we’re excited to present our project: **Predicting University Global Rank Using Machine Learning**.

My name is **Murat**, and I’m joined by my classmates **Yağız** and **Kerem**.

In this project, we aimed to demystify how global universities are ranked by building a machine learning system that can predict these rankings based on objective performance data."

---

## 0:30 - 2:00 | Slide 2: Problem Statement

**Murat:**
"So, why did we choose this topic?

University rankings are more than just a number. They directly influence millions of dollars in funding, attract top-tier talent, and determine student enrollment numbers globally. Being a 'Top 100' university is a massive economic driver.

However, the problem is that ranking methodologies are often **opaque, complex, and subjective**. Different organizations use wildly different formulas, leaving institutions guessing at what actually moves the needle.

**Our Goal** was to solve this by building a transparent, data-driven system. We wanted to see if we could purely use objective metrics—like publications and employment rates—to accurately predict a university's standing. Essentially, can we reverse-engineer the ranking process using Machine Learning?"

---

## 2:00 - 3:30 | Slide 3: Dataset & Features

**Murat:**
"To do this, we turned to the **Center for World University Rankings (CWUR)** dataset, sourced from Kaggle.

This dataset provided us with comprehensive data on over **2,200 universities**. It’s a rich dataset that allowed us to look at several key features:

* **Quality of Education:** Measured by the number of alumni who have won major international awards.
* **Alumni Employment:** The number of alumni who have held CEO positions at major companies.
* **Quality of Faculty:** Based on faculty distinctions and awards.
* And finally, **Research Output:** Which includes total publications and the number of highly cited papers.

We set up two target variables for our models:

1. A continuous **Score** from 0 to 100 for precise regression.
2. A **Ranking Category** (Elite, High, Average) to see if we could simply classify universities into broad tiers.

Now, I’ll hand it over to **Kerem** to walk you through how we actually built these models."

---

## 3:30 - 5:00 | Slide 4: Methodology

**Kerem:**
"Thanks, Murat.

Before feeding data into our models, we had to ensure it was clean and usable. Our **preprocessing pipeline** involved imputing missing values to retain as much data as possible, encoding categorical data like 'Country', and scaling all numerical features so that models wouldn't be biased by large numbers, like raw publication counts.

For the modeling phase, we tested three distinct algorithms to find the best fit:

1. **Linear & Logistic Regression:** We used these as our baseline to establish a minimum performance benchmark.
2. **Random Forest:** This allowed us to capture non-linear relationships and interactions between features.
3. **XGBoost:** Which is a powerful gradient boosting algorithm known for its high performance in structured data competitions.

We evaluated our Regression models using **RMSE** and **R-squared**, and our Classification models using **Accuracy** and **F1 Score**."

---

## 5:00 - 6:30 | Slide 5: Regression Results

**Kerem:**
"Moving on to our results. For the Regression task—predicting the exact score—**XGBoost** was the clear winner.

The data speaks for itself:

* We achieved an **R-squared value of 0.9809**. This is an exceptionally high score, meaning our model explains over 98% of the variance in the university scores.
* Our **RMSE was just 1.02**, indicating that our predictions are typically within 1 point of the actual score on a 100-point scale.

Interestingly, our feature importance analysis revealed that **Publications** and **Citations** were the most dominant predictors. This suggests that for this specific ranking system, research output is the single biggest factor driving a university's rank."

---

## 6:30 - 7:30 | Slide 6: Classification Results

**Kerem:**
"We saw similar success with our Classification task.

Again, **XGBoost** outperformed the others, achieving an **Accuracy of 97.95%** and an **F1 Score of roughly 0.98**.

The model was almost perfect at separating 'Elite' universities from 'Average' ones. This confirms that there is a very distinct, quantifiable threshold that separates top-tier institutions from the rest of the pack.

I'll now pass the mic to **Yağız** to show you how this looks in practice."

---

## 7:30 - 9:00 | Slide 7: Live Demo (Streamlit App)

**Yağız:**
"Thank you, Kerem.

We didn't just want to leave these models in a notebook. We wanted to make them accessible. So, we built an interactive **Streamlit Web Application**.

*(Gesture to screen/demo)*

As you can see here, the app allows any user—universities, students, or analysts—to adjust various parameters in real-time.

* You can use these sliders to change the number of publications, improve the alumni employment score, or increase faculty quality.
* Instantly, the model recalculates the predicted **Score** and assigns a **Ranking Category**.

It also visually compares the user's input against global averages, giving immediate feedback on where an institution stands compared to the competition."

---

## 9:00 - 9:45 | Slide 8: Conclusion

**Yağız:**
"To wrap things up, our project successfully demonstrated that Machine Learning can accurately reverse-engineer university rankings.

**Why does this matter?**
For universities, this is a strategic tool. Instead of guessing, an administration can look at our model and say, 'If we increase our research output by 10%, our global rank will likely jump by X positions.' It allows for targeted, data-backed decision making.

**Looking ahead**, we plan to improve this system by integrating real-time data from APIs and potentially adding 'social sentiment' analysis to capture the public perception of these universities."

---

## 9:45 - 10:00 | Slide 9: Questions?

**Yağız:**
"We’re happy to take any questions you might have about the data, the models, or the application.

**Thank you!**"
