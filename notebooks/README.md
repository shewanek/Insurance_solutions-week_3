# ACIS Marketing Analytics Project

This project is part of a challenge to perform data analysis and build machine learning models using historical insurance claim data for AlphaCare Insurance Solutions (ACIS) in South Africa. The aim is to optimize marketing strategies and identify low-risk targets for potential premium reductions.



---

### **Key Takeaways**

1. **Claims Prediction Performance**:
   - All models achieved perfect performance metrics (MSE = 0.0 and R² = 1.0), suggesting a possible issue with overfitting. Further investigation into model robustness and data variance is recommended.

2. **Premium Prediction Performance**:
   - **XGBoost** stands out with the best performance (MSE = 36.8, R² = 0.998), significantly outperforming other models. This highlights its ability to capture complex, non-linear relationships in the data effectively.
   - **Linear Regression** underperformed, indicating that simpler models may not suffice for this task due to the non-linear nature of the data.

3. **Feature Importance**:
   - **Calculated Premium Per Term** and **Premium Per Unit Sum Insured** are critical in determining premiums, with high feature importance scores. 
   - Features such as **Transaction Month** and **Postal Code** have minimal impact, suggesting that more focus should be placed on high-impact features for optimizing premium predictions.

4. **Actionable Insights**:
   - Utilize **XGBoost** for premium prediction to leverage its superior performance.
   - Reevaluate the claims prediction models to address potential overfitting.
   - Focus on key features like **Calculated Premium Per Term** and **Premium Per Unit Sum Insured** to refine pricing strategies and marketing efforts.

These insights will guide AlphaCare Insurance Solutions in enhancing its marketing strategies, optimizing pricing models, and ultimately improving its competitive positioning in the market.

---
