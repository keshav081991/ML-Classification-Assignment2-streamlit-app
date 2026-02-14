# ML-Classification-Assignment2-streamlit-app
Machine Learning classification assignment project implementing 6 ML models with Streamlit web deployment for prediction and model comparison.

Problem Statement : Banks run marketing campaigns to encourage customers to subscribe to term deposit schemes, but not every customer accepts the offer. Predicting which customers are more likely to subscribe helps the bank target the right customers and reduce marketing costs.
Dataset description:

The dataset includes 11,162 records and 17 input features(Including Target) such as:
•	Demographic information: age, job, marital status, education
•	Financial information: account balance, housing loan, personal loan
•	Campaign information: contact type, number of contacts (campaign), previous campaign outcome, call duration
•	Target variable: deposit (Yes / No), which indicates whether the customer subscribed to the term deposit

Models used:
Model	                Accuracy	Precision	Recall	   F1	       AUC	      MCC
5	XGBoost	            0.836095	0.913359	0.815482	0.849110	0.831956	0.672641
4	Random Forest	      0.827138	0.905728	0.801594	0.848172	0.824226	0.655459
0	Logistic Regression	0.793103	0.869568	0.782446	0.785380	0.783910	0.585462
2	KNN	                0.782803	0.851180	0.779808	0.760075	0.769815	0.564449
3	Naive Bayes	        0.770264	0.832118	0.745567	0.788191	0.766287	0.541546
1	Decision Tree     	0.769816	0.768687	0.767667	0.743205	0.755238	0.538353

## ML Model Name         :             Observation about model performance
'''
1. Logistic Regression  :          Logistic Regression showed good baseline performance with balanced precision and recall, but it could not capture 
                                   complex relationships in the dataset compared to ensemble models.
2. Decision Tree        :          Decision Tree produced lower performance because it tends to overfit the training data and does not generalize 
                                   as well as ensemble methods.
3. kNN                  :          kNN achieved moderate accuracy, but its performance depends heavily on the choice of neighbors and feature scaling, 
                                   which slightly affected the results.
4. Naive Bayes          :          Naive Bayes performed reasonably but gave slightly lower results because it assumes independence among features, 
                                   which is not always true in real datasets.
5. Random Forest (Ensemble) :      Random Forest improved the performance compared to single models by combining multiple decision trees, 
                                   resulting in better accuracy and MCC values.
6. XGBoost (Ensemble)	     :      XGBoost achieved the best performance among all models with the highest accuracy, AUC, and MCC, showing its strong 
                                  ability to capture complex patterns. Therefore, it was selected as the final deployed model.
'''
