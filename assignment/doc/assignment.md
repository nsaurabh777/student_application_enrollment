# Assignment Documentation

## Folder Structure
```
src: Code base
db: Input database
data: Output data
models: All model files
doc: documentation
venv: virtualenv
```


---

1. Clustering : Countries based on performance.

  - `Conversion Ratio = (Students Enrolled / Students Applied)*100`
  - This conversion ratio when clustered using K means algorithm, keeping the number of `clusters = 3`, gives the performance (High, medium and low) of the country
  - Once trained on the training set, we get a range of high performing countries
  - Use this algorithm to cluster unknown countries based on conversion ratio


2. Classification : Predict prospective students who will apply.

  - Used CatBoost & XGBoost classifiers to predict as there was enough data
  - There was high imbalance in the target variable
    ```
    0    27179
    1     2647
    ```
  - Hence had to oversample before training the data, using SMOTE over sampling technique. There are many more techniques, wherein we use combined strategy of over sampling and undersampling and pushing it via a pipeline to the model
  - Used Random Grid Search for hyperparameter optimization
  - Applied the parameters on various evaluation metrics
  - CatBoost has the capability of including categorical features
  - Even XGBoost has the capability, but does not really work as well as CatBoost does. Hence we need to encode the categorical to numerical values
  - We can use LabelEncoders and OneHotEncoders for this purpose. LabelEncoders are used when the data has ordinal values. Hence, used `pd.get_dummies()`, which is a OneHotEncoder in a way
  - This data is also over sampled and passed to the XGBoost algorithm
  - The data and the models are saved in the relevant folders in Base directory


  ---

  - Important Features:

    - country
    - undergrad_grade_points
    - course_name
    - experience

  ---

  - Appropriate Metrics for Validation

    - F1: Since it is a harmonic mean of Precision & Recall, you get a rough insight of your True Positive predictions based on the actuals and predicted positives
    - Recall: TP / (TP+FN). Out of all actual positives, how many of them are actually predicted as positive by the algorithm
    - True Negative Rate: TN / (TN + FP). Out of all actual negatives, how many of them are actually predicted as negative by the algorithm

    - The tradeoff is between Recall & TNR. In this example, we needed most of the students who will apply for a course, which means we needed less of False Negative cases (i.e. They actually applied for the course but the algorithm predicted it as the student did not apply). Hence `Recall` is the important evaluation metric in this case
    - Therefore, CatBoost algorithm (`model/catboost_baseline_auc.pkl`) is selected
    - Area Under the curve for this model is 0.58