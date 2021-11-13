#!/usr/bin/env python
# coding: utf-8

# ### Getting Started - Extraction & Transformations
# 
# #### Data:
# There is both a sqlite db, and a generic sql dump under *db/* so you can use whichever db you prefer. 
# - student_profiles contains information about a particular student (potential or actual) who has registered interested in a particular course, uniquely identified by a profile_id. For simplicity's sake, profiles are 1:1 with course and school, and the course and school the student expressed interested in is in their profile. 
# - course_applications contains one record for each course and school a student applied to.
# - course_enrollments contains one record for each course and school a student enrolled in.
# - student_profiles marked with is_test == true are the unannotated test sample. Their application/enrollment state is excluded to provide a basis for analyzing fit. 
# 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('reload_ext', 'autoreload')

# Libraries
import sqlite3
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import catboost
import xgboost as xgb

from utilities import evaluate_model, save_pickle_model, load_pickle_model


# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath("__file__")))
DB_DIR = os.path.join(BASE_DIR, "db")
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# In[2]:


# SQLite connection.
conn = sqlite3.connect(f"{DB_DIR}/funnel.db")

# Check tables in the database
tables = pd.read_sql_query("SELECT * FROM sqlite_master WHERE type='table';", conn)
tables


# In[3]:


# Extract student_profiles
student_profiles = pd.read_sql_query("SELECT * FROM student_profiles;", conn)
student_profiles.sample(3)


# In[4]:


# Extract profile_applications
profile_applications = pd.read_sql_query("SELECT * FROM profile_applications;", conn)
# profile_applications.columns = [f"{x}_applied" if x not in ['profile_id'] else x for x in profile_applications.columns]
profile_applications['has_applied'] = 1
print(profile_applications.shape)
profile_applications.sample(3)


# In[5]:


# Extract profile_enrollments
profile_enrollments = pd.read_sql_query("SELECT * FROM profile_enrollments;", conn)
# profile_enrollments.columns = [f"{x}_enrolled" if x not in ['profile_id'] else x for x in profile_enrollments.columns]
profile_enrollments['has_enrolled'] = 1
print(profile_enrollments.shape)
profile_enrollments.sample(3)


# In[6]:


# Merge and Clean data.
df = pd.merge(left=student_profiles,
              right=profile_applications[['profile_id', 'has_applied']],
              how='left',
              on=['profile_id']
             )

df = pd.merge(left=df,
              right=profile_enrollments[['profile_id', 'has_enrolled']],
              how='left',
              on=['profile_id']
             )

for col in ['has_applied', 'has_enrolled']:
    df[col] = df[col].fillna(0).astype(int)

df['undergrad_grade_points'] = df['undergrad_grade_points'].astype(int)
print(df.info())

df.sample(3)


# ###  1. Imagine that we want to segment our marketing campaigns at a country level by performance. 
# 
# Build a model that returns which countries we may want to consider together based purely on their performance.
# 
# - Describe your model and what led you to this approach.
# 

# In[7]:


# Country Wise conversion rate to get the performance
country_wise_df = pd.pivot_table(df,
                                 index=['country', 'is_test'],
                                 # columns=['is_test'],
                                 values=['has_applied', 'has_enrolled'],
                                 aggfunc={
                                     'has_enrolled': np.sum,
                                     'has_applied': np.sum,
                                 }
                                ).reset_index()

country_wise_df['conversion_rate'] = (country_wise_df['has_enrolled'] / country_wise_df['has_applied'])*100
country_wise_df.sort_values(['conversion_rate'], ascending=False)


# In[8]:


data = country_wise_df.loc[(country_wise_df['is_test']=='false'), ['conversion_rate', 'country']]
_X = data.drop(['country'], axis=1)
y = data['country']

# Data Processing
X = _X.values
X = np.nan_to_num(X)

sc = StandardScaler()

cluster_data = sc.fit_transform(X)

# Modeling
clusters = 3
model = KMeans(init = 'k-means++', 
               n_clusters = clusters, 
               n_init = 12)
model.fit(X)

labels = model.labels_

data['cluster_num'] = labels
data


# In[9]:


# Get Cluster-wise Performance Label
cluster_label_mapping = data.groupby('cluster_num').mean().sort_values('conversion_rate').reset_index()
performance_labels = ['low', 'medium', 'high']
cluster_label_mapping['performance'] = performance_labels
cluster_label_mapping


# In[10]:


# Train data's performance by Country

data = pd.merge(left=data,
                right=cluster_label_mapping[['cluster_num', 'performance']],
                how='left',
                on=['cluster_num']
               )

high_performing_countries_train = data.loc[(data['performance'] == 'high'), 'country'].unique()

data


# In[11]:


# Test data's performance by Country

data = country_wise_df.loc[(country_wise_df['is_test']=='true'), ['conversion_rate', 'country']]
_X = data.drop(['country'], axis=1)

# Data Processing
X = _X.values
X = np.nan_to_num(X)

sc = StandardScaler()

cluster_data = sc.fit_transform(X)

data['cluster_num'] = model.predict(X)
data = pd.merge(left=data,
                right=cluster_label_mapping[['cluster_num', 'performance']],
                how='left',
                on=['cluster_num']
               )

high_performing_countries_test = data.loc[(data['performance'] == 'high'), 'country'].unique()

data


# In[12]:


print(f"High performing countries in train dataset: {high_performing_countries_train}")
print(f"High performing countries in test dataset: {high_performing_countries_test}")


# ### 2. Imagine that we want to predict which of these prospective students will apply.
# 
# 
# Build a model to predict whether a prospective student will apply to a program (as in, will have a record in student_applications). 
# 
# - Describe your model and what led you to this approach.
# - What are its most important features?
# - What metrics are appropriate for validation on a problem like this? What are the most important tradeoffs.
# - Measure and visualize your models performance using whatever metric you think is most relevant.
# 
# Please send back code, models, annotations for the test sample, and a short document (markdown, .doc, google doc, etc) addressing the above questions.
# 

# In[13]:


# Splitting train and test dataset
train = df.loc[(df['is_test'] == "false")]
test = df.loc[(df['is_test'] == "true")]
print(f"train: {train.shape}\ntest: {test.shape}")
train.sample()


# #### CatBoost

# In[14]:


X = train.drop(['has_enrolled', 'has_applied', 'profile_id', 'is_test'], axis=1)
y = train['has_applied']

print(y.value_counts()) # Imbalanced dataset


# In[15]:


X = train.drop(['has_enrolled', 'has_applied', 'profile_id', 'is_test'], axis=1)
y = train['has_applied']

from imblearn.over_sampling import SMOTENC

# Nominal & Continuous
smotenc = SMOTENC([0,1,2,3,4,5],random_state = 42)
X_oversample, y_oversample = smotenc.fit_resample(X, y)
print(X_oversample.shape, y_oversample.shape)

# Splitting train and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_oversample, y_oversample, test_size=0.3, random_state=42)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)


# In[16]:


# Baseline CatBoost Classifier

cat_features = ['school', 'course_name', 'country', 
                'os_name', 'experience', 'lead_initiated']
cat = catboost.CatBoostClassifier(random_seed=42, iterations=5000, early_stopping_rounds=200, eval_metric='AUC')
cat_model = cat.fit(X_train, y_train,
                     eval_set=(X_val, y_val),
                     cat_features=cat_features,
                     plot=True,
                     verbose=0,
                    use_best_model=True,
                   )


# In[17]:


X_test = test.drop(['has_enrolled', 'has_applied', 'profile_id', 'is_test'], axis=1)
y_test = test['has_enrolled']

y_pred = cat.predict(X_test)

cat_df = pd.concat([test.reset_index(drop=True), pd.Series(y_pred, name='y_pred_cat_auc')], axis=1)
cat_df


# In[18]:


cat_metrics = evaluate_model(y_test, y_pred, "CatBoost")
cat_model_path = f"{MODEL_DIR}/catboost_baseline_auc.pkl"
save_pickle_model(cat_model, cat_model_path)

cat_metrics


# In[19]:


# Features' importance

feat_imp = cat_model.get_feature_importance(prettified=True)
print(feat_imp)
# Plotting top 5 features' importance

# plt.figure(figsize = (12,8))
sns.barplot(feat_imp['Importances'],feat_imp['Feature Id'], orient = 'h')
plt.show()


# In[20]:


train_data = catboost.Pool(X_train, cat_features=cat_features)

interaction = cat_model.get_feature_importance(train_data, type="Interaction")
column_names = X_train.columns.values 
interaction = pd.DataFrame(interaction, columns=["feature1", "feature2", "importance"])
interaction.feature1 = interaction.feature1.apply(lambda l: column_names[int(l)])
interaction.feature2 = interaction.feature2.apply(lambda l: column_names[int(l)])
interaction.head(20)


# In[21]:


cat_model.get_all_params()


# In[22]:


# # Preforming a Random Grid Search to find the best combination of parameters

# grid = {'iterations': [5000],
#         'learning_rate': [0.1, 0.2, 0.05],
#         'depth': [4, 6, 10, 16],
# #         'l2_leaf_reg': [1, 3, 5, 9]
#        }

# final_model = catboost.CatBoostClassifier(cat_features=cat_features,
#                                           verbose=False
#                                          )
# randomized_search_result = final_model.randomized_search(grid,
#                                                    X=X_train,
#                                                    y=y_train,
#                                                    verbose=False,
#                                                    plot=True)


# In[23]:


# Final Cat-Boost Regressor

params = {'iterations': 10000,
          'learning_rate': 0.05,
          'depth': 10,
#           'l2_leaf_reg': 1,
          'eval_metric':'F1',
          'early_stopping_rounds': 200,
          'verbose': 200,
          'random_seed': 42,
          'cat_features': cat_features
         }

cat_f = catboost.CatBoostClassifier(**params)
cat_model_f = cat_f.fit(X_train, y_train,
                     eval_set=(X_val, y_val),
                     plot=True,
                     verbose=False,
                     use_best_model=True)

catf_pred = cat_model_f.predict(X_test)


# In[24]:


cat_df = pd.concat([cat_df.reset_index(drop=True), pd.Series(catf_pred, name='y_pred_cat_f1')], axis=1)
cat_df


# In[25]:


catf_metrics = evaluate_model(y_test, catf_pred, 'catboost_grid')
print(catf_metrics['classification_report'])

catf_model_path = f"{MODEL_DIR}/catboost_grid_f1.pkl"
save_pickle_model(cat_model_f, catf_model_path)

catf_metrics


# In[26]:


# Features' importance

feat_imp = cat_model_f.get_feature_importance(prettified=True)
print(feat_imp)
# Plotting top 5 features' importance

# plt.figure(figsize = (12,8))
sns.barplot(feat_imp['Importances'],feat_imp['Feature Id'], orient = 'h')
plt.show()


# In[27]:


# Final Cat-Boost Regressor

params = {'iterations': 10000,
          'learning_rate': 0.05,
          'depth': 10,
#           'l2_leaf_reg': 1,
          'eval_metric': 'Precision',
          'early_stopping_rounds': 200,
          'verbose': 200,
          'random_seed': 42,
          'cat_features': cat_features
         }

cat_r = catboost.CatBoostClassifier(**params)
cat_model_r = cat_r.fit(X_train, y_train,
                     eval_set=(X_val, y_val),
                     plot=True,
                     verbose=False,
                     use_best_model=True)

catr_pred = cat_model_r.predict(X_test)


# In[28]:


cat_df = pd.concat([cat_df.reset_index(drop=True), pd.Series(catr_pred, name='y_pred_cat_precision')], axis=1)

cat_df.to_csv(f"{DATA_DIR}/test_annotation_catboost.csv", index=False)
cat_df


# In[29]:


catr_metrics = evaluate_model(y_test, catr_pred, 'catboost_grid')
print(catr_metrics['classification_report'])

catr_model_path = f"{MODEL_DIR}/catboost_grid_precision.pkl"
save_pickle_model(cat_model_r, catr_model_path)

catr_metrics


# In[30]:


# Features' importance

feat_imp = cat_model_r.get_feature_importance(prettified=True)
print(feat_imp)
# Plotting top 5 features' importance

# plt.figure(figsize = (12,8))
sns.barplot(feat_imp['Importances'],feat_imp['Feature Id'], orient = 'h')
plt.show()


# #### XGBoost

# In[31]:


X_cat = df.select_dtypes(include=['object'])
X_enc = X_cat.copy()

X_enc = pd.get_dummies(X_enc, columns=cat_features)
X_enc


# In[32]:


data = pd.concat([df.drop(cat_features + ['is_test', 'profile_id'], axis=1), X_enc], axis=1)
train = data.loc[(data['is_test'] == 'false')]
test = data.loc[(data['is_test'] == 'true')]
train.shape, test.shape


# In[33]:


X = train.drop(['profile_id', 'is_test', 'has_enrolled', 'has_applied'], axis=1)
y = train['has_applied']

from imblearn.over_sampling import SMOTE

# Oversampling the data
smote = SMOTE(random_state = 42)
X_oversampled, y_oversampled = smote.fit_resample(X, y)

X_train, X_val, y_train, y_val = train_test_split(X_oversampled, y_oversampled, test_size=0.3, random_state=42)
X_test, y_test = test.drop(['has_enrolled', 'has_applied'], axis=1), test['has_applied']

# X_train = X_train.drop(['profile_id', 'is_test'], axis=1)
# X_val = X_val.drop(['profile_id', 'is_test'], axis=1)
X_test = X_test.drop(['profile_id', 'is_test'], axis=1)

# XGB does not entertain '<' ',' '>' in column names
X_train.columns = X_train.columns.astype(str).str.replace('<', 'lt').str.replace('>', 'gt').str.replace(',', '')
X_val.columns = X_val.columns.astype(str).str.replace('<', 'lt').str.replace('>', 'gt').str.replace(',', '')
X_test.columns = X_test.columns.astype(str).str.replace('<', 'lt').str.replace('>', 'gt').str.replace(',', '')

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)


# In[34]:


xgb_clf = xgb.XGBClassifier(random_state=42, learning_rate=0.2)
xgb_clf.fit(X_train, y_train,
           eval_set=[(X_val, y_val)],
            eval_metric='logloss',
            verbose=False)


# In[35]:


y_pred_xgb = xgb_clf.predict(X_test)

xgb_metrics = evaluate_model(y_test, y_pred_xgb, 'XGBClassifier')
print(xgb_metrics['classification_report'])

xgb_model_path = f"{MODEL_DIR}/xgboost_baseline.pkl"
save_pickle_model(xgb_clf, xgb_model_path)

xgb_metrics


# In[36]:


xgb_df = pd.concat([test.reset_index(drop=True), pd.Series(y_pred_xgb, name='y_pred_xgb')], axis=1)
xgb_df.loc[(xgb_df['has_enrolled'] != xgb_df['y_pred_xgb'])]

xgb_df.to_csv(f"{DATA_DIR}/test_annotation_xgboost.csv", index=False)


# In[37]:


# XGBoost Feature Importance
feat_imp = xgb_clf.feature_importances_
print(feat_imp)
# Plotting top 5 features' importance

fig, ax = plt.subplots(figsize=(20, 10))
xgb.plot_importance(xgb_clf, ax=ax)
plt.show()


# ### Model Evaluation

# In[38]:


# Get ROC-AUC curve
fpr_cat = cat_metrics.get("fpr")
tpr_cat = cat_metrics.get("tpr")
thresholds_cat = cat_metrics.get("thresholds")
auc_cat = cat_metrics.get("roc_auc_score")

fpr_catf = catf_metrics.get("fpr")
tpr_catf = catf_metrics.get("tpr")
thresholds_catf = catf_metrics.get("thresholds")
auc_catf = catf_metrics.get("roc_auc_score")

fpr_catr = catr_metrics.get("fpr")
tpr_catr = catr_metrics.get("tpr")
thresholds_catr = catr_metrics.get("thresholds")
auc_catr = catr_metrics.get("roc_auc_score")

fpr_xgb = xgb_metrics.get("fpr")
tpr_xgb = xgb_metrics.get("tpr")
thresholds_xgb = xgb_metrics.get("thresholds")
auc_xgb = xgb_metrics.get("roc_auc_score")


plt.plot(fpr_cat, tpr_cat,'r-',label = 'CatBoost AUC: %.3f'%auc_cat)
plt.plot(fpr_catf, tpr_catf,'b-', label= 'CatBoostF1 AUC: %.3f'%auc_catf)
plt.plot(fpr_catr, tpr_catr,'g-', label= 'CatBoostRecall AUC: %.3f'%auc_catr)
plt.plot(fpr_xgb, tpr_xgb,'k-',label='XGBoost AUC: %.3f'%auc_xgb)
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[39]:


# Compare all the metrics based on test dataset
pd.DataFrame(({"xgb_metrics":xgb_metrics, "cat_metrics_auc":cat_metrics, "cat_metrics_f1":catf_metrics, "cat_metrics_precision":catr_metrics}))


# In[ ]:




