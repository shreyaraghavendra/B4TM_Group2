# %%
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# %%
# load data
df_X = pd.read_csv('../data/X.csv', index_col=0)
df_y = pd.read_csv('../data/y.csv', index_col=0)

X = df_X.to_numpy()
y = df_y.values.ravel()  # 0 is HER2+, 1 is HR+, 2 is Triple Negative

# %% [markdown]
# ## Lasso regression

# %%
# First Logistic Regression with L1 penalty
logreg1 = LogisticRegression(solver='liblinear', multi_class='ovr', C=1, max_iter=100, penalty='l1')
# logreg1.fit(X_train, y_train)
logreg1.fit(X, y)

# Create a boolean mask for features with non-zero coefficients in any class
features_first_round = np.any(logreg1.coef_ != 0, axis=0)
selected_features_first_round = df_X.columns[features_first_round].tolist()

# Apply the mask to reduce X to significant features only
# X_reduced_train = X_train[:, features_first_round]
# X_reduced_test = X_test[:, features_first_round]
X_reduced = X[:, features_first_round]

# Second Logistic Regression with L1 penalty on the reduced feature set
logreg2 = LogisticRegression(solver='liblinear', multi_class='ovr', C=1, max_iter=100, penalty='l1')
logreg2.fit(X_reduced, y)

# Identify features with non-zero coefficients in the second round
features_second_round = np.any(logreg2.coef_ != 0, axis=0)
selected_features_second_round = df_X.columns[features_first_round][features_second_round].tolist()

# Output selected features
print("Selected features in the first round:", selected_features_first_round, 
      "\n length:", len(selected_features_first_round))
print("Selected features in the second round:", selected_features_second_round,
      "\n length:", len(selected_features_second_round))

# # Evaluate model performance on the test set with reduced features
# y_pred_first = logreg1.predict(X_test)
# accuracy_first = accuracy_score(y_test, y_pred_first)
# print("Accuracy on the test set (First Model):", accuracy_first)

# y_pred_second = logreg2.predict(X_reduced_test)
# accuracy_second = accuracy_score(y_test, y_pred_second)
# print("Accuracy on the test set (Second Model):", accuracy_second)


# %% [markdown]
# ## RFECV LASSO

# %%
estimator = LogisticRegression(solver='liblinear', multi_class='ovr', C=1, max_iter=100, penalty='l1')  # best parameters from grid search
selector = RFECV(estimator, step=1, cv=StratifiedKFold(5), scoring='accuracy')

# Apply RFECV across the whole dataset (no need to split here as CV does it)
selector = selector.fit(X, y)

print("Optimal number of features: ", selector.n_features_)
selected_features = np.where(selector.support_)[0]
print("Selected features: ", selected_features)

# %% [markdown]
# ## Multiple Random Splits LASSO

# %%
num_splits = 10
selected_features_all = []

for seed in range(num_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    # Logistic Regression with L1 penalty
    model = LogisticRegression(solver='liblinear', penalty='l1', C=1, random_state=seed)
    model.fit(X_train, y_train)
    
    # Feature selection
    selector = SelectFromModel(model, prefit=True)
    selected_features = df_X.columns[selector.get_support()]
    selected_features_all.append(selected_features)

# Analyze feature stability
all_features = np.concatenate(selected_features_all)
features, counts = np.unique(all_features, return_counts=True)
feature_stability = pd.DataFrame({'Feature': features, 'Count': counts})

print(feature_stability.sort_values(by='Count', ascending=False))

# %%
# features in 5 or more splits
features_split = feature_stability[feature_stability['Count'] > 5].Feature.to_list()
len(features_split)
features_split

# %% [markdown]
# ## RFECV Random Forest

# %%
rf = RandomForestClassifier(n_estimators=20, random_state=42)
rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X, y)
print('Optimal number of features:', rfecv.n_features_)
selected_features = pd.Series(rfecv.support_, index=np.arange(X.shape[1]))
print("Selected features:\n", selected_features[selected_features == True].index.tolist())

# %% [markdown]
# ## Multiple Random Splits RF

# %%
num_splits = 10
selected_features_all = []

for seed in range(num_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    # Logistic Regression with L1 penalty
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    
    # Feature selection
    selector = SelectFromModel(model, prefit=True)
    selected_features = df_X.columns[selector.get_support()]
    selected_features_all.append(selected_features)

# Analyze feature stability
all_features = np.concatenate(selected_features_all)
features, counts = np.unique(all_features, return_counts=True)
feature_stability = pd.DataFrame({'Feature': features, 'Count': counts})

# %%
features_split = feature_stability[feature_stability['Count'] > 5].Feature.to_list()

# Find the common elements
common_features = set(selected_features).intersection(features_split)
print("Common features between RFECV and Multiple Splits for RF:", common_features,
      "\n length:", len(common_features))

# %%
features_split

# %% [markdown]
# ## Compare all feature lists

# %%
rf_rfe = [1874, 1877, 1883, 1884, 1887, 1896, 1902, 1906, 1909, 1910, 1956, 1972, 1973, 1976, 1977, 1997, 2004, 
          2018, 2019, 2026, 2027, 2058, 2063, 2065, 2085, 2112, 2113, 2116, 2124, 2140, 2167, 2183, 2184, 2185, 
          2207, 2211, 2213, 2288, 2433, 2435, 2464, 2465, 2481, 2485, 2492, 2528, 2529, 2546, 2547, 2593, 2603, 
          2609, 2614, 2641, 2643, 2655, 2669, 2677, 2690, 2691]

rf_split = [1242, 1672, 2036, 2183, 2184, 2207, 2211, 2221, 2223, 2675, 848, 855]

lasso_split = [1091, 118, 1656, 1678, 1900, 192, 2017, 2021, 2026, 2184, 2213, 2218, 2750, 2776, 2791,
               2817, 772, 791, 854]

lasso_rfe = [695, 791, 1061, 1559, 1643, 1656, 1678, 1900, 2024, 2026, 2184, 2210, 2213, 2750, 2825]

lasso_repeated = [118, 185, 189, 190, 192, 226, 230, 261, 415, 432, 486, 674, 695, 746, 765, 771, 772, 791, 
                  792, 818, 851, 854, 1009, 1035, 1059, 1061, 1062, 1091, 1110, 1160, 1191, 1243, 1559, 1562, 
                  1563, 1569, 1642, 1643, 1656, 1657, 1672, 1677, 1678, 1816, 1856, 1869, 1881, 1900, 1902, 1956, 
                  1973, 2017, 2021, 2024, 2026, 2058, 2099, 2184, 2189, 2210, 2213, 2218, 2219, 2276, 2328, 2329, 
                  2342, 2379, 2383, 2420, 2423, 2501, 2547, 2593, 2742, 2747, 2750, 2760, 2776, 2791, 2817, 2825, 2827, 2829]

# %% [markdown]
# List names	number of elements  
# LASSO-RFECV	15  
# LASSO-Relaxed	84  
# LASSO-Splits	19  
# RF-RFECV	60  
# RF-Splits	12  
# Overall number of unique elements	142  
# 
# Generated using: https://bioinformatics.psb.ugent.be/webtools/Venn/  
# 
# ![Venn Diagram](../figures/venn_result1714838963347537853.svg "Data Visualization")

# %%
from collections import Counter
combined_list = rf_rfe + rf_split + lasso_repeated + lasso_rfe + lasso_split
element_count = Counter(combined_list)

# Filter elements that appear in at least three lists
common_elements = [item for item, count in element_count.items() if count >= 4]
print(common_elements)

# %%
len(common_elements)

# %% [markdown]
# 2184: "NEUROD2", "AC087491.2", "PPP1R1B"  in all feature lists  
# BRCA1: 2208  not in feature lists  
# BRCA2: 1726   not in feature lists  
# HERC6, HERC5: 571 not in feature lists  
# HERPUD2: 921 not in feature lists  
# HERC4: 1354 not in feature lists  
# EGFR: 938 not in feature lists  


