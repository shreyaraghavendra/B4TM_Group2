# %% [markdown]
# ### Import packages and data

# %%
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# %%
df = pd.read_csv('../data/Train_call.txt', sep='\t', header=0, )
label = pd.read_csv('../data/Train_clinical.txt', sep='\t', index_col=0)
df

# %% [markdown]
# ### Data visualization

# %%
array_data = df.filter(regex='^Array\.\d+$')
# Extracting chromosome and position data
chromosome_position = df[['Chromosome', 'Start']].sort_values(by=['Chromosome', 'Start'])
# Mapping columns to simple ID numbers
array_data.columns = [int(col.split('.')[1]) for col in array_data.columns]

# Sort the columns by sample ID
array_data = array_data.reindex(sorted(array_data.columns), axis=1)

# Assuming 'df' and 'array_data' have already been defined and processed as before
chromosome_labels = df['Chromosome'].unique()  # Unique chromosome labels

# Creating the heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(array_data.T, cmap='viridis', cbar_kws={'label': 'Value'})

# Modify x-tick labels to show chromosome numbers only once per chromosome
# Create a new list for labels with positions
chromosome_positions = []
prev_chrom = None
for i, chrom in enumerate(df['Chromosome']):
    if chrom != prev_chrom:
        chromosome_positions.append(i)
        prev_chrom = chrom

# Set x-tick labels
ax.set_xticks([pos for pos in chromosome_positions])
ax.set_xticklabels(chromosome_labels)
plt.xticks(rotation=45)  # Rotate labels for better visibility

# Adding vertical lines after each chromosome
for pos in chromosome_positions[1:]:  # Skip the first position
    plt.axvline(x=pos, color='white')

plt.title('Heatmap of Array Values by Chromosome')
plt.xlabel('Chromosome Number')
plt.ylabel('Sample ID')
plt.show()


# %%
# correlation matrix for each gene
corr_matrix = df.iloc[4:].corr(method='pearson') 
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='viridis')
plt.title('Correlation matrix of raw data')

# %%
df_X = df.drop(['Chromosome', 'Start', 'End', "Nclone"], axis=1).T
X = np.array(df_X, dtype=np.float64)
le = LabelEncoder()

# Fit label encoder and return encoded labels
label['subgroup_encoded'] = le.fit_transform(label['Subgroup'])
y = label['subgroup_encoded']

df_y = pd.DataFrame(y)
df_X.to_csv('../data/X.csv', index=True)
df_y.to_csv('../data/y.csv', index=True)


# %%
categories = ["HER2+", "HR+", "Triple Neg"]

# Histogram of the labels
plt.figure(figsize=(7, 4))  
plt.hist(y, bins=len(set(y)), alpha=0.7, edgecolor='black')  # Creates the histogram
plt.title('Distribution of Clinical Breast Cancer Subtypes')  # Adds a title
plt.xlabel('Encoded Subtype')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.xticks(range(len(categories)), categories)  # Ensures there is a tick for each class
plt.show()  # Displays the plot

# %% [markdown]
# ### Non-Linear Feature Selection using Random Forest

# %%
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from scipy.stats import randint as sp_randint
import time
import numpy as np

def nested_cv_rfc(x_data, y_targets):
    # Random Forest classifier to be optimized
    rfc = RandomForestClassifier(random_state=2)

    # Define the parameter space for RandomizedSearchCV
    param_dist = {
        "n_estimators": sp_randint(10, 750),
        "max_leaf_nodes": sp_randint(20, 1000),
        "max_depth": sp_randint(20, 500),
        "min_samples_split": sp_randint(2, 250),
        "max_features": sp_randint(3, 100)
    }

    # Number of parameter settings that are sampled
    n_iter_search = 50
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=1)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=2)

    # Execute RandomizedSearchCV
    random_search = RandomizedSearchCV(rfc, param_distributions=param_dist,
                                       n_iter=n_iter_search, scoring=make_scorer(f1_score, average='micro'), cv=inner_cv, verbose=1)

    # Using outer CV for assessing the performance
    outer_scores = cross_val_score(random_search, x_data, y_targets, cv=outer_cv, scoring=make_scorer(f1_score, average='micro'))

    print("Nested CV score (mean):", np.mean(outer_scores))

    # Refitting on the entire dataset (can be omitted if only interested in performance estimate)
    random_search.fit(x_data, y_targets)  # This line will be executed with the best parameters found in the inner CV
    best_params_random = random_search.best_params_
    
    return best_params_random

# Usage: best_params = nested_cv_rfc(x_data, y_targets)


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# %%
m_rf = RandomForestClassifier(n_estimators = 20, max_features = 3, random_state=42).fit(X_train, y_train)
y_pred_std = m_rf.predict(X_test)

print(classification_report(y_test, y_pred_std, zero_division=0))

# %%
best_params = nested_cv_rfc(X_train, y_train)

# %%
m_rf_optimized = RandomForestClassifier(random_state=42, n_estimators = int(best_params["n_estimators"]), max_depth = int(best_params["max_depth"]), max_leaf_nodes = int(best_params["max_leaf_nodes"]), min_samples_split = int(best_params["min_samples_split"]), max_features = int(best_params["max_features"])).fit(X_train, y_train)
y_pred = m_rf_optimized.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))

# %%
def feature_importance(model, X_train):

    # Extract feature importances
    feature_importances = model.feature_importances_

    # Create generic feature names if not already provided
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns
    else:
        feature_names = [f"{i}" for i in range(X_train.shape[1])]

    # Sort the feature importances in descending order and get the indices
    indices = np.argsort(feature_importances)[::-1]

    # Select the top 50 features
    top_indices = indices[:100]
    
    # Adjust the size of the plot to accommodate 50 features nicely
    plt.figure(figsize=(15, 10))
    plt.title("Top 50 Feature Importances")
    plt.bar(range(100), feature_importances[top_indices], color="r", align="center")
    
    # Set the ticks to be the names of the top 50 features
    plt.xticks(range(100), [feature_names[i] for i in top_indices], rotation=90)
    plt.xlim([-1, 100])
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()

    return top_indices

# %%
selected_features_rf = feature_importance(m_rf_optimized, X_train)

# %%
def calculate_similarity(list1, list2):
    # Convert lists to sets
    set1 = set(list1)
    set2 = set(list2)
    
    # Calculate intersection and union
    intersection = set1 & set2
    union = set1 | set2
    
    # Calculate Jaccard similarity index
    if len(union) == 0:  # Prevent division by zero
        similarity = 0
    else:
        similarity = len(intersection) / len(union)
    
    return similarity, list(intersection)

# Example usage:
list_a = [1, 2, 3, 4, 5]
list_b = [4, 5, 6, 7, 8]

similarity, intersection = calculate_similarity(list_a, list_b)
print("Similarity index:", similarity)
print("Intersection of lists:", intersection)


# %%
selected_features = list(set(selected_features_l1) | set(selected_features_rf)) 
print(selected_features)
print(len(selected_features))

# %%
print(best_params)


