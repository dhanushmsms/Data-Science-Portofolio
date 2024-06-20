#!/usr/bin/env python
# coding: utf-8

# In[ ]:


sur#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/Users/dhanush/Desktop/Bussiness analytics /Sem 2/Data Mining/sales_data (1).csv')

# Display the first few rows of the DataFrame to verify it's loaded correctly
print(data.head())

#print all the column names
print(data.columns)



# In[2]:


#Data exploration

#1)Flag

#Count missing values in the 'flag' column
flag_na =data['flag'].isna().sum()
print("The na present is ",flag_na)


# Provide a summary of the 'flag' column (count of each unique value)
flag_summary = data['flag'].value_counts()
print("Summary of 'flag' column:")
print(flag_summary)

# 4. Create a bar plot showing the distribution of values in the 'flag' column
sns.countplot(x='flag', data=data, palette='viridis')
plt.title("Distribution of 'flag' Values")
plt.show()


# In[3]:


#2)gender

#Count missing values in the 'Gender' column
gender_na = data['gender'].isna().sum()
print("The na present is ",gender_na)

# Provide a summary of the 'Gender' column (count of each unique value)
age_summary = data['gender'].value_counts()
print("Summary of 'gender' column:")
print(age_summary)

# 4. Create a bar plot showing the distribution of values in the 'gender' column
sns.countplot(x='gender', data=data, palette='viridis')
plt.title("Distribution of gender")
plt.show()

#there is some unkown u in the data which has to removed or imputed 


# In[4]:


#3)education

#Count missing values in the 'education' column
edu_na = data['education'].isna().sum()
print("The na present is ",edu_na)


# Provide a summary of the 'education' column (count of each unique value)
education_summary = data['education'].value_counts()
print("Summary of 'education' column:")
print(education_summary)

# 4. Create a bar plot showing the distribution of values in the 'education' column
# Fill NA values with a placeholder for 'education'
data_filled_education = data.copy()
data_filled_education['education'] = data_filled_education['education'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'education' column, including NA values
sns.countplot(x='education', data=data_filled_education, palette='viridis')
plt.title("Distribution of 'education' (Including Missing Values)")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()


#there are so na which has to be removed 


# In[5]:


#4)house_val

#Count missing values in the 'house_val' column
house_val_na = data['house_val'].isna().sum()
print("The na present is ",house_val_na)


# Provide a summary of the 'house_val' column (count of each unique value)
house_val_summary = data['house_val'].value_counts()
print("Summary of 'house_val' column:")
print(house_val_summary)

# 4. Create a box plot showing the distribution of values in the 'house_val' column
sns.boxplot(x=data['house_val'])
plt.title("Boxplot of House Values")
plt.xlabel("House Value")
plt.show()

# need to normalise the data 


# In[6]:


#5)age

#Count missing values in the 'age' column
age_na = data['age'].isna().sum()
print("The na present is ",age_na)


# Provide a summary of the 'age' column (count of each unique value)
age_summary = data['age'].value_counts()
print("Summary of 'age' column:")
print(age_summary)

# 4. Create a bar plot showing the distribution of values in the 'education' column
sns.countplot(x='age', data=data, palette='viridis')
plt.title("Distribution of age")
plt.show()


# In[7]:


#6)online

#Count missing values in the 'online' column
online_na = data['online'].isna().sum()
print("The na present is ",online_na)


# Provide a summary of the 'online' column (count of each unique value)
online_summary = data['online'].value_counts()
print("Summary of 'online' column:")
print(online_summary)

# 4. Create a bar plot showing the distribution of values in the 'online' column
sns.countplot(x='online', data=data, palette='viridis')
plt.title("Distribution of online")
plt.show()


# In[8]:


#7)marriage

#Count missing values in the 'marriage' column
marriage_na = data['marriage'].isna().sum()
print("The na present is ",marriage_na)


# Provide a summary of the 'marriage' column (count of each unique value)
marriage_summary = data['marriage'].value_counts()
print("Summary of 'marriage' column:")
print(marriage_summary)

# 4. Create a bar plot showing the distribution of values in the 'marriage' column
data_filled = data.copy()
data_filled['marriage'] = data_filled['marriage'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'marriage' column, including NA values
sns.countplot(x='marriage', data=data_filled, palette='viridis')
plt.title("Distribution of 'marriage' (Including Missing Values)")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()


# In[9]:


#8) child

#Count missing values in the 'child' column
child_na = data['child'].isna().sum()
print("The na present is ",child_na)


# Provide a summary of the 'child' column (count of each unique value)
child_summary = data['child'].value_counts()
print("Summary of 'child' column:")
print(child_summary)

# 4. Create a bar plot showing the distribution of values in the 'child' column
data_filled = data.copy()
data_filled['child'] = data_filled['child'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'child' column, including NA values
sns.countplot(x='child', data=data_filled, palette='viridis')
plt.title("Distribution of child")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()




# In[10]:


#9) occupation

#Count missing values in the 'occupation' column
occupation_na = data['occupation'].isna().sum()
print("The na present is ",occupation_na)


# Provide a summary of the 'occupation' column (count of each unique value)
occupation_summary = data['occupation'].value_counts()
print("Summary of 'occupation' column:")
print(occupation_summary)

# 4. Create a bar plot showing the distribution of values in the 'occupation' column
data_filled = data.copy()
data_filled['occupation'] = data_filled['occupation'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'occupation' column, including NA values
sns.countplot(x='occupation', data=data_filled, palette='viridis')
plt.title("Distribution of occupation")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()



# In[11]:


#10) mortgage

#Count missing values in the 'mortgage' column
mortgage_na = data['mortgage'].isna().sum()
print("The na present is ",mortgage_na)


# Provide a summary of the 'mortgage' column (count of each unique value)
mortgage_summary = data['mortgage'].value_counts()
print("Summary of 'mortgage' column:")
print(mortgage_summary)

# 4. Create a bar plot showing the distribution of values in the 'occupation' column
data_filled = data.copy()
data_filled['mortgage'] = data_filled['mortgage'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'occupation' column, including NA values
sns.countplot(x='mortgage', data=data_filled, palette='viridis')
plt.title("Distribution of mortgage")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()



# In[12]:


#11) house_owner

#Count missing values in the 'house_owner' column
house_owner_na = data['house_owner'].isna().sum()
print("The na present is ",house_owner_na)


# Provide a summary of the 'mortgage' column (count of each unique value)
house_owner_summary = data['house_owner'].value_counts()
print("Summary of 'house_owner' column:")
print(house_owner_summary)

# 4. Create a bar plot showing the distribution of values in the 'house_owner' column
data_filled = data.copy()
data_filled['house_owner'] = data_filled['house_owner'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'house_owner' column, including NA values
sns.countplot(x='house_owner', data=data_filled, palette='viridis')
plt.title("Distribution of house_owner")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()


# In[13]:


#12) region

#Count missing values in the 'region' column
region_na = data['region'].isna().sum()
print("The na present is ",region_na)


# Provide a summary of the 'region' column (count of each unique value)
region_summary = data['region'].value_counts()
print("Summary of 'region' column:")
print(region_summary)

# 4. Create a bar plot showing the distribution of values in the 'region' column
data_filled = data.copy()
data_filled['region'] = data_filled['region'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'region' column, including NA values
sns.countplot(x='region', data=data_filled, palette='viridis')
plt.title("Distribution of region")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()


# In[14]:


#13) fam_income

#Count missing values in the 'fam_income' column
fam_income_na = data['fam_income'].isna().sum()
print("The na present is ",fam_income_na)


# Provide a summary of the 'fam_income' column (count of each unique value)
fam_income_summary = data['fam_income'].value_counts()
print("Summary of 'region' column:")
print(fam_income_summary)

# 4. Create a bar plot showing the distribution of values in the 'fam_income' column
data_filled = data.copy()
data_filled['fam_income'] = data_filled['fam_income'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'fam_income' column, including NA values
sns.countplot(x='fam_income', data=data_filled, palette='viridis')
plt.title("Distribution of fam_income")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()


# In[15]:


#Missing information
missing_info = data.isnull().sum()
print("Missing Information:")
print(missing_info)

# Calculate missing percentage
missing_percentage = 100 * data.isnull().sum() / len(data)
print("Missing Information(%):")
print(missing_percentage)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Actual count of missing values (left y-axis)
color = 'red'
ax1.bar(data.columns, data.isnull().sum(), color=color, label='Count of Missing Values')
ax1.set_ylabel('Count of Missing Values', color=color)
plt.xticks(rotation=45)
ax1.tick_params(axis='y', labelcolor=color)

# Percentage of missing values (right y-axis)
ax2 = ax1.twinx()
color = 'black'
ax2.plot(data.columns, missing_percentage, color=color, marker='o', label='Percentage of Missing Values')
ax2.set_ylabel('Percentage of Missing Values', color=color)
plt.xticks(rotation=45)
ax2.tick_params(axis='y', labelcolor=color)

# Title and labels
plt.title('Bar Chart: Missing Values with Percentage')
plt.xlabel('Variables')
plt.xticks(rotation=45)
plt.legend(loc='upper left')

plt.show()#


# In[16]:


#Addresing data quality issues 
#As observed from the data exploartion the data quality issues are as follows and there are addressed as follows :

#1) Gender
# As observed there are about 1151 'U' obersved in which u is not defined in the data dictornary will we impute it with male as it is the most occuring mode

new_data = data.copy()

# Impute 'U' values with 'M' in the 'gender' column of the new DataFrame
new_data['gender'] = new_data['gender'].replace('U', 'M')

# Verify the imputation in the new DataFrame
print(new_data['gender'].value_counts())

# Create a bar plot showing the distribution of values in the 'gender' column
sns.countplot(x='gender', data=new_data, palette='viridis')
plt.title("Distribution of gender")
plt.show()


# In[17]:


#2)Education 

#Education has 741 NA valus which will be imputed by mode of the education 

# Calculate the mode of the 'education' column
education_mode = new_data['education'].mode()[0]

# Replace NA values in the 'education' column with the mode
new_data['education'] = new_data['education'].fillna(education_mode)

# Verify the imputation by checking if there are any NA values left
print(new_data['education'].isna().sum())

#plot to verify 
data_filled_education = new_data.copy()
data_filled_education['education'] = data_filled_education['education'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'education' column, including NA values
sns.countplot(x='education', data=data_filled_education, palette='viridis')
plt.title("Distribution of 'education' ")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()


#Count missing values in the 'education' column
edu_na = new_data['education'].isna().sum()
print("The na present is ",edu_na)


# Provide a summary of the 'education' column (count of each unique value)
education_summary = new_data['education'].value_counts()
print("Summary of 'education' column:")
print(education_summary)


# In[18]:


#3) Marriage 
#There are NA values which must be replaced by mode

# Calculate the mode of the 'education' column
Marriage_mode = new_data['marriage'].mode()[0]

# Replace NA values in the 'education' column with the mode
new_data['marriage'] = new_data['marriage'].fillna(Marriage_mode)

# Verify the imputation by checking if there are any NA values left
print(new_data['marriage'].isna().sum())


#plot to verify 
data_filled_marriage= new_data.copy()
data_filled_marriage['marriage'] = data_filled_marriage['marriage'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'education' column, including NA values
sns.countplot(x='marriage', data=data_filled_marriage, palette='viridis')
plt.title("Distribution of 'marriage' ")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()

marriage_summary = new_data['marriage'].value_counts()
print("Summary of 'marriage' column:")
print(marriage_summary)


# In[19]:


#4)Child
#there is 0 an undefined entry as they are only 127 entries they can be deleted 

# Remove rows where 'child' column has the value '0'
new_data = new_data[new_data['child'] != '0']

# Verify the removal by checking the unique values left in the 'child' column
print(new_data['child'].unique())

data_filled = new_data.copy()
data_filled['child'] = data_filled['child'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'child' column, including NA values
sns.countplot(x='child', data=data_filled, palette='viridis')
plt.title("Distribution of child")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()

#Count missing values in the 'child' column
child_na = new_data['child'].isna().sum()
print("The na present is ",child_na)


# Provide a summary of the 'child' column (count of each unique value)
child_summary = new_data['child'].value_counts()
print("Summary of 'child' column:")
print(child_summary)



# In[20]:


#5)house_owner
#there are 3377 NA which must be replaced by mode 

house_owner_mode = new_data['house_owner'].mode()[0]


# Replace NA values in the 'education' column with the mode
new_data['house_owner'] = new_data['house_owner'].fillna(house_owner_mode)

# Verify the imputation by checking if there are any NA values left
print(new_data['house_owner'].isna().sum())


#plot to verify 
data_filled_house_owner = new_data.copy()
data_filled_house_owner['house_owner'] = data_filled_house_owner['house_owner'].fillna('Missing')

# Create a bar plot showing the distribution of values in the 'education' column, including NA values
sns.countplot(x='house_owner', data=data_filled_house_owner, palette='viridis')
plt.title("Distribution of 'house_owner' ")
plt.xticks(rotation=45)  # Optional: Rotate labels if they overlap
plt.show()

house_owner_summary = new_data['house_owner'].value_counts()
print("Summary of 'house_owner' column:")
print(house_owner_summary)


# In[21]:


#correlation and bivarient analysis for 5 variables 


# In[22]:


#Age vs flag 
# 'age' is categorical in this dataset,
plt.figure(figsize=(12, 6))
sns.countplot(x='age', hue='flag', data=new_data, palette='viridis')
plt.title('Purchase Behavior Across Different Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Purchased', loc='upper right')
plt.tight_layout()

# Prepare data for Chi-square test
contingency_table = pd.crosstab(new_data['age'], new_data['flag'])

# Perform Chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Display the test results
print(f'Chi-square Statistic: {chi2}, p-value: {p_value}')
# Print the results
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {p_value}")
print("Expected Frequencies:", expected)

plt.show()


# In[23]:


#Gender VS Flag (purchased)
purchase_by_gender = new_data.groupby('gender')['flag'].value_counts().unstack()
print(purchase_by_gender)

# Now, for the visual representation
plt.figure(figsize=(8, 6))
sns.countplot(x='gender', hue='flag', data=new_data)
plt.title('Purchase Behavior by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Purchased')
plt.show()

#chisq test 
# Prepare data for Chi-square test
contingency_table_gen = pd.crosstab(new_data['gender'], new_data['flag'])

# Perform Chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table_gen)

# Display the test results
print(f'Chi-square Statistic: {chi2}, p-value: {p_value}')
# Print the results
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {p_value}")
print("Expected Frequencies:", expected)


# In[24]:


#Education vs Flag

plt.figure(figsize=(10, 6))
sns.countplot(x='education', hue='flag', data=new_data, palette='Set2')
plt.title('Purchase Behavior by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.legend(title='Purchased')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Now, for the Chi-square test of independence
contingency_table = pd.crosstab(new_data['education'], new_data['flag'])

# Perform Chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Display the test results
print(f'Chi-square Statistic: {chi2}, p-value: {p_value}')
# Print the results
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {p_value}")
print("Expected Frequencies:", expected)


# In[25]:


#Marriage vs Flag

plt.figure(figsize=(8, 6))
sns.countplot(x='marriage', hue='flag', data=new_data, palette='coolwarm')
plt.title('Purchase Behavior by Marriage Status')
plt.xlabel('Marriage Status')
plt.ylabel('Count')
plt.legend(title='Purchased')
plt.show()

# Now, for the Chi-square test of independence
contingency_table = pd.crosstab(new_data['marriage'], new_data['flag'])

# Perform Chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Display the test results
print(f'Chi-square Statistic: {chi2}, p-value: {p_value}')
# Print the results
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {p_value}")
print("Expected Frequencies:", expected)


# In[26]:


#occupation vs Flag

# Visual analysis for 'occupation'
plt.figure(figsize=(10, 6))
sns.countplot(x='occupation', hue='flag', data=new_data, palette='viridis')
plt.title('Purchase Behavior by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.legend(title='Purchased')
plt.xticks(rotation=45)
plt.show()

# Now, for the Chi-square test of independence
contingency_table = pd.crosstab(new_data['occupation'], new_data['flag'])

# Perform Chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Display the test results
print(f'Chi-square Statistic: {chi2}, p-value: {p_value}')
# Print the results
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {p_value}")
print("Expected Frequencies:", expected)


# In[ ]:


#Converting all character as factors

# Loop through each column and convert object type columns to categorical
for column in new_data.columns:
    if new_data[column].dtype == 'object':
        new_data[column] = new_data[column].astype('category')

# Verify the conversion by printing the data types of the DataFrame columns
print(new_data.dtypes)


# In[27]:


# Set random seed for reproducibility
np.random.seed(40412492)


# In[28]:


from sklearn.model_selection import train_test_split

#splitting data into test and train 
X = new_data.drop('flag', axis=1)
y = new_data['flag']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40412492)


# In[29]:


#Scaling the data 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Identifying categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Define transformers for the preprocessing step
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Fit the preprocessor and transform the training and testing data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# In[30]:


#import necessary lib 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# In[33]:


#Model 1 
# Logistic Regression model with hyperparameters
logistic_regression_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(C=1.0, max_iter=10000, random_state=40412492))
])

# Train the model
logistic_regression_model.fit(X_train, y_train)

# Predict on the testing data
y_pred = logistic_regression_model.predict(X_test)
y_proba = logistic_regression_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}')

# Plotting Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Adjusting for the ValueError in the roc_curve function
from sklearn.metrics import roc_curve, auc

# Calculate the ROC curve data
fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label='Y')
roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[44]:


#Model 2
from sklearn.ensemble import RandomForestClassifier

# Define the pipeline for the Random Forest model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=40412492))
])

# Setup the grid search for the Random Forest model
param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, scoring='accuracy', verbose=1)

#Grid search
# Perform the grid search
grid_search_rf.fit(X_train, y_train)

# Extract the best estimator
best_rf = grid_search_rf.best_estimator_

#Run model
# Predict on the testing data using the best model
y_pred_best_rf = best_rf.predict(X_test)

# Since RandomForest does not have predict_proba method by default, check before calling it
y_proba_best_rf = best_rf.predict_proba(X_test)[:, 1] if hasattr(best_rf, 'predict_proba') else None

# Calculate metrics
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
precision_best_rf = precision_score(y_test, y_pred_best_rf, average='weighted', zero_division=0)
recall_best_rf = recall_score(y_test, y_pred_best_rf, average='weighted')
f1_best_rf = f1_score(y_test, y_pred_best_rf, average='weighted')

print(f'Best Random Forest Metrics:\nAccuracy: {accuracy_best_rf:.4f}\nPrecision: {precision_best_rf:.4f}')
print(f'Recall: {recall_best_rf:.4f}\nF1 Score: {f1_best_rf:.4f}')

#plot cf and roc 

cm_best_rf = confusion_matrix(y_test, y_pred_best_rf)
sns.heatmap(cm_best_rf, annot=True, fmt="d", cmap="Greens")
plt.title('Confusion Matrix for Best Random Forest')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

if y_proba_best_rf is not None:
    fpr_best_rf, tpr_best_rf, _ = roc_curve(y_test, y_proba_best_rf, pos_label='Y')
    roc_auc_best_rf = auc(fpr_best_rf, tpr_best_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_best_rf, tpr_best_rf, color='darkorange', lw=2, label=f'Best Random Forest ROC curve (area = {roc_auc_best_rf:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Best Random Forest')
    plt.legend(loc="lower right")
    plt.show()




