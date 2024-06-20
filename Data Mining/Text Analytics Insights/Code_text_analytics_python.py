#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC


# In[2]:


#load the data set 

# Path to the Excel file
file_path = '/Users/dhanush/Desktop/Bussiness analytics /Sem 2/Data Mining/ASS_2/A_II_Emotion_Data_Student_Copy_Final.xlsx'

# Load the dataset
data = pd.read_excel(file_path)

# Display the first few rows of the dataset to check it
print(data.head())


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
# To check missing values in each column
missing_values = data.isnull().sum()

# Print out the missing values count per column
print("Missing values per column:")
print(missing_values)

# Calculate the percentage of missing values for each column
total_rows = len(data)
missing_percentages = (missing_values / total_rows) * 100

# Visualize these as a part of exploratory data analysis
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size as necessary
missing_percentages.plot(kind='bar', color='skyblue')
plt.ylabel('Percentage of Missing Values')
plt.title('Missing Data Analysis')
plt.xticks(rotation=45)  # Rotate labels to avoid overlap
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines for better readability
plt.show()


# In[5]:


#data preprocessing / text cleaning 
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = text.lower()  # Lowercase text
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words with 1 or 2 letters
    text = re.sub(r'[^a-z\s]', '', text)  # Keep text with letters and spaces

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)




# In[6]:


#To check the data
data['Cleaned_reviews'] = data['text_reviews_'].apply(clean_text)
print(data)


# In[7]:


#Lets split the data to unlabbeled and lablled data 

unlabeled_data = data[data['emotions_'].isna()][['Cleaned_reviews']]
unlabeled_data['emotions_'] = -1
print(unlabeled_data)


# In[8]:


# Define labeled data as data where "Sentiment" is not missing
# Unlabeled data
# Define labeled data as data where "Sentiment" is not missing
labeled_data = data[data['emotions_'].notna() & (data['emotions_'] != 'NaN')]
# Extract labels from labeled_data
y_labeled = labeled_data['emotions_']
y_unlabeled = unlabeled_data['emotions_']
X_labeled = labeled_data['Cleaned_reviews']
X_unlabeled = unlabeled_data['Cleaned_reviews']


# In[9]:


labeled_data


# In[10]:


unlabeled_data


# In[11]:


#Supervised Machine Learning 

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Parameters for vectorization
vectorizer_params = dict(ngram_range=(1, 2), min_df=1, max_df=0.8)

# Random state for all classifiers
random_state = 40412492

# 1)Pipeline for Support Vector Machine (SVM)
svm_params = dict(C=1.0, kernel='linear', gamma='auto', probability=True, random_state=random_state)
svm_pipeline = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", SVC(**svm_params))
])

# 2)Pipeline for Logistic Regression
lr_params = dict(C=1.0, penalty='l2', solver='liblinear', random_state=random_state)
lr_pipeline = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", LogisticRegression(**lr_params))
])

# 3)Pipeline for Random Forest
rf_params = dict(n_estimators=100, max_depth=None, random_state=random_state)
rf_pipeline = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", RandomForestClassifier(**rf_params))
])

# 4)Pipeline for Gradient Boosting
gb_params = dict(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state)
gb_pipeline = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", GradientBoostingClassifier(**gb_params))
])

# 5)Pipeline for K-Nearest Neighbors (KNN)
knn_params = dict(n_neighbors=5, weights='uniform')
knn_pipeline = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", KNeighborsClassifier(**knn_params))
])

# 6) Pipeline for Decision Tree
dt_params = dict(max_depth=None, random_state=random_state)
dt_pipeline = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", DecisionTreeClassifier(**dt_params))
])





# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, stratify=y_labeled, random_state=40412492)


# In[13]:


# Assuming all imports and pipelines definition are correct and placed appropriately in the script

def eval_metrics_to_dataframe(pipelines, X_train, y_train, X_test, y_test):
    results = []
    for name, pipeline in pipelines:
        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Extract relevant metrics
        metrics = {
            'Model': name,
            'Accuracy': report['accuracy'],
            'Macro Precision': report['macro avg']['precision'],
            'Macro Recall': report['macro avg']['recall'],
            'Macro F1-score': report['macro avg']['f1-score']
        }

        # Append metrics to results list
        results.append(metrics)

    # Create DataFrame from results
    df_results = pd.DataFrame(results)
    return df_results

# Define pipelines
pipelines = [
    ("Support Vector Machine", svm_pipeline),
    ("Logistic Regression", lr_pipeline),
    ("Random Forest", rf_pipeline),
    ("Gradient Boosting", gb_pipeline),
    ("K-Nearest Neighbors", knn_pipeline),
    ("Decision Tree", dt_pipeline)
]

# Get DataFrame of results
results_df = eval_metrics_to_dataframe(pipelines, X_train, y_train, X_test, y_test)


# In[14]:


results_df


# In[ ]:


#Semi supervsied learning 


# In[18]:


from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

gb_params = dict(n_estimators=100, learning_rate=0.1, max_depth=3)

st_pipeline_gradient_boosting = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", SelfTrainingClassifier(GradientBoostingClassifier(**gb_params), verbose=True))
])
#

vectorizer_params = dict(ngram_range=(1, 2), min_df=1, max_df=0.8)
svm_params = dict(C=1.0, kernel='linear', gamma='auto', probability=True)

st_pipeline_svm = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", SelfTrainingClassifier(SVC(**svm_params), verbose=True))
])

from sklearn.linear_model import LogisticRegression

lr_params = dict(C=1.0, penalty='l2', solver='liblinear')

st_pipeline_logistic_regression = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", SelfTrainingClassifier(LogisticRegression(**lr_params), verbose=True))
])

from sklearn.ensemble import RandomForestClassifier

rf_params = dict(n_estimators=100, max_depth=None)

st_pipeline_random_forest = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", SelfTrainingClassifier(RandomForestClassifier(**rf_params), verbose=True))
])

from sklearn.ensemble import GradientBoostingClassifier

gb_params = dict(n_estimators=100, learning_rate=0.1, max_depth=3)

st_pipeline_gradient_boosting = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", SelfTrainingClassifier(GradientBoostingClassifier(**gb_params), verbose=True))
])

from sklearn.neighbors import KNeighborsClassifier

knn_params = dict(n_neighbors=5, weights='uniform')

st_pipeline_knn = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", SelfTrainingClassifier(KNeighborsClassifier(**knn_params), verbose=True))
])

from sklearn.tree import DecisionTreeClassifier

dt_params = dict(max_depth=None)

st_pipeline_decision_tree = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", SelfTrainingClassifier(DecisionTreeClassifier(**dt_params), verbose=True))
])



# In[21]:


test_indices = X_test.index
#print("TEST INDICES",test_indices)
# Exclude test data from X_labeled and y_labeled based on the identified indices
X_labeled_filtered = X_labeled.drop(index=test_indices, errors='ignore')
y_labeled_filtered = y_labeled.drop(index=test_indices, errors='ignore')
# Concatenate the filtered labeled data with the unlabeled data
X=X_combined = pd.concat([X_labeled_filtered, X_unlabeled])
y=y_combined = pd.concat([y_labeled_filtered, y_unlabeled])


# In[22]:


#Define the mapping for labels
label_mapping = {'anger': 1, 'disgust': 2, 'fear': 3, 'joy':4, 'sadness': 5, 'surprise':6 ,'neutral': 0, -1:-1 }
# Apply the mapping to labels
y = [label_mapping[label] for label in y]
#print(y)
y_test = [label_mapping[label] for label in y_test]
#print(y_test)


# In[31]:


import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict

def run_pipelines(pipelines, X, y, X_test, y_test):
    results = []
    for name, pipeline in pipelines.items():
        print(f"Running {name} pipeline...")
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.append([name, accuracy, precision, recall, f1])
    return pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Define pipelines dictionary
pipelines = {
    "SVM": st_pipeline_svm,
    "Decision Tree": st_pipeline_decision_tree,
    "Random Forest": st_pipeline_random_forest,
    "KNN": st_pipeline_knn,
    "Logistic Regression": st_pipeline_logistic_regression,
    "Gradient Boosting": st_pipeline_gradient_boosting
}

# Run pipelines and get results
results_df = run_pipelines(pipelines, X_combined, y, X_test, y_test)
print(results_df)


# In[32]:


# Run pipelines and get results
print(results_df)


# In[25]:


# Assuming data is your DataFrame containing the text reviews and emotions
for index, row in data.iterrows():
    if row['emotions_'] not in ['surprise', 'joy', 'neutral', 'sadness', 'fear', 'disgust', 'anger']:
        predicted_emotion = st_pipeline_gradient_boosting.predict([row['text_reviews_']])
        data.at[index, 'emotions_'] = predicted_emotion[0]  # Assign the predicted emotion to the DataFrame


data['emotions_'] = data['emotions_'].map({
    1: 'anger',
    2: 'disgust',
    3: 'fear',
    4: 'joy',
    5: 'sadness',
    6: 'surprise',
    0: 'neutral',
    -1: -1
}).fillna(data['emotions_'])



# In[30]:


get_ipython().system('pip install wordcloud')

import wordcloud
from wordcloud import WordCloud

# pie chart for the emotions_
data["emotions_"].value_counts().plot(kind="pie")

# word cloud for text reviews
emotions = ["surprise", "joy", "neutral", "sadness", "fear", "disgust", "anger"]
colors = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples']
num_plots = len(emotions)
fig, axs = plt.subplots(1, num_plots, figsize=(15, 5))

# Iterate over each emotion and create a WordCloud plot
for i, emotion in enumerate(emotions):
    # Filter text for the current emotion
    filtered_text = data.loc[data['emotions_'] == emotion, 'Cleaned_reviews']

    # Join the filtered text into a single string using " "
    meta_text = " ".join(filtered_text)

    # Generate WordCloud for the current emotion
    wc = WordCloud(width=400, height=200, colormap=colors[i]).generate(meta_text)

    # Display WordCloud plot in the corresponding subplot
    axs[i].imshow(wc, interpolation='bilinear')
    axs[i].set_title(emotion.capitalize())  # Set title with capitalized emotion name
    axs[i].axis('off')


plt.tight_layout()
plt.show()


# In[33]:


data['emotions_'].value_counts()


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns

## Histogram for Star Ratings
plt.figure(figsize=(8, 6))
# Changed color to 'blue' and removed palette because 'kde=True' does not use palette
sns.histplot(data['star_rating_'], kde=True, color='blue')
plt.title('Distribution of Star Ratings')
plt.xlabel('Star Rating')
plt.ylabel('Frequency')
plt.show()

## Boxplot for Star Ratings by Brand
plt.figure(figsize=(10, 6))
# Changed palette to 'viridis' for a different color gradient
sns.boxplot(x='brand_name_', y='star_rating_', data=data, palette='viridis')
plt.title('Star Rating Distribution by Brand')
plt.xlabel('Brand')
plt.ylabel('Star Rating')
plt.show()




# In[42]:


import matplotlib.pyplot as plt
import seaborn as sns

## Bar plot for Emotion vs Brand
plt.figure(figsize=(10, 6))
# Changed estimator to count to plot the count of each emotion category
sns.countplot(x='brand_name_', hue='emotions_', data=data, palette='muted')
plt.title('Emotion Distribution by Brand')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotating x-axis labels for better readability
plt.legend(title='Emotion')
plt.show()


# In[ ]:




