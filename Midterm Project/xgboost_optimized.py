# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
import spacy
import en_core_web_sm  # Need to run: python -m spacy download en_core_web_sm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ks_2samp, ranksums
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# Set NLTK data path (optional)
nltk.data.path.append('/home/tvy5242/nltk_data')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load spaCy model
nlp = en_core_web_sm.load()

# Read datasets
train_df = pd.read_csv('/home/tvy5242/llm/Processed_Training.csv')
test_df = pd.read_csv('/home/tvy5242/llm/Processed_Testing.csv')

# Combine training and test datasets for easier feature extraction
all_df = pd.concat([train_df, test_df], ignore_index=True)

# Extract xi, xj, and LLM labels
xi_texts = all_df['xi'].astype(str)
xj_texts = all_df['xj'].astype(str)
llm_labels = all_df['LLM']

# Define feature extraction functions
def get_vocabulary_size(text):
    cleaned_text = text.replace("\n", " ").strip()
    tokens = word_tokenize(cleaned_text.lower())
    vocab = set(tokens)
    return len(vocab)

def get_pos_distribution(text):
    cleaned_text = text.replace("\n", " ").strip()
    tokens = word_tokenize(cleaned_text)
    pos_tags = nltk.pos_tag(tokens)
    pos_counts = {}
    total_count = len(pos_tags)
    for word, pos in pos_tags:
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    pos_freq = {k: v / total_count for k, v in pos_counts.items()}
    return pos_freq

def get_dependency_distribution(text):
    cleaned_text = text.replace("\n", " ").strip()
    doc = nlp(cleaned_text)
    dep_counts = {}
    total_count = len(doc)
    for token in doc:
        dep = token.dep_
        dep_counts[dep] = dep_counts.get(dep, 0) + 1
    dep_freq = {k: v / total_count for k, v in dep_counts.items()}
    return dep_freq

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    cleaned_text = text.replace("\n", " ").strip()
    sentiment = sia.polarity_scores(cleaned_text)
    return sentiment['compound']

# Initialize feature lists
xi_vocab_sizes = []
xi_pos_features = []
xi_dep_features = []
xi_sentiment_scores = []

xj_vocab_sizes = []
xj_pos_features = []
xj_dep_features = []
xj_sentiment_scores = []

# Extract features for xi and xj
print("Extracting features for xi and xj...")

for xi_text, xj_text in zip(xi_texts, xj_texts):
    try:
        # xi features
        xi_vocab_sizes.append(get_vocabulary_size(xi_text))
        xi_pos_features.append(get_pos_distribution(xi_text))
        xi_dep_features.append(get_dependency_distribution(xi_text))
        xi_sentiment_scores.append(get_sentiment_score(xi_text))
    except Exception as e:
        print(f"Error processing xi_text: {e}")
        xi_vocab_sizes.append(0)
        xi_pos_features.append({})
        xi_dep_features.append({})
        xi_sentiment_scores.append(0)
    try:
        # xj features
        xj_vocab_sizes.append(get_vocabulary_size(xj_text))
        xj_pos_features.append(get_pos_distribution(xj_text))
        xj_dep_features.append(get_dependency_distribution(xj_text))
        xj_sentiment_scores.append(get_sentiment_score(xj_text))
    except Exception as e:
        print(f"Error processing xj_text: {e}")
        xj_vocab_sizes.append(0)
        xj_pos_features.append({})
        xj_dep_features.append({})
        xj_sentiment_scores.append(0)

# Calculate cosine similarity between xi and xj
print("Calculating cosine similarity between xi and xj...")

# Combine xi and xj texts for TF-IDF vectorization
combined_texts = xi_texts.tolist() + xj_texts.tolist()

# Fit TF-IDF vectorizer on combined texts
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts)

# Split TF-IDF matrix back into xi and xj
xi_tfidf = tfidf_matrix[:len(xi_texts)]
xj_tfidf = tfidf_matrix[len(xi_texts):]

# Calculate cosine similarity
cosine_similarities = []

for i in range(len(xi_texts)):
    xi_vec = xi_tfidf[i]
    xj_vec = xj_tfidf[i]
    cosine_sim = cosine_similarity(xi_vec, xj_vec)[0][0]
    cosine_similarities.append(cosine_sim)

# Convert POS and dependency features to DataFrame
xi_pos_df = pd.DataFrame(xi_pos_features).fillna(0).add_prefix('xi_pos_')
xj_pos_df = pd.DataFrame(xj_pos_features).fillna(0).add_prefix('xj_pos_')

xi_dep_df = pd.DataFrame(xi_dep_features).fillna(0).add_prefix('xi_dep_')
xj_dep_df = pd.DataFrame(xj_dep_features).fillna(0).add_prefix('xj_dep_')

# Create feature DataFrame
print("Creating feature DataFrame...")

features_df = pd.DataFrame()

# Add vocabulary size features
features_df['xi_vocab_size'] = xi_vocab_sizes
features_df['xj_vocab_size'] = xj_vocab_sizes

# Add sentiment scores
features_df['xi_sentiment'] = xi_sentiment_scores
features_df['xj_sentiment'] = xj_sentiment_scores

# Add cosine similarity
features_df['cosine_similarity'] = cosine_similarities

# Merge POS and dependency features
features_df = pd.concat([features_df, xi_pos_df, xj_pos_df, xi_dep_df, xj_dep_df], axis=1)

# Handle missing values
features_df = features_df.fillna(0)

# Add LLM labels
features_df['LLM'] = llm_labels.values

# Statistical analysis
print("Performing statistical analysis...")

# Vocabulary analysis (ANOVA)
print("\nVocabulary Analysis (ANOVA):")
f_stat_xi, p_value_xi = f_oneway(*[features_df[features_df['LLM'] == llm]['xi_vocab_size'] for llm in features_df['LLM'].unique()])
f_stat_xj, p_value_xj = f_oneway(*[features_df[features_df['LLM'] == llm]['xj_vocab_size'] for llm in features_df['LLM'].unique()])

print(f"xi_vocab_size ANOVA result: F={f_stat_xi:.4f}, p={p_value_xi:.4e}")
print(f"xj_vocab_size ANOVA result: F={f_stat_xj:.4f}, p={p_value_xj:.4e}")

# Tukey post-hoc test
print("\nTukey Post-Hoc Test (for xj_vocab_size):")
vocab_tukey = pairwise_tukeyhsd(endog=features_df['xj_vocab_size'], groups=features_df['LLM'], alpha=0.05)
print(vocab_tukey)

# POS and dependency analysis (Kolmogorov-Smirnov test)
print("\nPOS and Dependency Analysis (Kolmogorov-Smirnov Test):")
llm_list = features_df['LLM'].unique()
alpha = 0.05

# POS features
pos_feature_names = xi_pos_df.columns.tolist() + xj_pos_df.columns.tolist()
pos_results = {}
for feature in pos_feature_names:
    groups = [features_df[features_df['LLM'] == llm][feature] for llm in llm_list]
    for i in range(len(llm_list)):
        for j in range(i+1, len(llm_list)):
            ks_stat, p_value = ks_2samp(groups[i], groups[j])
            pos_results[(feature, llm_list[i], llm_list[j])] = (ks_stat, p_value)

# Apply Bonferroni correction
num_tests = len(pos_results)
corrected_alpha = alpha / num_tests

print("\nPOS Feature KS Test Results (with Bonferroni Correction):")
for key, (ks_stat, p_value) in pos_results.items():
    feature, llm1, llm2 = key
    significant = p_value < corrected_alpha
    if significant:
        print(f"{feature} between {llm1} and {llm2}: KS statistic={ks_stat:.4f}, p-value={p_value:.4e} *")
    else:
        print(f"{feature} between {llm1} and {llm2}: KS statistic={ks_stat:.4f}, p-value={p_value:.4e}")

# Dependency features
dep_feature_names = xi_dep_df.columns.tolist() + xj_dep_df.columns.tolist()
dep_results = {}
for feature in dep_feature_names:
    groups = [features_df[features_df['LLM'] == llm][feature] for llm in llm_list]
    for i in range(len(llm_list)):
        for j in range(i+1, len(llm_list)):
            ks_stat, p_value = ks_2samp(groups[i], groups[j])
            dep_results[(feature, llm_list[i], llm_list[j])] = (ks_stat, p_value)

# Apply Bonferroni correction
num_tests = len(dep_results)
corrected_alpha = alpha / num_tests

print("\nDependency Feature KS Test Results (with Bonferroni Correction):")
for key, (ks_stat, p_value) in dep_results.items():
    feature, llm1, llm2 = key
    significant = p_value < corrected_alpha
    if significant:
        print(f"{feature} between {llm1} and {llm2}: KS statistic={ks_stat:.4f}, p-value={p_value:.4e} *")
    else:
        print(f"{feature} between {llm1} and {llm2}: KS statistic={ks_stat:.4f}, p-value={p_value:.4e}")

# Sentiment analysis (Wilcoxon signed-rank test)
print("\nSentiment Analysis (Wilcoxon Signed-Rank Test):")
sentiment_features = ['xi_sentiment', 'xj_sentiment']
sentiment_results = {}
for feature in sentiment_features:
    groups = [features_df[features_df['LLM'] == llm][feature] for llm in llm_list]
    for i in range(len(llm_list)):
        for j in range(i+1, len(llm_list)):
            stat, p_value = ranksums(groups[i], groups[j])
            sentiment_results[(feature, llm_list[i], llm_list[j])] = (stat, p_value)

# Apply Bonferroni correction
num_tests = len(sentiment_results)
corrected_alpha = alpha / num_tests

print("\nSentiment Analysis Results (with Bonferroni Correction):")
for key, (stat, p_value) in sentiment_results.items():
    feature, llm1, llm2 = key
    significant = p_value < corrected_alpha
    if significant:
        print(f"{feature} between {llm1} and {llm2}: statistic={stat:.4f}, p-value={p_value:.4e} *")
    else:
        print(f"{feature} between {llm1} and {llm2}: statistic={stat:.4f}, p-value={p_value:.4e}")

# LLM attribution classification task
print("\nTraining XGBoost classifier for LLM attribution...")

# Prepare dataset
X = features_df.drop(['LLM'], axis=1)
y = features_df['LLM']

# Split dataset back into training and testing sets
X_train = X.iloc[:len(train_df)]
X_test = X.iloc[len(train_df):]
y_train = y.iloc[:len(train_df)]
y_test = y.iloc[len(train_df):]

# Label encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train XGBoost classifier with eval_set parameter
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

eval_set = [(X_train, y_train_encoded), (X_test, y_test_encoded)]
xgb_model.fit(X_train, y_train_encoded, eval_set=eval_set, verbose=False)

# Get evaluation results during training (optional, can be removed if not needed)
# results = xgb_model.evals_result()

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Model evaluation
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

# Output evaluation metrics (Accuracy, Recall, Precision, F1 Score)
accuracy = accuracy_score(y_test_encoded, y_pred)
recall = recall_score(y_test_encoded, y_pred, average='weighted')
precision = precision_score(y_test_encoded, y_pred, average='weighted')
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
conf_mat = confusion_matrix(y_test_encoded, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# End of script
print("Script finished. The generated confusion matrix is saved as 'confusion_matrix.png'. The classification report has been displayed in the console.")
