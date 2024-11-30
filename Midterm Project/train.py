# 导入必要的库
import pandas as pd
import numpy as np
import nltk
import spacy
import en_core_web_sm  # 需要先运行：python -m spacy download en_core_web_sm
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

# 设置 NLTK 数据路径（可选）
nltk.data.path.append('/home/tvy5242/nltk_data')

# 下载必要的 NLTK 数据
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('stopwords')

# 加载 spaCy 模型
nlp = en_core_web_sm.load()

# 读取数据集
train_df = pd.read_csv('/home/tvy5242/llm/Processed_Training.csv')
test_df = pd.read_csv('/home/tvy5242/llm/Processed_Testing.csv')

# 合并训练集和测试集以方便特征提取
all_df = pd.concat([train_df, test_df], ignore_index=True)

# 提取 xi、xj 和 LLM 标签
xi_texts = all_df['xi'].astype(str)
xj_texts = all_df['xj'].astype(str)
llm_labels = all_df['LLM']

# 定义特征提取函数
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

# 初始化特征列表
xi_vocab_sizes = []
xi_pos_features = []
xi_dep_features = []
xi_sentiment_scores = []

xj_vocab_sizes = []
xj_pos_features = []
xj_dep_features = []
xj_sentiment_scores = []

# 提取 xi 和 xj 的特征
print("正在提取 xi 和 xj 的特征...")

for xi_text, xj_text in zip(xi_texts, xj_texts):
    try:
        # xi 特征
        xi_vocab_sizes.append(get_vocabulary_size(xi_text))
        xi_pos_features.append(get_pos_distribution(xi_text))
        xi_dep_features.append(get_dependency_distribution(xi_text))
        xi_sentiment_scores.append(get_sentiment_score(xi_text))
    except Exception as e:
        print(f"处理 xi_text 时出错: {e}")
        xi_vocab_sizes.append(0)
        xi_pos_features.append({})
        xi_dep_features.append({})
        xi_sentiment_scores.append(0)
    try:
        # xj 特征
        xj_vocab_sizes.append(get_vocabulary_size(xj_text))
        xj_pos_features.append(get_pos_distribution(xj_text))
        xj_dep_features.append(get_dependency_distribution(xj_text))
        xj_sentiment_scores.append(get_sentiment_score(xj_text))
    except Exception as e:
        print(f"处理 xj_text 时出错: {e}")
        xj_vocab_sizes.append(0)
        xj_pos_features.append({})
        xj_dep_features.append({})
        xj_sentiment_scores.append(0)

# 计算 xi 和 xj 之间的余弦相似度
print("正在计算 xi 和 xj 之间的余弦相似度...")

# 将 xi 和 xj 文本组合用于 TF-IDF 向量化
combined_texts = xi_texts.tolist() + xj_texts.tolist()

# 在组合文本上拟合 TF-IDF 矢量化器
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts)

# 将 TF-IDF 矩阵拆分回 xi 和 xj
xi_tfidf = tfidf_matrix[:len(xi_texts)]
xj_tfidf = tfidf_matrix[len(xi_texts):]

# 计算余弦相似度
cosine_similarities = []

for i in range(len(xi_texts)):
    xi_vec = xi_tfidf[i]
    xj_vec = xj_tfidf[i]
    cosine_sim = cosine_similarity(xi_vec, xj_vec)[0][0]
    cosine_similarities.append(cosine_sim)

# 将 POS 和依赖关系特征转换为 DataFrame
xi_pos_df = pd.DataFrame(xi_pos_features).fillna(0).add_prefix('xi_pos_')
xj_pos_df = pd.DataFrame(xj_pos_features).fillna(0).add_prefix('xj_pos_')

xi_dep_df = pd.DataFrame(xi_dep_features).fillna(0).add_prefix('xi_dep_')
xj_dep_df = pd.DataFrame(xj_dep_features).fillna(0).add_prefix('xj_dep_')

# 创建特征 DataFrame
print("正在创建特征 DataFrame...")

features_df = pd.DataFrame()

# 添加词汇量特征
features_df['xi_vocab_size'] = xi_vocab_sizes
features_df['xj_vocab_size'] = xj_vocab_sizes

# 添加情感得分
features_df['xi_sentiment'] = xi_sentiment_scores
features_df['xj_sentiment'] = xj_sentiment_scores

# 添加余弦相似度
features_df['cosine_similarity'] = cosine_similarities

# 合并 POS 和依赖关系特征
features_df = pd.concat([features_df, xi_pos_df, xj_pos_df, xi_dep_df, xj_dep_df], axis=1)

# 处理缺失值
features_df = features_df.fillna(0)

# 添加 LLM 标签
features_df['LLM'] = llm_labels.values

# 统计分析
print("正在进行统计分析...")

# 词汇分析（ANOVA）
print("\n词汇分析（ANOVA）：")
f_stat_xi, p_value_xi = f_oneway(*[features_df[features_df['LLM'] == llm]['xi_vocab_size'] for llm in features_df['LLM'].unique()])
f_stat_xj, p_value_xj = f_oneway(*[features_df[features_df['LLM'] == llm]['xj_vocab_size'] for llm in features_df['LLM'].unique()])

print(f"xi_vocab_size ANOVA 结果：F={f_stat_xi:.4f}, p={p_value_xi:.4e}")
print(f"xj_vocab_size ANOVA 结果：F={f_stat_xj:.4f}, p={p_value_xj:.4e}")

# Tukey 事后检验
print("\nTukey 事后检验（针对 xj_vocab_size）：")
vocab_tukey = pairwise_tukeyhsd(endog=features_df['xj_vocab_size'], groups=features_df['LLM'], alpha=0.05)
print(vocab_tukey)

# 词性和依赖关系分析（Kolmogorov-Smirnov 检验）
print("\n词性和依赖关系分析（Kolmogorov-Smirnov 检验）：")
llm_list = features_df['LLM'].unique()
alpha = 0.05

# POS 特征
pos_feature_names = xi_pos_df.columns.tolist() + xj_pos_df.columns.tolist()
pos_results = {}
for feature in pos_feature_names:
    groups = [features_df[features_df['LLM'] == llm][feature] for llm in llm_list]
    for i in range(len(llm_list)):
        for j in range(i+1, len(llm_list)):
            ks_stat, p_value = ks_2samp(groups[i], groups[j])
            pos_results[(feature, llm_list[i], llm_list[j])] = (ks_stat, p_value)

# 应用 Bonferroni 校正
num_tests = len(pos_results)
corrected_alpha = alpha / num_tests

print("\n词性特征的 KS 检验结果（使用 Bonferroni 校正）：")
for key, (ks_stat, p_value) in pos_results.items():
    feature, llm1, llm2 = key
    significant = p_value < corrected_alpha
    if significant:
        print(f"{feature} 在 {llm1} 和 {llm2} 之间：KS 统计量={ks_stat:.4f}, p值={p_value:.4e} *")
    else:
        print(f"{feature} 在 {llm1} 和 {llm2} 之间：KS 统计量={ks_stat:.4f}, p值={p_value:.4e}")

# 依赖关系特征
dep_feature_names = xi_dep_df.columns.tolist() + xj_dep_df.columns.tolist()
dep_results = {}
for feature in dep_feature_names:
    groups = [features_df[features_df['LLM'] == llm][feature] for llm in llm_list]
    for i in range(len(llm_list)):
        for j in range(i+1, len(llm_list)):
            ks_stat, p_value = ks_2samp(groups[i], groups[j])
            dep_results[(feature, llm_list[i], llm_list[j])] = (ks_stat, p_value)

# 应用 Bonferroni 校正
num_tests = len(dep_results)
corrected_alpha = alpha / num_tests

print("\n依赖关系特征的 KS 检验结果（使用 Bonferroni 校正）：")
for key, (ks_stat, p_value) in dep_results.items():
    feature, llm1, llm2 = key
    significant = p_value < corrected_alpha
    if significant:
        print(f"{feature} 在 {llm1} 和 {llm2} 之间：KS 统计量={ks_stat:.4f}, p值={p_value:.4e} *")
    else:
        print(f"{feature} 在 {llm1} 和 {llm2} 之间：KS 统计量={ks_stat:.4f}, p值={p_value:.4e}")

# 情感分析（Wilcoxon 符号秩检验）
print("\n情感分析（Wilcoxon 符号秩检验）：")
sentiment_features = ['xi_sentiment', 'xj_sentiment']
sentiment_results = {}
for feature in sentiment_features:
    groups = [features_df[features_df['LLM'] == llm][feature] for llm in llm_list]
    for i in range(len(llm_list)):
        for j in range(i+1, len(llm_list)):
            stat, p_value = ranksums(groups[i], groups[j])
            sentiment_results[(feature, llm_list[i], llm_list[j])] = (stat, p_value)

# 应用 Bonferroni 校正
num_tests = len(sentiment_results)
corrected_alpha = alpha / num_tests

print("\n情感分析结果（使用 Bonferroni 校正）：")
for key, (stat, p_value) in sentiment_results.items():
    feature, llm1, llm2 = key
    significant = p_value < corrected_alpha
    if significant:
        print(f"{feature} 在 {llm1} 和 {llm2} 之间：统计量={stat:.4f}, p值={p_value:.4e} *")
    else:
        print(f"{feature} 在 {llm1} 和 {llm2} 之间：统计量={stat:.4f}, p值={p_value:.4e}")

# 可视化情感得分（保留该部分，因为用户要求只生成混淆矩阵和评估表格，可移除）
# 因为用户只需要两个输出，删除所有其他可视化部分

# LLM 归因分类任务
print("\n训练 XGBoost 分类器进行 LLM 归因...")

# 准备数据集
X = features_df.drop(['LLM'], axis=1)
y = features_df['LLM']

# 将数据集划分回训练集和测试集
X_train = X.iloc[:len(train_df)]
X_test = X.iloc[len(train_df):]
y_train = y.iloc[:len(train_df)]
y_test = y.iloc[len(train_df):]

# 标签编码
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 训练 XGBoost 分类器，添加 eval_set 参数
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

eval_set = [(X_train, y_train_encoded), (X_test, y_test_encoded)]
xgb_model.fit(X_train, y_train_encoded, eval_set=eval_set, verbose=False)

# 获取训练过程中的评估结果（可选，如果不需要绘制训练过程，可以删除）
# results = xgb_model.evals_result()

# 绘制训练过程中的对数损失曲线（删除，因为用户不需要）
# plt.figure(figsize=(10, 6))
# plt.plot(results['validation_0']['mlogloss'], label='训练集')
# plt.plot(results['validation_1']['mlogloss'], label='测试集')
# plt.xlabel('轮次')
# plt.ylabel('对数损失')
# plt.title('XGBoost 训练过程中的对数损失')
# plt.legend()
# plt.savefig('xgboost_training_logloss.png')
# plt.show()

# 在测试集上进行预测
y_pred = xgb_model.predict(X_test)

# 评估模型
print("\n分类报告：")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

# 输出评估指标（Accuracy、Recall、Precision、F1 Score）
accuracy = accuracy_score(y_test_encoded, y_pred)
recall = recall_score(y_test_encoded, y_pred, average='weighted')
precision = precision_score(y_test_encoded, y_pred, average='weighted')
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print(f"Accuracy（准确率）：{accuracy:.4f}")
print(f"Recall（召回率）：{recall:.4f}")
print(f"Precision（精确率）：{precision:.4f}")
print(f"F1 Score：{f1:.4f}")

# 绘制混淆矩阵
conf_mat = confusion_matrix(y_test_encoded, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Prediction')  # 设置为英文
plt.ylabel('Actual')      # 设置为英文
plt.title('Confusion Matrix')  # 设置为英文
plt.savefig('confusion_matrix.png')
plt.show()

# 结束
print("脚本运行完毕。生成的混淆矩阵保存在 'confusion_matrix.png'。分类报告已显示在控制台。")
