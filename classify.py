import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, decomposition
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('/Users/christianbouwens/Documents/sideprojects/NLP_kaggle/train.csv')
test_df = pd.read_csv('/Users/christianbouwens/Documents/sideprojects/NLP_kaggle/test.csv')

train_df[train_df["target"] == 0]["text"].values[1]
train_df[train_df["target"] == 1]["text"].values[1]



# feature extraction:

# number of words or common words in distaster tweets

count_vec = feature_extraction.text.CountVectorizer()


count_train_vec = count_vec.fit_transform(train_df["text"])
count_test_vec = count_vec.transform(test_df['text'])
clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf,
    count_train_vec,
    train_df["target"],
    cv=3, scoring="f1")
scores

# model did okay, lets run it with Tfidf
term_freq = feature_extraction.text.TfidfVectorizer()
freq_train_vec = term_freq.fit_transform(train_df["text"])
freq_test_vec = term_freq.transform(test_df["text"])
scores = model_selection.cross_val_score(clf,
    freq_train_vec,
    train_df["target"],
    cv=3,
    scoring="f1")

# can we combine both features?
# horizontally stack features:
import scipy.sparse

X = scipy.sparse.hstack([count_train_vec, freq_train_vec])

scores = model_selection.cross_val_score(clf,
    X,
    train_df["target"],
    cv=3,
    scoring="f1")
# pretty much the same score, if anything it seems slightly lower.

# try SVD (LSA in this case):
svd = decomposition.TruncatedSVD()

freq_svd = svd.fit_transform(freq_train_vec)
scores = model_selection.cross_val_score(clf,
    freq_svd, train_df["target"],
    cv=3,
    scoring="f1")
# that didn't do much of anything. how about we use logistic regression?

# how about using a different learning algorithm?
lgrg = linear_model.LogisticRegression()
scores = model_selection.cross_val_score(lgrg,
    freq_svd,
    train_df["target"],
    cv=3,
    scoring="f1")


# XGBoost it?
import xgboost as xgb
# this score is on the training set, how does it perform on the testing set?

