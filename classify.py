import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, decomposition
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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

hashmat = feature_extraction.text.HashingVectorizer()
hashmat_train = hashmat.fit_transform(train_df["text"])
scores = model_selection.cross_val_score(clf,
    hashmat_train,
    train_df["target"],
    cv=3,
    scoring="f1")
# hashmat already gives a decent approximation.
# what if we provide all features to an XGBclassifier
import xgboost as xgb

# features:

hashmat_train
count_train_vec
freq_train_vec


X = scipy.sparse.hstack([count_train_vec, freq_train_vec, hashmat_train])
y = train_df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# set up quick random XGBClassifier
xgbclass = xgb.XGBClassifier()

params = {
    'objective':['binary:logistic'],
    'learning_rate':[0.01, 0.05, 0.1, 0.15, 0.5],
    'colsample_bytree':[0.01, 0.1, 0.3, 0.6],
    'max_depth':[3, 4, 7, 10, 15, 20],
    'alpha':[0.01,0.1, 1, 3, 5, 7, 10, 15],
    'lambda':[0.01,0.1,0.5,1],
    'gamma':[0.01, 0.1, 0.5, 1, 2, 4, 5,],
    'subsample':[0.5,0.7,1]
}

random_search = RandomizedSearchCV(
    xgbclass,
    param_distributions = params,
    cv = 3,
    scoring = "f1")

random_search.fit(X_train, y_train)

random_search.best_score_
random_search.score(X_test, y_test)

y_train_hat = random_search.predict(X_train)
y_test_hat = random_search.predict(X_test)

print('train performance')
print('--------------------------------')
print(classification_report(y_train, y_train_hat))

print('test performance')
print('--------------------------------')
print(classification_report(y_test, y_test_hat))

print('ROC_AUC score')
print('--------------------------------')
print(roc_auc_score(y_test, y_test_hat))
print('')

print('Confusion Matrix')
print('--------------------------------')
print(confusion_matrix(y_test, y_test_hat))
# this score is on the training set, how does it perform on the testing set?

