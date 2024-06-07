# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import chardet
import re

# %%
with open('emails.csv', 'rb') as f:
    result = chardet.detect(f.read())

df = pd.read_csv('emails.csv', encoding=result['encoding'])

# %%
df['text'] = df['text'].str.lower()

# %%
df['text'] = df['text'].apply(lambda text: re.sub(r'[^a-zA-Z\s]', '', text))
df['text'] = df['text'].apply(lambda text: re.sub(r'\bsubject\b', '', text, flags=re.IGNORECASE))

# %%
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
X = X.toarray()
# print(vectorizer.get_feature_names_out())
y = np.array(df['spam'])
y

# %%
#Dataset to be used to train our model

emails = X
labels = y
N = emails.shape[0]

#Split dataset for internal training and testing for tuning hyperparameters

emails_train, emails_test, labels_train, labels_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

emails_train_nba = np.where(emails != 0, 1, emails)

# %%
spam_emails = [np.array(x) for i, x in enumerate(emails_train_nba) if labels[i] == 1]
ham_emails = [np.array(x) for i, x in enumerate(emails_train_nba) if labels[i] == 0]
# laplace smoothing
spam_emails.append(np.ones(spam_emails[0].shape))
spam_emails = np.array(spam_emails)
ham_emails.append(np.ones(ham_emails[0].shape))
ham_emails = np.array(ham_emails)
p_being_spam = len(spam_emails)/(len(spam_emails) + len(ham_emails))
prob_spam = spam_emails.sum(axis=0)/len(spam_emails)
prob_ham = ham_emails.sum(axis=0)/len(ham_emails)

# %%
# ans_labels = np.zeros(len(labels_test))
w_naive = np.log(prob_spam*(1-prob_ham)/(prob_ham*(1-prob_spam)), dtype=np.float64)
b_naive = np.log(p_being_spam/(1-p_being_spam), dtype=np.float64) + np.sum(np.log((1-prob_spam)/(1-prob_ham)))
# for i, email in enumerate(emails_test):
#     email = np.where(email != 0, 1, email)
#     if np.dot(w_naive, email) + b_naive >= 0:
#         ans_labels[i] = 1

# %%
# error_nba = np.sum(np.abs(ans_labels - labels_test))
# print("Accuracy: ",1- (error_nba/len(labels_test)))

# %%
def sigmoid (x):
    y = x
    # Since exp(20) is a large number, we need to limit y within a reasonable range to avoid overflow
    #Also It does not affect sigmoid function as sigmoid(20) is extremely close to 1
    y = np.where(y > 20, 20, y)

    # Since exp(-20) is a small number, we need to limit y within a reasonable range to avoid underflow
    #Also It does not affect sigmoid function as sigmoid(-20) is extremely close to 0
    y = np.where(y < -20, -20, y)
    return 1/(1 + np.exp(-y))
def sgn(x):
    if x>0:
        return 1
    else:
        return 0
def gradient(x, y, w):
    y_new = sigmoid(np.dot(x, w))
    return np.dot(x.T, y - y_new)
def update(w, x, y, stepsize):
    return w + stepsize * gradient(x, y, w)

# %%
stepsize = 0.01
w = np.random.rand(emails.shape[1])
error = []
for i in range(100):
    w = update(w, emails, labels, stepsize)

# %%
# ans_labels_lr = np.zeros(len(labels_test))
# for i, email in enumerate(emails_test):
#     # email = np.where(email != 0, 1, email)
#     ans_labels_lr[i] = sgn(np.dot(email, w))

# %%
# test_error = np.sum(np.abs(ans_labels_lr - labels_test))
# print("Accuracy: ", 1-(test_error/len(labels_test)))

# %% [markdown]
# SVM

# %%
clf = svm.LinearSVC(C= 1, max_iter=10000)
clf.fit(emails, labels)
# pred = clf.predict(emails_test)

# %%
# error_svm = np.sum(np.abs(pred - labels_test))
# print("Accuracy: ", 1 - (error_svm/len(labels_test)))

# %% [markdown]
# Gaussian Naive Bayes Algorithm

# %%
# spam_indices = labels_train == 1
# ham_indices = labels_train == 0
# mean_spam = np.sum(emails_train[spam_indices], axis=0) / np.sum(spam_indices)
# mean_ham = np.sum(emails_train[ham_indices], axis=0) / np.sum(ham_indices)
# mean_i = []
# for i in range(len(labels_train)):
#     if labels_train[i] == 1:
#         mean_i.append(mean_spam)
#     else:
#         mean_i.append(mean_ham)


# %%
# mean_i = np.array(mean_i)
# x_new = emails_train - mean_i
# cov = np.matmul(x_new.T, x_new, dtype='float64') / len(labels_train)

# %%
# for d in range(cov.shape[0]):
#     cov[d][d] += 1e-6

# %%
# p = np.sum(labels_train) / len(labels_train)
# cov_inv = np.linalg.inv(cov)

# %%
# emails_test_0 = emails_test - mean_ham
# emails_test_1 = emails_test - mean_spam
# a_1 = np.matmul(emails_test_0, cov_inv)
# b_1 = [np.dot(a_1[i], emails_test_0[i]) for i in range(len(emails_test_0))]
# b_1 = np.array(b_1)
# a_2 = np.matmul(emails_test_1, cov_inv)
# b_2 = [np.dot(a_2[i], emails_test_1[i]) for i in range(len(emails_test_1))]
# b_2 = np.array(b_2)
# a_3 = np.log(p/(1-p))
# ans = (b_1 - b_2) + a_3
# ans_labels_gnba = np.where(ans > 0, 1, 0)

# %%
# ans_labels_gnba

# %%
# error_gnba = np.sum(np.abs(ans_labels_gnba - labels_test))
# error_gnba

# %%
import os
directory = './test'
txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
test_emails = []
for file in txt_files:
    with open(f'{directory}/{file}', 'r') as f:
        test_emails.append(f.read())

# %%
test_emails = vectorizer.transform(test_emails)
test_emails = test_emails.toarray()
pred_svm = clf.predict(test_emails)
pred_lr = [sgn(np.dot(email, w)) for email in test_emails]
pred_nba = np.zeros(len(test_emails))
for i, email in enumerate(test_emails):
    email = np.where(email != 0, 1, email)
    if np.dot(w_naive, email) + b_naive >= 0:
        pred_nba[i] = 1

# %%
# test_emails_0 = test_emails - mean_ham
# test_emails_1 = test_emails - mean_spam
# a_1 = np.matmul(test_emails_0, cov_inv)
# b_1 = [np.dot(a_1[i], test_emails_0[i]) for i in range(len(test_emails_0))]
# b_1 = np.array(b_1)
# a_2 = np.matmul(test_emails_1, cov_inv)
# b_2 = [np.dot(a_2[i], test_emails_1[i]) for i in range(len(test_emails_1))]
# b_2 = np.array(b_2)
# a_3 = np.log(p/(1-p))
# ans = (b_1 - b_2) + a_3
# pred_gnba = np.where(ans > 0, 1, 0)

# %%
pred_lr = np.array(pred_lr)
pred_svm, pred_lr, pred_nba

# %%
final_pred = pred_svm + pred_lr + pred_nba
final_pred = np.where(final_pred >= 2, 1, 0)

# %%
final_pred

# %%
with open('predictions.txt', 'w') as f:
    for i in range(len(final_pred)):
        f.write(f"Email {i+1} : {final_pred[i]}\n")


