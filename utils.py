import os
import string
import numpy as np


def load_emails(path):
    texts = []
    for entry in os.listdir(path):
        is_spam = 0
        if entry[0] == 's':
            is_spam = 1
        with open(os.path.join(path, entry), 'r') as mail:
            next(mail)
            word_count = {}
            words = mail.read().replace('\n', '').split()
            for word in words:
                striped = word.strip(string.punctuation)
                if striped.isalpha():
                    if striped not in word_count:
                        word_count[striped] = 1
                    else:
                        word_count[striped] += 1
            texts.append((word_count, is_spam))
    return texts


def select_feature(data, size):
    word_classes = {}
    spams = 0
    hams = 0
    for word_count, is_spam in data:
        if is_spam:
            spams += 1
        else:
            hams += 1
        for word in word_count:
            if word not in word_classes:
                word_classes[word] = [0, 0]
            word_classes[word][is_spam] += 1
    sorted_features = sorted(word_classes,
                             key=lambda x: conditional(word_classes[x][1], word_classes[x][0], spams, hams),
                             reverse=True)
    return sorted_features[0: min(size, len(sorted_features))]


def conditional(count_spam, count_ham, spams, hams):
    entropy = 0
    if count_spam > 0:
        entropy += count_spam * np.log(count_spam / (count_spam + count_ham))
    if count_ham > 0:
        entropy += count_ham * np.log(count_ham / (count_spam + count_ham))
    if spams - count_spam > 0:
        entropy += (spams - count_spam) * np.log((spams - count_spam) / (spams + hams - count_spam - count_ham))
    if hams - count_ham > 0:
        entropy += (hams - count_ham) * np.log((hams - count_ham) / (spams + hams - count_spam - count_ham))
    return entropy


def evaluate(eval_model, eval_data):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for text, is_spam in eval_data:
        prediction = eval_model.predict(text)
        if is_spam == 1:
            if prediction == 1:
                true_pos += 1
            else:
                false_neg += 1
        else:
            if prediction == 1:
                false_pos += 1
            else:
                true_neg += 1
    eval_precision = true_pos / (true_pos + false_pos)
    eval_recall = true_pos / (true_pos + false_neg)
    false_pos_rate = false_pos / (false_pos + true_neg)
    false_neg_rate = false_neg / (true_pos + false_neg)
    return eval_precision, eval_recall, false_pos_rate, false_neg_rate



