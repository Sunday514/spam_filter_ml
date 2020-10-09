import numpy as np
from sklearn import svm


# svm classifier
class SVM:
    def __init__(self, kernel):
        self.clf = svm.SVC(kernel=kernel)
        self.feature_indices = {}

    def fit(self, texts, features):
        # transform dictionary to vector
        self.feature_indices = {features[i]: i for i in range(len(features))}
        train_data = np.zeros((len(texts), len(self.feature_indices)))
        labels = np.zeros((len(texts),))
        for i, (word_count, is_spam) in enumerate(texts):
            for word, count in word_count.items():
                if word in self.feature_indices:
                    train_data[i, self.feature_indices[word]] = count
            labels[i] = is_spam
        self.clf.fit(train_data, labels)

    def predict(self, text):
        test_data = np.zeros((1, len(self.feature_indices)))
        for word, count in text.items():
            if word in self.feature_indices:
                test_data[0, self.feature_indices[word]] = count
        label = self.clf.predict(test_data)
        return int(np.squeeze(label))


