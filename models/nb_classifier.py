import numpy as np


# Naive Bayes classifier
class NB:

    def __init__(self, classifier_type):
        if classifier_type not in ['Binomial BF', 'Multinomial BF', 'Multinomial TF']:
            raise ValueError('Invalid classifier type.')
        self.classifier_type = classifier_type
        self.spam_prob = {}
        self.ham_prob = {}

    def fit(self, texts, features):
        spam_dict = {}
        ham_dict = {}
        texts_spam = 0
        texts_ham = 0
        # Laplace smoothing
        for word in features:
            spam_dict[word] = 1
            ham_dict[word] = 1
        for word_count, is_spam in texts:
            if is_spam == 1:
                texts_spam += 1
                for word in word_count:
                    if word in spam_dict:
                        if self.classifier_type[0] == 'B':
                            spam_dict[word] += 1
                        elif self.classifier_type[0] == 'M':
                            spam_dict[word] += word_count[word]
            else:
                texts_ham += 1
                for word in word_count:
                    if word in ham_dict:
                        if self.classifier_type[0] == 'B':
                            ham_dict[word] += 1
                        elif self.classifier_type[0] == 'M':
                            ham_dict[word] += word_count[word]
        if self.classifier_type[0] == 'B':
            spam_count = texts_spam + 2
            ham_count = texts_ham + 2
        else:
            spam_count = sum(spam_dict.values())
            ham_count = sum(ham_dict.values())
        for word in spam_dict:
            self.spam_prob[word] = spam_dict[word] / spam_count
        for word in ham_dict:
            self.ham_prob[word] = ham_dict[word] / ham_count

    def predict(self, text):
        # spam_rate is the prior probability that an email is spam, 0.5 by default
        predict_spam = 0
        predict_ham = 0
        # Binomial BF
        if self.classifier_type[0] == 'B':
            for word in self.spam_prob:
                if word in text:
                    predict_spam += np.log(self.spam_prob[word])
                else:
                    predict_spam += np.log(1 - self.spam_prob[word])
            for word in self.ham_prob:
                if word in text:
                    predict_ham += np.log(self.ham_prob[word])
                else:
                    predict_ham += np.log(1 - self.ham_prob[word])
        # Multinomial NB
        elif self.classifier_type[0] == 'M':
            for word in text:
                xi = text[word]
                # Multinomial BF
                if self.classifier_type[-2] == 'B':
                    xi = 1
                if word in self.spam_prob:
                    predict_spam += xi * np.log(self.spam_prob[word])
                if word in self.ham_prob:
                    predict_ham += xi * np.log(self.ham_prob[word])
        # return 1 if we classify this email as spam and 0 if not
        if predict_spam > predict_ham:
            return 1
        return 0
