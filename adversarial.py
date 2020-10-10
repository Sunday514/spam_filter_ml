import numpy as np
from models.nb_classifier import NB

import os
import utils


# the adversary that tries to cheat the NB classifier model by adding words
class Attacker:
    def __init__(self, model):
        if model.classifier_type != 'Multinomial BF':
            raise ValueError('Invalid classifier type.')
        self.features = []
        self.log_odds = []
        for word in model.spam_prob:
            self.features.append(word)
            self.log_odds.append(int((np.log(model.spam_prob[word]) - np.log(model.ham_prob[word])) * 100))

    def attack(self, text, max_cost):
        one_hot = self.text2vec(text)
        cost, add_words = self.get_list(one_hot, max_cost)
        if add_words is not None:
            for i in add_words:
                text[self.features[i - 1]] = 1
            return cost
        return 0

    def get_list(self, one_hot, max_cost):
        gap = self.get_gap(one_hot)
        add_list = None
        cost, add_indices = self.find_mcc(one_hot, len(one_hot), gap)
        if gap > 0 and cost <= max_cost:
            add_list = add_indices
        return cost, add_list

    def text2vec(self, text):
        one_hot = []
        for word in self.features:
            if word in text:
                one_hot.append(1)
            else:
                one_hot.append(0)
        return one_hot

    def get_gap(self, one_hot):
        gap = 0
        for i, present in enumerate(one_hot):
            if present == 1:
                gap += self.log_odds[i]
        return gap

    # find the minimum cost to change the text to non-spam
    def find_mcc(self, one_hot, i, gap):
        if gap <= 0:
            return 0, []
        if i == 0:
            return np.inf, None
        min_cost = np.inf
        min_list = None
        if one_hot[i - 1] == 0 and self.log_odds[i - 1] < 0:
            cur_cost, cur_list = self.find_mcc(one_hot, i - 1, gap + self.log_odds[i - 1])
            if cur_cost + 1 < min_cost:
                min_cost = cur_cost + 1
                min_list = cur_list
                min_list.append(i)
        else:
            cur_cost, cur_list = self.find_mcc(one_hot, i - 1, gap)
            if cur_cost < min_cost:
                min_cost = cur_cost
                min_list = cur_list
        return min_cost, min_list


class Defender(NB):
    def __init__(self):
        super(Defender, self).__init__('Multinomial BF')
        self.adversary = Attacker(self)

    def fit(self, texts, features):
        super(Defender, self).fit(texts, features)
        self.adversary = Attacker(self)

    def predict(self, text):
        one_hot = self.adversary.text2vec(text)
        originals = self.get_original(one_hot)
        spam_prob_vec = list(self.spam_prob.values())
        ham_prob_vec = list(self.ham_prob.values())
        predict_spam = 0
        predict_ham = 0
        exp_predict_spam = 0
        for i in range(len(one_hot)):
            if one_hot[i] == 1:
                predict_spam += np.log(spam_prob_vec[i])
        for i in range(len(one_hot)):
            if one_hot[i] == 1:
                predict_ham += np.log(ham_prob_vec[i])
        if predict_spam <= predict_ham or len(originals) == 0:
            exp_predict_spam += np.exp(predict_spam)
        for original in originals:
            predict_original = 0
            for i in range(len(original)):
                if original[i] == 1:
                    predict_original += np.log(spam_prob_vec[i])
            exp_predict_spam += np.exp(predict_original)
        if exp_predict_spam > np.exp(predict_ham):
            return 1
        return 0

    def get_original(self, one_hot):
        originals = []
        for i in range(len(one_hot)):
            if one_hot[i] == 1:
                one_hot[i] = 0
                _, mcc_list = self.adversary.get_list(one_hot, 1)
                if mcc_list is not None and mcc_list[0] == i + 1:
                    originals.append(one_hot.copy())
                one_hot[i] = 1
        return originals
