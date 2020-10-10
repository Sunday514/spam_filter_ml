import os
import pandas as pd
from .models.nb_classifier import NB
from .models.svm_classifier import SVM
from .adversarial import Attacker, Defender
from .utils import *

if __name__ == '__main__':
    # load data
    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'lemm_stop')
    train_emails = []
    for i in range(1, 10):
        train_emails.extend(load_emails(os.path.join(base_path, 'part' + str(i))))
    test_emails = load_emails(os.path.join(base_path, 'part10'))
    # top 10 features
    top_10 = select_feature(train_emails, 10)
    print('top 10 features:', top_10)

    # create naive bayes models
    nb_models = {
        'Binomial NB with BF': NB('Binomial BF'),
        'Multinomial NB with BF': NB('Multinomial BF'),
        'Multinomial NB with TF': NB('Multinomial TF')
    }
    # train ans test models
    nb_results = {}
    for feature_size in [10, 100, 1000]:
        for model_name, model in nb_models.items():
            # select features
            features = select_feature(train_emails, feature_size)
            model.fit(train_emails, features)
            precision, recall, _, _ = evaluate(model, test_emails)
            nb_results[model_name + ', N=' + str(feature_size)] = [precision, recall]
    nb_results_df = pd.DataFrame(nb_results, index=['precision', 'recall']).T
    # print results
    print(nb_results_df)

    # create svm models
    svm_models = {
        'Linear SVM': SVM('linear'),
        'RBF kernel SVM': SVM('rbf')
    }
    # train ans test models
    svm_results = {}
    for feature_size in [10, 100, 1000]:
        for model_name, model in svm_models.items():
            # select features
            features = select_feature(train_emails, feature_size)
            model.fit(train_emails, features)
            precision, recall, _, _ = evaluate(model, test_emails)
            svm_results[model_name + ', N = ' + str(feature_size)] = [precision, recall]
    svm_results_df = pd.DataFrame(svm_results, index=['Precision', 'Recall']).T
    # print results
    print(svm_results_df)

    # adversarial test
    baseline = NB('Multinomial BF')
    baseline.fit(train_emails, top_10)
    _, _, _, base_bnr_before = evaluate(baseline, test_emails)
    print('Before attack:')
    print('False negative rate of the baseline NB classifier = ', base_bnr_before)
    # attack launched
    attacker = Attacker(baseline)
    total_cost = 0
    for email, _ in test_emails:
        total_cost += attacker.attack(email, 10)
    print('Attack Launched......')
    print('Average cost by attacker = ', total_cost / len(test_emails))
    _, _, _, base_fnr_after = evaluate(baseline, test_emails)
    print('After attack:')
    print('False negative rate of the baseline NB classifier = ', base_fnr_after)
    # update the classifier to defend the attack
    updated = Defender()
    updated.fit(train_emails, top_10)
    _, _, updated_fpr, updated_fnr = evaluate(updated, test_emails)
    print('Classifier updated......')
    print('False negative rate of the updated NB classifier = ', updated_fnr)
    print('False positive rate of the updated NB classifier = ', updated_fpr)
