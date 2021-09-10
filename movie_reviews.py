'''
From https://pythonprogramming.net/text-classification-nltk-tutorial/
'''

import os
import nltk
import random
from nltk.corpus import movie_reviews
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

print(f'There are {len(documents)} total reviews')

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print('Most Common: ', all_words.most_common(15))
print(all_words['ridiculous'], ' occurrences of ridiculous')

#   Take the frequency distribution and return the first 3000 most common.
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)  #   Is the word there or not?

    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category)
               for (rev, category) in documents]

# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)

os.chdir('classifiers')
with open(f'naivebayes.pickle', 'wb') as f:
    pickle.dump(classifier, f)


classifiers_dict = {'Multinomial Naive Bayes': SklearnClassifier(MultinomialNB()),
'Bernoulli Naive Bayes': SklearnClassifier(BernoulliNB()),
'Logistic Regression Classifier': SklearnClassifier(LogisticRegression()),
'Stochastic Gradient Descent Classifier': SklearnClassifier(SGDClassifier()),
'Support Vector Curve Classifier': SklearnClassifier(SVC()),
'Linear Support Vector Curve Classifier': SklearnClassifier(LinearSVC()),
'Nu SVC Classifier': SklearnClassifier(NuSVC())}

for name, wrapped_classifier in classifiers_dict.items():
    wrapped_classifier.train(training_set)
    print(f'{name} percent: {(nltk.classify.accuracy(wrapped_classifier, testing_set))*100}')
    with open(f'{name}.pickle', 'wb') as f:
        pickle.dump(wrapped_classifier, f)