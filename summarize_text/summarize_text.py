'''
Natural Language Toolkit book:
https://www.nltk.org/book/

'''

import nltk

'''Stemming'''
porter = nltk.stem.porter.PorterStemmer()

print(porter.stem('running'))
print(porter.stem('jumping'))
print(porter.stem('fishing'))

