from nltk.corpus import gutenberg
from nltk import ConditionalFreqDist
from random import choice

cfd = ConditionalFreqDist()

prev_word = None
for word in gutenberg.words('bible-kjv.txt'):
    cfd[prev_word][word] += 1
    prev_word = word

word = 'I'
i = 1

print(list(cfd[word].values()))

while i < 10:
    print (word)
    lwords =list(cfd[word].keys())
    follower = choice(lwords)
    word = follower
    i += 1