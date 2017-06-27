from nltk.corpus import gutenberg
from nltk import FreqDist
import matplotlib.pyplot as plt

fd = FreqDist()

for word in gutenberg.words('bible-kjv.txt'):
    fd[word] += 1
    
print (fd.N())

print (fd.B())

for word in list(fd.keys()):
    print (word, fd[word])

fd2 = FreqDist()
for text in gutenberg.fileids():
    for word in gutenberg.words(text):
        fd2[word] += 1

ranks = []
freqs = []

for rank, word in enumerate(fd2):
    ranks.append(rank+1)
    freqs.append(fd2[word])

plt.loglog(ranks, freqs)
plt.xlabel('frequency(f)',fontsize=14, fontweight='bold')
plt.ylabel('rank(r)', fontsize=14, fontweight='bold')
plt.grid(True)
plt.show()
