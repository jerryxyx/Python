from nltk.corpus import brown, stopwords
from nltk import ConditionalFreqDist
cfd = ConditionalFreqDist()

stopwords_list = stopwords.words('english')

def is_noun(tag):
    return tag.lower() in ['nn','nns','nn$','nn-tl','nn+bez','nn+hvz', 'nns$','np','np$','np+bez','nps','nps$','nr','np-tl','nrs','nr$']

for sentence in brown.tagged_sents():
    for (index, tagtuple) in enumerate(sentence):
        (token, tag) = tagtuple
        token = token.lower()
        if token not in stopwords_list and is_noun(tag):
            window = sentence[index+1:index+5]
            for (window_token, window_tag) in window:
                window_token = window_token.lower()
                if window_token not in stopwords_list and is_noun(window_tag):
                    cfd[token][window_token] += 1

print (cfd['left'].max())
print (cfd['life'].max())
print (cfd['man'].max())
print (cfd['woman'].max())
print (cfd['boy'].max())
print (cfd['girl'].max())
print (cfd['male'].max())
print (cfd['ball'].max())
print (cfd['doctor'].max())
print (cfd['road'].max())