import pandas as pd
outlook=['sunny','overcast','rainy']
temperature=['hot','mild','cool']
humidity=['high','normal']
windy=['TRUE','FALSE']
weather_dataset=[[o,t,h,w] for o in outlook for t in temperature for h in humidity for w in windy]
play=[0,0,1,1,1,0,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1]
df=pd.DataFrame(weather_dataset,columns=['outlook','temperature','humidity','windy'])

#DataFrame.get_dummies
dummy_encoding = pd.get_dummies(df)
import statsmodels.api as sm
X=sm.add_constant(dummy_encoding)
logit=sm.Logit(play,X)
result=logit.fit()

#feature_extraction.DictVectorizer
from sklearn import feature_extraction
dict_representation=[{varname:var for varname,var in zip(['outlook','temperature','humidity','windy'],row)} for row in weather_dataset]
vectorizer = feature_extraction.DictVectorizer()
print(vectorizer.fit_transform(dict_representation))

#feature_extraction.text.CountVectorizer
corpus = [' The quick fox jumped over the lazy dog', 'I sought a dog wondering around with a bird', 'My dog is named Fido']
textual = feature_extraction.text.CountVectorizer()
print(textual.fit_transform(corpus))
