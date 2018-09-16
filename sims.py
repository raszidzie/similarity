import gensim
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
documents = []
str1 = input("Enter Str1:  ")
documents.append(str1)
str2 = input("Enter Str2:  ")
documents.append(str2)


gen_strings = [[w.lower() for w in  wordpunct_tokenize((text))] 
            for text in documents]



        
dictionary = gensim.corpora.Dictionary(gen_strings )
print("Number of words in dictionary:",len(dictionary))
for i in range(len(dictionary)):
    print(i, dictionary[i])


simlist = [dictionary.doc2bow(gen_strings) for gen_strings in gen_strings]
print(simlist)

tf_idf = gensim.models.TfidfModel(simlist)
print(tf_idf)
s = 0
for i in simlist:
    s += len(i)
print(s)
sims = gensim.similarities.Similarity('',tf_idf[simlist],
                                      num_features=len(dictionary))
print(sims)
print(type(sims))
str3 = input("Enter str3:  ")
query_doc = [w.lower() for w in word_tokenize(str3)]
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)
print( 'Similarity:',sims[query_doc_tf_idf]*100, '%')


