import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split as ttp

def tokenize(text,stop_words = None):

    tokens = word_tokenize(text)
    stemming = []
    for i in tokens:
        stemming.append(PorterStemmer().stem(i))
    filtered_stemming = []
    for i in range(len(stemming)):
        stemming[i] = stemming[i].lower()
        if stemming[i] not in stop_words or stop_words == None:
            filtered_stemming.append(stemming[i])
    filtered_stemming =' '.join(filtered_stemming)
    return filtered_stemming

ds = pd.read_csv("onion-or-not.csv")

X = ds.iloc[:,0]
Y = ds.iloc[:,1].values

stop_words=set(stopwords.words("english"))

"""Natural Language Preprocessing"""

stemmed_data =[]
for i in range(np.size(X)):
    stemmed_data += [tokenize(X[i],stop_words)]

"""END"""


"""tf-idf vectorizer"""

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(stemmed_data)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

"""END"""

"""Training and test data"""

X = df.iloc[:,:].values
Y = np.reshape(Y,[np.size(Y),1])

X_training, X_testing, Y_training, Y_testing = ttp(X,Y,test_size=0.25,random_state= 1234)

"""END"""

"""Neural Network"""

mlp = MLPClassifier(hidden_layer_sizes=(8,4),max_iter=200)
mlp.fit(X_training,Y_training)

prediction = mlp.predict(X_testing)

print(classification_report(Y_testing,prediction))
print(confusion_matrix(Y_testing,prediction))

"""End"""