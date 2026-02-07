import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))

def stemming(txt):
    stemmed_content = re.sub("[^a-zA-Z]", " ", txt) # remove all things that is not letter
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    cleaned =[]
    for word in stemmed_content:
        if word not in stopwords:
            cleaned.append(stemmer.stem(word))
    return ' '.join(cleaned)