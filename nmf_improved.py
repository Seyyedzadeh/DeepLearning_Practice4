from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
from sklearn.datasets import fetch_20newsgroups
import re
import nltk
from nltk.corpus import stopwords


categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]


def clean_text(text):

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    all_names = set(names.words())

    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in all_names]

    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text



groups = fetch_20newsgroups(subset='all', categories=categories)

all_names = set(names.words())

data_cleaned = [clean_text(doc) for doc in groups.data]

count_vector = CountVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)

data = count_vector.fit_transform(data_cleaned)

t = 3
nmf = NMF(n_components=t, random_state=42)

nmf.fit(data)

terms = count_vector.get_feature_names_out()


for topic_idx, topic in enumerate(nmf.components_):
        print("Topic {}:" .format(topic_idx))
        print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

