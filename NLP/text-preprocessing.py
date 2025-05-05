import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')

Sample_text = "Hello, how are you doing today? I hope you are having a great day!"
print("\nSample Text ")
print(Sample_text)

# Tokenization 
tokenized_text = word_tokenize(Sample_text)
print("\nTokenized Text ")
print(tokenized_text)

# Stemming 
stemmer = PorterStemmer()
stemmed_text = [stemmer.stem(word) for word in word_tokenize(Sample_text)]
#stemmed_text = " ".join(stemmed_text)
print("\nStemmed Text ")
print(stemmed_text)

# Lemmatization 
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word) for word in word_tokenize(Sample_text)]
#lemma_text = " ".join(lemmas)
print("\nLemmatized Text ")
print(lemmas)

