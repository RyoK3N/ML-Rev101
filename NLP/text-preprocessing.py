import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import bigrams, trigrams, ngrams as nltk_ngrams
import re 
from nltk.collections import defaultdict
#from nltk.corpus import reuters

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

# Clean the text then tokenize and create n grams 

# Cleaned text  
# Remove punctuation
cleaned_text = re.sub(r'[^\w\s]','',Sample_text)
# print("\nCleaned Text ")
# print(cleaned_text)

#Tokenize the cleaned text 
cleaned_tokens = word_tokenize(cleaned_text)
print("\n Cleaned Tokens ")
print(cleaned_tokens)

#Create n-grams
#unigrams
unigrams = cleaned_tokens
print("\nUnigrams")
print(unigrams)

#bigrams
n = 2  # Changed from 3 to 2 for proper bigrams
bigrams = list(nltk_ngrams(unigrams, n))
print("\nBigrams")
print(bigrams)

#trigrams
n = 3
trigrams = list(nltk_ngrams(unigrams, n))
print("\nTrigrams")
print(trigrams)

#ngrams
n = int(input("Enter the value of n :"))
custom_ngrams = list(nltk_ngrams(unigrams, n))  # Changed variable name to avoid conflict
print("\nN-grams")
print(custom_ngrams)

# N-gram language model 
print("\nBuilding language model from training data...")

# Text Data for N-gram language model 
data_path = "/Users/code.ai/Desktop/Workspace/Revision/RAG/ref_doc.txt"

# Read and preprocess the data
with open(data_path, 'r') as file:
    text = file.read().lower()  # Convert to lowercase

# Clean the text - keep only letters, numbers and spaces
cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
# Split into sentences to avoid creating n-grams across sentences
sentences = cleaned_text.split('.')

# Create trigrams from each sentence
all_trigrams = []
for sentence in sentences:
    # Tokenize the sentence
    tokens = word_tokenize(sentence.strip())
    if len(tokens) >= 3:  # Only process sentences with at least 3 words
        sentence_trigrams = list(nltk_ngrams(tokens, 3))
        all_trigrams.extend(sentence_trigrams)

# Build the language model
model = defaultdict(lambda: defaultdict(lambda: 0))

# Count the occurrences
for w1, w2, w3 in all_trigrams:
    model[(w1, w2)][w3] += 1

# Transform counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

# Show examples from the model
print("\nExample word pairs from the training data (with their next words):")
example_pairs = list(model.keys())[:5]  # Show first 5 word pairs
for pair in example_pairs:
    next_words = list(model[pair].keys())[:3]  # Show up to 3 possible next words
    print(f"'{pair[0]} {pair[1]}' -> possible next words: {next_words}")

def predict_nextword(w1, w2):
    w3_prob = model[w1,w2]
    if w3_prob:
        return max(w3_prob, key=w3_prob.get)
    else:
        return "No prediction available - word pair not found in training data"

print("\nEnter the first two words of the sentence:")
print("(Note: Try using some of the example pairs shown above)")
w1 = input("First word: ")
w2 = input("Second word: ")

next_word = predict_nextword(w1, w2)
if isinstance(next_word, str) and "No prediction" in next_word:
    print("\nWarning:", next_word)
else:
    print("\nThe predicted next word is:", next_word)

