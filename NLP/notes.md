# 🔸 2. Natural Language Processing (NLP)
Text preprocessing (tokenization, stemming, lemmatization)

### Text preprocessing : 
This is the process of converting raw text into a structured format that can be used for further processing or ingestion into a model.
This process is important because it helps to remove noise and other irrelevant information from the data(text) to make the data fit for modelling.


### Tokenization :
Tokenization is the process of breaking down a text into smaller units called tokens for easier machine analysis

#### Types of tokenization :
1. Word tokenization : This method breaks text down into individual words. It's the most common approach and is particularly effective for languages with clear word boundaries like English.
2. Character tokenization : Here, the text is segmented into individual characters. This method is beneficial for languages that lack clear word boundaries or for tasks that require a granular analysis, such as spelling correction.
3. Subword tokenization: Striking a balance between word and character tokenization, this method breaks text into units that might be larger than a single character but smaller than a full word. For instance, "Chatbots" could be tokenized into "Chat" and "bots". This approach is especially useful for languages that form meaning by combining smaller units or when dealing with out-of-vocabulary words in NLP tasks.

### Stemming :
Stemming is the process of reducing a word to its root or base form also knoe as a stem , this involves removing the affixes(prefixes and suffixes) from a word to create a standard form.
It aims to improve text analysis by simplifying words, making it easier to compare and process them.


### Lemmatization :
Lemmatization is the process of reducing a word to its base form also known as lemma( or dictionary form) , its a technique used to group different inflected forms of a word into a single
represention for a more accurate language analysis.

### N-grams
N-grams are contiguous sequences of n items from a given sample of text or data.These items can be words, letters or base pairs according to the application.
Its mostly used for language modelling and text generation.

#### Types of N-grams :
1. Unigrams : These are single words or tokens.
2. Bigrams : These are two consecutive words or tokens.
3. Trigrams : These are three consecutive words or tokens.
4. N-grams : These are n consecutive words or tokens.

#### N-gram language model :
This model predicts the probability of a given n-gram within a sequence of words. It calculated by calculating the P(w|h) where w is the word and h is the history of words.




### TF-IDF


