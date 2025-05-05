# Importing necessary libraries
using CSV
using DataFrames
using TextAnalysis
using WordTokenizers
using Languages: English
using Snowball
using JSON

# Load the comprehensive lemmatization dictionary
const LEMMA_DICT = Dict{String,String}(
    pairs(JSON.parsefile(joinpath(@__DIR__, "lemma_dict.json")))
)

# Define a simple lemmatization function
function lemmatize(word::String)
    # Convert to lowercase for consistency
    word_lower = lowercase(word)
    # Return the lemma if it exists in the dictionary, otherwise return the original word
    return get(LEMMA_DICT, word_lower, word)
end

# Sample text
sample_text = "Hello, how are you doing today? I hope you are having a great day!"
println("Sample Text:")
println(sample_text)

# Tokenization
tokenized_text = WordTokenizers.tokenize(sample_text)
println("\nTokenized Text:")
println(tokenized_text)

# Stopwords removal
# Using TextAnalysis for stopwords
stop_words = TextAnalysis.stopwords(English())
filtered_text = filter(word -> !(lowercase(word) in stop_words), tokenized_text)
println("\nFiltered Text:")
println(filtered_text)

# Stemming
stemmer = Stemmer("english")
stemmed_text = [stem(stemmer, word) for word in filtered_text if !isempty(word) && !ispunct(first(word))]
println("\nStemmed Text:")
println(stemmed_text)

# Lemmatization
lemmatized_text = [lemmatize(word) for word in filtered_text if !isempty(word) && !ispunct(first(word))]
println("\nLemmatized Text:")
println(lemmatized_text)

