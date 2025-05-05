# NLP in Julia
In this readme, I will be using the `TextAnalysis` package to perform text preprocessing tasks in Julia.
Also there would be a detailed explaination on how to run the code and get the output

### Tokenization
To tokenize the text, we will be using the `WordTokenizers` package.

### Stemming
To stem the text, we will be using the `Snowball` package.

### Lemmatization
To lemmatize we use python's nltk library to extract lemma  and generate all the synsets and then use the `WordNet` package to get the lemma for the words , and then we use the `JSON` package to save the lemma dictionary.

We use the `generate_lemma_dict.py` file to generate the lemma dictionary, and then we use the `lemma_dict.json` file to load the lemma dictionary in the `text-preprocessing.jl` file to lemmatize the text.

### Stopwords Removal
To remove the stopwords we use the `TextAnalysis` package.

### Installing the packages 

```julia -e 'using Pkg; Pkg.add("TextAnalysis"); Pkg.add("WordTokenizers"); Pkg.add("Snowball"); Pkg.add("JSON"); Pkg.add("Languages"); Pkg.add("CSV"); Pkg.add("DataFrames")'
```

### Running the code

#### Generate the lemma dictionary

```python generate_lemma_dict.py```

#### Run the text-preprocessing.jl file

```julia text-preprocessing.jl```

### Make sure that julia is installed in your system
### Make sure that the `generate_lemma_dict.py` file is in the same directory as the `text-preprocessing.jl` file
### Make sure that the `lemma_dict.json` file is in the same directory as the `text-preprocessing.jl` file


