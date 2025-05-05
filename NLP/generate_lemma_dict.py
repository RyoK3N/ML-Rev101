import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import json

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wn.ADJ,
        "N": wn.NOUN,
        "V": wn.VERB,
        "R": wn.ADV
    }
    return tag_dict.get(tag, wn.NOUN)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Create dictionary to store lemmas
lemma_dict = {}

# Get all synsets
for synset in wn.all_synsets():
    # Get all lemmas for the synset
    for lemma in synset.lemmas():
        word = lemma.name()
        pos = synset.pos()
        
        # Get the base form
        base_form = lemmatizer.lemmatize(word, pos=pos)
        
        # Skip if word is the same as its lemma
        if word != base_form:
            lemma_dict[word] = base_form
        
        # Handle derived forms
        derived_forms = lemma.derivationally_related_forms()
        for derived in derived_forms:
            derived_word = derived.name()
            if derived_word != base_form:
                lemma_dict[derived_word] = base_form

# Add common contractions
contractions = {
    "n't": "not",
    "'s": "be",
    "'m": "be",
    "'re": "be",
    "'ve": "have",
    "'ll": "will",
    "'d": "would",
    "wo": "will",
    "ca": "can",
    "sha": "shall"
}
lemma_dict.update(contractions)

# Save to JSON file
with open('./lemma_dict.json', 'w') as f:
    json.dump(lemma_dict, f, indent=4)

print(f"Generated dictionary with {len(lemma_dict)} entries") 