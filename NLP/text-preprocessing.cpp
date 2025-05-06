#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include <algorithm>
using namespace std;

// Function to load lemma dictionary from JSON file
map<string, string> load_lemma_dict() {
    map<string, string> lemma_dict;
    ifstream file("NLP/lemma_dict.json");
    string line;
    while (getline(file, line)) {
        // Simple JSON parsing (for basic "word": "lemma" format)
        size_t pos1 = line.find("\"");
        size_t pos2 = line.find("\"", pos1 + 1);
        size_t pos3 = line.find("\"", pos2 + 1);
        size_t pos4 = line.find("\"", pos3 + 1);
        
        if (pos1 != string::npos && pos2 != string::npos && 
            pos3 != string::npos && pos4 != string::npos) {
            string word = line.substr(pos1 + 1, pos2 - pos1 - 1);
            string lemma = line.substr(pos3 + 1, pos4 - pos3 - 1);
            lemma_dict[word] = lemma;
        }
    }
    return lemma_dict;
}

// Function to convert a word to lowercase
string to_lower(string word) {
    transform(word.begin(), word.end(), word.begin(), ::tolower);
    return word;
}

// Function to stem a single word
string stem_word(const string& word, const map<string, string>& lemma_dict) {
    string lower_word = to_lower(word);
    auto it = lemma_dict.find(lower_word);
    if (it != lemma_dict.end()) {
        return it->second;
    }
    return lower_word;  // Return original word if not found in dictionary
}

// Function to tokenize the text
vector<string> tokenize(const string& text) {
    vector<string> tokens;
    stringstream ss(text);
    string token;
    while(ss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

// Function to stem all tokens
vector<string> stem(const vector<string>& tokens) {
    static const map<string, string> lemma_dict = load_lemma_dict();
    vector<string> stemmed_tokens;
    for(const auto& token : tokens) {
        stemmed_tokens.push_back(stem_word(token, lemma_dict));
    }
    return stemmed_tokens;
}

int main() {
    string sample_text = "Hi, This is a sample text for text preprocessing";
    
    // Tokenization
    vector<string> tokens = tokenize(sample_text);
    cout << "Tokenized text:" << endl;
    for(const auto& t : tokens) {
        cout << t << endl;
    }
    
    // Stemming
    vector<string> stemmed_tokens = stem(tokens);
    cout << "\nStemmed text:" << endl;
    for(const auto& t : stemmed_tokens) {
        cout << t << endl;
    }
    
    return 0;
}

//Compile the code
//g++ -o text_preprocessing text-preprocessing.cpp
//Run the code
//./text_preprocessing

