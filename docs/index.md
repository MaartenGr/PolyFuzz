# PolyFuzz
<img src="logo.png">

**`PolyFuzz`** performs fuzzy string matching, string grouping, and contains extensive evaluation functions. 
PolyFuzz is meant to bring fuzzy string matching techniques together within a single framework.

Currently, methods include Levenshtein distance with RapidFuzz, a character-based n-gram TF-IDF, word embedding
techniques such as FastText and GloVe, and 🤗 transformers embeddings. 

The philosophy of PolyFuzz is: `Easy to use yet highly customizable`. It is a string matcher tool that requires only 
a few lines of code but that allows you customize and create your own models. 