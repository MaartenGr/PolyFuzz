## **v0.4.0**


* Added new models (SentenceTransformers, Gensim, USE, Spacy)
* Added `.fit`, `.transform`, and `.fit_transform` methods
* Added `.save` and `PolyFuzz.load()`


**SentenceTransformers**  
```python
from polyfuzz.models import SentenceEmbeddings
distance_model = SentenceEmbeddings("all-MiniLM-L6-v2")
model = PolyFuzz(distance_model)
```

**Gensim**  
```python
from polyfuzz.models import GensimEmbeddings
distance_model = GensimEmbeddings("glove-twitter-25")
model = PolyFuzz(distance_model)
```

**USE**  
```python
from polyfuzz.models import USEEmbeddings
distance_model = USEEmbeddings("https://tfhub.dev/google/universal-sentence-encoder/4")
model = PolyFuzz(distance_model)
```

**Spacy**  
```python
from polyfuzz.models import SpacyEmbeddings
distance_model = SpacyEmbeddings("en_core_web_md")
model = PolyFuzz(distance_model)
```


**fit, transform, fit_transform**  
Add `fit`, `transform`, and `fit_transform` in order to use PolyFuzz in production [#34](https://github.com/MaartenGr/PolyFuzz/issues/34)

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from polyfuzz import PolyFuzz

train_words = ["apple", "apples", "appl", "recal", "house", "similarity"]
unseen_words = ["apple", "apples", "mouse"]

# Fit
model = PolyFuzz("TF-IDF")
model.fit(train_words)

# Transform
results = model.transform(unseen_words)
```

In the code above, we fit our TF-IDF model on `train_words` and use `.transform()` to match the words in `unseen_words` to the words that we trained on in `train_words`. 

After fitting our model, we can save it as follows:

```python
model.save("my_model")
```

Then, we can load our model to be used elsewhere:

```python
from polyfuzz import PolyFuzz

model = PolyFuzz.load("my_model")
```


## **v0.3.4**

- Make sure that when you use two lists that are exactly the same, it will return 1 for identical terms:

```python
from polyfuzz import PolyFuzz

from_list = ["apple", "house"]
model = PolyFuzz("TF-IDF")
model.match(from_list, from_list)
```

This will match each word in `from_list` to itself and give it a score of 1. Thus, `apple` will be matched to `apple` and 
`house` will be mapped to `house`. However, if you input just a single list, it will try to map them within the list without 
mapping to itself:

```python
from polyfuzz import PolyFuzz

from_list = ["apple", "apples"]
model = PolyFuzz("TF-IDF")
model.match(from_list)
```

In the example above, `apple` will be mapped to `apples` and not to `apple`. Here, we assume that the user wants to 
find the most similar words within a list without mapping to itself. 

## **v0.3.3**  
- Update numpy to "numpy>=1.20.0" to prevent [this](https://github.com/MaartenGr/PolyFuzz/issues/23) and this [issue](https://github.com/MaartenGr/PolyFuzz/issues/21)
- Update pytorch to "torch>=1.4.0,<1.7.1" to prevent save_state_warning error   

## **v0.3.2**  
- Fix exploding memory usage when using `top_n`   

## **v0.3.0**  
- Use `top_n` in `polyfuzz.models.TFIDF` and `polyfuzz.models.Embeddings`   

## **v0.2.2**  
- Update grouping to include all strings only if identical lists of strings are compared  

## **v0.2.0**  
- Update naming convention matcher --> model  
- Update documentation  
- Add basic models to grouper  
- Fix issues with vector order in cosine similarity  
- Update naming of cosine similarity function  

## **v0.1.0**
- Additional tests  
- More thorough documentation  
- Prepare for public release  

## **v0.0.1**
- First release of `PolyFuzz`
- Matching through:
    - Edit Distance
    - TF-IDF
    - Embeddings
    - Custom models
- Grouping of results with custom models
- Evaluation through precision-recall curves

