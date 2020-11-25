<img src="images/logo.png" width="70%" height="70%"/>

[![PyPI - Python](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)](https://pypi.org/project/keybert/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/keybert/blob/master/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/polyfuzz)](https://pypi.org/project/polyfuzz/)
[![Build](https://img.shields.io/github/workflow/status/MaartenGr/polyfuzz/Code%20Checks/master)](https://pypi.org/project/polyfuzz/)

**`PolyFuzz`** performs fuzzy string matching, string grouping, and contains extensive evaluation functions. 
PolyFuzz is meant to bring fuzzy string matching techniques together within a single framework.

Currently, methods include Levenshtein distance with RapidFuzz, a character-based n-gram TF-IDF, word embedding
techniques such as FastText and GloVe, and finally 🤗 transformers embeddings. 

You can use your own **custom models** for both the fuzzy string matching as well as the string grouping. 

Corresponding medium post can be found [here]().


<a name="installation"/></a>
## Installation
You can install **`PolyFuzz`** via pip:
 
```bash
pip install polyfuzz
```

This will install the base dependencies and excludes any deep learning/embedding models. 

If you want to be making use of 🤗 Transformers, install the additional additional `Flair` dependency:

```bash
pip install polyfuzz[flair]
```

<a name="gettingstarted"/></a>
## Getting Started

For a quick start check the section below. For a more in depth overview of the possibilities of **`PolyFuzz`** 
you can check the full documentation [here](https://maartengr.github.io/PolyFuzz/) or you can follow along 
with the notebook [here](https://github.com/MaartenGr/PolyFuzz/blob/master/notebooks/Overview.ipynb).

### Quick Start

The main goal of `PolyFuzz` is to allow the user to perform different methods for matching strings. 
We start by defining two lists, one to map from and one to map to. We are going to be using `TF-IDF` to create 
n-grams on a character level in order to compare similarity between strings. 

We only have to instantiate `PolyFuzz` with `TF-IDF` and match the lists:


```python
from polyfuzz import PolyFuzz

from_list = ["apple", "apples", "appl", "recal", "house", "similarity"]
to_list = ["apple", "apples", "mouse"]

model = PolyFuzz("TF-IDF").match(from_list, to_list)
```  

**NOTE**: Simply use "EditDistance" or "Embeddings" to quickly access Levenshtein and FastText respectively. 

The resulting matches can be accessed through `model.get_matches()`:

```python
>>> model.get_matches()
         From      To  Similarity
0       apple   apple    1.000000
1      apples  apples    1.000000
2        appl   apple    0.783751
3       recal    None    0.000000
4       house   mouse    0.587927
5  similarity    None    0.000000

``` 


### Precision Recall Curve  
Next, we would like to see how well our model is doing on our data. Although this method is unsupervised, 
we can use the similarity score as a proxy for the accuracy of our model (assuming we trust that similarity score).

A minimum similarity score might be used to identify when a match could be considered to be correct. 
For example, we can assume that if a similarity score pass 0.95 we are quite confident that the matches are correct. 
This minimum similarity score can be defined as `Precision` since it shows you how precise we believe the matches are at a minimum.

`Recall` can then be defined as as the percentage of matches found at a certain minimum similarity score. 
A high recall means that for a certain minimum precision score, we find many matches.

Creating the visualizations is as simple as:

```python
model.visualize_precision_recall()
```
<img src="images/tfidf.png" width="100%" height="100%"/>

### Group Matches
We can group the matches `To` as there might be significant overlap in strings in our to_list. 
To do this, we calculate the similarity within strings in to_list and use `single linkage` to then 
group the strings with a high similarity.

```python
model.group(link_min_similarity=0.75)
```

When we extract the new matches, we can see an additional column `Group` in which all the `To` matches were grouped to:

```python
>>> model.get_matches()
	      From	To	    Similarity	Group
0	     apple	apple	1.000000	apples
1	    apples	apples	1.000000	apples
2	      appl	apple	0.783751	apples
3	     recal	None	0.000000	None
4	     house	mouse	0.587927	mouse
5	similarity	None	0.000000	None
```

As can be seen above, we grouped apple and apples together to `apple` such that when a string is mapped to `apple` it 
will fall in the cluster of `[apples, apple]` and will be mapped to the first instance in the cluster which is `apples`.

For example, `appl` is mapped to apple and since apple falls into the cluster `[apples, apple]`, `appl` will be mapped to `apples`.

## 🤗 Transformers

With `Flair`, we can use all 🤗 Transformers that are publicly available. We simply have to instantiate any Flair
WordEmbedding method and pass it through PolyFuzzy:  

```python
from polyfuzz.models import Embeddings
from flair.embeddings import TransformerWordEmbeddings

bert = TransformerWordEmbeddings('bert-base-multilingual-cased')
bert_matcher = Embeddings(bert, matcher_id="BERT", min_similarity=0)
model = PolyFuzz("TF-IDF")
```

For a full list of transformer models see [this](# https://huggingface.co/transformers/pretrained_models.html) link. 

## Multiple Models
You might be interested in running multiple models with different matchers and different parameters in order to compare the best results.
Fortunately, PolyFuzz allows you to exactly do this!

Below, you will find all models currently implemented in PolyFuzz and are compared against one another.

```python
from polyfuzz.models import EditDistance, TFIDF, Embeddings
from flair.embeddings import TransformerWordEmbeddings

bert = TransformerWordEmbeddings('bert-base-multilingual-cased')
bert_matcher = Embeddings(bert, min_similarity=0)
tfidf_matcher = TFIDF(min_similarity=0)
edit_matcher = EditDistance()

matchers = [bert_matcher, tfidf_matcher, edit_matcher]
```

Then, we simply call `PolyFuzz` with all matchers and visualize the results:

```
model = PolyFuzz(matchers).match(from_list, to_list)
model.visualize_precision_recall()
```

<img src="images/multiple_models.png" width="100%" height="100%"/>


## Custom Grouper
We can even use one of the `polyfuzz.models` to be used as the grouper in case you would like to use 
something else than the standard TF-IDF matcher:

```python
model = PolyFuzz("TF-IDF").match(from_list, to_list)
base_edit_grouper = EditDistance(n_jobs=1)
model.group(base_edit_grouper)
```

## Custom Models
Although the options above are a great solution for comparing different models, what if you have developed your own? 
What if you want a different similarity/distance measure that is not defined in PolyFuzz? 
That is where custom models come in. If you follow the structure of PolyFuzz's BaseMatcher 
you can quickly implement any model you would like.

Below, we are implementing the ratio similarity measure from RapidFuzz.

```python
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from polyfuzz.models import BaseMatcher


class MyModel(BaseMatcher):
    def match(self, from_list, to_list):
        # Calculate distances
        matches = [[fuzz.ratio(from_string, to_string) / 100 for to_string in to_list] 
                    for from_string in from_list]
        
        # Get best matches
        mappings = [to_list[index] for index in np.argmax(matches, axis=1)]
        scores = np.max(matches, axis=1)
        
        # Prepare dataframe
        matches = pd.DataFrame({'From': from_list,'To': mappings, 'Similarity': scores})
        return matches
```
Then, we can simply create an instance of MyModel and pass it through PolyFuzz:
```python
custom_matcher = MyModel()
model = PolyFuzz(custom_matcher)
```

## References