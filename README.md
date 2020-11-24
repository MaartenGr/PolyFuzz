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


<a name="gettingstarted"/></a>
## 1. Getting Started
[Back to ToC](#toc)

```python
from polyfuzz import PolyFuzz

from_list = ["apple", "apples", "appl", "recal", "house", "similarity"]
to_list = ["apple", "apples", "mouse"]

model = PolyFuzz("TF-IDF").match(from_list, to_list)

```  

The resulting topics can be accessed through `model.get_matches()`:

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


## References