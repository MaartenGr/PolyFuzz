# Datasets
There are two datasets prepared for you to play around with:
* Company Names
* Movie Titles

## Movie Titles
This data is retrieved from:  
* https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset  
* https://www.kaggle.com/shivamb/netflix-shows  

It contains Netflix and IMDB movie titles that can be matched against each other. 
Where IMDB has 80852 movie titles and Netflix has 6172 movie titles.

You can use them as follows:

```python
from polyfuzz import PolyFuzz
from polyfuzz.datasets import load_movie_titles

data = load_movie_titles()
model = PolyFuzz("TF-IDF").match(data["Netflix"], data["IMDB"])
```

## Company Names
This data is retrieved from https://www.kaggle.com/dattapiy/sec-edgar-companies-list?select=sec__edgar_company_info.csv 
and contains 100_000 company names to be matched against each other. 

This is a different use case than what you have typically seen so far. We often see two different lists compared 
with each other. Here, you can use this dataset to compare the company names with themselves in order to clean 
them up. 

You can use them as follows:

```python
from polyfuzz import PolyFuzz
from polyfuzz.datasets import load_company_names

data = load_company_names()
model = PolyFuzz("TF-IDF").match(data, data)
```

PolyFuzz will recognize that the lists are similar and that you are looking to match the titles with themselves. 
It will ignore any comparison a string has with itself, otherwise everything will get mapped to itself. 
