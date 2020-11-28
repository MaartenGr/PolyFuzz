import json
import requests
from typing import List, Mapping


def load_movie_titles() -> Mapping[str, List[str]]:
    """ Load Netflix and IMDB movie titles to be matched against each other

    Retrieved from:
        https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset
        https://www.kaggle.com/shivamb/netflix-shows

    Preprocessed such that it only contains the title names where
    IMDB has 80852 titles and Netflix has 6172

    Returns:
         data: a dictionary with two keys: "Netflix" and "IMDB" where
               each value contains a list of movie titles
    """
    url = 'https://github.com/MaartenGr/PolyFuzz/raw/master/data/movie_titles.json'
    resp = requests.get(url)
    data = json.loads(resp.text)
    return data


def load_company_names() -> List[str]:
    """ Load company names to be matched against each other.

    Retrieved from:
        https://www.kaggle.com/dattapiy/sec-edgar-companies-list?select=sec__edgar_company_info.csv

    Preprocessed such that it only contains 100_000 company names.

    Returns:
        data: a list of company names
    """
    url = 'https://github.com/MaartenGr/PolyFuzz/raw/master/data/company_names.json'
    resp = requests.get(url)
    data = json.loads(resp.text)
    return data
