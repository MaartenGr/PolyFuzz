v0.3.3  
- Update numpy to "numpy>=1.20.0" to prevent [this](https://github.com/MaartenGr/PolyFuzz/issues/23) and this [issue](https://github.com/MaartenGr/PolyFuzz/issues/21)
- Update pytorch to "torch>=1.4.0,<1.7.1" to prevent save_state_warning error   

v0.3.2  
- Fix exploding memory usage when using `top_n`   

v0.3.0  
- Use `top_n` in `polyfuzz.models.TFIDF` and `polyfuzz.models.Embeddings`   

v0.2.2  
- Update grouping to include all strings only if identical lists of strings are compared  

v0.2.0  
- Update naming convention matcher --> model  
- Update documentation  
- Add basic models to grouper  
- Fix issues with vector order in cosine similarity  
- Update naming of cosine similarity function  

v0.1.0  
- Additional tests  
- More thorough documentation  
- Prepare for public release  

v0.0.1  
- First release of `PolyFuzz`
- Matching through:
    - Edit Distance
    - TF-IDF
    - Embeddings
    - Custom models
- Grouping of results with custom models
- Evaluation through precision-recall curves

