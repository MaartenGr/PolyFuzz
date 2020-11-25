# Custom Grouper
The basic grouper is a TF-IDF implementation that uses single linkage to group the strings 
you mapped to together. 

With the customizability philosophy of PolyFuzz in mind it is not unexpected that you can also 
use any of the models, and even custom models, as your grouper!

Here, we use Edit Distance instead of TF-IDF to group the strings we mapped to:

```python
from polyfuzz import PolyFuzz
from polyfuzz.models import EditDistance

from_list = ["apple", "apples", "appl", "recal", "house", "similarity"]
to_list = ["apple", "apples", "mouse"]

model = PolyFuzz("TF-IDF").match(from_list, to_list)

# Custom grouper
base_edit_grouper = EditDistance(n_jobs=1)
model.group(base_edit_grouper)
```

And that is it! We have now grouped our matches we mapped to together using Edit Distance instead of TF-IDF.
