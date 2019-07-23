# BP-FIS
Bayesian personalized feature interaction selection

## Requirement
- pytrec_eval (https://github.com/cvangysel/pytrec_eval)
- pytorch 1.0

**sample run**
```python
cd code
python personal.py PFM --gpu --rank --batch 128 --dir ../dataset -d movielens
```
