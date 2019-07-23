# BP-FIS
Bayesian personalized feature interaction selection
- source code
- movielens dataset

## Requirement
- pytrec_eval (https://github.com/cvangysel/pytrec_eval)
- pytorch >= 1.0

## Running
### check the running parameters
```python
python personal.py -h
```
### sample run

```python
cd code
python personal.py PFM --gpu --rank --batch 128 --dir ../dataset -d movielens
```
