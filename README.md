# BP-FIS
Bayesian personalized feature interaction selection
- source code
- movielens dataset

## Requirement
- pytrec_eval (https://github.com/cvangysel/pytrec_eval)
- pytorch >= 1.0

## Running
### Check the running parameters
```python
python personal.py -h
```
### Sample run

Please unzip movielens.zip and run the following:

```python
cd code
python personal.py PFM --gpu --rank --batch 128 --dir ../dataset -d movielens --save
```
