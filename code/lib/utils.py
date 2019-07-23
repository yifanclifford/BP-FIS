import numpy as np
import pytrec_eval
import torch


class Evaluator:
    def __init__(self, metrics):
        self.result = {}
        self.metrics = metrics

    def evaluate(self, predict, test):
        evaluator = pytrec_eval.RelevanceEvaluator(test, self.metrics)
        self.result = evaluator.evaluate(predict)

    def show(self, metrics):
        result = {}
        for metric in metrics:
            res = pytrec_eval.compute_aggregated_measure(metric, [user[metric] for user in self.result.values()])
            result[metric] = res
            # print('{}={}'.format(metric, res))
        return result

    def show_all(self):
        key = next(iter(self.result.keys()))
        keys = self.result[key].keys()
        return self.show(keys)


def mmread(R, type='float32'):
    row = R.row.astype(int)
    col = R.col.astype(int)
    val = torch.from_numpy(R.data.astype(type))
    index = torch.from_numpy(np.row_stack((row, col)))
    m, n = R.shape
    return torch.sparse.FloatTensor(index, val, torch.Size([m, n]))


def trace(A=None, B=None):
    if A is None:
        print('please input pytorch tensor')
        val = None
    elif B is None:
        val = torch.sum(A * A)
    else:
        val = torch.sum(A * B)
    return val


# def data2test(users, items, ratings):
#     return {str(u): {str(i): int(1) for i, r in zip(items[users == u], ratings[users == u]) if r == 1} for u in
#             np.unique(users)}
def ind2hot(X, N):
    m, n = X.shape
    col = X.flatten().cpu().numpy()
    row = np.arange(m)
    row = row.repeat(n)
    H = torch.zeros([m, N], dtype=torch.uint8)
    H[row, col] = 1
    return H


def sort2query(run):
    n = len(run)
    return {str(int(run[j])): float(1.0 / (j + 1)) for j in range(n)}
