import numpy as np
import pytrec_eval
import torch
from sortedcontainers import SortedDict
from torch.nn import functional
from torch.utils.data import DataLoader
import math
import scipy.stats


class Evaluator:
    def __init__(self, metrics):
        self.result = None
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


def evaluate_mse_file(load_data, out_path, result_path, test_method='valid'):
    rating = load_data.get_rating(test_method)
    predict = np.loadtxt(result_path)
    num_sample = len(rating)
    diff = rating - predict
    mse = np.sum(diff * diff) / num_sample
    with open(out_path, 'a') as file:
        file.write('{}\n'.format(mse))


def evaluate_rank_file(load_data, result_path):
    users = load_data.data['test']['user']
    predict = np.loadtxt(result_path)
    test_loader = DataLoader(np.unique(users), batch_size=1, shuffle=True)
    run = dict()
    test = dict()
    with torch.no_grad():
        for user in test_loader:
            user = user.item()
            data = load_data.get_user(user, 'test')
            idx = load_data.get_user_idx(user, 'test')
            rating = data['rating']
            item = np.array(data['item'])
            test[str(user)] = {str(i): int(1) for i in item[rating == 1]}
            pred = predict[idx]
            run[str(user)] = {str(i): float(v) for i, v in zip(item, pred)}

    evaluator = Evaluator({'recall', 'recip_rank_cut'})
    evaluator.evaluate(run, test)
    result = evaluator.show(
        ['recall_1', 'recall_5', 'recall_10', 'recip_rank_cut_1', 'recip_rank_cut_5', 'recip_rank_cut_10'])
    return result


def evaluate_rank(load_data, fm, args):
    fm.eval()
    users = load_data.data['test']['user']
    test_loader = DataLoader(np.unique(users), batch_size=1, shuffle=True)
    run = dict()
    test = dict()
    with torch.no_grad():
        for user in test_loader:
            data = load_data.get_user(user.item(), 'test')
            feature = torch.tensor(data['feature']).to(args.device)
            user = torch.tensor(data['user']).to(args.device)
            rating = data['rating']
            item = np.array(data['item'])
            test[str(user)] = {str(i): int(1) for i in item[rating == 1]}
            predict = fm.predict(feature, user).cpu().numpy()
            run[str(user)] = {str(i): float(v) for i, v in zip(item, predict)}

    evaluator = Evaluator({'recall', 'recip_rank_cut'})
    evaluator.evaluate(run, test)
    result = evaluator.show(
        ['recall_1', 'recall_5', 'recall_10', 'recip_rank_cut_1', 'recip_rank_cut_5', 'recip_rank_cut_10'])
    return result


def output_predict(load_data, fm, args, file):
    fm.eval()
    N = len(load_data.data['test']['rating'])
    test_loader = DataLoader(np.arange(N), batch_size=args.batch, shuffle=False)
    torch.no_grad()
    with open(file, 'w') as f:
        for x in test_loader:
            feature = torch.tensor(load_data.data['test']['feature'][x])
            user = torch.tensor(load_data.data['test']['user'][x]).to(args.device)
            predict = fm.predict(feature, user).detach().cpu().numpy()
            for r in predict:
                f.write('%.4f\n' % r)


def significant(res1, res2, measure):
    query_ids = list(set(res1.keys()) & set(res2.keys()))
    score1 = [res1[query_id][measure] for query_id in query_ids]
    score2 = [res2[query_id][measure] for query_id in query_ids]
    print(scipy.stats.ttest_rel(score1, score2))


def file2run(load_data, file):
    users = load_data.data['test']['user']
    predict = np.loadtxt(file)
    test_loader = DataLoader(np.unique(users), batch_size=1, shuffle=False)
    run = dict()
    test = dict()
    with torch.no_grad():
        for user in test_loader:
            user = user.item()
            data = load_data.get_user(user, 'test')
            idx = load_data.get_user_idx(user, 'test')
            rating = data['rating']
            item = np.array(data['item'])
            test[str(user)] = {str(i): int(1) for i in item[rating == 1]}
            pred = predict[idx]
            run[str(user)] = {str(i): float(v) for i, v in zip(item, pred)}
    evaluator = Evaluator({'recall', 'recip_rank_cut'})
    evaluator.evaluate(run, test)
    return evaluator.result

# def evaluate_rank(load_data, fm, args, test_method='valid'):
#     fm.eval()
#     users = list(load_data.data[test_method].keys())
#     test_loader = DataLoader(users, batch_size=1, shuffle=True)
#     run = dict()
#     test = dict()
#     with torch.no_grad():
#         for user in test_loader:
#             print(user)
#             user = user.item()
#             data = load_data.get_user(user, test_method)
#             if len(data['feature']) < args.N:
#                 continue
#             feature = torch.tensor(data['feature'][0:args.N], dtype=torch.int64).to(args.device)
#             rating = data['rating'][0:args.N]
#             item = np.array(data['item'][0:args.N], dtype='int')
#             test[str(user)] = {str(i): int(1) for i in item[rating == 1]}
#             predict = fm.predict(feature, user).cpu().numpy()
#             run[str(user)] = {str(i): float(v) for i, v in zip(item, predict)}
#     evaluator = Evaluator({'recall', 'recip_rank_cut'})
#     evaluator.evaluate(run, test)
#     result = evaluator.show(
#         ['recall_5', 'recall_10', 'recall_15', 'recall_20', 'recip_rank_cut_5', 'recip_rank_cut_10',
#          'recip_rank_cut_15',
#          'recip_rank_cut_20'])
#     return result
