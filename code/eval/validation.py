import argparse
import os.path
import torch
import numpy as np
import scipy
import lib.LoadCF as DATA
from eval.evaluation import evaluate_mse_file, evaluate_rank_file, evaluate_rank, significant
from lib.fm import PFM, PNFM
from torch.utils.data import DataLoader
from eval.evaluation import Evaluator


# def validation_fm(load_data, args):
#     type = 'rank' if args.rank else 'rating'
#     dir_path = '{}/{}/result/{}/{}'.format(args.dir, args.dataset, args.algo, type)
#     N = len(os.listdir(dir_path))
#     out_path = os.path.dirname(dir_path) + '/valid_fm'
#     for n in range(1, N + 1):
#         result_path = '{}/pred{}.txt'.format(dir_path, n)
#         evaluate_mse_file(load_data, out_path, result_path)

def evaluate_file(load_data, args):
    dir_path = '{}/result/{}'.format(args.dir, args.dataset)
    for file in os.listdir(dir_path):
        if file.startswith('FM'):
            result = evaluate_rank_file(load_data, dir_path + '/' + file)
            line = ''
            for val in result.values():
                line += '\t{}'.format(val)
            file = file[:-4]
            file = file.replace('_', '\t')
            print(file + line)


def evalute_model(load_data, args):
    Machine = eval(args.algo)
    dir_path = '{}/{}/model/'.format(args.dir, args.dataset)
    args.rate = 1
    args.device = 'cpu'
    args.L = 1
    args.l1 = 0
    args.l2 = 0

    for file in os.listdir(dir_path):
        if not file.endswith('.tmp') and args.algo in file:
            args.k = eval(file.split('_')[2])
            args.layer = [args.k]
            fm = Machine(args)
            fm.load_state_dict(torch.load(dir_path + '/' + file))
            result = evaluate_rank(load_data, fm, args)
            line = ''
            for val in result.values():
                line += '\t{}'.format(val)
            file = file.replace('_', '\t')
            print(file + line)






# if __name__ == "__main__":
#     dir_path = ''
#     data_path = ''
#     data = DATA.LoadCF(dir_path, data_path, True)
#     args.m = data.num_user
#     args.d = data.dimension
#     args.n = data.nnz
#     # eval_function = eval('validation_{}'.format(args.algo))
#     if args.model:
#         evalute_model(data, args)
#     else:
#         evaluate_file(data, args)

# def evaluate_mse(load_data, fm, args, test='valid'):
#     users = load_data.data[test]['user']
#     fm.eval()
#     M = np.unique(users)
#     user_loss = SortedDict()
#     test_loader = DataLoader(M, batch_size=1, shuffle=True)
#     test_loss = 0
#     num_sample = 0
#     with torch.no_grad():
#         for user in test_loader:
#             user = user.item()
#             rating, feature, _ = load_data.get_user(user, test)
#             feature = torch.from_numpy(feature).to(args.device)
#             rating = torch.from_numpy(rating).to(args.device)
#             predict = fm.predict(feature, user)
#             loss = functional.mse_loss(predict, rating, reduction='sum').item()
#             test_loss += loss
#             user_loss[user] = loss / feature.shape[0]
#             num_sample += feature.shape[0]
#     path = args.dir + '/' + args.dataset + '/result/{}'.format(fm.name())
#     with open(path, 'w') as file:
#         file.write(json.dumps(user_loss, indent=1))
#     print('rmse: {}'.format(test_loss / num_sample))
#
#
# def evaluate_rank(load_data, fm, args, test_method='valid'):
#     users = load_data.data[test_method]['user']
#     fm.eval()
#     M = np.unique(users)
#     test_loader = DataLoader(M, batch_size=1, shuffle=True)
#     run = dict()
#     test = dict()
#     with torch.no_grad():
#         for user in test_loader:
#             user = user.item()
#             rating, feature, item = load_data.get_user(user, test_method)
#             feature = torch.from_numpy(feature).to(args.device)
#             test[str(user)] = {str(i): int(1) for i in item[rating == 1]}
#             predict = fm.predict(feature, user).cpu().numpy()
#             run[str(user)] = {str(i): float(v) for i, v in zip(item, predict)}
#     evaluator = Evaluator({'recall', 'recip_rank_cut'})
#     evaluator.evaluate(run, test)
#     result = evaluator.show(
#         ['recall_5', 'recall_10', 'recall_15', 'recall_20', 'recip_rank_cut_5', 'recip_rank_cut_10',
#          'recip_rank_cut_15',
#          'recip_rank_cut_20'])
#     print(result)


# def evaluate_mse_file(load_data, args, result_path, test_method='valid'):
#     users = load_data.data[test_method]['user']
#     M = np.unique(users)
#     result = np.loadtxt(result_path)
#     user_loss = SortedDict()
#     test_loader = DataLoader(M, batch_size=1, shuffle=True)
#     test_loss = 0
#     num_sample = 0
#     with torch.no_grad():
#         for user in test_loader:
#             user = user.item()
#             idx = load_data.get_user_index(user, test_method)
#             rating = load_data.data[test_method]['rating'][idx]
#             predict = result[idx]
#             loss = trace(rating - predict)
#             test_loss += loss
#             user_loss[user] = loss / len(idx)
#             num_sample += len(idx)
#     path = args.dir + '/' + args.dataset + '/result/{}'.format(fm.name())
#     with open(path, 'w') as file:
#         file.write(json.dumps(user_loss, indent=1))
#     print('rmse: {}'.format(test_loss / num_sample))


# def personalized_evaluate(data, fm):
#     user = data.valid['user']
#     test_loader = DataLoader(np.unique(user), batch_size=1, shuffle=True)
#     fm.eval()
#     user_loss = [None] * len(np.unique(user))
#     with torch.no_grad():
#         for idx, u in enumerate(test_loader):
#             rating, feature, _ = data.get_user(u.item(), 'valid')
#             x = mmread(feature.tocoo()).to(args.device)
#             r = torch.from_numpy(rating.astype('float32')).to(args.device)
#             predict = fm(x, u)
#             predict = torch.clamp(predict, -1, 1)
#             loss = functional.mse_loss(predict, r)
#             user_loss[u] = loss.item()
#     path = args.dir + '/' + args.dataset + '/result/{}'.format(fm.name())
#     np.savetxt(path, user_loss)
#
#
# def rmse_evaluate(data, fm):
#     feature = data.valid['feature']
#     rating = data.valid['rating']
#     user = data.valid['user']
#     test_loader = DataLoader(np.arange(len(rating)), batch_size=100, shuffle=True)
#     train_loss = 0
#     for idx, sample in enumerate(test_loader):
#         x = mmread(feature[sample].tocoo()).to(args.device)
#         u = user[sample]
#         r = torch.from_numpy(rating[sample].astype('float32')).to(args.device)
#         predict = fm(x, u)
#         loss = functional.mse_loss(predict, r, reduction='sum')
#         train_loss += loss.item()
#     return train_loss / len(rating)
#
#
# def recommendation_evaluate(data, fm):
#     user = data.valid['user']
#     test_loader = DataLoader(np.unique(user), batch_size=1, shuffle=True)
#     fm.eval()
#     user_loss = [None] * len(np.unique(user))
#     run = {}
#     test = {}
#     with torch.no_grad():
#         for idx, u in enumerate(test_loader):
#             rating, feature, item = data.get_user(u.item(), 'valid')
#             x = mmread(feature.tocoo()).to(args.device)
#             i = item[rating == 1]
#             test[str(u)] = {str(i): 1}
#             predict = fm(x, u)
#             idx = torch.argsort(predict, descending=True)
#             run[str(u)] = sort2query(idx)
#
#             user_loss[u] = loss.item()
#     path = args.dir + '/' + args.dataset + '/result/{}'.format(fm.name())
#     np.savetxt(path, user_loss)
#     return 0
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='neural factorization machines')
#
#     parser.add_argument('algo', help='specify algorithm', choices=['FM', 'PFM'])
#     parser.add_argument('--gpu', action='store_true', default=False,
#                         help='enables CUDA training')
#     parser.add_argument('--dir', help='dataset directory', default='/Users/chenyifan/jianguo/dataset/sigir-2019')
#     parser.add_argument('-d', '--dataset', help='specify dataset', type=str, default='test')
#     parser.add_argument('-k', help='parameter k', type=int, default=20)
#     parser.add_argument('-l', '--lamb', help='parameter lambda', type=float, default=.01)
#     parser.add_argument('--rate', help='prior probabilities', nargs='+', type=float, default=[0.5, 0.6, 0.8])
#     parser.add_argument('-u', '--user', help='specify the user', type=int, default=1)
#     parser.add_argument('--rank', help='whether to train a ranking model', action='store_true')
#     parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
#     # parser.add_argument('--batch', type=int, default=10, help='input batch size for training (default: 1)')
#
#     args = parser.parse_args()
#     args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
#     args.L = 1
#     Model = eval(args.algo)
#
#     data = DATA.LoadCF(args.dir, args.dataset, args.rank)
#     args.m = data.m
#     args.d = data.d
#     fm = FM(args)
#
#     type = 'ranking' if args.rank else 'rating'
#     path = '{}/{}/model/{}/fm_{}_{}_{}'.format(args.dir, args.dataset, type, args.k, args.lamb, args.lr)
#     fm.load_state_dict(torch.load(path))
#     fm.to(args.device)
#
#     # feature = data.valid['feature']
#     # rating = data.valid['rating']
#     res = 0
#     if args.algo == 'FM':
#         if args.rank:
#             res = recommendation_evaluate(data, fm)
#         else:
#             res = rmse_evaluate(data, fm)
#         # user = data.valid['user']
#         # test_loader = DataLoader(np.unique(user), batch_size=1, shuffle=True)
#         #
#         # fm.eval()
#         # user_loss = [None] * args.m
#         # with torch.no_grad():
#         #     for idx, u in enumerate(test_loader):
#         #         rating, feature = data.get_user(u.item(), 'valid')
#         #         x = mmread(feature.tocoo()).to(args.device)
#         #         r = torch.from_numpy(rating.astype('float32')).to(args.device)
#         #         predict = fm(x, u)
#         #         predict = torch.clamp(predict, -1, 1)
#         #         loss = functional.mse_loss(predict, r)
#         #         user_loss[u] = loss.item()
#         # path = args.dir + '/' + args.dataset + '/result/{}'.format(fm.name())
#         # np.savetxt(path, user_loss)
#         # rmse = sum(user_loss) / args.m
#         # print(rmse)
#     else:
#         pfm = PIS(fm, args)
#         path = '{}/{}/model/{}'.format(args.dir, args.dataset, pfm.name())
#         pfm.load_state_dict(torch.load(path))
#         pfm.to(args.device)
#         rating, feature = data.get_user(args.user, 'valid')
#         test_loader = DataLoader(np.arange(len(rating)), batch_size=1, shuffle=True)
#         test_loss = 0
#         pfm.eval()
#         with torch.no_grad():
#             for idx, sample in enumerate(test_loader):
#                 x = torch.from_numpy(feature[sample].todense()).squeeze(0).to(args.device) > 0
#                 r = rating[sample]
#                 predict = pfm.prediction(x)
#                 predict = torch.clamp(predict, -1, 1)
#                 diff = predict - float(r)
#                 loss = diff * diff
#                 test_loss += loss.item()
#         print('rmse of pfm:{}'.format(test_loss / len(rating)))
#
#     name = 'rank' if args.rank else 'rmse'
#     path = '{}/{}/result/{}'.format(args.dir, args.dataset, name)
#     line = '{}\t{}\t{}\r\n'.format(fm.name(sep='\t'), args.lr, res)
#     with open(path, 'a') as f:
#         f.write(line)
