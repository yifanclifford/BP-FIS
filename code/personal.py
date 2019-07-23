import argparse
import numpy as np
import torch
import lib.LoadCF as DATA
import math

from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader
from lib.fm import PNFM, PFM
from eval.evaluation import Evaluator
import os
from tqdm import tqdm


def evaluate_rmse(test_method):
    pfm.eval()
    test_loss = 0
    N = len(load_data.data[test_method]['rating'])
    test_loader = DataLoader(np.arange(N), batch_size=args.batch, shuffle=True)
    with torch.no_grad():
        for x in test_loader:
            feature = torch.tensor(load_data.data[test_method]['feature'][x])
            rating = torch.tensor(load_data.data[test_method]['rating'][x]).to(args.device)
            user = torch.tensor(load_data.data[test_method]['user'][x]).to(args.device)
            predict = pfm.predict(feature, user)
            predict = torch.clamp(predict, 0, 1)
            test_loss += functional.mse_loss(predict, rating, reduction='sum').item()
        test_loss /= N
    return math.sqrt(test_loss)


def evaluate_rank(test_method):
    pfm.eval()
    users = load_data.data[test_method]['user']
    test_loader = DataLoader(np.unique(users), batch_size=1, shuffle=True)
    run = dict()
    test = dict()
    with torch.no_grad():
        for user in test_loader:
            data = load_data.get_user(user.item(), test_method)
            feature = torch.tensor(data['feature']).to(args.device)
            user = torch.tensor(data['user']).to(args.device)
            rating = data['rating']
            item = np.array(data['item'])
            test[str(user)] = {str(i): int(1) for i in item[rating == 1]}
            predict = pfm.predict(feature, user).cpu().numpy()
            run[str(user)] = {str(i): float(v) for i, v in zip(item, predict)}

    evaluator = Evaluator({'recall'})
    evaluator.evaluate(run, test)
    result = evaluator.show(['recall_1'])
    return result['recall_1']


# def pre_train(epoch):
#     train_loss = 0
#     pfm.train()
#     for idx, x in enumerate(rating_loader):
#         weight_optimizer.zero_grad()
#         feature = torch.tensor(load_data.data['train']['feature'][x]).to(args.device)
#         rating = torch.tensor(load_data.data['train']['rating'][x]).to(args.device)
#         predict = pfm.pretrain(feature)
#         loss = loss_function(predict, rating, reduction='sum')
#         loss.backward()
#         train_loss += loss.item()
#         weight_optimizer.step()
#         if args.log != 0 and idx % args.log == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, idx * args.batch, num_rating, 100. * idx * args.batch / num_rating, loss.item()))
#     print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_rating))


def weight_train(epoch):
    train_loss = 0
    pfm.train()
    for idx, x in enumerate(tqdm(rating_loader)):
        weight_optimizer.zero_grad()
        feature = torch.tensor(load_data.data['train']['feature'][x])
        rating = torch.tensor(load_data.data['train']['rating'][x]).to(args.device)
        user = torch.tensor(load_data.data['train']['user'][x]).to(args.device)
        predict = pfm(feature, user)
        loss = loss_function(predict, rating, reduction='sum')
        loss.backward()
        train_loss += loss.item()
        # print(torch.nonzero(fm.pro.grad > 0))
        weight_optimizer.step()
        # if args.log != 0 and idx % args.log == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, idx * args.batch, num_rating, 100. * idx * args.batch / num_rating, loss.item()))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_rating))


def select_train(epoch):
    train_loss = 0
    pfm.train()
    for idx, user in enumerate(tqdm(user_loader)):
        user = user.item()
        select_optimizer.zero_grad()
        data = load_data.get_user(user, 'train')
        feature = torch.tensor(data['feature']).to(args.device)
        rating = torch.tensor(data['rating']).to(args.device)
        predict = pfm.select(feature, user)
        loss = loss_function(predict, rating.unsqueeze(-1), reduction='mean')
        loss.backward()
        train_loss += loss.item()
        select_optimizer.step()
        pfm.clamp()
        if args.log != 0 and idx % args.log == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx, num_user, 100. * idx / num_user, loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_user))


def train(iteration):
    valid_rmse = []
    test_recall = []

    pfm.initial()
    for epoch in range(20):
        weight_train(epoch)
        valid_result = evaluate_rmse('valid')
        recall_result = evaluate_rank('test')
        valid_rmse.append(valid_result)
        test_recall.append(recall_result)
        print('valid={:.5f}, recall={:.5}'.format(valid_result, recall_result))
        if args.save:
            pfm.cpu()
            torch.save(pfm.state_dict(),
                       '{}/{}/model/{}_{}_{}.tmp'.format(args.dir, args.dataset, args.algo, args.k, epoch))
            pfm.to(args.device)

    best_rmse = min(valid_rmse)
    index = valid_rmse.index(best_rmse)
    print('best: iter={}, valid={:.5f}, recall={:.4f}'.format(index, valid_rmse[index], test_recall[index]))
    src = '{}/{}/model/{}_{}_{}.tmp'.format(args.dir, args.dataset, args.algo, args.k, index)
    dest = '{}/{}/model/{}_{}_{}'.format(args.dir, args.dataset, args.algo, iteration, args.k)
    os.rename(src, dest)
    dir = '{}/{}/model'.format(args.dir, args.dataset)
    for file in os.listdir(dir):
        if '{}_{}'.format(args.algo, args.k) in file and file.endswith('.tmp'):
            os.remove(dir + '/' + file)

    if args.record:
        line = '{}\t{}\t{}\t{}\t{}\t{:.5f}\t{:.5f}\n'.format(args.algo, args.dataset, args.k, iteration, index,
                                                             best_rmse,
                                                             test_recall[index])
        path = '{}/{}/result/{}'.format(args.dir, args.dataset, args.algo)
        print('write result to {}'.format(path))
        with open(path, 'a') as f:
            f.write(line)


# def print_select():
#     users = load_data.data['train']['user']
#     pfm.eval()
#     M = np.unique(users)
#     test_loader = DataLoader(M, batch_size=1, shuffle=True)
#     with torch.no_grad():
#         for idx, user in enumerate(test_loader):
#             user = user.item()
#             rating, feature, _ = load_data.get_user(user, 'train')
#             feature = torch.from_numpy(feature).to(args.device)
#             print('{}: {}'.format(user, pfm.pro[user][feature]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='neural factorization machines')
    parser.add_argument('algo', choices=['PFM', 'PNFM'], help='choose factorization machines')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dir', help='dataset directory',
                        default='../dataset')
    parser.add_argument('-d', '--dataset', help='specify dataset', type=str, default='test')
    parser.add_argument('-k', help='parameter k', type=int, default=64)
    parser.add_argument('--lr1', help='learning rate for weights', type=float, default=1e-3)
    parser.add_argument('--lr2', help='learning rate for probability', type=float, default=1e-2)
    parser.add_argument('--l1', help='regularization parameter for weights', type=float, default=0)
    parser.add_argument('--l2', help='regularization parameter for parameters', type=float, default=1e-3)
    parser.add_argument('--log', help='log interval', type=int, default=1)
    parser.add_argument('-m', '--maxiter', help='max number of iteration', type=int, default=1)
    parser.add_argument('-L', type=int, default=1, help='number of samples')
    parser.add_argument('--rate', help='prior probabilities', type=float, default=0.8)
    parser.add_argument('--loss', help='loss function, currently support binary and mse', choices=['log', 'mse'],
                        default='mse')
    parser.add_argument('--rank', help='evaluate with ranking metrics', action='store_true')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--record', help='record the result', action='store_true')
    parser.add_argument('--layer', nargs='+', help='number of neurals in each layer', type=int, default=[20])
    parser.add_argument('--load', action='store_true', help='whether to load pretrained')
    parser.add_argument('--pretrain', action='store_true', help='whether to pretrain the model')
    parser.add_argument('--batch', type=int, default=1, help='input batch size for training (default: 1)')
    parser.add_argument('--save', action='store_true', help='whether to save the model')
    args = parser.parse_args()

    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    load_data = DATA.LoadCF(args.dir, args.dataset, args.rank)
    load_data.statistics()

    args.m = load_data.num_user
    args.d = load_data.dimension
    args.n = load_data.nnz

    Machine = eval(args.algo)
    pfm = Machine(args)
    if args.load:
        pfm = Machine(args)
        pfm.load_state_dict(torch.load('{}/{}/model/{}_{}'.format(args.dir, args.dataset, args.algo, args.k)))

    pfm.to(args.device)

    users = np.unique(load_data.data['train']['user'])
    num_user = len(users)
    num_rating = len(load_data.data['train']['rating'])

    user_loader = DataLoader(users, batch_size=1, shuffle=True)
    rating_loader = DataLoader(np.arange(num_rating), batch_size=args.batch, shuffle=True)
    params = []
    for name, param in pfm.named_parameters():
        if name != 'pro':
            params.append(param)
    weight_optimizer = optim.Adam(params, lr=args.lr1)
    select_optimizer = optim.Adam(list([pfm.pro]), lr=args.lr2)
    loss_function = functional.mse_loss if args.loss == 'mse' else functional.binary_cross_entropy_with_logits

    # weight_train(1)
    # train_result = evaluate_rmse('train')
    pfm.initial()
    valid_result = evaluate_rmse('valid')
    print('initial: valid={:.5f}'.format(valid_result))

    train(0)

    for iter in range(args.maxiter):
        if args.save:
            path = '{}/{}/model/{}_{}_{}'.format(args.dir, args.dataset, args.algo, iter, args.k)
            print('load model from ' + path)
            pfm.cpu()
            pfm.load_state_dict(torch.load(path))
            pfm.to(args.device)

        for epoch in range(5):
            select_train(epoch)

        train(iter + 1)

    # result = evaluate_rank(load_data, fm, args)
    # print(result)
    # if args.save:
    #     path = args.dir + '/' + args.dataset + '/model/pfm_{}_{}'.format(args.k, args.user)
    #     pfm.cpu()
    #     torch.save(pfm.state_dict(), path)
