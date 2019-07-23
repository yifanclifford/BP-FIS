import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.nn.parameter import Parameter
from torch.distributions import normal
from lib.utils import ind2hot


def sparse2dense(sparse_mat):
    num_variable = 0
    for f in sparse_mat:
        num_variable = max(num_variable, f.nnz)
    dense = np.full((sparse_mat.shape[0], num_variable), -1, dtype='int')
    rows, cols = sparse_mat.nonzero()
    for row, col in zip(rows, cols):
        dense[row, col] = sparse_mat[row, col]
    return dense


# def idx2hot(X, N):
#     m, n = X.shape
#     col = X.flatten()
#     row = np.arange(m)
#     row = row.repeat(n)
#     H = np.zeros([m, N], dtype='uint8')
#     H[row, col] = True
#     return H

# class FM(nn.Module):
#     def __init__(self, args):
#         super(FM, self).__init__()
#         self.V = Parameter(torch.rand(args.d, args.k))
#         self.P = Parameter(torch.rand(args.m, args.k))
#         self.w = Parameter(torch.rand(args.d))
#         self.b = Parameter(torch.rand(args.m))
#         self.k = args.k
#         self.lamb = args.lamb
#
#     def forward(self, x, u):
#         sumV = torch.matmul(x, self.V)
#         sqV = torch.matmul(x, self.V * self.V)
#         predict = torch.matmul(self.P[u].unsqueeze(1), sumV.unsqueeze(2))
#         predict = predict.squeeze(-1).squeeze(-1)
#         predict += self.b[u]
#         predict += torch.matmul(x, self.w.unsqueeze(1)).squeeze(-1)
#         predict += torch.sum(sumV * sumV - sqV, dim=1) / 2
#         return predict
#
#     def regularization(self, u):
#         return self.lamb * (trace(self.V) + trace(self.P[u]) + trace(self.b[u]))
#
#     def name(self, sep='_'):
#         return 'FM{}{}{}{}'.format(sep, self.k, sep, self.lamb)

class FM(nn.Module):
    def __init__(self, args):
        super(FM, self).__init__()
        randn = normal.Normal(0, 0.01)
        self.V = Parameter(randn.sample(sample_shape=[args.d, args.k]))
        self.w = Parameter(randn.sample(sample_shape=[args.d]))
        self.w0 = Parameter(randn.sample())

    def forward(self, x):
        V = self.V[x]
        w = self.w[x]
        sumV = torch.sum(V, dim=1)
        sqV = torch.sum(V * V, dim=1)
        predict = torch.sum(w, dim=1) + self.w0
        predict += torch.sum(sumV * sumV - sqV, dim=1) / 2
        return predict

    def regularization(self):
        return torch.norm(self.V)


class SFM(FM):
    def __init__(self, args):
        super(SFM, self).__init__(args)

    def regularization(self):
        return torch.sum(torch.norm(self.V, dim=0))


class PIS(nn.Module):
    def __init__(self, args):
        super(PIS, self).__init__()
        self.V = Parameter(torch.empty(args.d, args.k))
        self.pro = Parameter(torch.full((args.m, args.d), args.rate))
        self.w = Parameter(torch.empty(args.d))
        self.w0 = Parameter(torch.empty(1))
        self.rate = args.rate
        idx = torch.tensor([[i, j] for i in range(args.n) for j in range(i + 1, args.n)], dtype=torch.int64).to(
            args.device)
        self.row = idx[:, 0]
        self.col = idx[:, 1]
        self.d = args.d
        self.k = args.k
        self.L = args.L
        self.l1 = args.l1
        self.l2 = args.l2
        self.device = args.device
        self.num_feature = args.n
        self.num_interaction = int(args.n * (args.n - 1) / 2)

    def initial(self):
        nn.init.normal_(self.V.data, 0, 0.01)
        nn.init.normal_(self.w.data, 0, 0.01)
        nn.init.normal_(self.w0.data, 0, 0.01)

    def clamp(self):
        self.pro.data[self.pro.data < 0] = 0
        self.pro.data[self.pro.data > 1] = 1

    def predict(self, x, u):
        d = self.pro.shape[1]
        onehot = ind2hot(x, d).to(self.device)
        pro = self.pro[u]
        pi = pro[onehot].reshape(x.shape)
        V = self.V[x]
        interaction = V.unsqueeze(1) * V.unsqueeze(2)
        mu, _ = self.inference(interaction[:, self.row, self.col])
        w = self.w[x]
        Pi = torch.matmul(pi.unsqueeze(2), pi.unsqueeze(1))
        Pi += (1 - Pi) * (pi.unsqueeze(2) + pi.unsqueeze(1)) * Pi
        Pi = Pi[:, self.row, self.col]
        second = 0
        second += torch.sum(mu * Pi, dim=1)
        return second + torch.sum(w, dim=1) + self.w0

    def pretrain(self, x):
        V = self.V[x]
        interaction = V.unsqueeze(1) * V.unsqueeze(2)
        mu, sigma = self.inference(interaction[:, self.row, self.col])
        w = self.w[x]
        second = 0
        for l in range(self.L):
            interaction = self.reparameterize(mu, sigma)
            second += torch.sum(interaction * self.rate * self.rate, dim=1)
        return second / self.L + torch.sum(w, dim=1) + self.w0 + self.l1 * PIS.weight_reg(mu, sigma)

    def forward(self, x, u):
        d = self.pro.shape[1]
        onehot = ind2hot(x, d).to(self.device)
        pro = self.pro[u]
        pi = pro[onehot].reshape(x.shape)
        V = self.V[x]
        interaction = V.unsqueeze(1) * V.unsqueeze(2)
        mu, sigma = self.inference(interaction[:, self.row, self.col])
        w = self.w[x]
        Pi = torch.matmul(pi.unsqueeze(2), pi.unsqueeze(1))
        Pi = Pi[:, self.row, self.col]
        second = 0
        for l in range(self.L):
            interaction = self.reparameterize(mu, sigma)
            S = self.interaction_selection(pi, Pi)
            second += torch.sum(interaction * S, dim=1)
        return second / self.L + torch.sum(w, dim=1) + self.w0 + self.l1 * PIS.weight_reg(mu, sigma)

    def select(self, x, u):
        V = self.V[x].detach()
        interaction = V.unsqueeze(1) * V.unsqueeze(2)
        mu, sigma = self.inference(interaction[:, self.row, self.col])
        w = self.w[x].detach()
        pi = self.pro[u].squeeze(0)[x]
        Pi = torch.matmul(pi.unsqueeze(2), pi.unsqueeze(1))
        Pi = Pi[:, self.row, self.col]
        second = 0
        for l in range(self.L):
            var_interaction = PIS.reparameterize(mu, sigma)
            S = self.interaction_selection(pi, Pi)
            sumV = torch.sum(var_interaction * S, dim=1).unsqueeze(-1)
            second += Pi * (sumV + (1 - S) * var_interaction)
            second += (1 - Pi) * (sumV - S * var_interaction)
        second /= self.L
        return second + torch.sum(w, dim=1).unsqueeze(-1) + self.w0.detach() + self.l2 * PIS.select_reg(pi)

    def output_interaction(self, x, u):
        pi = self.pro[u].squeeze(0)[x]
        Pi = torch.matmul(pi.unsqueeze(2), pi.unsqueeze(1))
        V = self.V[x].detach()
        interaction = V.unsqueeze(1) * V.unsqueeze(2)
        mu, _ = self.inference(interaction[:, self.row, self.col])
        return pi, Pi, mu

    def interaction_selection(self, pi, Pi):
        eps = torch.rand(self.num_feature).to(self.device)
        s = (pi >= eps).type(torch.float32)
        eps = torch.rand(self.num_interaction).to(self.device)
        S = s.unsqueeze(1) + s.unsqueeze(2)
        S = S[:, self.row, self.col] + (Pi >= eps).type(torch.float32)
        S[S < 2] = 0
        S[S > 0] = 1
        return S

    @staticmethod
    def select_reg(pi):
        eps = 1e-8
        return torch.sum(pi * torch.log(pi + eps) + (1 - pi) * torch.log(1 - pi + eps))

    @staticmethod
    def weight_reg(mu, sigma):
        return -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    @staticmethod
    def reparameterize(mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def inference(self, V):
        pass
    # def weight_reg(self, x):
    #     w = self.w[x]
    #     V = self.V[x]
    #     return trace(w) * self.alpha + trace(V) * self.beta


class PFM(PIS):
    def __init__(self, args):
        super(PFM, self).__init__(args)
        self.mu_linear = nn.Linear(args.k, 1)
        self.sigma_linear = nn.Linear(args.k, 1)

    def inference(self, V):
        mu = self.mu_linear(V).squeeze(-1)
        # mu = torch.sum(V, dim=-1)
        sigma = self.sigma_linear(V).squeeze(-1)
        return mu, sigma


class PNFM(PIS):
    def __init__(self, args):
        super(PNFM, self).__init__(args)
        self.num_layer = len(args.layer)
        self.mu_linear = nn.Linear(args.layer[self.num_layer - 1], 1)
        self.sigma_linear = nn.Linear(args.layer[self.num_layer - 1], 1)
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(args.k, args.layer[0]))
        for l in range(self.num_layer - 1):
            self.mlp.append(nn.Linear(args.layer[l], args.layer[l + 1]))

    def inference(self, V):
        for l in range(self.num_layer):
            V = functional.relu(self.mlp[l](V))
        mu = self.mu_linear(V).squeeze(-1)
        sigma = self.sigma_linear(V).squeeze(-1)
        return mu, sigma
# class PFM(nn.Module):
#     def __init__(self, fm, args):
#         super(PFM, self).__init__()
#         self.rate0 = 1
#         self.rate1 = 1
#         self.rate2 = 1
#         if len(args.rate) > 0:
#             self.rate0 = args.rate[0]
#         if len(args.rate) > 1:
#             self.rate1 = args.rate[1]
#         if len(args.rate) > 2:
#             self.rate2 = args.rate[2]
#         self.p0 = Parameter(torch.tensor(self.rate0))
#         self.p1 = Parameter(torch.full((args.d,), self.rate1))
#         self.p = Parameter(fm.P[args.user])
#         self.V = fm.V.detach().to(args.device)
#         self.b = fm.b[args.user].detach().to(args.device)
#         self.mu_linear = nn.Linear(args.k, 1)
#         self.sigma_linear = nn.Linear(args.k, 1)
#         self.device = args.device
#         self.L = args.L
#         self.user = args.user
#         self.lamb = args.lamb
#         self.k = args.k
#         self.mu = None
#         self.sigma = None
#         self.Mu = None
#         self.Sigma = None
#         self.pi = None
#         self.Pi = None
#
#     def interaction_selection(self, row, col):
#         pi = self.pi.detach()
#         n = len(pi)
#         eps = torch.rand(n).to(self.device)
#         s = (pi >= eps).type(torch.float32)
#         num = int(n * (n - 1) / 2)
#         eps = torch.rand(num).to(self.device)
#         S = s.view(n, 1) + s.view(1, n)
#         S = S[row, col] + (self.Pi.detach() >= eps).type(torch.float32)
#         S[S < 2] = 0
#         S = S > 0
#         return s > 0, S
#
#     def prediction(self, x):
#         V = self.V[x]
#         n = V.shape[0]
#         idx = torch.tensor([[i, j] for i in range(n) for j in range(i + 1, n)], dtype=torch.int64).to(self.device)
#         row = idx[:, 0]
#         col = idx[:, 1]
#         pV = self.p * V
#         self.mu = self.mu_linear(pV).squeeze(-1)
#         pV = pV.unsqueeze(1) * V
#         pV = pV[row, col, :]
#         self.Mu = self.mu_linear(pV).squeeze(-1)
#         self.pi = self.p1[x]
#         self.Pi = torch.matmul(self.pi.view(n, 1), self.pi.view(1, n))
#         self.Pi += torch.matmul(self.pi.pow(2).view(n, 1), self.pi.view(1, n))
#         self.Pi -= torch.matmul(self.pi.pow(3).view(n, 1), self.pi.pow(2).view(1, n))
#         self.Pi += self.Pi.t()
#         self.Pi = self.Pi[row, col]
#         self.Pi = self.Pi * (1 - self.p0) + self.p0
#         self.pi = self.pi * (1 - self.p0) + self.p0
#
#         predict = self.b * self.p0
#         predict += torch.sum(self.mu * self.pi)
#         predict += torch.sum(torch.matmul(self.Mu * self.Pi, pV))
#         return predict
#
#     def weight_forward(self, x):
#         V = self.V[x]
#         n = V.shape[0]
#         idx = torch.tensor([[i, j] for i in range(n) for j in range(i + 1, n)], dtype=torch.int64).to(self.device)
#         row = idx[:, 0]
#         col = idx[:, 1]
#         pV = self.p * V
#         self.mu = self.mu_linear(pV).squeeze(-1)
#         self.sigma = self.sigma_linear(pV).squeeze(-1)
#         pV = pV.unsqueeze(1) * V
#         pV = pV[row, col, :]
#         self.Mu = self.mu_linear(pV).squeeze(-1)
#         self.Sigma = self.sigma_linear(pV).squeeze(-1)
#
#         # s_u = 1
#         pred1 = 0
#         for l in range(self.L):
#             w = self.reparameterize(self.mu, self.sigma)
#             W = self.reparameterize(self.Mu, self.Sigma)
#             pred1 += torch.sum(w) + torch.sum(torch.matmul(W, pV))
#         pred1 = pred1 / self.L + self.b
#
#         # s_u = 0
#         pred0 = 0
#         self.pi = self.p1[x]
#         self.Pi = torch.matmul(self.pi.view(n, 1), self.pi.view(1, n))
#         self.Pi = self.Pi[row, col]
#         for l in range(self.L):
#             s, S = self.interaction_selection(row, col)
#             if not s.any():
#                 continue
#             w = self.reparameterize(self.mu[s], self.sigma[s])
#             pred0 += torch.sum(w)
#             if not S.any():
#                 continue
#             W = self.reparameterize(self.Mu[S], self.Sigma[S])
#             pred0 += torch.sum(torch.matmul(W, pV[S]))
#         pred0 = pred0 / self.L
#
#         return pred1 * self.p0 + pred0 * (1 - self.p0)
#
#     def select_forward(self, x):
#         V = self.V[x]
#         self.pi = self.p1[x]
#         n = V.shape[0]
#         idx = torch.tensor([[i, j] for i in range(n) for j in range(i + 1, n)], dtype=torch.int64).to(self.device)
#         row = idx[:, 0]
#         col = idx[:, 1]
#         pV = self.p.detach() * V
#         self.mu = self.mu_linear(pV).squeeze(-1)
#         self.sigma = self.sigma_linear(pV).squeeze(-1)
#         pV = pV.unsqueeze(1) * V
#         pV = pV[row, col, :]
#         self.Mu = self.mu_linear(pV).squeeze(-1)
#         self.Sigma = self.sigma_linear(pV).squeeze(-1)
#         self.Pi = torch.matmul(self.pi.view(n, 1), self.pi.view(1, n))
#         self.Pi = self.Pi[row, col]
#
#         num = int(n * (n - 1) / 2)
#         predict = [0] * num
#
#         # s_u = 0
#         for l in range(self.L):
#             s, S = self.interaction_selection(row, col)
#             if not s.any():
#                 continue
#             w = self.reparameterize(self.mu[s], self.sigma[s])
#             pred = torch.sum(w)
#             W = self.reparameterize(self.Mu, self.Sigma)
#             for i in range(num):
#                 predict[i] = pred
#                 si = torch.zeros(num, dtype=torch.uint8).to(self.device)
#                 si[i] = 1
#                 predict[i] += torch.sum(torch.matmul(W[S + si], pV[S + si])) * self.Pi[i]
#                 si = torch.ones(num, dtype=torch.uint8).to(self.device)
#                 si[i] = 0
#                 if not (S * si).any():
#                     continue
#                 predict[i] += torch.sum(torch.matmul(W[S * si], pV[S * si])) * (1 - self.Pi[i])
#
#         return torch.stack(predict) / self.L * (1 - self.p0.detach())
#
#     def weight_reg(self):
#         KLD = -0.5 * torch.sum(1 + self.sigma - self.mu.pow(2) - self.sigma.exp())
#         KLD += -0.5 * torch.sum(1 + self.Sigma - self.Mu.pow(2) - self.Sigma.exp())
#         return KLD * self.lamb
#
#     def select_reg(self):
#         KLD = torch.sum(
#             self.pi * torch.log(self.pi / self.rate1) + (1 - self.pi) * torch.log((1 - self.pi) / (1 - self.rate1)))
#         KLD += torch.sum(
#             self.Pi * torch.log(self.Pi / self.rate2) + (1 - self.Pi) * torch.log((1 - self.Pi) / (1 - self.rate2)))
#         return KLD * self.lamb
#
#     def reparameterize(self, mu, sigma):
#         std = torch.exp(0.5 * sigma)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu)
#
#     def name(self):
#         return 'pfm_{}_{}'.format(self.k, self.user)


# class PFM(nn.Module):
#     def __init__(self, args):
#         super(PFM, self).__init__()
#         self.P = Parameter(torch.rand(args.m, args.k))
#         self.V = torch.rand(args.d + 1, args.k)
#         self.V[-1] = 0
#         self.V = Parameter(self.V)
#         self.pi = Parameter(torch.rand(args.m, args.d))
#         self.L = args.L
#         self.d = args.d
#         self.k = args.k
#         self.device = args.device
#         self.layer = args.layer
#         self.mu = None
#         self.sigma = None
#         self.Mu = None
#         self.Sigma = None
#
#         darray = [args.k] + args.layer
#         self.l = len(darray)
#
#         self.infer = nn.ModuleList()
#         for i in range(self.l - 1):
#             self.infer.append(nn.Linear(darray[i], darray[i + 1]))
#         self.mu = nn.Linear(darray[self.l - 1], 1)
#         self.sigma = nn.Linear(darray[self.l - 1], 1)
#
#         self.gen = nn.ModuleList()
#         for i in range(self.l - 1):
#             self.gen.append(nn.Linear(darray[i], darray[i + 1]))
#
#     def reparameterize(self, mu, sigma):
#         std = torch.exp(0.5 * sigma)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu)
#
#     def weight_forward(self, x, u):
#
#         dx = sparse2dense(x)
#         m, n = dx.shape
#         v = self.V[dx.flatten()].view(m, n, self.k).detach()  # get V
#         pu = self.P[u].view(m, 1, self.k).detach()  # get pu
#         pi = self.pi[u].detach()
#         rows, cols = dx.nonzero()
#         pi = pi[rows, cols].reshape(m, n)  # get pi
#         Pi = torch.matmul(pi.view(m, n, 1), pi.view(m, 1, n))
#         pv = v * pu
#         self.mu, self.sigma = self.encode(pv)
#         self.Mu, self.Sigma = self.encode(pv.unsqueeze(2) * v.unsqueeze(1))
#         # W = torch.matmul(pu * self.V, self.V.t())
#         # Mu = self.mu(W)
#         # Sigma = self.sigma(W)
#         r = 0
#         for l in range(self.L):
#             self.train()
#             eps = torch.rand(n).to(self.device)
#             s = (pi >= eps).type(torch.float32)
#             r += torch.sum(self.reparameterize(self.mu, self.sigma), dim=1) * s
#             eps = torch.rand(n, n)
#             S = s.view(m, n, 1) + s.view(m, 1, n)
#             S[S == 2] = 0
#             S = torch.matmul(s.view(m, n, 1), s.view(m, 1, n)) + S * (Pi >= eps).type(torch.float32)
#             W = self.reparameterize(self.Mu, self.Sigma) * S
#             interaction = v.unsqueeze(2) * v.unsqueeze(1) * W.unsqueeze(3)
#             interaction = torch.sum(interaction, dim=[1, 2])
#             self.eval()
#             r += self.decode(interaction)
#         return r / self.L
#
#     # def select_forward(self, x, u):
#     #     for l in range(self.L):
#     #         eps = torch.rand(n).to(self.device)
#     #         s = (pi >= eps).type(torch.float32)
#     #         r += torch.sum(self.reparameterize(self.mu, self.sigma), dim=1) * s
#     #         eps = torch.rand(n, n)
#     #         S = s.view(m, n, 1) + s.view(m, 1, n)
#     #         S[S == 2] = 0
#     #     pass
#
#     def encode(self, x):
#         h = x
#         for i in range(self.l - 1):
#             h = functional.relu(self.infer[i](h))
#         return self.mu(h).squeeze(-1), self.sigma(h).squeeze(-1)
#
#     def decode(self, x):
#         h = x
#         for i in range(self.l - 1):
#             h = functional.relu(self.gen[i](h))
#         return torch.sum(h, dim=1)
