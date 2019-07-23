import numpy as np


class LoadCF(object):
    def __init__(self, dir, dataset, rank):
        dir = dir + '/' + dataset + '/'
        suffix = '.rank' if rank else '.rating'
        self.trainfile = dir + "train" + suffix
        self.validfile = dir + "valid" + suffix
        self.testfile = dir + "test" + suffix
        self.num_user = 0
        self.num_item = 0
        self.dimension = 0
        self.nnz = 0
        N1 = self.prepare(self.trainfile)
        N2 = self.prepare(self.validfile)
        N3 = self.prepare(self.testfile)
        self.num_user += 1
        self.num_item += 1 - self.num_user
        self.dimension += 1
        self.num_rating = N1 + N2 + N3
        # print('m={}, d={}'.format(self.m, self.d))
        self.data = dict()
        self.data['train'] = self.read_feature(self.trainfile, N1)
        self.data['valid'] = self.read_feature(self.validfile, N2)
        self.data['test'] = self.read_feature(self.testfile, N3)

    def statistics(self):
        print('num_user={}\nnum_item={}\nnum_rating={}\nnum_feature={}'.format(self.num_user,
                                                                               self.num_item,
                                                                               self.num_rating,
                                                                               self.dimension))

    def prepare(self, file):
        with open(file) as f:
            line = f.readline()
            self.nnz = len(line.strip().split(' ')) - 1
            line_num = 0
            while line:
                features = line.strip().split(' ')
                user = int(features[1].split(':')[0])
                item = int(features[2].split(':')[0])
                self.num_user = user if self.num_user < user else self.num_user
                self.num_item = item if self.num_item < item else self.num_item
                for feature in features[2:]:
                    fid = int(feature.split(':')[0])
                    self.dimension = fid if self.dimension < fid else self.dimension
                line = f.readline()
                line_num += 1
        return line_num

    def read_feature(self, file, N):
        f = open(file, 'r')
        rating = np.empty(N, dtype='float32')
        user = np.empty(N, dtype='int64')
        item = np.empty(N, dtype='int64')
        feature = np.empty((N, self.nnz), dtype='int64')
        for l, line in enumerate(f):
            params = line.strip().split(' ')
            r = float(params[0])
            u = int(params[1].split(':')[0])
            i = int(params[2].split(':')[0])
            rating[l] = r
            user[l] = u
            item[l] = i
            for idx, param in enumerate(params[1:]):
                id = int(param.split(':')[0])
                feature[l, idx] = id
        return {'user': user, 'item': item, 'rating': rating, 'feature': feature}

    def get_user_idx(self, u, name='train'):
        data = self.data[name]
        idx = data['user'] == u
        return idx

    def get_user(self, u, name='train'):
        data = self.data[name]
        idx = data['user'] == u
        return {'user': data['user'][idx], 'item': data['item'][idx], 'rating': data['rating'][idx],
                'feature': data['feature'][idx]}

    # def get_rating(self, name='train'):
    #     feature = self.data[name]
    #     rating = [r for user in feature for r in feature[user]['rating']]
    #     return np.array(rating)
# if name == 'train':
#     user = self.train['user']
#     idx = np.flatnonzero(user == u)
#     return self.train['rating'][idx], self.train['feature'][idx], self.train['item'][idx]
# elif name == 'valid':
#     user = self.valid['user']
#     idx = np.flatnonzero(user == u)
#     return self.valid['rating'][idx], self.valid['feature'][idx], self.valid['item'][idx]
# else:
#     user = self.test['user']
#     idx = np.flatnonzero(user == u)
#     return self.test['rating'][idx], self.test['feature'][idx], self.test['item'][idx]
