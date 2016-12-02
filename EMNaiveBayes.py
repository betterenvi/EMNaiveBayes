import sys, os, collections, copy
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

class EMNaiveBayes(object):
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon

    def _init_params(self, X, K):
        self.X, self.K = X, K
        self.N, self.M = X.shape
        self.num_iter = 0
        self.q = [sorted(list(set(X[:, j]))) for j in range(self.M)]
        self.nq = np.array([len(v) for v in self.q])
        self.q2number = [Series(range(self.nq[j]), index=self.q[j]) for j in range(self.M)]
        self.X_number = np.array([self.q2number[j][self.X[:, j]] for j in range(self.M)]).T # some kind of onehot encoding. Use natural numbers instead of {0, 1}
        #self.X_onehot = pd.get_dummies(DataFrame(self.X)) # ok?
        self.X_onehot = np.zeros((self.N, sum(self.nq)))
        col_idx = 0
        for j in range(self.M):
            for l in range(self.nq[j]):
                self.X_onehot[:, col_idx + l] = (self.X_number[:, j] == l).astype(int)
            col_idx += self.nq[j]
        return self

    def _init_theta(self, random_state=0):
        self.theta_dim = K + K * sum(self.nq)
        if random_state == None:
            return self._init_theta_uniform()
        else:
            return self._init_theta_random(random_state)

    def _init_theta_random(self, random_state):
        eps = 1e-2
        tmp = np.random.rand(self.K) + eps
        self.pyk = tmp / tmp.sum()
        self.A = list()
        for j in range(self.M):
            tmp = np.random.rand(self.K, self.nq[j]) + eps
            self.A.append((tmp.T / tmp.sum(axis=1)).T)
        return self

    def _init_theta_uniform(self):
        self.pyk = np.ones(K) / K
        self.A = list()
        for j in range(self.M): # A[j][k, l] means: for the kth class, the jth feature' value is the lth value of the feature. i.e. P(a_jl | y_k)
            pj = pd.value_counts(self.X_number[:, j]).sort_index() / float(self.N)
            self.A.append(np.array([copy.deepcopy(pj) for _ in range(self.K)]))
        return self

    def _iterate(self):
        '''
        Ev_ik = P(y_k | x_i, \theta^{(t)}) / Z_i
        '''
        self.num_iter += 1
        self.pyk_old = copy.deepcopy(self.pyk)
        self.A_old = copy.deepcopy(self.A)

        Ev = np.ones((self.N, self.K))
        for j in range(self.M):
            Ev *= (self.A[j][:, self.X_number[:, j]]).T

        Ev *= self.pyk
        self.Ev = (Ev.T / Ev.sum(axis=1).astype(float)).T # normalize

        # update theta
        Ev_k = self.Ev.sum(axis=0).astype(float)
        self.pyk = Ev_k / float(self.N)
        EvT = self.Ev.T
        col_idx = 0
        for j in range(self.M):
            for l in range(self.nq[j]):
                self.A[j][:, l] = (EvT * self.X_onehot[:, col_idx + l]).sum(axis=1) / Ev_k
            col_idx += self.nq[j]
        return self

    def _is_convergent(self):
        delta = np.zeros(self.theta_dim)
        delta[:self.K] = self.pyk - self.pyk_old
        delta_idx = self.K
        for j in range(self.M):
            k_nqj = self.K * self.nq[j]
            new_delta_idx = delta_idx + k_nqj
            delta[delta_idx:new_delta_idx] = (self.A[j] - self.A_old[j]).reshape(k_nqj)
            delta_idx = new_delta_idx
        sq_delta = (delta ** 2).sum()
        return sq_delta < self.epsilon

    def fit(self, X, K, max_iter=100):
        self._init_params(X, K)
        self._init_theta()
        for i in range(max_iter):
            self._iterate()
            if self._is_convergent():
                break
        return self

if __name__ == '__main__':
    X, K = np.array([['A', 'B', 'A'], ['B', 'A', 'A'], ['B', 'A', 'C']]), 2
    m = EMNaiveBayes(epsilon=1e-5)
    self = m
    m.fit(X, K)
