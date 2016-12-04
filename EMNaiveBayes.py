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
        # self.X_number is some kind of onehot encoding. Use natural numbers instead of {0, 1}
        self.X_number = np.array([self.q2number[j][self.X[:, j]] for j in range(self.M)]).T
        # self.X_onehot = pd.get_dummies(DataFrame(self.X)) # ok?
        self.X_onehot = np.zeros((self.N, sum(self.nq)))
        col_idx = 0
        for j in range(self.M):
            for l in range(self.nq[j]):
                self.X_onehot[:, col_idx + l] = (self.X_number[:, j] == l).astype(int)
            col_idx += self.nq[j]
        return self

    def _init_theta(self, random_state=0):
        self.theta_dim = self.K + self.K * sum(self.nq)
        if random_state == None:
            return self._init_theta_uniform()
        elif random_state < 0:
            return self._init_theta_uniform_plus_normal()
        else:
            return self._init_theta_random(random_state)

    def _init_theta_random(self, random_state):
        '''
        \theta = (pyk, A)
        pyk is a 1d array, and A is a list of 2d array
        A[j][k, l] means: for the kth class, the probability that the jth feature' value is the lth value of the feature.
                        i.e. P(a_jl | y_k)
        '''
        eps = 1e-2
        tmp = np.random.rand(self.K) + eps
        self.pyk = tmp / tmp.sum()
        self.A = list()
        for j in range(self.M):
            tmp = np.random.rand(self.K, self.nq[j]) + eps
            self.A.append((tmp.T / tmp.sum(axis=1)).T)
        return self

    def _init_theta_uniform(self):
        self.pyk = np.ones(self.K) / self.K
        self.A = list()
        for j in range(self.M):
            # A[j][k, l] = P(a_jl | y_k)
            pj = pd.value_counts(self.X_number[:, j]).sort_index() / float(self.N)
            self.A.append(np.array([copy.deepcopy(pj) for _ in range(self.K)]))
        return self

    def _init_theta_uniform_plus_normal(self):
        tmp = np.ones(self.K) / self.K + np.random.randn(self.K)
        self.pyk = tmp / tmp.sum()
        self.A = list()
        for j in range(self.M):
            # A[j][k, l] = P(a_jl | y_k)
            pj = pd.value_counts(self.X_number[:, j]).sort_index() / float(self.N)
            tmp = np.array([copy.deepcopy(pj) for _ in range(self.K)])
            tmp += np.random.randn(self.K, self.nq[j])
            tmp = (tmp.T / tmp.sum(axis=1)).T # normalize
            self.A.append(tmp)
        return self

    def _iterate(self):
        '''
        E(v_{ik}) = P(y_k | x_i, \theta^{(t)})
        P(y_k) = \frac{\Sigma_{i=1}^{N} E(v_{ik})}{N}
        P(a_{jl} | y_k) = \frac{\Sigma_{i=1}^{N} E(v_{ik}) * u_i^{(j, l)}}{\Sigma_{i=1}^{N} E(v_{ik})}
        u_i^{(j, l)} =  1, if the ith sample's jth feafure took the lth value of the feature
                        0, otherwise
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
        '''
        check if || \theta - \theta_old || < \epsilon
        '''
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

    def _calc_overall_feature_entropy(self):
        '''
        overall feature entropy
        \Sigma_{j=1}^M \Sigma_{l=1}^{nq_j} - \frac{ \Sigma_{i=1}^N u_i^{(j, l)} }{N} * log( \frac{ \Sigma_{i=1}^N u_i^{(j, l)} }{N} )
        '''
        tmp = self.X_onehot.sum(axis=0).astype(float) / self.N
        self.overall_feature_entropy = 0 - (tmp * np.log(np.where(tmp == 0, 1, tmp))).sum()
        return self

    def _calc_clustered_feature_entropy(self):
        '''
        clustered feature entropy
        partition the samples into K classes, calculate overall feature entropy for each class
        and then calculate weighted sum of these entropy
        '''
        tmp = 0
        for j in range(self.M):
            tmp -= (self.pyk * (self.A[j] * np.log(np.where(self.A[j] == 0, 1, self.A[j]))).T).sum()
        self.clustered_feature_entropy = tmp
        return self

    def _calc_ground_feature_entropy(self, Y):
        '''
        ground feature entropy
        same as clustered feature entropy, except that the partition is done by using ground truth Y
        '''
        self.Y_g = Y
        self.yk_g = sorted(list(set(Y)))
        self.y2number_g = Series(range(len(self.yk_g)), index=self.yk_g)
        self.Y_number_g = np.array(self.y2number_g[Y])
        self.Y_onehot_g = np.zeros((self.N, self.K))
        for k in range(self.K):
            self.Y_onehot_g[:, k] = (self.Y_number_g == k).astype(int)
        self.nk_g = np.array(pd.value_counts(self.Y_number_g).sort_index())
        self.pyk_g = self.nk_g / float(self.N)
        tmp = 0
        for k in range(self.K):
            p_ajl_yk = (self.X_onehot.T * self.Y_onehot_g[:, k]).sum(axis=1) / float(self.nk_g[k])
            tmp -= self.pyk_g[k] * (p_ajl_yk * np.log(np.where(p_ajl_yk == 0, 1, p_ajl_yk))).sum()
        self.ground_feature_entropy = tmp
        return self

    def _print_evaluation(self):
        print '''Feature Entropy:\n
        Overall:\t%.4f\n
        Ground: \t%.4f\n
        Clustered:\t%.4f\n''' % (
            self.overall_feature_entropy,
            self.ground_feature_entropy,
            self.clustered_feature_entropy)
        return self

    def _correspondence_analysis(self):
        try:
            import mca
        except Exception, e:
            print e
            return
        # clustered class
        self.Y = self.Ev.argmax(axis=1)
        self.yk = range(self.K)
        self.Y_onehot = np.zeros((self.N, self.K))
        self.y2number = Series(range(len(self.yk)), index=self.yk)
        self.Y_number = np.array(self.y2number[self.Y])
        self.Y_onehot = np.zeros((self.N, self.K))
        for k in range(self.K):
            self.Y_onehot[:, k] = (self.Y_number == k).astype(int)

        self.count = DataFrame(self.Y_onehot.T.dot(self.Y_onehot_g))
        self.ca = mca.mca(self.count)


    def fit(self, X, K, max_iter=100):
        self._init_params(X, K)
        self._init_theta()
        for t in range(max_iter):
            self._iterate()
            if self._is_convergent():
                break
        return self

    def evaluate(self, Y):
        self._calc_overall_feature_entropy()
        self._calc_ground_feature_entropy(Y)
        self._calc_clustered_feature_entropy()
        self._print_evaluation()
        return self

if __name__ == '__main__':
    # toy data as an example. For more, please go to main.py
    X = np.array([['A', 'B', 'A'], ['B', 'A', 'A'], ['B', 'A', 'C']])
    Y = np.array(['P', 'Q', 'P'])
    K = len(set(Y))
    m = EMNaiveBayes(epsilon=1e-5)
    m.fit(X, K)
    m.evaluate(Y)
