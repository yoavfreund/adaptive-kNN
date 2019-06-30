from collections import defaultdict

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


# Node of K-D Tree, corresponds to a specialist.
class TreeNode:
    
    def __init__(self, bounds):
        self.bounds = bounds
        self.left = self.right = None
        self.split_point = self.split_dim = None
        self.label_cnts = defaultdict(int)
    
    def split(self, x, d):
        self.split_point = x
        self.split_dim = d
        bounds_left = self.bounds.copy()
        bounds_right = self.bounds.copy()
        bounds_left[d][1] = x[d]
        bounds_right[d][0] = x[d]
        self.left = TreeNode(bounds_left)
        self.right = TreeNode(bounds_right)

        
class KDTree:
    
    def __init__(self, X, Y, feat_names=None, tgt_names=None):
        self.num_feats = X.shape[1]
#         self.dim_order = np.random.permutation(self.num_feats)
        self.dim_order = np.arange(self.num_feats)
        self.num_tgts = len(set(Y))
        if feat_names is None:
            feat_names = ['x%d' % i for i in range(self.num_feats)]
        if tgt_names is None:
            tgt_names = list(set(Y))
        self.feat_names = list(feat_names)
        self.tgt_names = list(tgt_names)
        both = list(zip(X, Y))
        np.random.shuffle(both)
        self.X, self.Y = zip(*both)
        self.X, self.Y = np.array(self.X), np.array(self.Y)
        self.root = TreeNode(np.array([[self.X[:, d].min(), self.X[:, d].max()]
                                       for d in range(X.shape[1])]))
        for i in range(self.X.shape[0]):
            self.insert(self.X[i], self.Y[i])
    
    def insert(self, x, y):
        d = self.dim_order[0]
        node = self.root
        while node.split_point is not None:
            if y not in [-1, None, 'NA']:
                node.label_cnts[y] += 1
            d = node.split_dim
            side = 'left' if x[d] < node.split_point[d] else 'right'
            node = getattr(node, side)
        idx = np.where(self.dim_order == d)[0][0]
        d = self.dim_order[(idx + 1) % x.size]
        # d = (d + 1) % x.size # split across each dimension sequentially
        # d = np.random.randint(x.size) # pick a random dimension each time
        node.split(x, d)
        node.left.label_cnts[y] += 1
        node.right.label_cnts[y] += 1
    
    # get sequence of specialists converging to {x}
    def get_seq(self, x):
        seq = []
        d = 0
        node = self.root
        while node.split_point is not None:
            seq.append(node)
            d = node.split_dim
            node = node.left if x[d] < node.split_point[d] else node.right
        return seq[::-1]
    
    # get prediction on sample
    def get_pred(self, x, conf=1.0, verbose=False):
        for n in self.get_seq(x):
            cnts = n.label_cnts
            pred = max(self.tgt_names, key=lambda k: cnts[k])
            k = sum(cnts.values())
            eta = (cnts[pred] / k) - (1 / self.num_tgts)
            delta = conf / np.sqrt(k)
            if verbose:
                print('%s \t eta=%.3f \t delta=%.3f \t k=%d' % (pred, eta, delta, k))
            if eta >= delta:
                return pred
    
    ## VISUALIZATION METHODS FOR IRIS PLANTS (2D) DATA
    # visualize dataset
    def viz_data(self):
        for y in [-1]+list(range(self.num_tgts)):
            plt.scatter(self.X[self.Y == y][:, 0], self.X[self.Y == y][:, 1],
                        label=(self.tgt_names+['removed label'])[y])
        plt.title('Iris Plants Dataset')
        plt.xlabel(self.feat_names[0])
        plt.ylabel(self.feat_names[1])
        plt.legend(loc='best')
    
    # visualize entire decision tree
    def viz_tree(self, depth=np.inf):
        # recursively DFS through tree adding dividers to visualization
        dim = self.X.shape[1]
        assert dim == 2
        def recurse(node, d, rec_depth=0):
            if rec_depth > depth:
                return
            if node is None:
                return
            [plt.vlines, plt.hlines][d]([node.split_point[d]], *node.bounds[1-d])
            # make new specialist contexts for recursion
            d = (d + 1) % dim
            recurse(node.left, d, rec_depth+1)
            recurse(node.right, d, rec_depth+1)
        recurse(self.root, 0)
    
    # visualize the sequence of specialists induced by decision tree for a single point
    def viz_point(self, x, ax):
        ax.scatter([x[0]], [x[1]], color='black', s=300, marker='x')
        dim = x.size
        assert dim == 2
        d = 0
        rects = []
        for spec in self.get_seq(x):
            rect = Rectangle(spec[:, 0], spec[0, 1] - spec[0, 0], spec[1, 1] - spec[1, 0])
            rects.append(rect)
            d = (d + 1) % dim
        pc = PatchCollection(rects, facecolor='none', alpha=1, edgecolor='black')
        ax.add_collection(pc)
