import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle



def reloadtest():
    print('gggg')


# Node of K-D Tree, corresponds to a specialist.
class TreeNode:
    
    def __init__(self, bounds):
        self.bounds = bounds
        self.left = self.right = None
        self.split_point = self.split_dim = None
    
    def split(self, x, d):
        self.split_point = x
        self.split_dim = d
        bounds_left = self.bounds.copy()
        bounds_right = self.bounds.copy()
        bounds_left[d][1] = x
        bounds_right[d][0] = x
        self.left = TreeNode(bounds_left)
        self.right = TreeNode(bounds_right)

    # propagate bias from children
    def prop_bias(self):
        if self.split_point is None:
            return
        pass
        
class KDTree:
    
    def __init__(self, X, Y, feat_names=None, tgt_names=None):
        self.feat_names = list(feat_names)
        self.tgt_names = list(tgt_names)
        both = list(zip(X, Y))
        np.random.shuffle(both)
        self.X, self.Y = zip(*both)
        self.X, self.Y = np.array(self.X), np.array(self.Y)
        self.root = TreeNode(np.array([[self.X[:, d].min(), self.X[:, d].max()]
                                   for d in range(X.shape[1])]))
        for i in range(self.X.shape[0]):
            self.insert(self.X[i])
    
    def insert(self, x):
        d = 0
        node = self.root
        while node.split_point:
            side = 'left' if x[d] < node.split_point else 'right'
            node = getattr(node, side)
            d = (d + 1) % x.size
        node.split(x[d], d)
    
    # get sequence of specialists converging to {x}
    def get_seq(self, x):
        seq = []
        d = 0
        node = self.root
        while node.split_point:
            seq.append(node.bounds)
            node = node.left if x[d] < node.split_point else node.right
            d = (d + 1) % x.size
        return seq
    
    ## VISUALIZATION METHODS
    # visualize dataset
    def viz_data(self):
        for y in [-1, 0, 1, 2]:
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
        vlines, hlines = [], []
        def recurse(node, d, rec_depth=0):
            if rec_depth > depth:
                return
            if node is None:
                return
            [plt.vlines, plt.hlines][d]([node.split_point], *node.bounds[1-d])
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
