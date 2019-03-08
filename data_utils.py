import numpy as np
import scipy
from scipy.spatial import ConvexHull

class ConvexHullPoints(object):
    def __init__(self, ndim=2):
        self.ndim = ndim
        self.points = []
        self.hull_points = []
        
    def __call__(self, num_points):
        self.num_points = num_points
        self.points = np.random.rand(self.num_points, self.ndim)
        hull_indices = ConvexHull(points=self.points)
        self.hull_points = self.points[hull_indices.vertices]
        return self.points, self.hull_points, hull_indices.vertices
    
    def get_batch(self, batchsize=100, max_points=1000):
        batch = [self.__call__(np.random.randint(3,max_points)) for i in range(batchsize)]
        return batch
    
    def get_fixed_batch(self, batchsize=100, num_points=10):
        batch = np.asanyarray([self.__call__(num_points) for i in range(batchsize)])
        return np.asanyarray(batch[:][:,0]), batch[:][:,1], batch[:][:,2]