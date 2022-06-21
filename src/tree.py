import numpy as np


class ClusterNode():

    def __init__(self, id, left, right, count, sample_probability):

        self.id = id
        self.left = left
        self.right = right
        self.count = count
        self.sample_probability = sample_probability

    def get_id(self):
        return self.id
    
    def get_left(self):
        return self.left

    def get_right(self):
        return self.right
    
    def get_count(self):
        return self.count

    def get_sample_probability(self):
        return self.sample_probability

class Tree():

    def __init__(self, node, nodelist):

        self.node = node
        self.left = None
        self.right = None
        self.nodelist = nodelist

        # self.cluster_node = ClusterNode(node.get_id(), node.get_left(), node.get_right(), node.get_count(), 1)
        # self.cluster_nodelist = [self.cluster_node]
        self.cluster_nodelist = []

    def construct_tree(self, node):

        # base case
        if node is None:
            return None
    
        # create a new node with the same data as the root node
        cluster_node = ClusterNode(node.get_id(), None, None, node.get_count(), 0.5)
    
        # clone the left and right subtree
        cluster_node.left = self.construct_tree(node.left)
        cluster_node.right = self.construct_tree(node.right)

        self.cluster_nodelist.append(cluster_node)
    
        # return cloned root node
        return cluster_node

    # Function to print the inorder traversal on a given binary tree
    def inorder(self, node):
    
        if node is None:
            return
    
        # recur for the left subtree
        self.inorder(node.left)
    
        # print the current node's data
        print(node.sample_probability, end = ' ')
    
        # recur for the right subtree
        self.inorder(node.right)