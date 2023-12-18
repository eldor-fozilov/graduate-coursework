# some initial imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import mnist_dataset as mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist.init()

class ContinualMNIST:
    def __init__(self, seed=2023, task_order:list =None,):
        """
            Initializes the ContinualMNIST dataset.
            Dataset loading.
            
            Args:
            - seed (int): Seed for random operations.
            - task_order (list): List of integers defining the task order. 
                                For example: [0, 1, 2, 3, ...] or [3, 2, 1, 0, ...], you could set any orders.
                                It will create tasks using pairs of these numbers.
                                Default task order is: [(0, 1, 2, 3, 4)  (5, 6, 7, 8, 9)]
        """
        torch.manual_seed(seed)
        mnist.init()
        self.x_train, self.t_train, self.x_test, self.t_test = mnist.load()
        self.task_data = []
        if task_order is not None:
            if not all([0 <= i < 10 for i in task_order]):
                raise ValueError("All values in task_order must be between 0 and 9.")
            self.task_classes_arr = [(task_order[i], task_order[i+1]) for i in range(0, len(task_order), 2)]
        else:
            self.task_classes_arr =  [(0, 1, 2, 3, 4),  (5, 6, 7, 8, 9)]
        self.tasks_num = len(self.task_classes_arr)
        self.prepare_tasks()

    def prepare_tasks(self):
        '''
            Organizes data into tasks based on specified class labels.
        '''
        for task_classes in self.task_classes_arr:
            train_mask = np.isin(self.t_train, task_classes)
            test_mask = np.isin(self.t_test, task_classes)
            x_train_task, t_train_task = self.x_train[train_mask], self.t_train[train_mask]
            x_test_task, t_test_task = self.x_test[test_mask], self.t_test[test_mask]

            self.task_data.append((x_train_task, t_train_task, x_test_task, t_test_task))


    def print_data_distribution(self):
        '''
            Displays basic statistics of the dataset.
        '''
        print("-" * 50)
        print("x_train dim and type: ", self.x_train.shape, self.x_train.dtype)
        print("t_train dim and type: ", self.t_train.shape, self.t_train.dtype)
        print("x_test dim and type: ", self.x_test.shape, self.x_test.dtype)
        print("t_test dim and type: ", self.t_test.shape, self.t_test.dtype)
        
        for i in range(10):
            print(f'Size of label {i} in training : {len(self.t_train[self.t_train == i])}')
            print(f'Size of label {i} in testing : {len(self.t_test[self.t_test == i])}')
            print("-" * 50)
