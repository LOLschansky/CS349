import math
from random import choice, choices
from copy import deepcopy
import starter

test_data = [['1',['1','0','0']], ['0',['1','0','1']], ['3',['1','1','1']], ['3',['1','1','1']], ['5',['1','1','1']]]

# Hard-coded tolerance:
max_distance = 110
# Current tolerance
total_distance = float('inf')
# Hard-coded number of classes
num_classes = 10
# Hard-coded number of attributes
num_attributes = 784

means = choices(test_data, k=2)
print("initial means", means)
new_means = {}
# Iterates over means
for idx, mean in enumerate(means):
    # Adds mean to class 0, 1, 2, ...
    new_means[str(idx)] = mean
print("new means", new_means)
 # Counter for printing
ite = 0 
                
# Continue until convergence
while total_distance > max_distance:
            
    # Set the old mean to be the new means that were just calculated
    old_means = deepcopy(new_means)
            
    # Create classes dictionary to store examples corresponding to each class
    classes = {}
    for idx in range(10):
        classes[str(idx)] = []