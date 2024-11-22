import csv
import math
from random import sample

###############################
# Find distance
###############################   
def find_distance(a, b, metric):
    
    # If euclidean, use euclidean distance
    if metric == "euclidean":
        distance = euclidean(a, b)
    # If cosim, use cosine similarity
    elif metric == "cosim":
        distance = cosim(a, b)
        
    return distance

###############################
# Find mean of vectors
###############################                 
def find_mean(vecs):        
    '''
    Finds the mean of a list of vectors
    '''
    # Initialize zeros for the average vector
    avg = [0 for _ in range(len(vecs[0]))]
    
    # Iterate over all vectors in the list
    for vec in vecs:
        # Add from every index to the average vector
        for idx, _ in enumerate(vecs[0]):
            avg[idx] += int(vec[idx])
    # Divide every index by the length of the list of vectors
    for idx, _ in enumerate(avg):
        avg[idx] /= len(vecs)
        
    return avg

###############################
# Verify Vector Lengths
###############################
def verify(a, b):
    if len(a) != len(b):
        raise Exception("The length of vector a is different from the length of vector b!")
    else:
        return

###############################
# Euclidean Distance
###############################
def euclidean(a,b):
    '''
    Takes in vectors a and b as lists, and returns the euclidian distance
    as a float scalar.
    ''' 
    # Verify vector lengths
    verify(a, b)
    
    # Find Euclidean Distance
    dist = math.sqrt(sum((int(a_i) - int(b_i)) ** 2 for a_i, b_i in zip(a, b)))
        
    return dist
        
###############################
# Cosine Similarity
###############################
def cosim(a,b):
    '''
    Takes in vectors a and b as lists, and returns the cosine similarity
    as a float scalar.
    '''
    # Verify vector lengths
    verify(a, b)
    
    norm_a = euclidean(a, [0 for _ in range(len(a))])
    norm_b = euclidean(b, [0 for _ in range(len(b))])
    dot_a_b = sum(int(a_i) * int(b_i) for a_i, b_i in zip(a, b))
    
    return dot_a_b / (norm_a * norm_b)

###############################
# Pearson Correlation
###############################
def pearson(a, b):
    '''
    Takes in vectors a and b as lists, and returns the pearson correlation
    as a float scalar.
    '''
    # Verify vector lengths
    verify(a, b)
    
    # Find the means of a and b
    mu_a = sum(a) / len(a)
    mu_b = sum(b) / len(b)
    # Find the square root of the sum of all distances for a
    a_squared_sum = 0
    for a_i in a:
        a_squared_sum += (a_i - mu_a) ** 2
    # Find the square root of the sum of all distances for b
    b_squared_sum = 0
    for b_i in b:
        b_squared_sum += (b_i - mu_b) ** 2
    # Calculate the denominator
    denominator = math.sqrt(a_squared_sum) * math.sqrt(b_squared_sum)
    # Calculate the numerator
    numerator = 0
    for i in range(len(a)):
        numerator += (a[i] - mu_a) * (b[i] - mu_b)
    # Find the Pearson Correlation coefficient
    r_ab = numerator / denominator
    
    return r_ab

###############################
# Hamming Distance
###############################
def hamming(a, b):
    '''
    Takes in vectors a and b as lists, and returns the hamming distance as
    an int
    '''
    # Verify vector lengths
    verify(a, b)
    
    # Initialize hamming distance
    ham = 0
    # Iterate over a and b
    for a_i, b_i in zip(a, b):
        if a_i != b_i:
            ham += 1

    return ham

###############################
# Process Height
###############################
def process_height(example):
    height_list = example[1][1].split('-')
    if len(height_list) == 1:
        height = -1
    else:
        # Get the height in inches
        height = int(height_list[0]) * 12 + int(height_list[1])
    return height

###############################
# Process Attribute
###############################
def process_attribute(value):
    if value == '':
        attribute = -1
    else:
        attribute = float(value)
    return attribute
    
###############################
# Process Data
###############################
def process_data(data_set):
    '''
    Takes in a data set returned from read_data, and turns each attribute into an integer
    value in order to use K-Nearest Neighbors. For the position, each position is placed into
    the list and if the user plays that position, the corresponding index is 1.
    '''
    positions_set = set()
    height_list = []
    weight_list = []
    yd40_list = []
    vertical_list = []
    bench_list = []
    # Iterate over all data
    for example in data_set:
        positions_set.add(example[1][0])
        height_list.append(process_height(example))
        weight_list.append(process_attribute(example[1][2]))
        yd40_list.append(process_attribute(example[1][3]))
        vertical_list.append(process_attribute(example[1][4]))
        bench_list.append(process_attribute(example[1][5]))
    # Assign each position a location in the vector for positions
    positions_dict = {pos: idx for idx, pos in enumerate(positions_set)}
    # Find the min and max values for position, height, weight, yd40, vertical, and bench to normalize values
    height_min = min([val for val in height_list if val > 0])
    height_max = max(height_list)
    weight_min = min([val for val in weight_list if val > 0])
    weight_max = max(weight_list)
    yd40_min = min([val for val in yd40_list if val > 0])
    yd40_max = max(yd40_list)
    vertical_min = min([val for val in vertical_list if val > 0])
    vertical_max = max(vertical_list)
    bench_min = min([val for val in bench_list if val > 0])
    bench_max = max(bench_list)
    
    # Iterate over all data
    processed_data_set = []
    for example in data_set:
        # Get the round
        round = example[0]
        # If the round is empty, the user was not drafted. Set it to "round 8" to represent not getting drafted.
        if round == '':
            round_int = -100
        # Otherwise, turn it into an integer
        else:
            round_int = int(round)
        # Positions list initialization
        positions_list = [0] * len(positions_set)
        # Set the position for the given player
        positions_list[positions_dict[example[1][0]]] = 5
        height = process_height(example)
        if height != -1:
            # Normalize the height
            height = (height - height_min) / (height_max - height_min)
        # Convert weight into float value
        weight = process_attribute(example[1][2])
        if weight != -1:
            # Normalize the weight
            weight = (weight - weight_min) / (weight_max - weight_min)
        # Convert 40yd into float value
        yd40 = process_attribute(example[1][3])
        if yd40 != -1:
            # Normalize the 40 yard dash
            yd40 = (yd40 - yd40_min) / (yd40_max - yd40_min)
        # Convert Vertical into float value
        vertical = process_attribute(example[1][4])
        if vertical != -1:
            # Normalize the vertical
            vertical = (vertical - vertical_min) / (vertical_max - vertical_min)
        # Convert bench into float value
        bench = process_attribute(example[1][5])
        if bench != -1:
            # Normalize the bench
            bench = (bench - bench_min) / (bench_max - bench_min)
            
        processed_data_set.append([round_int, positions_list + [height, weight, yd40, vertical, bench]])

    return processed_data_set

###############################
# Read Data
###############################
def read_combine_data(file_name, include_undrafted=True):
    '''
    Takes in a name for an NFL combine csv file as a string, and returns a list of examples, 
    where each example is a list of length 2, with the first index being the draft 
    round and the second index being a list of combine results in the order 
    [Position, Height, Weight, 40 Yard Dash, Vertical, Bench Press Repetions].
    '''
    
    data_set = []
    with open(file_name,'rt') as file:
        # Map every row to dictionaries using the header row as keys
        reader = csv.DictReader(file)
        # Iterate over every row to add each player to the data_set list
        for row in reader:
            if include_undrafted:
                data_set.append([row['Round'], [row['Pos'], row['Height'], row['Weight'], row['40yd'], row['Vertical'], row['Bench']]])
            else:
                if row['Round'] != "":
                    data_set.append([row['Round'], [row['Pos'], row['Height'], row['Weight'], row['40yd'], row['Vertical'], row['Bench']]])
        return data_set
                
    return(data_set)

################################
# Split Data
################################
def split_data(data, test=False):
    '''
    Takes in data from read_combine_data and returns three datasets: a training set,
    validation set, and a test set. The size of each set is selected by hyperparameters.
    '''
    
    # Hyperparameters for dataset sizes
    if not test:
        train_size = int(len(data) // 1.1)
        valid_size = 1
        test_size = len(data) - train_size - valid_size
    else:
        train_size = len(data) // 3
        valid_size = len(data) // 3
        test_size = len(data) - train_size - valid_size
    
    # Randomized list of data
    data_randomized = sample(data, len(data))
    
    train_set = data_randomized[:train_size]
    valid_set = data_randomized[train_size:train_size+valid_size]
    test_set = data_randomized[train_size+valid_size:]
    return train_set, valid_set, test_set

def accuracy(confusion_matrix):
    for error in range(0, 5):
        correct = 0
        total = 0
        for idx, row in enumerate(confusion_matrix):
            error_range = list(range(max(0, idx-error), min(len(confusion_matrix), idx+error+1)))
            correct += sum([row[idx] for idx in error_range])
            total += sum(row)
        print("Error:", error, "| Accuracy: ", correct/total)

def metrics(confusion_matrix):
    precisions = []
    recalls = []
    f1s = []
    
    # Iterate over the entire confusion matrix
    for idx in range(len(confusion_matrix)):
        # True positives
        tp = confusion_matrix[idx][idx]
        # False positives
        fp = sum(row[idx] for row in confusion_matrix) - tp
        # False negatives
        fn = sum(confusion_matrix[idx]) - tp
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision / recall) / (precision + recall) if (precision + recall) != 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        print("Draft Round:", idx)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
    
        
    return precisions, recalls, f1s
        


if __name__ == "__main__":
    confusion_matrix = [[5, 2, 12, 4, 7, 2, 10, 0],
                        [2, 5, 9, 7, 6, 2, 7, 0],
                        [3, 8, 5, 8, 4, 6, 12, 0],
                        [6, 8, 12, 10, 10, 2, 4, 0],
                        [2, 6, 6, 7, 7, 3, 7, 0],
                        [0, 4, 10, 5, 4, 2, 7, 0],
                        [1, 3, 6, 3, 2, 2, 8, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]
    # confusion_matrix = [[5, 3, 10, 0, 7, 4, 0, 28],
    #                     [1, 0, 6, 0, 3, 1, 0, 27],
    #                     [4, 0, 2, 0, 3, 2, 0, 28],
    #                     [5, 2, 6, 1, 3, 2, 0, 28],
    #                     [3, 0, 1, 2, 2, 2, 1, 20],
    #                     [1, 1, 2, 3, 3, 1, 0, 24],
    #                     [0, 1, 3, 2, 1, 1, 0, 23],
    #                     [10, 3, 15, 3, 6, 7, 1, 108]]
    metrics(confusion_matrix)
    accuracy(confusion_matrix)
    # data = read_combine_data("data/nfl_combine_2010_to_2023.csv")
    # processed_data = process_data(data)
    # train_set, valid_set, test_set = split_data(processed_data)
    
    
