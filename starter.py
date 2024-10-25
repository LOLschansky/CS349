import math
from random import choice, choices
from copy import deepcopy

########################################################
#
# Part 1
#
########################################################

"""
Question 2:
    In order to process our data, our team elected to transform the grayscale pixel data
    to binary, setting pixels with values greater than or equal to 128 to 1 and all other
    values to 0. This makes the math operations faster and reduces the complexity of the
    data being manipulated, but still keeps the important edge information required to 
    make decisions. Moreover, it reduces the impact that minor intensity noise can have 
    on the overall classification.
    In terms of hyper-parameters, we opted to use a k value of 10, since larger k values
    reduce susceptibility to noise and reduces the risk of overfitting. We used all of
    the observations to have as large of a dataset as possible. Finally, we did not use
    default labels. If a given image was identified to be multitple classes (equal number
    of nearest neighbors for multiple classes), we randomly selected one of them. For 
    example, with a k of 5, and the nearest neighbors are 1, 1, 7, 7, 2, we would randomly
    selected between 1 and 7.
    
    Confusion Matrix (using Euclidean Distance):
        0 :  [17,  0,  1,  0,  0,  0,  0,  0,  0,  0]
        1 :  [ 0, 27,  0,  0,  0,  0,  0,  0,  0,  0]
        2 :  [ 0,  2, 15,  0,  0,  0,  0,  2,  0,  0]
        3 :  [ 0,  2,  0, 16,  0,  0,  0,  0,  0,  0]
        4 :  [ 0,  0,  0,  0, 23,  0,  0,  0,  0,  2]
        5 :  [ 0,  1,  0,  2,  0,  8,  1,  0,  0,  1]
        6 :  [ 0,  0,  0,  0,  0,  0, 13,  0,  0,  0]
        7 :  [ 0,  1,  0,  0,  0,  0,  0, 23,  0,  0]
        8 :  [ 1,  1,  1,  1,  0,  1,  0,  0, 16,  0]
        9 :  [ 0,  2,  0,  0,  1,  0,  1,  1,  0, 17]
        
    In total, our model had 87.5% accuracy on the test data set. For the most part, it was
    good at identifying the majority of classes, with the numbers 1 and 6 being classified 
    correctly 100% of the time. The only major exceptions came in the form of the number 5 
    and to a lesser extent, the number 8. Most notably, the model only classified true 5's
    61% of the time, meaning that it was difficult to distinguish it from the other numbers.
    Also, numbers 2, 3, and 9 had some overlap with class 1, meaning that they might have
    shared some similar features. To improve upon this, we could potentially introduce more
    training data for the numbers 5 and 8 for the model to train on.

""" 
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
        Exception("The length of vector a is different from the length of vector b!")
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
# K-Nearest Neighbors
###############################
def knn(train, query, metric):
    '''
    Takes in a training dataset as a list of examples (format in read_data), a query dataset, which is also
    a list of examples (format in read_data), as well as a metric, return a list of labels for the query
    dataset.
    '''
    
    # If incorrect metric entered, return
    if metric != "euclidean" and metric != "cosim":
        Exception("Incorrect metric entered! Please try again.")
        return
    # If correct metric entered, continue
    else:
        query_labels = []
        # Iterate over all query points
        for query_idx, query_point in enumerate(query):
            print("Current query: ", query_idx)
            distances = []
            # Unused, because we are predicting the label
            query_point_label = query_point[0]
            # Retrieve attribute values for given query point
            query_point_attribute_vals = query_point[1]
            # Iterate over all examples in the training set to find the k nearest neighbors
            for example_idx, example in enumerate(train):
                # print("Current example: ", example_idx)
                # Get the label for a given example
                example_label = example[0]
                # Get the attribute values for a given example
                example_attribute_vals = example[1]
                # Find the distance between the example and the query point
                distance = find_distance(example_attribute_vals, query_point_attribute_vals, metric)
                # Add to distances list as a tuple with the distance and label of the example from the training set
                distances.append((distance, example_label))
            
            # Implement knn, but for now we are just doing k = 1
            k = 10
            k_nearest_cnt = 0
            k_nearest = ""
            k_nearest_neighbors = {}
            print("Now evaluating query to find nearest neighbors")
            # Iterate over k neighbors
            for _ in range(k):
                # Find label with shortest distance
                min_tuple = min(distances, key = lambda x: x[0])
                # Find index of label with shortest distance
                min_idx = distances.index(min_tuple)
                # Append the label of with the shortest distance
                k_nearest_neighbors[min_tuple[1]] = k_nearest_neighbors.get(min_tuple[1], 0) + 1
                # Check if this is the most common occurence
                if k_nearest_neighbors[min_tuple[1]] > k_nearest_cnt:
                    k_nearest = min_tuple[1]
                    k_nearest_cnt = k_nearest_neighbors[min_tuple[1]]                                
                # Remove tuple from the list if searching for k > 1
                distances.pop(min_idx)
            # Iterate over k_nearest_neighbors to find all classes with count equal to the mode, in case there are multiple
            mode_k_nearest_neighbors = []
            for k_class in list(k_nearest_neighbors):
                # If count of class is equal, add it to the mode list
                if k_nearest_neighbors[k_class] == k_nearest_cnt:
                    mode_k_nearest_neighbors.append(k_class)
            # Select a random item from the mode list
            k_nearest = choice(list(mode_k_nearest_neighbors))
            print("Result: ", k_nearest)
            # Append result to query labels result
            query_labels.append([k_nearest, query_point_attribute_vals]) 
    
    return(query_labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
###############################
# K-Means
###############################
def kmeans(train,query,metric):
    '''
    Takes in a training dataset as a list of examples (format in read_data), a query dataset, which is also
    a list of examples (format in read_data), as well as a metric, return a list of labels for the query
    dataset.
    To implement K-Means, our implementation ignores the labels in the training set
    '''
    # If incorrect metric entered, return
    if metric != "euclidean" and metric != "cosim":
        Exception("Incorrect metric entered! Please try again.")
        return
    
    # Hard-coded tolerance:
    max_distance = 110
    # Current tolerance
    total_distance = float('inf')
    # Hard-coded number of classes
    num_classes = 10
    # Hard-coded number of attributes
    num_attributes = 784
    
    # Step 1: Randomly Select Means
    
    # Randomly select means
    means = [mean[1] for mean in choices(train, k=num_classes)]
            
    # Counter for printing
    ite = 0 

    # net distance of all points to all means
    net_distance = float('inf')
   
    # Continue until convergence
    while net_distance > max_distance:
        
        # Create classes dictionary to store examples corresponding to each class
        classes = [[] for _ in range(num_classes)]
        
        # Step 2: Calculate distance to means for every data point
        net_distance = 0
        
        # Iterate over all examples in the training data set
        for example in train:
            # Get the attribute values for a given example
            example_attribute_vals = example[1]
            # Initialize min distance and min mean variables to keep track of shortest distance
            min_distance = float('inf')
            min_mean = -1

            # Iterate over all of the current means to find the closest one to this point
            for idx, mean in enumerate(means):

                # Find the distance between the example and the mean
                distance = find_distance(example_attribute_vals, mean, metric)

                if distance < min_distance:
                    min_mean = idx
                    min_distance = distance
            
            net_distance += min_distance

            # Step 3: Assign class labels based upon shortest distance
            classes[min_mean].append(example[1])
        
        # Step 4: Update means
        
        # Iterate overall classes in the dictionary to find the mean of each one
        means = []
            
        # Iterate over all of the classes
        for myclass in classes:
            print(len(myclass))
            if len(myclass) == 0:
                means.append(choices(train, k=1)[0][1])
            else:
                means.append(find_mean(myclass))

        print("Iteration: ", ite) 
        print(net_distance) 
        input('next')
        ite += 1        
        # Update current tolerance
        


    query_labels = []            
    for query_point in query:
        min_distance = float('inf')
        min_mean = ""
        for mean in new_means:
            distance = find_distance(query_point[1], new_means[mean][1], metric)
            if distance < min_distance:
                min_mean = mean
                min_distance = distance
        query_labels.append([min_mean, query_point[1]])
        
    return query_labels
            
###############################
# Test K-Means
###############################
def test_kmeans(train_dataset, query_dataset):
    '''
    Take in a query dataset (format in read_data) and the labeled query dataset
    (returned from knn function). Returns accuracy as a float.
    Also prints out 10x10 confusion matrix.
    '''
    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
    query_labels = kmeans(train_dataset, query_dataset, "euclidean")
    correct = 0
    for idx, query in enumerate(query_labels):
        # Add to confusion matrix
        confusion_matrix[int(query_dataset[idx][0])][int(query[0])] += 1
        # Check if correct
        if query[0] == query_dataset[idx][0]:
            correct += 1
            print("Correct! Current Accuracy: ", correct / (idx + 1))
        else:
            print("Incorrect! Current Accuracy: ", correct / (idx + 1))
    # Calculate overall accuracy
    accuracy = correct / len(query_dataset)
    print("Total Accuracy: ", accuracy)
    print("Confusion Matrix:")
    for idx in range(10):
        print(idx, ": ", confusion_matrix[idx])                   


###############################
# Test KNN
###############################
def test_knn(train_dataset, query_dataset):
    '''
    Take in a query dataset (format in read_data) and the labeled query dataset
    (returned from knn function). Returns accuracy as a float.
    Also prints out 10x10 confusion matrix.
    '''
    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
    query_labels = knn(train_dataset, query_dataset, "euclidean")
    correct = 0
    for idx, query in enumerate(query_labels):
        # Add to confusion matrix
        confusion_matrix[int(query_dataset[idx][0])][int(query[0])] += 1
        # Check if correct
        if query[0] == query_dataset[idx][0]:
            correct += 1
            print("Correct! Current Accuracy: ", correct / (idx + 1))
        else:
            print("Incorrect! Current Accuracy: ", correct / (idx + 1))
    # Calculate overall accuracy
    accuracy = correct / len(query_dataset)
    print("Total Accuracy: ", accuracy)
    print("Confusion Matrix:")
    for idx in range(10):
        print(idx, ": ", confusion_matrix[idx])

###############################
# Process Data
###############################
def process_data(data_set):
    '''
    Takes in a data set returned from read_data, and turns it from grayscale to
    binary. The boundary is at 128. If the pixel is greater than or equal to 128, 
    it is set to 1. If less than, it is set to 0.
    '''
    # Iterate over all data
    processed_data_set = []
    for example in data_set:
        # Create processed example
        processed_pixels = []
        # print(example)
        for pixel in example[1]:
            if int(pixel) >= 128:
                processed_pixels.append('1')
            else:
                processed_pixels.append('0')
        processed_example = [example[0], processed_pixels]
        processed_data_set.append(processed_example)

    return processed_data_set

###############################
# Read Data
###############################
def read_data(file_name):
    '''
    Takes in a csv file name, and returns a list of examples, where each example 
    is a list of length 2, with the first index is the class and the second index 
    is the list of pixels
    '''
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
                
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    train_dataset = process_data(read_data("mnist_train.csv"))
    query_dataset = process_data(read_data("mnist_test.csv"))
    # test(train_dataset, query_dataset)
    # print(find_mean([[0,0,1], [1,0,0]]))
    test_kmeans(train_dataset, query_dataset)
    
if __name__ == "__main__":
    main()
    