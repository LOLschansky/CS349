from random import choice
from helpers import find_distance, process_data, read_combine_data, split_data
import copy

###############################
# K-Nearest Neighbors
###############################
def knn(train, query, metric):
    '''
    Takes in a training dataset as a list of examples (format in read_combine_data), a query dataset, which is also
    a list of examples (format in read_combine_data), as well as a metric, return a list of labels for the query
    dataset.
    '''
    
    # If incorrect metric entered, return
    if metric != "euclidean" and metric != "cosim":
        raise Exception("Incorrect metric entered! Please try again.")
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
                example_attribute_vals = copy.deepcopy(example[1])
                # Make a copy of query_point_attribute_vals since we are modifying it
                temp_query_point_attribute_vals = copy.deepcopy(query_point_attribute_vals)
                # Determine if any values are -1. If so, find their indices and remove from both lists.
                for attribute_vals in [temp_query_point_attribute_vals, example_attribute_vals]:
                    indices = [idx for idx, val in enumerate(attribute_vals) if val == -1]
                    for idx in reversed(indices):
                        del temp_query_point_attribute_vals[idx]
                        del example_attribute_vals[idx]       
                
                # Find the distance between the example and the query point
                distance = find_distance(example_attribute_vals, temp_query_point_attribute_vals, metric)
                # Add to distances list as a tuple with the distance and label of the example from the training set
                distances.append((distance, example_label))
            
            # Implement knn, but for now we are just doing k = 1
            k = 10
            k_nearest_cnt = 0
            k_nearest = ""
            k_nearest_neighbors = {}
            # print("Now evaluating query to find nearest neighbors")
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
            # print("Result: ", k_nearest)
            # Append result to query labels result
            query_labels.append([k_nearest, query_point_attribute_vals]) 
    
    return query_labels 

###############################
# Test KNN
###############################
def test_knn(train_dataset, query_dataset, error=0):
    '''
    Take in a query dataset (format in read_combine_data) and a query dataset.
    Returns accuracy as a float. Also prints out 10x10 confusion matrix.
    '''
    confusion_matrix = [[0 for _ in range(9)] for _ in range(9)]
    query_labels = knn(train_dataset, query_dataset, "euclidean")
    
    correct = 0
    for idx, query in enumerate(query_labels):
        # Error checking
        if int(query_dataset[idx][0]) < 0 or int(query_dataset[idx][0]) >= len(confusion_matrix):
            label1 = 8
        else:
            label1 = int(query_dataset[idx][0])
        if int(query[0]) < 0 or int(query[0]) >= len(confusion_matrix[0]):
            label2 = 8
        else:
            label2 = int(query[0])
        # Add to confusion matrix
        confusion_matrix[label1][label2] += 1
        # Check if correct
        error_range = list(range(query[0] - error, query[0] + error + 1))
        if query_dataset[idx][0] in error_range:
            correct += 1
            # print("Correct! Current Accuracy: ", correct / (idx + 1))
        else:
            # print("Incorrect! Current Accuracy: ", correct / (idx + 1))
            pass
    # Calculate overall accuracy
    accuracy = correct / len(query_dataset)
    print("Total Accuracy: ", accuracy)
    print("Confusion Matrix:")
    for idx in range(1,9):
        print(idx, ": ", confusion_matrix[idx][1:])
        
###############################
# KNN Accuracy
###############################   
def knn_accuracy(recommendations, query_user):
    # Iterate through the query_user data set and only include examples where recommendations overlap
    for query in query_user:
        query_reviews = query_user[query].reviews
        # tp, fp, and fn are tricky... not based on ratings themselves
        tp = 0 # true positives are recs that user has seen (not necessarily rated highly)
        fp = 0 # false positives are recs that user hasn't seen
        fn = 0 # false negatives are movies that user has seen but weren't recommended
        # print(query_reviews)
        for movie_id in recommendations:
            if movie_id in query_reviews:
                tp +=1
            else: 
                fp +=1
        fn = len(query_reviews) - len(recommendations)
        if fn < 0:
            fn = 0
    precision = tp/(tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*(precision * recall) / (precision + recall)
        
    return precision, recall, f1
            
def main():
    # Get data from CSV file
    data = read_combine_data("data/nfl_combine_2010_to_2023.csv", include_undrafted=True)
    # Process data to convert it to numerical values and normalize
    processed_data = process_data(data)
    # Split dataset up into training, validation, and testing sets
    train_set, valid_set, test_set = split_data(processed_data, test=False)
    test_knn(train_set, test_set, error=0)
    
"""
When only considering players that got drafted:
Total Accuracy (Correct within 2 rounds):  0.63003663003663
Confusion Matrix:
1 :  [5, 2, 12, 4, 7, 2, 10, 0]
2 :  [2, 5, 9, 7, 6, 2, 7, 0]
3 :  [3, 8, 5, 8, 4, 6, 12, 0]
4 :  [6, 8, 12, 10, 10, 2, 4, 0]
5 :  [2, 6, 6, 7, 7, 3, 7, 0]
6 :  [0, 4, 10, 5, 4, 2, 7, 0]
7 :  [1, 3, 6, 3, 2, 2, 8, 0]
8 :  [0, 0, 0, 0, 0, 0, 0, 0]
When considering all players:
Total Accuracy (Correct within 2 rounds):  0.4013921113689095
Confusion Matrix:
1 :  [5, 3, 10, 0, 7, 4, 0, 28]
2 :  [1, 0, 6, 0, 3, 1, 0, 27]
3 :  [4, 0, 2, 0, 3, 2, 0, 28]
4 :  [5, 2, 6, 1, 3, 2, 0, 28]
5 :  [3, 0, 1, 2, 2, 2, 1, 20]
6 :  [1, 1, 2, 3, 3, 1, 0, 24]
7 :  [0, 1, 3, 2, 1, 1, 0, 23]
8 :  [10, 3, 15, 3, 6, 7, 1, 108]

This accuracy is pretty bad.
"""
    
if __name__ == "__main__":
    main()
    