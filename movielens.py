from starter import knn, kmeans, test, find_distance
from random import choice

class user:
    def __init__(self, age, gender, occupation):
        self.age = age
        self.gender = gender
        self.occupation = occupation
        self.reviews = {}
        # Key: movie_id
        # Value: full_rating class (including movie_id, rating, title, and genre)

class full_rating:
    def __init__(self, movie_id, rating, title, genre):
        self.movie_id = movie_id
        self.rating = rating
        self.title = title
        self.genre = genre

###############################
# MovieLens Data Parser
###############################   
def movielens_parse(input_file="movielens.txt"):
    
    # Initialize dictionary to hold all training data
    movielens_by_user = {}
    # Key: user_id
    # Value: user class (including age, gender, occupation, and reviews dictionary)
    
    with open(input_file, 'r') as file:
        # Get the column names
        fields = file.readline().replace('\n', '').split('\t')
        # Initialize variable to find max movie_id
        max_movie_id = 0
        # Iterate over the entire file
        for line in file:
            # Parse line from movielens.txt into list
            fields = line.replace('\n', '').split('\t')
            # Update max_movie_id
            if max_movie_id < int(fields[1]):
                max_movie_id = int(fields[1])
            # Use the user_id to add all fields besides user_id
            if fields[0] not in movielens_by_user:
                movielens_by_user[fields[0]] = user(age=fields[5], gender=fields[6], occupation=fields[7])
            movielens_by_user[fields[0]].reviews[fields[1]] = full_rating(movie_id=fields[1], rating=fields[2], title=fields[3], genre=fields[4])
    
    return movielens_by_user, max_movie_id

###############################
# MovieLens Recommender
###############################  
def movielens_recommender(query_user, all_users, metric, max_movies):
    # Find users who also liked the same movies
    # Look for movies they rated highly that our user has not watched yet
    # If incorrect metric entered, return
    if metric != "euclidean" and metric != "cosim":
        Exception("Incorrect metric entered! Please try again.")
        return
    # If correct metric entered, continue
    else:
        # Iterate over all query reviews (should only be one user)
        for query in query_user:
            # Get the queried user data (user class with fields age, gender, occupation, and reviews)
            query_data = query_user[query]
            # Put the reviews into a list where the index is the movie ID, and the value at that index is the rating. If the movie
            #   is not rated, a 0 is put in that location.
            query_reviews = [0 for _ in range(max_movies)]
            for review in query_data.reviews:
                query_reviews[int(review)] = query_data.reviews[review].rating
            print("Current query: ", query)
            # Initialize list to store distances
            distances = [] 
            # Iterate over all examples in the training set to find the k nearest neighbors
            for user in all_users:               
                # Skip if it's the same user
                if user == query:
                    continue
                else:
                    # Get data from the user from the training set
                    user_data = all_users[user]
                    # Put the reviews into a list where the index is the movie ID, and the value at that index is the rating. If the movie
                    #   is not rated, a 0 is put in that location.
                    user_reviews = [0 for _ in range(max_movies+1)]
                    for review in user_data.reviews:
                        user_reviews[int(review)] = user_data.reviews[review].rating
                    # Find distance between query user and user from training set
                    distance = find_distance(query_reviews, user_reviews, metric)
                    # Add to distances list as a tuple with the distance and user from the training set
                    distances.append((distance, user))
                    
            # Implement KNN
            k = 10
            k_nearest_cnt = 0
            k_nearest = ""
            k_nearest_neighbors = {}
            print("Now evaluating query to find nearest neighbors")
            # Iterate over k neighbors
            for _ in range(k):
                # Find user with shortest distance
                min_tuple = min(distances, key = lambda x: x[0])
                # Find index of user with shortest distance
                min_idx = distances.index(min_tuple)
                # Append the user_id with the shortest distance
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
            nearest_user = choice(list(mode_k_nearest_neighbors))
            print("Closest Neighbor: ", nearest_user)
        
        # Using k_nearest, which is the user_id of the closest neighbor to the query user,
        #   find the M highest rated movies that the query user has not yet watched
        # Hyper-parameter of number of movies to recommend
        m = 20
        nearest_user_reviews = all_users[nearest_user].reviews
        # Add user reviews to sorted list
        nearest_user_reviews_list = sorted(nearest_user_reviews.values(), key=lambda x: int(x.rating))
        print(nearest_user_reviews_list)
        recommendations = []
        while len(recommendations) < m:
            # print(len(nearest_user_reviews_list))
            recommendation = nearest_user_reviews_list.pop()
            if str(recommendation.movie_id) not in list(query_user.values())[0].reviews:
                print(recommendation)
                recommendations.append(recommendation.movie_id)
            else:
                continue
            
        return recommendations

###############################
# MovieLens Accuracy
###############################   
def movielens_accuracy(recommendations, query_user):
    #iterate through the query_user data set and only include examples where recommendations overlap
    for query in query_user:
        query_reviews = query_user[query].reviews
        tp = 0
        fp = 0
        fn = 0
        for movie_id in recommendations:
            if movie_id in query_reviews:
                rating = query_reviews[movie_id].rating
                #true positive means > 3
                if rating > 3:
                    tp +=1 
                else:
                    fp +=1
                #if greater than 3, add to TP, else add to FP
        precision = tp/(tp + fp)
        #recall = tp / (tp + fn)
        #f1 = 2*(precision * recall) / (precision + recall)
        
    return precision

if __name__ == "__main__":
    all_users, max_movie_id = movielens_parse(input_file="movielens.txt")
    one_user, _ = movielens_parse(input_file="test_a.txt")
    recommendations = movielens_recommender(one_user, all_users, "euclidean", max_movie_id)
    #precision = movielens_accuracy(recommendations, one_user)
    print(recommendations)
    #print(precision)
