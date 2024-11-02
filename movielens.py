from starter import knn, kmeans, test, find_distance
from random import choice
import math

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
def movielens_recommender(query_user, all_users, metric, max_movies, repeat_recs):
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
            k = 50
            m = 10
            #dict containing average movie ratings for all movies among k nearest neighbors. key: movie_id. value: [ratings, count, avg]
            avg_movie_ratings = {}
            print("Now evaluating query to find nearest neighbors")
            
            # Iterate over k neighbors
            for _ in range(k):
                # Find user with shortest distance to query
                min_tuple = min(distances, key = lambda x: x[0])
                #dict of reviews for the kth neighbor (key: movie_id. value: full rating)
                kth_reviews = all_users[min_tuple[1]].reviews
                #create dict of avg_movie_ratings, where first element is sum of ratings for that id, second element is count, third is avg.
                for movie_id in kth_reviews:
                    if movie_id not in avg_movie_ratings:
                        avg_movie_ratings[movie_id] = [0, 0, 0]
                        avg_movie_ratings[movie_id][0] = int(kth_reviews[movie_id].rating)
                        avg_movie_ratings[movie_id][1] = 1
                        avg_movie_ratings[movie_id][2] = int(kth_reviews[movie_id].rating)
                    else:
                        avg_movie_ratings[movie_id][0] = avg_movie_ratings[movie_id][0] + int(kth_reviews[movie_id].rating)
                        avg_movie_ratings[movie_id][1] = int(avg_movie_ratings[movie_id][1]) + 1
                        avg_movie_ratings[movie_id][2] = avg_movie_ratings[movie_id][0] / float(avg_movie_ratings[movie_id][1])
                print(min_tuple)
                min_idx = distances.index(min_tuple)
                #remove user w/ minimum distance
                distances.pop(min_idx)
            #sort our dictionary by avg distance (descending order)
            sorted_avg_ratings = dict(sorted(avg_movie_ratings.items(), key=lambda item: item[1][2], reverse=True))
    
            #recommend m movies w/ highest avg ratings
            recommendations = []
            if repeat_recs: # meaning we can recommend movies a user has already seen (necessary for validaiton)
                recommendations = list(sorted_avg_ratings.keys())[:m]
            else : # meaning we can't recommend movies a user has already seen (for practical use)
                for rec in sorted_avg_ratings:
                    if rec not in list(query_user.values())[0].reviews:
                        recommendations.append(rec)
                recommendations = recommendations[:m]
    
        return recommendations


###############################
# MovieLens Recommender
###############################  
def movielens_recommender_improved(query_user, all_users, metric, max_movies):
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
            query_reviews = [0 for _ in range(max_movies + 1)]
            for review in query_data.reviews:
                query_reviews[int(review)] = query_data.reviews[review].rating
            query_reviews.append(query_data.age)
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
                    user_reviews.append(user_data.age)
                    # Find distance between query user and user from training set
                    distance = find_distance(query_reviews, user_reviews, metric)
                    # Add to distances list as a tuple with the distance and user from the training set
                    distances.append((distance, user))
                    
            # Implement KNN
            k = 10
            m = 10
            k_nearest_cnt = 0
            k_nearest = ""
            #k_nearest_neighbors = {}
            new_knn = []
            #dict containing average movie ratings for all movies among k nearest neighbors. key: movie_id. value: [ratings, count, avg]
            avg_movie_ratings = {}
            print("Now evaluating query to find nearest neighbors")
            
            # Iterate over k neighbors
            for _ in range(k):
                # Find user with shortest distance to query
                min_tuple = min(distances, key = lambda x: x[0])
                #append user_id to list
                new_knn.append(min_tuple[1])
                #get dict of reviews for the kth neighbor (key: movie_id. value: full rating)
                kth_reviews = all_users[min_tuple[1]].reviews
                
                for movie_id in kth_reviews:
                    if movie_id not in avg_movie_ratings:
                        avg_movie_ratings[movie_id] = [0, 0, 0]
                        avg_movie_ratings[movie_id][0] = int(kth_reviews[movie_id].rating)
                        avg_movie_ratings[movie_id][1] = 1
                        avg_movie_ratings[movie_id][2] = int(kth_reviews[movie_id].rating)
                    else:
                        avg_movie_ratings[movie_id][0] = avg_movie_ratings[movie_id][0] + int(kth_reviews[movie_id].rating)
                        avg_movie_ratings[movie_id][1] = int(avg_movie_ratings[movie_id][1]) + 1
                        avg_movie_ratings[movie_id][2] = avg_movie_ratings[movie_id][0] / float(avg_movie_ratings[movie_id][1])
                print(min_tuple)
                min_idx = distances.index(min_tuple)
                distances.pop(min_idx)
                
            sorted_avg_ratings = dict(sorted(avg_movie_ratings.items(), key=lambda item: item[1][2], reverse=True))
            # first_m_items = list(sorted_avg_ratings.items())[:m]
            # for key, value in first_m_items:
            #     print(key, value)
            print(sorted_avg_ratings)
            print("recommendations:")
            #recommendations is a list storing ids of movie recs
            recommendations = []
            if test_accuracy: # meaning we can recommend movies a user has already seen (necessary for validaiton)
                recommendations = list(sorted_avg_ratings.keys())[:m]
            else : # meaning we can't recommend movies a user has already seen (for practical use)
                for rec in sorted_avg_ratings:
                    if rec not in list(query_user.values())[0].reviews:
                        recommendations.append(rec)
        #         print(recommendation)
        #         recommendations.append(recommendation.movie_id)
                
            print(recommendations)
                
                
        #         # Find index of user with shortest distance
        #         min_idx = distances.index(min_tuple)
        #         # Append the user_id with the shortest distance
        #         k_nearest_neighbors[min_tuple[1]] = k_nearest_neighbors.get(min_tuple[1], 0) + 1
        #         # Check if this is the most common occurence
        #         if k_nearest_neighbors[min_tuple[1]] > k_nearest_cnt:
        #             k_nearest = min_tuple[1]
        #             k_nearest_cnt = k_nearest_neighbors[min_tuple[1]]                                
        #         # Remove tuple from the list if searching for k > 1
        #         distances.pop(min_idx)
        #     # Iterate over k_nearest_neighbors to find all classes with count equal to the mode, in case there are multiple
        #     mode_k_nearest_neighbors = []
        #     for k_class in list(k_nearest_neighbors):
        #         # If count of class is equal, add it to the mode list
        #         if k_nearest_neighbors[k_class] == k_nearest_cnt:
        #             mode_k_nearest_neighbors.append(k_class)
        #     # Select a random item from the mode list
        #     nearest_user = choice(list(mode_k_nearest_neighbors))
        #     print("Closest Neighbor: ", nearest_user)
        
        # # Using k_nearest, which is the user_id of the closest neighbor to the query user,
        # #   find the M highest rated movies that the query user has not yet watched
        # # Hyper-parameter of number of movies to recommend
        # m = 20
        # nearest_user_reviews = all_users[nearest_user].reviews
        # # Add user reviews to sorted list
        # nearest_user_reviews_list = sorted(nearest_user_reviews.values(), key=lambda x: int(x.rating))
        # print(nearest_user_reviews_list)
        # recommendations = []
        # while len(recommendations) < m:
        #     # print(len(nearest_user_reviews_list))
        #     recommendation = nearest_user_reviews_list.pop()
        #     if str(recommendation.movie_id) not in list(query_user.values())[0].reviews:
        #         print(recommendation)
        #         recommendations.append(recommendation.movie_id)
        #     else:
        #         continue
        
        return recommendations

###############################
# MovieLens Accuracy
###############################   
def movielens_accuracy(recommendations, query_user):
    #iterate through the query_user data set and only include examples where recommendations overlap
    for query in query_user:
        query_reviews = query_user[query].reviews
        # tp, fp, and fn are tricky... not based on ratings themselves
        tp = 0 # true positives are recs that user has seen (not necessarily rated highly)
        fp = 0 # false positives are recs that user hasn't seen
        fn = 0 # false negatives are movies that user has seen but weren't recommended
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

if __name__ == "__main__":
    all_users, max_movie_id = movielens_parse(input_file="movielens.txt")
    one_user, _ = movielens_parse(input_file="test_a.txt")
    #test_accuracy is a boolean. True: we can recommend movies a user has seen, False: we can't
    repeat_recs = True
    recommendations = movielens_recommender(one_user, all_users, "euclidean", max_movie_id, repeat_recs)
    #recommendations = movielens_recommender_improved(one_user, all_users, "euclidean", max_movie_id)
    #precision, recall, f1 = movielens_accuracy(recommendations, one_user)
    print(recommendations)
    #print(precision, recall, f1)
