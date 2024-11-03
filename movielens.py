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
    
    # Initialize set to hold all genres
    movielens_genres = set()
    # Initialize set to hold all genders
    movielens_genders = set()
    # Initialize set to hold all occupations
    movielens_occupations = set()
    
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
            # Add genre to genres set
            movielens_genres.add(fields[4])
            # Add gender to genders set
            movielens_genders.add(fields[6])
            # Add occupation to occupations set
            movielens_occupations.add(fields[7]) 
    
    return movielens_by_user, max_movie_id, movielens_genres, movielens_genders, movielens_occupations

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
def movielens_recommender_improved(query_user, all_users, metric, max_movies, repeat_recs, genres, genders, occupations):
    # Find users who also liked the same movies
    # Look for movies they rated highly that our user has not watched yet
    # If incorrect metric entered, return
    if metric != "euclidean" and metric != "cosim":
        Exception("Incorrect metric entered! Please try again.")
        return
    # If correct metric entered, continue
    else:
        # Occupation hyper-parameter
        o = 30
        # Gender hyper-parameter
        g = 1
        
        # Create dictionary to map each occupation to an index. For example, marketing : 0, executive : 1, etc.
        occupations_loc = {value: index for index, value in enumerate(occupations)}
        # Create dictionary to map each occupation to an index. For example, M : 0, F : 1, etc.
        genders_loc = {value: index for index, value in enumerate(genders)}
        
        # Iterate over all query reviews (should only be one user)
        for query in query_user:
            # Get the queried user data (user class with fields age, gender, occupation, and reviews)
            query_data = query_user[query]
            # Put the reviews into a list where the index is the movie ID, and the value at that index is the rating. If the movie
            #   is not rated, a 0 is put in that location.
            # TODO: Should it be 0? Or does 3 make more sense?
            query_reviews = [0 for _ in range(max_movies+1)]
            for review in query_data.reviews:
                query_reviews[int(review)] = query_data.reviews[review].rating
            # Add a user's age to the list for comparison
            query_reviews.append(query_data.age)
            # Add a user's gender to the list for comparison
            query_reviews += [0 if idx != occupations_loc[query_user[query].occupation] else o for idx in range(len(occupations))]
            # Add a user's gender to the list for comparison
            query_reviews += [0 if idx != genders_loc[query_user[query].gender] else g for idx in range(len(genders))]            
                        
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
                    # Add all reviews to the user reviews list
                    for review in user_data.reviews:
                        user_reviews[int(review)] = user_data.reviews[review].rating
                    # Add a user's age to the list for comparison
                    user_reviews.append(user_data.age)
                    # Add a user's gender to the list for comparison
                    user_reviews += [0 if idx != occupations_loc[user_data.occupation] else o for idx in range(len(occupations))]
                    # Add a user's gender to the list for comparison
                    user_reviews += [0 if idx != genders_loc[user_data.gender] else g for idx in range(len(genders))]            
                    
                    # Find distance between query user and user from training set
                    distance = find_distance(query_reviews, user_reviews, metric)
                    # Add to distances list as a tuple with the distance and user from the training set
                    distances.append((distance, user))
                    
            # Implement KNN
            k = 10
            m = 10
            k_nearest_cnt = 0
            k_nearest = ""
            # Initialize list 
            # Dictionary containing average movie ratings for all movies among k nearest neighbors. 
            #   Key: movie_id
            #   Values: [ratings, count, avg]
            avg_movie_ratings = {}
            print("Now evaluating query to find nearest neighbors")
            
            # Iterate over k neighbors
            for _ in range(k):
                # Find user with shortest distance to query
                min_tuple = min(distances, key = lambda x: x[0])
                # Get dict of reviews for the kth neighbor (key: movie_id. value: full_rating class)
                kth_reviews = all_users[min_tuple[1]].reviews
                
                # Iterate over all of the reviews for the kth neighbor
                for movie_id in kth_reviews:
                    # If the movie has not been added to the recommendations, create a new dictionary entry
                    if movie_id not in avg_movie_ratings:
                        avg_movie_ratings[movie_id] = [0, 0, 0]
                        avg_movie_ratings[movie_id][0] = int(kth_reviews[movie_id].rating)
                        avg_movie_ratings[movie_id][1] = 1
                        avg_movie_ratings[movie_id][2] = int(kth_reviews[movie_id].rating)
                    # Otherwise, update the current average rating for that movie
                    else:
                        avg_movie_ratings[movie_id][0] = avg_movie_ratings[movie_id][0] + int(kth_reviews[movie_id].rating)
                        avg_movie_ratings[movie_id][1] = int(avg_movie_ratings[movie_id][1]) + 1
                        avg_movie_ratings[movie_id][2] = avg_movie_ratings[movie_id][0] / float(avg_movie_ratings[movie_id][1])
                
                # print(min_tuple)
                # Find the location of the closest user identified earlier and remove it
                distances.pop(distances.index(min_tuple))
                
            # Create list of ratings, with each entry being the tuple (movie_id, [total_ratings, count, avg_rating])
            #   Sort the rating based on the average rating from highest to lowest
            sorted_avg_ratings = sorted(avg_movie_ratings.items(), key=lambda item: item[1][2], reverse=True)
            # for idx, value in enumerate(sorted_avg_ratings):
            #     print(idx, value)
            #recommendations is a list storing ids of movie recs
            recommendations = []
            # If repeat recs is true, we can recommend movies a user has already seen (necessary for validaiton)
            if repeat_recs: 
                recommendations = [rating[0] for rating in sorted_avg_ratings][:m]
            # If we don't recommend movies that a user has already seen (practical use)
            else :
                for rec in sorted_avg_ratings:
                    if rec[0] not in list(query_user.values())[0].reviews:
                        recommendations.append(rec)
                            
        return recommendations

###############################
# MovieLens Accuracy
###############################   
def movielens_accuracy(recommendations, query_user):
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

if __name__ == "__main__":
    all_users, max_movie_id, all_genres, all_genders, all_occupations = movielens_parse(input_file="movielens.txt")
    # print(all_genres)
    # print(all_genders)
    # print(all_occupations)
    # input()
    one_user, _, _, _, _ = movielens_parse(input_file="train_a.txt")
    #test_accuracy is a boolean. True: we can recommend movies a user has seen, False: we can't
    repeat_recs = True
    # recommendations = movielens_recommender(one_user, all_users, "euclidean", max_movie_id, repeat_recs)
    recommendations = movielens_recommender_improved(query_user=one_user, 
                                                     all_users=all_users, 
                                                     metric="euclidean", 
                                                     max_movies=max_movie_id, 
                                                     repeat_recs=repeat_recs,
                                                     genders=all_genders,
                                                     genres=all_genres,
                                                     occupations=all_occupations
                                                     )
    print(recommendations)
    precision, recall, f1 = movielens_accuracy(recommendations, one_user)
    print(precision, recall, f1)
