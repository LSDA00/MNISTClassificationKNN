# Lucas Allen, 5004607031, HW #2
# KNN classification problem working on MNIST handwritten digit dataset 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter

#import training and test data
training_data = pd.read_csv('http://mkang.faculty.unlv.edu/teaching/CS422_622/HW2/MNIST_training.csv')
test_data = pd.read_csv('http://mkang.faculty.unlv.edu/teaching/CS422_622/HW2/MNIST_test.csv')

#show the first 10 rows of training data
# print(training_data.head(10))

#collect labels into array for later and drop label column from training data
labels_for_training_data = training_data['label']
training_data = training_data.drop(['label'], axis = 1)

#show the first 10 rows of training data without labels
# print(training_data.head(10))

#1. ... Compute distances or similarity (Euclidean ... ) with the training data
#function to find the Euclidean distance between two points: sqrt[(x2-x1)^2 + (y2-y1)^2]
def euclidean(point_a, point_b): 
    sum_distance = 0
    for a,b in zip(point_a, point_b): 
        distance_squared = (a-b)**2
        sum_distance = distance_squared + sum_distance
    euclidean_distance = np.sqrt(sum_distance)
    return euclidean_distance

#2. Find the K-nearest neighbors and decide the majority class  
#find the labels of the nearest neighbors, using k = 9 because gave the best accuracy for this dataset after testing 
def nn_labels(distances_list, labels_list, k = 9): 
    distances_list = distances_list.reshape(-1,1)
    labels_list = labels_list.reshape(-1,1)
    distances_and_labels = np.concatenate((distances_list, labels_list), axis =1)
    labels_dataframe = pd.DataFrame(distances_and_labels, columns = ['distance', 'label'])
    labels_dataframe = labels_dataframe.sort_values('distance')
    return labels_dataframe['label'].head(k).values
    
#find which label of the nearest neighbors is most common
def most_common_label(labels_arr):
    mc_label = Counter(labels_arr).most_common(1)
    return mc_label[0][0]

#knn function to find k nearest neighbors of the unknwon point and predict the unknown points value based off of the majority neighbors label
def knn(unknown_point, known_points, known_point_labels, k): 
    known_points_count = known_points.shape[0]
    found_distance = []

    for i in range(known_points_count):
        distance = euclidean(unknown_point, known_points.iloc[i])
        found_distance.append(distance)
    knn_labels = nn_labels(np.array(found_distance), np.array(known_point_labels), k)
    prediction = most_common_label(knn_labels)
    return prediction

#show the first 10 row of test data
#print(test_data.head(10))

#drop the label from test data
test_data_labels = test_data['label']
test_data = test_data.drop(['label'], axis=1)

#show the first 10 rows of test data without labels
#print(test_data.head(10))

#3, 4, 5, 6 Compare the prediction with the fround truth in test data and show the accuracy of your KNN
#function to check the accuracy of our prediction by comparing the ground truth values in test_data to predicted points from knn function
#accuracy = correct predictions / total # of elements in test data (50)
def check_prediction(k):
    correct_classifications = 0
    incorrect_classifications = 0 
    accuracy = 0.000
    for i in range (50): 
        unknown_point = test_data.iloc[i]

        correct_answers = test_data_labels.values[i]
        predicted = knn(unknown_point, training_data.head(949), labels_for_training_data.head(949), k)
        print("Ground truth value: ", float(correct_answers), "Predicted Value: ", knn(unknown_point, training_data.head(949), labels_for_training_data.head(949), k))
        if correct_answers == predicted:
            correct_classifications +=1
        else: 
            incorrect_classifications +=1
    accuracy = correct_classifications / 50
    print("Correct classifications: ", correct_classifications, "Incorrect classifications: ", incorrect_classifications)
    return accuracy
#show accuracy 
print("Accuracy: ", check_prediction(9))



    

