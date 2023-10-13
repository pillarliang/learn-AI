import nltk
import numpy as np
import math
from nltk.corpus import twitter_samples
from os import getcwd
from utils import build_freqs, process_tweet

nltk.download('twitter_samples')
nltk.download('stopwords')

filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

# prepare the data
# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
train_pos = all_positive_tweets[:4000]
test_pos = all_positive_tweets[4000:]
train_neg = all_negative_tweets[:4000]
test_neg = all_negative_tweets[4000:]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# Create the numpy array of positive labels and negative labels.
train_y = np.append(np.ones((len(train_pos), 1)),
                    np.zeros((len(train_neg), 1)),
                    axis=0)
test_y = np.append(np.ones((len(test_pos), 1)),
                   np.zeros((len(test_neg), 1)),
                   axis=0)

# create frequency dictionary
freqs = build_freqs(train_x, train_y)


# 1. design a model
def sigmoid(z):
    """
    Args:
        z (_type_): _description_
    """
    return 1 / (1 + math.exp(-z))


# 2. define loss from trainning data -- cross entropy error
def loss_func(samples_num, prediction, truth_value):
    """
    Args:
        samples_num (int): the number of trainning samples
        prediction (_type_): predict value
        truth_value (_type_): truth label
    """
    return -1 / samples_num * (
        np.dot(truth_value.T, np.log(prediction)) + np.dot(
            (1 - truth_value).T, np.log(1 - prediction)))


# 3. Optimization -- gradient descent
def gradientDescent(x, y, theta, alpha, num_iters):
    """
    Args:
        x (m, n+1): samples. +1 -> add bias; n=2-> positive or negative
        y (m, 1): label/ground truth
        theta (n+1, 1): parameter/weight vector
        alpha (float): learn rate
        num_iters (int):
    """
    # the number of samples
    for i in range(0, num_iters):
        sample_num = x.shape[0]

        z = np.dot(x, theta)
        h = sigmoid(z)  # prediction

        # calculate the cost function
        loss = loss_func(sample_num, h, y)

        # update the theta
        theta = theta - (alpha / sample_num) * np.dot(x.T, (h - y))

    return loss, theta


# 2. extract the features
def extract_features(tweet, freqs, process_tweet=process_tweet):
    """
    Args:
        tweet (string): a string containing one tweet
        freqs (_type_): a dictionary corresponding to the frequencies of each tuple (word, label)
        process_tweet (_type_, optional): _description_. Defaults to process_tweet.
    Output:
        x: a feature vector of dimension (1,3)
    """
    word_list = process_tweet(tweet)

    # 3 elements for [bias, positive, negative] counts
    x = np.zeros(3)

    # bias term is set to 1
    x[0] = 1

    for word in word_list:
        x[1] += freqs.get((word, 1), 0)
        x[2] += freqs.get((word, 0), 0)

    x = x[None, :]  # adding batch dimension for further processing
    return x


# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
loss, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)


# test your logistic regression
def predict_tweet(tweet, freqs, theta):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))

    return y_pred


def test_logistic_regression(test_x,
                             test_y,
                             freqs,
                             theta,
                             predict_tweet=predict_tweet):
    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)

    accuracy = (np.array(y_hat) == np.squeeze(test_y)).sum() / len(test_y)

    return accuracy


tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

# Predict with your own Tweet
# Feel free to change the tweet below
my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'
print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else:
    print('Negative sentiment')
