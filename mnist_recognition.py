import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed
np.random.seed(0)

def convert_to_one_hot(Y, no_class):
    """
    Convert Y from shape (m, 1) into (m, no_class) with each column correspond to the actual digit of example
    Args:
        Y(m, 1): label
        no_class: number of catergorical class
    Returns:
        Y_one_hot: (m, no_class)
    """
    return np.eye(no_class)[Y.reshape(-1)]


# Import data
train_set = pd.read_csv("/Users/haophung/Google Drive/Senior/Soft computing/digit-recognizer/train.csv")
X_test = pd.read_csv("/Users/haophung/Google Drive/Senior/Soft computing/digit-recognizer/test.csv")

# Separate x and y in training set
Y_train = train_set["label"]
X_train = train_set.drop(labels=["label"], axis=1)

# Explore dataset
# Shape of
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)

# Print first 5 labels in Y_train
print(Y_train.head())

# Count member in each class
print(Y_train.value_counts())

# Plot countplot
sns.countplot(Y_train)

# Normalize dataset
X_train = X_train.to_numpy()/255.0
X_test = X_test.to_numpy()/255.0

# Convert to one-hot
Y_train = convert_to_one_hot(Y_train.to_numpy(), 10) # (42000, 10)
# print(Y_train[:10,:])

# Split 42000 examples in  training set into 
# (90%: 37800) training examples and (10%: 4200) validate exemples
rand_index = np.random.choice(range(X_train.shape[0]), 4200, replace = False)
X_val, Y_val = X_train[rand_index,:], Y_train[rand_index,:] # (4200, 784) (4200, 10)

# Update training set
remaning_index = [i for i in range(X_train.shape[0]) if i not in rand_index]
X_train, Y_train = X_train[remaning_index], Y_train[remaning_index] # (37800, 784) (37800, 10)

print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)

# Tranpose train, val and test set
X_train, Y_train = X_train.T, Y_train.T # (784, 37800) (10, 37800)
X_val, Y_val = X_val.T, Y_val.T # (784, 4200) (10, 4200)
X_test = X_test.T # (784, 28000)


print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)
print(X_test.shape)





plt.show()



