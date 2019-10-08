import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nn_utils import *
from neural_network import *
import pickle 
from datetime import datetime
import argparse

# Set seed
np.random.seed(0)

# Create ArgumentParser
ap = argparse.ArgumentParser()
ap.add_argument("-train", "--train_images", required=True, help="Path to train images")
ap.add_argument("-test", "--test_images", required=True, help="Path to test images")
ap.add_argument("-nx", "--nx", type=int, default=784, help="Input dimension")
ap.add_argument("-nh1", "--nh1", type=int, default=64, help="Number of units in 1st hidden layer")
ap.add_argument("-nh2", "--nh2", type=int, default=0, help="Number of units in 2nd hidden layer")
ap.add_argument("-epoches", "--no_epoches", type=int, default=50, help="Number of epoches to train")
ap.add_argument("-batch", "--batch_size", type=int, default=64, help="Batch size")
ap.add_argument("-lr", "--learning_rate", type=float, default=0.5, help="Learning rate")
ap.add_argument("-decay", "--weight_decay", type=float, default=5e-4, help="L2 regularization")
ap.add_argument("-loss", "--loss_function", type=str, default='softmax_cross_entropy', help="softmax_cross_entropy (multi-class) or binary_cross_entropy (2 classes)")

# Parse args
args = ap.parse_args()

def main(args):
    train_path = args.train_images
    test_path = args.test_images
    # Import data
    (X_train, Y_train), X_test = load_data(train_path, test_path)

    # Explore dataset
    # Shape of
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("Y_train:", Y_train.shape)

    # Print first 5 labels in Y_train
    print("First 5 labels of trainig set:", Y_train.head())

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

    # Tranpose train, val and test set
    X_train, Y_train = X_train.T, Y_train.T # (784, 37800) (10, 37800)
    X_val, Y_val = X_val.T, Y_val.T # (784, 4200) (10, 4200)
    X_test = X_test.T # (784, 28000)


    print("X_train, Y_train:", X_train.shape, Y_train.shape)
    print("X_val, Y_val:", X_val.shape, Y_val.shape)
    print("X_test:", X_test.shape)

    # Layer dimension include input hidden and output layer
    if args.nh2 == 0:
        layer_dims = [args.nx, args.nh1, 10]
    else:
        layer_dims = [args.nx, args.nh1, args.nh2, 10]

    # Init Neural_network hyper-parameter
    nn = Neural_network(layer_dims=layer_dims)
    nn.no_epoches = args.no_epoches
    nn.learning_rate = args.learning_rate
    nn.loss_function_name = args.loss_function
    nn.weight_decay = args.weight_decay
    nn.batch_size = args.batch_size

    # Set X, Y
    nn.X = X_train
    nn.Y = Y_train

    params = nn.model(plot_learning_curve=True)

    # Store model
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dtime = now.strftime("%d%m%Y-%H%M%S")
    print("date and time =", dtime)
    with open('models/'+dtime, 'wb') as f:
        pickle.dump(params, f)

    acc_train = nn.evaluate()

    # Set X, Y for validating
    nn.X = X_val
    nn.Y = Y_val
    acc_val = nn.evaluate()

    print("Accuracy on training set:", acc_train)
    print("Accuracy on val set:", acc_val)

    # Predict test set
    nn.X = X_test
    Y_hat = nn.predict().reshape(-1, 1)

    ImageId = np.arange(1, Y_hat.shape[0]+1).reshape(-1, 1)
    Y_hat = np.hstack((ImageId, Y_hat))

    df = pd.DataFrame(Y_hat, columns=['ImageId', 'Label'])
    # Save to csv
    df.to_csv('sample_submission.csv', index=False) # index=False Avoid pd create extra index col

    plt.show()

if __name__ == '__main__':
    main(args)