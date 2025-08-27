import numpy as np
import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train a simple neural network.")
    parser.add_argument("--train", type=str, required=True, help="Training CSV file path")
    parser.add_argument("--test", type=str, required=True, help="Testing CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path for predictions")
    args = parser.parse_args()

    return args.train,args.test,args.output

train,test,out = get_args()

def save_output(file, predictions):
    df = pd.DataFrame(predictions.T)
    df.to_csv(file, index=False)
    print(f"Predictions saved to {file}")

def init_nn():
    n_x = 2 #input
    n_h = 4 #hidden --set back to 4 neur later/ best output at 32 neurons 1 layer, slow on my pc
    n_y = 1  #output num

    np.random.seed(73)#consistent output
    W1 = np.random.randn(n_h, n_x) * 0.1
    b1 = np.zeros((n_h, 1))                #0.01 was too low
    W2 = np.random.randn(n_y, n_h) * 0.1
    b2 = np.zeros((n_y, 1))
    return W1 , b1, W2 , b2

def relu(Z):
    return np.maximum(Z,0)

def sigmoid(x):
    return 1 / (1+np.exp((-x)))#tanh function could be used as well

def sigmoid_gradient(x):
    s = sigmoid(x)
    return s * (1 - s)

def binary_loss_func(y,y_hat):#ytrue and ylabel == prediction, label \
    #------doesnt account for log (0)  --> same thing as dZ2 = A2 - Y->
    #  which is the simplified deriv of loss function expressed in backprop
    return - np.mean(y*
                     np.log(y_hat)
                     +(1-y)
                     *np.log(1-y_hat)
                    )

def der_relu(Z):
    return (Z> 0).astype(int)
#the derivative of the  relu function is either 1 or 0, as it converges to 0 at Z<0,could use leaky relu

def forward_prop(X, W1, b1, W2, b2):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)

    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    cached = (Z1,A1,Z2,A2)

    return A2 ,cached

def backwards_propogation(Z1,A1,Z2,A2,Y,W1,W2,X ):
    m = Y.shape[1]

    dZ2 = A2 - Y
    dW2 =(1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2,axis= 1, keepdims=True)
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * der_relu(Z1)
    dW1 = (1 / m) * dZ1@X.T
    db1 = (1 / m) * np.sum(dZ1,axis= 1, keepdims=True)

    return dW1,db1,dW2,db2

def get_predictions(A2):
    return (A2 > 0.5).astype(int)

def get_accuracy(preds, Y):
    return np.mean(preds == Y)

def update_params(W1,W2,b1,b2,dW1,dW2,db1,db2,alpha=0.1):
    # take in init params plus their derivative. ;a = learning rate
    #n == new params
    nW1 = W1 - alpha * dW1
    nb1 = b1 - alpha * db1
    nW2 = W2 - alpha * dW2
    nb2 = b2 - alpha * db2

    return (nW1,nb1,nW2,nb2 )

def load_file(file):
    f = pd.read_csv(file)
    data, labels = f.iloc[:,:2], f.iloc[:,2:]
    return  data , labels

def load_test(file):
    return  pd.read_csv(file)

def load_file_split(file=train):
    f = pd.read_csv(file)
    size = len(f)
    size_train = int(size * 0.75)
    size_test = size - size_train

    f_train = f.iloc[:size_train, :]
    f_test = f.iloc[size_train:, :]

    train_data = f_train.iloc[:, :2]
    train_label = f_train.iloc[:, 2:]
    test_data = f_test.iloc[:, :2]
    test_label = f_test.iloc[:, 2:]

    return train_data, train_label, test_data, test_label

def gradient_descent(X,Y,epochs ,alpha):
    W1 ,b1 , W2, b2 = init_nn()
    for epoch in range(1,epochs+1):

        A2, cached = forward_prop(X, W1, b1, W2, b2)
        Z1, A1, Z2, A2_cached = cached

        dW1, db1, dW2, db2 = backwards_propogation(Z1, A1, Z2, A2_cached, Y, W1, W2, X)
        W1, b1, W2, b2 = update_params(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha)
#this should achieve global minima, given data and requirements

        if epoch %50 == 0 :
            print(f"iteration: {epoch}" )
            print(f"Arrucary:  {get_accuracy(get_predictions(A2),Y)}")
            print(f"\nDifferent formula for loss {binary_loss_func(Y,A2)}\n")

    return W1, b1, W2, b2



def test_neural_network():
    test_X = load_test(test)
    A2,_ = forward_prop(test_X.T.values,W1, b1, W2, b2) #we dont need cache here,just unpack accordingly
    accuracy = get_predictions(A2)
    save_output(out, get_predictions(A2))
    for i in range(A2.shape[1]):
        print(f"Sample {i+1}: Probability={A2[0,i]:.4f}, Predicted Class={accuracy[0,i]}")
    print(f"Avg confidence ------>{np.mean(A2)}")
    print(f"Min prob: {A2.min():.4f}, Max prob: {A2.max():.4f}")
    #implement histogram to track distribution of values from 0.0 to 1.0 in step 0.1

def test_neural_network_2():
    A2, _ = forward_prop(X_test.T.values, W1, b1, W2, b2)
    preds = get_predictions(A2)
    print(f"Test Accuracy: {get_accuracy(preds, Y_test.values.T):.4f}")
    print(f"Avg confidence ------> {np.mean(A2):.4f}")
    print(f"Min prob: {A2.min():.4f}, Max prob: {A2.max():.4f}")
    save_output(out, preds)
    for i in range( A2.shape[1]):  #
        print(f"Sample {i+1}: Probability={A2[0,i]:.4f}, Predicted={preds[0,i]}, Actual={Y_test.values[i,0]}")
X_train, Y_train, X_test, Y_test = load_file_split()
X_train = (X_train - X_train.mean()) / X_train.std()
X_test  = (X_test - X_test.mean()) / X_test.std()

W1, b1, W2, b2 = gradient_descent(X_train.T.values, Y_train.values.T, 2000, 0.1)

def main():
    test_neural_network()
    test_neural_network_2()

if __name__ == "__main__":
    main()
