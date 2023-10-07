import numpy as np
import math
from matplotlib import pyplot as plt


class mlp(object):
    def __init__(self, shape, activate_function, loss, reg_lam=0):
        input_size, hidden_size, output_size = shape

        self.layer = len(shape)
        self.reg_lambda = reg_lam
        self.hidden_size = hidden_size

        self.act = [self.activation(f) for f in activate_function]
        self.derivate = [self.act_deri(f) for f in activate_function]
        self.error = self.loss_function(loss)

        self.weights = [np.random.randn(input_size, hidden_size) * 0.01,
                        np.random.randn(hidden_size, output_size) * 0.01]
        self.delta_weights = [np.zeros((input_size, hidden_size)), np.zeros((hidden_size, output_size))]  # single layer
        self.bias = [np.zeros((1, hidden_size)), np.zeros((1, output_size))]
        self.delta_bias = [np.zeros((1, hidden_size)), np.zeros((1, output_size))]  # single layer

    def __call__(self, data, label, lr):
        net_out = self.forward(data)
        loss, _ = self.error(net_out[-1], label)

        accuracy = np.mean(np.argmax(net_out[-1], axis=1) == np.argmax(label, axis=1))

        self.backpropagation(net_out, label)
        self.step(lr)

        return loss, accuracy

    def prediction(self, test_data):
        net_out = self.forward(test_data)
        return np.argmax(net_out[-1], axis=1)

    def forward(self, x):
        # net = [x.copy(), None, None]
        net_out = [x.copy(), None, None]
        for i in range(self.layer - 1):
            net = np.dot(net_out[i], self.weights[i]) + self.bias[i]
            net_out[i + 1] = self.act[i](net)
        return net_out

    def backpropagation(self, net_out, label):
        delta = [None, None]
        y = net_out[-1]
        delta[-1] = self.derivate[-1](y, label)

        for i in np.arange(self.layer - 2, -1, -1):
            self.delta_weights[i] = np.dot(net_out[i].T, delta[i])
            self.delta_bias[i] = np.sum(delta[i], axis=0, keepdims=True)
            if i - 1 > -1:
                delta[i - 1] = np.dot(delta[i], self.weights[i].T) * self.derivate[i - 1](net_out[i])

    def step(self, lr):
        for i in range(self.layer - 1):
            self.weights[i] -= lr * self.delta_weights[i]
            self.bias[i] -= lr * self.delta_bias[i]

    def activation(self, name):
        z = None
        if name == "Sigmoid":  # Sigmoid
            def sigmoid(x):
                z = 1.0 / (1.0 + math.exp(-x))
                return z

            return sigmoid
        elif name == "ReLU":  # ReLU
            def relu(x):
                if np.isscalar(x):
                    z = np.max(x, 0)
                else:
                    zero_aux = np.zeros(x.shape)
                    meta_z = np.stack((x, zero_aux), axis=-1)
                    z = np.max(meta_z, axis=-1)
                return z

            return relu
        elif name == "Softmax":
            def softmax(x):
                exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_x / np.sum(exp_x, axis=1, keepdims=True)
            return softmax
        else:
            print("Activation function name invalid.")
            exit(1)

    def act_deri(self, func_name):
        if func_name == 'Sigmoid':
            def sigmoid_derivative(x):
                g = math.exp(-x) / (1.0 + math.exp(-x)) ** 2
                return g

            return sigmoid_derivative
        elif func_name == 'ReLU':
            def relu_derivative(x):
                g = 1 * (x > 0)
                return g

            return relu_derivative
        elif func_name == 'Softmax':  # which is actually softmax cross entropy loss derivative
            def softmax_cross_entropy(x, label):
                g = x - label
                return g
            return softmax_cross_entropy
        else:
            print("Derivate function not found.")
            exit(1)

    def loss_function(self, name):
        # BCEwithLogits
        # L = - d * math.log(1 / (1 + math.exp(-y))) - (1-d) * math.log(1 - 1 / (1 + math.exp(-y)))
        if name == "Cross_Entropy":
            epsilon = 1e-12

            def cross_entropy(y, d):
                y = np.clip(y, epsilon, 1. - epsilon)
                N = y.shape[0]
                L = -np.sum(d * np.log(y + 1e-9)) / N
                return L, 1

            return cross_entropy
        elif name == "MSE":
            def mse(y, d):
                L = 0.5 * (d - y) ** 2
                der = y - d
                return L, der

            return mse
        else:
            print("Loss name invalid")
            exit(1)


def convert_to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # Data Processing
    data = pd.read_csv('https://archive.ics.uci.edu/static/public/109/data.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y_onehot = convert_to_one_hot(y, len(np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # Initialization
    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = y_onehot.shape[1]
    shape = [input_dim, hidden_dim, output_dim]
    actvation_function = ['ReLU', 'Softmax']
    loss_function = 'Cross_Entropy'
    lr = 0.001

    model = mlp(shape, actvation_function, loss_function)

    num_epochs = 100
    loss_hist = []
    accuracy_hist = []
    for epoch in range(num_epochs):
        loss, acc = model(X_train, y_train, lr)
        loss_hist.append(loss)
        accuracy_hist.append(acc)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f} | Accuracy: {acc:.4f}')

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.plot(loss_hist, label="training loss", color=color)
    ax1.tick_params(axis='y')
    ax1.legend(loc="center right")

    ax2 = ax1.twinx()  # 创建一个共享同一x轴的第二个y轴
    color = 'tab:orange'
    ax2.set_ylabel('training set accuracies')  # 我们已经处理了x轴标签
    ax2.plot(accuracy_hist, label="training set accuracy", color=color)
    ax2.tick_params(axis='y')
    ax2.legend(loc="center left")

    fig.tight_layout()  # 否则右侧的y轴标签可能会被稍微裁剪
    plt.title("Cross Entropy Loss and Training Set Accuracy")
    plt.savefig("Wine_Classification_Result.png")
