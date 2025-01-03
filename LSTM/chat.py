import numpy as np

class LSTM:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights for input gate
        self.W_i = np.random.randn(hidden_dim, input_dim)
        self.U_i = np.random.randn(hidden_dim, hidden_dim)
        self.b_i = np.zeros((hidden_dim, 1))

        # Initialize weights for forget gate
        self.W_f = np.random.randn(hidden_dim, input_dim)
        self.U_f = np.random.randn(hidden_dim, hidden_dim)
        self.b_f = np.zeros((hidden_dim, 1))

        # Initialize weights for output gate
        self.W_o = np.random.randn(hidden_dim, input_dim)
        self.U_o = np.random.randn(hidden_dim, hidden_dim)
        self.b_o = np.zeros((hidden_dim, 1))

        # Initialize weights for cell candidate
        self.W_c = np.random.randn(hidden_dim, input_dim)
        self.U_c = np.random.randn(hidden_dim, hidden_dim)
        self.b_c = np.zeros((hidden_dim, 1))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        return LSTM.sigmoid(x) * (1 - LSTM.sigmoid(x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def dtanh(x):
        return 1 - np.tanh(x) ** 2

    def forward(self, x, h_prev, c_prev):
        # Input gate
        i = self.sigmoid(np.dot(self.W_i, x) + np.dot(self.U_i, h_prev) + self.b_i)
        
        # Forget gate
        f = self.sigmoid(np.dot(self.W_f, x) + np.dot(self.U_f, h_prev) + self.b_f)
        
        # Output gate
        o = self.sigmoid(np.dot(self.W_o, x) + np.dot(self.U_o, h_prev) + self.b_o)
        
        # Cell candidate
        c_hat = self.tanh(np.dot(self.W_c, x) + np.dot(self.U_c, h_prev) + self.b_c)

        # Cell state
        c = f * c_prev + i * c_hat

        # Hidden state
        h = o * self.tanh(c)
        print(c)
        

        return h, c, (i, f, o, c_hat, c)

    def backward(self, x, h_prev, c_prev, dh_next, dc_next, cache):
        i, f, o, c_hat, c = cache

        # Gradients of output gate
        do = dh_next * self.tanh(c)
        dWo = np.dot(do * self.dsigmoid(o), x.T)
        dUo = np.dot(do * self.dsigmoid(o), h_prev.T)
        dbo = np.sum(do * self.dsigmoid(o), axis=1, keepdims=True)

        # Gradients of cell state
        dc = dc_next + dh_next * o * self.dtanh(c)
        dc_hat = dc * i
        di = dc * c_hat
        df = dc * c_prev

        # Gradients of input gate
        dWi = np.dot(di * self.dsigmoid(i), x.T)
        dUi = np.dot(di * self.dsigmoid(i), h_prev.T)
        dbi = np.sum(di * self.dsigmoid(i), axis=1, keepdims=True)

        # Gradients of forget gate
        dWf = np.dot(df * self.dsigmoid(f), x.T)
        dUf = np.dot(df * self.dsigmoid(f), h_prev.T)
        dbf = np.sum(df * self.dsigmoid(f), axis=1, keepdims=True)

        # Gradients of cell candidate
        dWc = np.dot(dc_hat * self.dtanh(c_hat), x.T)
        dUc = np.dot(dc_hat * self.dtanh(c_hat), h_prev.T)
        dbc = np.sum(dc_hat * self.dtanh(c_hat), axis=1, keepdims=True)

        # Accumulate parameter gradients
        dW = {
            'W_i': dWi, 'U_i': dUi, 'b_i': dbi,
            'W_f': dWf, 'U_f': dUf, 'b_f': dbf,
            'W_o': dWo, 'U_o': dUo, 'b_o': dbo,
            'W_c': dWc, 'U_c': dUc, 'b_c': dbc
        }

        # Gradient of input and previous hidden state
        dx = (np.dot(self.W_i.T, di) + np.dot(self.W_f.T, df) +
              np.dot(self.W_o.T, do) + np.dot(self.W_c.T, dc_hat))
        dh_prev = (np.dot(self.U_i.T, di) + np.dot(self.U_f.T, df) +
                   np.dot(self.U_o.T, do) + np.dot(self.U_c.T, dc_hat))
        dc_prev = f * dc

        return dx, dh_prev, dc_prev, dW

    def update_parameters(self, gradients, lr):
        for param in gradients:
            setattr(self, param, getattr(self, param) - lr * gradients[param])

# Example Usage
np.random.seed(0)
lstm = LSTM(input_dim=3, hidden_dim=1)
x= np.array([[1],[2],[3]])
#x = np.random.randn(3, 1)  # Input vector
print(x)
timesteps = 10
h, c = np.zeros((1, 1)), np.zeros((1, 1))
for t in range(timesteps):
    h, c, cache = lstm.forward(x, h, c)