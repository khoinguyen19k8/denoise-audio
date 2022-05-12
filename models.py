from turtle import forward
import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    Please reference arguments needed for RNN here: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    Paramaters:
    --------------------
    input_size: Number of features in input
    hidden_size: Number of hidden units
    output_size: Number of features in output

    This is a basic implementation of an RNN network. The main problem is that this model will output a sequence with the same
    length as input. If it receives a seq of length 5500 it will output a seq of length 5500. We need to somehow output a seq of length
    11000 in our final model
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, batch_first = True, **kwargs)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()

    
    def forward(self, x):
        pred, hidden = self.rnn(x)
        pred = self.activation(self.linear(x))
        return pred

class LSTM(nn.Module):
    """
    Please reference arguments needed for RNN here: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    Paramaters:
    --------------------
    input_size: Number of features in input
    hidden_size: Number of hidden units
    output_size: Number of features in output

    This is a basic implementation of a LSTM network. The main problem is that this model will output a sequence with the same
    length as input. If it receives a seq of length 5500 it will output a seq of length 5500. We need to somehow output a seq of length
    11000 in our final model
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, batch_first = True, **kwargs)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()

    
    def forward(self, x):
        pred, hidden = self.lstm(x)
        pred = self.activation(self.linear(x))
        return pred