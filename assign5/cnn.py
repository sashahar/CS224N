#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, e_char, e_word): #possible params embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init CNN.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Size of hidden units (dimensionality)
        """
        super(CNN, self).__init__()
        self.e_char = e_char
        self.e_word = e_word
        self.conv = nn.Conv1d(self.e_char, self.e_word, kernel_size = 5, padding = 1)

    def forward(self, x_emb) -> torch.Tensor:
        """ Forward Pass for Highway Layer. Maps xconv_out to x_highway

        @param x_emb (torch.Tensor): input to network (batch_size, e_char, m_word)
        """
        x_conv = self.conv(x_emb)
        # print(self.conv.state_dict()['bias'])
        # print(x_conv)
        self.maxPool = nn.MaxPool1d(kernel_size = x_conv.shape[2])
        x_conv_out = self.maxPool(F.relu(x_conv)).squeeze()
        return x_conv_out




    ### END YOUR CODE
