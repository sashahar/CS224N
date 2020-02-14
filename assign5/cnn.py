#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, input_shape, hidden_size, f_channels, dropout_rate=0.2): #possible params embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init CNN.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Size of hidden units (dimensionality)
        """
         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size = 5, padding = 1)
         self.maxPool = nn.MaxPool1d

    def forward(self, x_emb) -> torch.Tensor:
        """ Forward Pass for Highway Layer. Maps xconv_out to x_highway

        @param x_emb (torch.Tensor): input to network (dim embed_size)
        """
        x_conv = self.conv(x_emb)
        x_conv_out = self.maxPool(F.relu(x_conv))




    ### END YOUR CODE
