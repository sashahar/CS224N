#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, embed_size, dropout_rate=0.2):
        """ Init Highway Model.

        @param embed_size (int): Embedding size (dimensionality)
        """
        super(Highway, self).__init__()
        self.dropout_rate = dropout_rate
        self.W_proj = nn.Linear(embed_size, embed_size, bias=True)
        self.W_gate = nn.Linear(embed_size, embed_size, bias=True)
        self.dropout =nn.Dropout(p=self.dropout_rate)


    def forward(self, x_conv) -> torch.Tensor: #possible params: source: List[List[str]], target: List[List[str]]
        """ Forward Pass for Highway Layer. Maps xconv_out to x_highway

        @param x_conv (torch.Tensor): input to network (dim embed_size)
        """
        x_proj = F.relu(self.W_proj(x_conv))
        x_gate = F.log_softmax(self.W_gate(x_conv), dim=-1)
        x_highway = x_gate*x_proj + (1-x_gate)*x_proj
        x_word_emb = self.dropout(x_highway)
        return x_word_emb






    ### END YOUR CODE
