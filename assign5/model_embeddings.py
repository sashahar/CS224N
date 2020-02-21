#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, word_embed_size, vocab, dropout_rate = 0.3):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>'] # notice that in assignment 4 vocab is of type (Vocab), not (VocabEntry) as assignment 5.
        # self.embeddings = nn.Embedding(len(vocab.src), word_embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1h

        self.vocab = vocab
        self.e_char = 50
        self.dropout_rate = dropout_rate
        self.word_embed_size = word_embed_size
        self.embedding = nn.Embedding(len(self.vocab.char2id), self.e_char, padding_idx=self.vocab.char_pad)
        self.cnn = CNN(e_char = self.e_char, e_word = self.word_embed_size)
        self.highway = Highway(embed_size = word_embed_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        ### END YOUR CODE

    def forward(self, x_padded):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        sentence_length, batch_size, max_word_length = x_padded.shape
        input_tensor = self.embedding(x_padded)
        input_tensor = input_tensor.permute(0, 1, 3, 2).contiguous().reshape((sentence_length*batch_size, self.e_char,-1))
        x_conv_out = self.cnn(input_tensor)
        x_highway = self.highway(x_conv_out)
        x_word_embed = self.dropout(x_highway)
        output = x_word_embed.reshape((sentence_length, batch_size, -1))
        return output

        ### END YOUR CODE
