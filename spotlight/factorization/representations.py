"""
Classes defining user and item latent representations in
factorization models.
"""

import torch.nn as nn

import torch.nn.functional as F

from spotlight.layers import ScaledEmbedding, ZeroEmbedding


class BilinearNet(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.

    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 user_embedding_layer=None, item_embedding_layer=None, sparse=False):

        super(BilinearNet, self).__init__()

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(num_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   sparse=sparse)

        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        return dot + user_bias + item_bias


class MixtureNet(nn.Module):
    """
    A representation that models users as mixtures-of-tastes.

    This is accomplished via maintaining multiple taste
    embeddings for each user as well as attention vectors
    that match each taste to the items it describes.

    For a full description of the model, see [5]_.

    Parameters
    ----------

    num_users: int
        Number of users to be represented.
    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer, and the number of hidden
        units in the LSTM layer.
    num_mixtures: int, optional
        Number of mixture components (distinct user tastes) that
        the network should model.

    References
    ----------

    .. [5] Kula, Maciej. "Mixture-of-tastes Models for Representing
       Users with Diverse Interests" https://arxiv.org/abs/1711.08379 (2017)
    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 num_mixtures=4, sparse=False):

        super(MixtureNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_mixtures = num_mixtures

        self.taste_embeddings = ScaledEmbedding(num_users,
                                                embedding_dim * num_mixtures,
                                                sparse=sparse)
        self.attention_embeddings = ScaledEmbedding(num_users,
                                                    embedding_dim * num_mixtures,
                                                    sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items,
                                               embedding_dim,
                                               sparse=sparse)

        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        item_embedding = self.item_embeddings(item_ids)

        batch_size, embedding_size = item_embedding.size()

        user_tastes = (self.taste_embeddings(user_ids)
                       .resize(batch_size,
                               self.num_mixtures,
                               embedding_size))
        user_attention = (self.attention_embeddings(user_ids)
                          .resize(batch_size,
                                  self.num_mixtures,
                                  embedding_size))

        attention = (F.softmax((user_attention *
                                item_embedding.unsqueeze(1).expand_as(user_attention))
                               .sum(2)).unsqueeze(2).expand_as(user_attention))
        weighted_preference = (user_tastes * attention).sum(1)

        dot = (weighted_preference * item_embedding).sum(1)

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return dot + user_bias + item_bias
