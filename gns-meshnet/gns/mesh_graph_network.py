from typing import List
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:
  """Build a MultiLayer Perceptron.

  Args:
    input_size: Size of input layer.
    layer_sizes: An array of input size for each hidden layer.
    output_size: Size of the output layer.
    output_activation: Activation function for the output layer.
    activation: Activation function for the hidden layers.

  Returns:
    mlp: An MLP sequential container.
  """
  # Size of each layer
  layer_sizes = [input_size] + hidden_layer_sizes
  if output_size:
    layer_sizes.append(output_size)

  # Number of layers
  nlayers = len(layer_sizes) - 1

  # Create a list of activation functions and
  # set the last element to output activation function
  act = [activation for i in range(nlayers)]
  act[-1] = output_activation

  # Create a torch sequential container
  mlp = nn.Sequential()
  for i in range(nlayers):
    mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
                                             layer_sizes[i + 1]))
    mlp.add_module("Act-" + str(i), act[i]())

  return mlp


class Encoder(nn.Module):

    def __init__(
            self,
            nnode_in_features: int,
            nnode_out_features: int,
            nedge_in_features: int,
            nedge_out_features: int,
            nmlp_layers: int,
            mlp_hidden_dim: int):

        super(Encoder, self).__init__()
        # Encode node features as an MLP
        self.node_fn = nn.Sequential(*[build_mlp(nnode_in_features,
                                                 [mlp_hidden_dim
                                                  for _ in range(nmlp_layers)],
                                                 nnode_out_features),
                                       nn.LayerNorm(nnode_out_features)])
        # Encode edge features as an MLP
        self.edge_fn = nn.Sequential(*[build_mlp(nedge_in_features,
                                                 [mlp_hidden_dim
                                                  for _ in range(nmlp_layers)],
                                                 nedge_out_features),
                                       nn.LayerNorm(nedge_out_features)])

    def forward(
            self,
            x: torch.tensor,
            edge_features: torch.tensor):

        return self.node_fn(x), self.edge_fn(edge_features)


class InteractionNetwork(MessagePassing):

    def __init__(
            self,
            nnode_in: int, nnode_out: int,  # node-related
            nedge_in: int,  nedge_out: int,  # edge-related
            nmlp_layers: int, mlp_hidden_dim: int):  # mlp-related

        # Aggregate features from neighbors
        super(InteractionNetwork, self).__init__(aggr='add')
        # Node MLP
        self.node_fn = nn.Sequential(*[build_mlp(nnode_in + nedge_out,
                                                 [mlp_hidden_dim
                                                  for _ in range(nmlp_layers)],
                                                 nnode_out),
                                       nn.LayerNorm(nnode_out)])
        # Edge MLP
        self.edge_fn = nn.Sequential(*[build_mlp(nnode_in + nnode_in + nedge_in,
                                                 [mlp_hidden_dim
                                                  for _ in range(nmlp_layers)],
                                                 nedge_out),
                                       nn.LayerNorm(nedge_out)])