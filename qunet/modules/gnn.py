from ..config   import Config
from typing import Any, Callable, Optional, Sequence, Union
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj
from torch import Tensor, LongTensor
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
import torch, torch.nn as nn

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}

class DynEdgeConv(EdgeConv):
    """Dynamical edge convolution layer."""

    def __init__(
        self,
        *arg, **kwargs,
    ):
        """Construct `DynEdgeConv`.
        """
        self.cfg = Config(
            mlp=[3,256],
            features_subset=slice(0, 3),
            nb_neighbors=8,
            aggr = "add",
            update_adjacency = True                              
        )   
        self.cfg = self.cfg(*arg, **kwargs)   # add, change properties 

        # Base class constructor
        super().__init__(nn=self.cfg.nn, aggr=self.cfg.aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = self.cfg.nb_neighbors
        self.features_subset = self.cfg.features_subset
        self.update_adjacency = self.cfg.update_adjacency

    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass."""
        if self.update_adjacency:
            # Update adjacency
            edge_index = knn_graph(
                x=x[:,self.features_subset],
                k=self.nb_neighbors,
                batch=batch,
            ).to(x.device)

        # Standard EdgeConv forward pass
        x = super().forward(x, edge_index)
        return x, edge_index

class GraphNet(nn.Module):
    def __init__(self, *arg, **kvargs):        
        super().__init__()  
        self.cfg = Config(
            name = 'GraphNet',
            nb_inputs = 3, # ����� ��� ������ ��������� �����
            layers = [     # ������ DynEdgeConv                       
                       Config(mlp=[16,16],                  # ���� MLP � DynEdgeConv
                              features_subset=slice(0, 3),  # �������� ��� �� ������� ��������� ������
                              nb_neighbors=4),              # ����� ������� ��� ������ ������� �����
                       Config(mlp=[16,16], 
                              features_subset=slice(0, 3), 
                              nb_neighbors=4),
            ],
            post_processing = [16,16],
            global_pooling = ["min","max","mean"],
            readout = [128,10]                               
        )   

        self.cfg = self.cfg(*arg, **kvargs)   # add, change properties    
        self._activation = torch.nn.LeakyReLU()  
        self._build_layers()   
        self._build_postprocessing()
        self._build_readout()

    def _build_layer(self, nb_in, layer_cfg):
        layers = []
        layer_sizes = [nb_in] + list(layer_cfg.mlp)
        for ix, (nb_in, nb_out) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            if ix == 0:
                nb_in *= 2
            layers.append(torch.nn.Linear(nb_in, nb_out))
            layers.append(self._activation)

        layer_cfg.nn = torch.nn.Sequential(*layers)
        conv_layer = DynEdgeConv(layer_cfg)
        return conv_layer, nb_out

    def _build_layers(self):
        self._conv_layers = torch.nn.ModuleList()
        nb_inputs = self.cfg.nb_inputs
        for layer_conf in self.cfg.layers:            
            conv_layer, nb_out = self._build_layer(nb_inputs, layer_conf)
            self._conv_layers.append(conv_layer)
            nb_inputs = nb_out 

    def _build_postprocessing(self):
        nb_inputs = self.cfg.nb_inputs
        for layer_conf in self.cfg.layers:      
            nb_inputs += layer_conf.mlp[-1]
            
        layer_sizes = [nb_inputs] + list(
                self.cfg.post_processing
            )
        post_processing_layers = []
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
                post_processing_layers.append(torch.nn.Linear(nb_in, nb_out))
                post_processing_layers.append(self._activation)  
        self._post_processing = torch.nn.Sequential(*post_processing_layers)

    def _build_readout(self):
        nb_inputs = self.cfg.post_processing[-1]*len(self.cfg.global_pooling)
        layer_sizes = [nb_inputs] + list(
                self.cfg.readout
            )
        read_out_layers = []
        for idx, (nb_in, nb_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                read_out_layers.append(torch.nn.Linear(nb_in, nb_out))
                if idx < len(layer_sizes)-2:
                    read_out_layers.append(self._activation)  
        self._readout = torch.nn.Sequential(*read_out_layers)  

    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        pooled = []
        for pooling_scheme in self.cfg.global_pooling:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, index=batch, dim=0)
            if isinstance(pooled_x, tuple) and len(pooled_x) == 2:
                # `scatter_{min,max}`, which return also an argument, vs.
                # `scatter_{mean,sum}`
                pooled_x, _ = pooled_x
            pooled.append(pooled_x)
        pooled = torch.cat(pooled, dim=1)
        return pooled

    def forward(self, data):
        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # DynEdge-convolutions
        skip_connections = [x]
        for conv_layer in self._conv_layers: 
            x, edge_index = conv_layer(x, edge_index, batch)
            skip_connections.append(x)
        # Skip-cat
        x = torch.cat(skip_connections, dim=1)
        # Post-processing        
        x = self._post_processing(x)
        # Global pooling  
        x = self._global_pooling(x, batch=batch)
        # Read-out
        x = self._readout(x)
        return x

