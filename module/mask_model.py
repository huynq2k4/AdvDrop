# from module.inv_loss import *
# from utils import sparse_dense_mul
# from typing import Optional, Tuple, Union

# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.typing import NoneType  # noqa
# from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
# from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
# from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric.utils import add_self_loops, degree
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
# from torch_geometric.utils import is_sparse, is_torch_sparse_tensor

# from typing import Optional, Tuple

# import torch

# def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
#     if dim < 0:
#         dim = other.dim() + dim
#     if src.dim() == 1:
#         for _ in range(0, dim):
#             src = src.unsqueeze(0)
#     for _ in range(src.dim(), other.dim()):
#         src = src.unsqueeze(-1)
#     src = src.expand(other.size())
#     return src


# def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
#                 out: Optional[torch.Tensor] = None,
#                 dim_size: Optional[int] = None) -> torch.Tensor:
#     index = broadcast(index, src, dim)
#     if out is None:
#         size = list(src.size())
#         if dim_size is not None:
#             size[dim] = dim_size
#         elif index.numel() == 0:
#             size[dim] = 0
#         else:
#             size[dim] = int(index.max()) + 1
#         out = torch.zeros(size, dtype=src.dtype, device=src.device)
#         return out.scatter_add_(dim, index, src)
#     else:
#         return out.scatter_add_(dim, index, src)


# def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
#                 out: Optional[torch.Tensor] = None,
#                 dim_size: Optional[int] = None) -> torch.Tensor:
#     return scatter_sum(src, index, dim, out, dim_size)


# def scatter_mul(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
#                 out: Optional[torch.Tensor] = None,
#                 dim_size: Optional[int] = None) -> torch.Tensor:
#     return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


# def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
#                  out: Optional[torch.Tensor] = None,
#                  dim_size: Optional[int] = None) -> torch.Tensor:
#     out = scatter_sum(src, index, dim, out, dim_size)
#     dim_size = out.size(dim)

#     index_dim = dim
#     if index_dim < 0:
#         index_dim = index_dim + src.dim()
#     if index.dim() <= index_dim:
#         index_dim = index.dim() - 1

#     ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
#     count = scatter_sum(ones, index, index_dim, None, dim_size)
#     count[count < 1] = 1
#     count = broadcast(count, out, dim)
#     if out.is_floating_point():
#         out.true_divide_(count)
#     else:
#         out.div_(count, rounding_mode='floor')
#     return out


# def scatter_min(
#         src: torch.Tensor, index: torch.Tensor, dim: int = -1,
#         out: Optional[torch.Tensor] = None,
#         dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#     return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


# def scatter_max(
#         src: torch.Tensor, index: torch.Tensor, dim: int = -1,
#         out: Optional[torch.Tensor] = None,
#         dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#     return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


# def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
#             out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
#             reduce: str = "sum") -> torch.Tensor:
#     if reduce == 'sum' or reduce == 'add':
#         return scatter_sum(src, index, dim, out, dim_size)
#     if reduce == 'mul':
#         return scatter_mul(src, index, dim, out, dim_size)
#     elif reduce == 'mean':
#         return scatter_mean(src, index, dim, out, dim_size)
#     elif reduce == 'min':
#         return scatter_min(src, index, dim, out, dim_size)[0]
#     elif reduce == 'max':
#         return scatter_max(src, index, dim, out, dim_size)[0]
#     else:
#         raise ValueError

# class Mask_Model_Attention(MessagePassing):
#     def __init__(self, args):
#         super().__init__(aggr='add')
#         self.embed_size = args.embed_size
#         self.embed_h = args.att_dim
#         self.Q = Linear(self.embed_size, self.embed_h)
#         self.K = Linear(self.embed_size, self.embed_h)
#         #self.W = Linear(2*self.embed_size, 1)
#         self.gumble_tau = args.gumble_tau
#         self.device = torch.device(args.cuda)
#         self.args = args
    
#     def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
#         decomposed_layers = 1 if self.explain else self.decomposed_layers

#         for hook in self._propagate_forward_pre_hooks.values():
#             res = hook(self, (edge_index, size, kwargs))
#             if res is not None:
#                 edge_index, size, kwargs = res

#         size = self.__check_input__(edge_index, size)

#         if decomposed_layers > 1:
#                 user_args = self.__user_args__
#                 decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
#                 decomp_kwargs = {
#                     a: kwargs[a].chunk(decomposed_layers, -1)
#                     for a in decomp_args
#                 }
#                 decomp_out = []

#         for i in range(decomposed_layers):
#             if decomposed_layers > 1:
#                 for arg in decomp_args:
#                     kwargs[arg] = decomp_kwargs[arg][i]

#             coll_dict = self.__collect__(self.__user_args__, edge_index,
#                                             size, kwargs)

#             msg_kwargs = self.inspector.distribute('message', coll_dict)
#             for hook in self._message_forward_pre_hooks.values():
#                 res = hook(self, (msg_kwargs, ))
#                 if res is not None:
#                     msg_kwargs = res[0] if isinstance(res, tuple) else res
#             out = self.message(**msg_kwargs)

#             for hook in self._message_forward_hooks.values():
#                 res = hook(self, (msg_kwargs, ), out)
#                 if res is not None:
#                     out = res

#             if self.explain:
#                 explain_msg_kwargs = self.inspector.distribute(
#                     'explain_message', coll_dict)
#                 out = self.explain_message(out, **explain_msg_kwargs)

#             if decomposed_layers > 1:
#                     decomp_out.append(out)
#         if decomposed_layers > 1:
#                 out = torch.cat(decomp_out, dim=-1)
#         return out 

#     def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         norm = deg[col]
#         x_norm = F.normalize(x, p=2., dim=-1)
#         # propagate_type: (x: Tensor, x_norm: Tensor)
#         return self.propagate(edge_index, x=x, x_norm = x_norm,norm=norm, size=None)

#     def message(self, x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor,norm,
#                 index: Tensor, ptr: OptTensor,
#                 size_i: Optional[int]) -> Tensor:
#         # apply transformation layers
#         Query = self.Q(x_norm_i)
#         Keys = self.K(x_norm_j)

#         # alpha = torch.squeeze(self.W(torch.cat([x_norm_i,x_norm_j],dim=1)))
#         # dot product of each query-key pair
#         alpha = (Keys * Query).sum(dim=-1)
#         # # apply gumble
#         # gumble_G = torch.log(-torch.log(torch.rand(alpha.shape[0]).to(self.device)))
#         # alpha = (alpha - gumble_G) / self.gumble_tau
#         alpha = (alpha - scatter(alpha,index, dim=0, reduce='mean')[index])
        

#         #alpha = (alpha - torch.mean(alpha))/torch.sqrt(torch.var(alpha))
#         # softmax
#         #alpha = softmax(alpha, index, ptr, size_i)
#         #alpha = alpha*norm*self.args.keep_prob
#         #return torch.clamp(alpha, min=0, max=1) 
#         return torch.sigmoid(alpha) 
#     def reset_parameters(self):
#         self.Q.reset_parameters()
#         self.K.reset_parameters()

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, scatter
from torch.nn import Linear
from typing import Optional

class Mask_Model_Attention(MessagePassing):
    def __init__(self, args):
        super().__init__(aggr='add')
        self.embed_size = args.embed_size
        self.embed_h = args.att_dim
        self.Q = Linear(self.embed_size, self.embed_h)
        self.K = Linear(self.embed_size, self.embed_h)
        self.gumble_tau = args.gumble_tau
        self.device = torch.device(args.cuda)
        self.args = args

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        norm = deg[col]
        x_norm = F.normalize(x, p=2., dim=-1)
        # propagate_type: (x: Tensor, x_norm: Tensor, norm: Tensor)
        return self.propagate(edge_index, x=x, x_norm=x_norm, norm=norm)

    def message(self, x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor, norm: Tensor, index: Tensor, ptr: Optional[Tensor], size_i: Optional[int]) -> Tensor:
        # apply transformation layers
        Query = self.Q(x_norm_i)
        Keys = self.K(x_norm_j)

        # dot product of each query-key pair
        alpha = (Keys * Query).sum(dim=-1)
        alpha = (alpha - scatter(alpha, index, dim=0, reduce='mean')[index])

        # apply sigmoid
        return torch.sigmoid(alpha)

    def reset_parameters(self):
        self.Q.reset_parameters()
        self.K.reset_parameters()
