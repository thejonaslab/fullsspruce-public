import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from fullsspruce.model.net_util import * 

class UncertaintyLinLayer(nn.Module):
    def __init__(self, layer_n, d, attn_n):
        super( UncertaintyLinLayer, self).__init__()

        self.l1 = nn.Linear(d, d)
        self.l2 = nn.Linear(d, attn_n)

    def forward(self, x, attn):
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return torch.sum(y * attn, dim=-1).unsqueeze(-1)        
        

def bootstrap_perm_compute(x_1, input_idx, num_obs = 1,
                           var_eps=1e-5,
                           training=True):
    """
    shape is MIX_N, BATCH_SIZE, ....
    compute bootstrap by taking the first num_obs instances of a permutation
    """
    MIX_N = x_1.shape[0]
    BATCH_N = x_1.shape[1]
    
    if training:
        x_zeros = np.zeros(x_1.shape)
        for i, idx in enumerate(input_idx):
            rs = np.random.RandomState(idx).permutation(MIX_N)[:num_obs]
            for j in range(num_obs):
                x_zeros[rs[j], i] = 1
        mask = torch.Tensor(x_zeros).to(x_1.device)
        x_1_sub = mask * x_1
        x_1_sub = x_1_sub.sum(dim=0)/ num_obs
    else:
        x_1_sub = x_1.mean(dim=0)
    # x_1_sub = torch.stack([x_1[v, v_i] for v_i, v in enumerate(idx)])
    if MIX_N > 1:
        std = torch.sqrt(torch.var(x_1, dim=0) + var_eps)
    else:
        std = torch.ones_like(x_1_sub) * var_eps
    return x_1_sub, std




class CouplingUncertainty(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None, 
                 int_d = None, layer_n = None, 
                 mixture_n = 5,
                 mixture_num_obs_per=1,
                 resnet=True, 
                 gml_class = 'GraphMatLayers',
                 gml_config = {}, 
                 init_noise=1e-5,
                 init_bias = 0.0, agg_func=None, GS=1, OUT_DIM=1,
                 input_norm='batch',
                 input_norm_e='batch', out_std= False, 
                 resnet_out = False, resnet_blocks = (3, ), 
                 resnet_d = 128,
                 resnet_norm = 'layer',
                 resnet_dropout = 0.0,
                 decode_class = 'DecodeEdge', 
                 decode_config = {},
                 inner_norm=None, 
                 out_std_exp = False, 
                 force_lin_init=False, 
                 use_random_subsets=True,
                 embed_edge = False,
                 embed_vert = False,
                 final_use_softmax = True,
                 coupling_output_kinds = 9, 
                 ):
        
        """
        GraphVertConfigBootstrap with multiple max outs
        """
        self.int_d = int_d 
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_out_n=", g_feature_out_n)

        super( CouplingUncertainty, self).__init__()
        # self.gml = eval(gml_class)(g_feature_n, g_feature_out_n, 
        #                            resnet=resnet, noise=init_noise,
        #                            agg_func=parse_agg_func(agg_func), 
        #                            norm = inner_norm, 
        #                            GS=GS,
        #                            **gml_config)

        if input_norm == 'batch':
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == 'layer':
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        if input_norm == 'batch':
            self.input_e_norm = MaskedBatchNorm1d(GS)
        elif input_norm == 'layer':
            self.input_e_norm = nn.LayerNorm(GS)
        else:
            self.input_e_norm = None

        # self.resnet_out = resnet_out 
        # if not resnet_out:
        #     self.mix_out = nn.ModuleList([nn.Linear(g_feature_out_n[-1], OUT_DIM) for _ in range(mixture_n)])
        # else:
        #     self.mix_out = nn.ModuleList([ResNetRegressionMaskedBN(g_feature_out_n[-1], 
        #                                                            block_sizes = resnet_blocks, 
        #                                                            INT_D = resnet_d, 
        #                                                            FINAL_D=resnet_d,
        #                                                            norm = resnet_norm,
        #                                                            dropout = resnet_dropout, 
        #                                                            OUT_DIM=OUT_DIM) for _ in range(mixture_n)])


        #self.coupling_out = nn.Linear(g_feature_out_n[-1], 1)
        self.decode_edge = eval(decode_class)(**decode_config)

        self.out_std = out_std
        self.out_std_exp = False

        self.use_random_subsets = use_random_subsets
        self.mixture_num_obs_per = mixture_num_obs_per

        self.e_embed = nn.Linear(GS, int_d)
        self.embed_edge = embed_edge
        
        self.v_embed = nn.Linear(g_feature_n, int_d)
        self.embed_vert = embed_vert
        
        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        if init_bias > 0:
                            nn.init.normal_(m.bias, 0, init_bias)
                        else:
                            nn.init.constant_(m.bias, 0)


        self.coupling_types_embed = nn.Embedding(coupling_output_kinds,
                                                 coupling_output_kinds)

        self.ull = nn.ModuleList([UncertaintyLinLayer(2, int_d, coupling_output_kinds) for _ in range(mixture_n)])

        self.final_use_softmax = final_use_softmax
        
    def forward(self, adj, vect_feat, input_mask, input_idx, adj_oh,
                return_g_features = False, also_return_g_features = False,
                coupling=None, coupling_mask=None, coupling_types=None,
                passthrough_coupling_types_encoded = None, 
                **kwargs):

        # for k, v in kwargs.items():
        #     print(f"extra {k}: {v.shape}")
        G = adj
        
        BATCH_N, MAX_N, _ = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, 
                                             vect_feat, 
                                             input_mask)
        

        x_1 = torch.zeros(BATCH_N, MAX_N, 1).to(adj.device)
        std = torch.zeros(BATCH_N, MAX_N, 1).to(adj.device)
        
        adj_as_e = adj.permute(0, 2, 3, 1)

        if self.input_e_norm is not None:
            adj_as_e = self.input_e_norm(adj_as_e)
            
        if self.embed_edge:
            e_pad = self.e_embed(adj_as_e)
        else:
            e_pad = F.pad(adj_as_e, (0, self.int_d-adj_as_e.shape[-1]),
                          'constant', 0)

        if self.embed_vert:
            v_pad = self.v_embed(vect_feat)
        else:
        
            v_pad = F.pad(vect_feat, (0,  self.int_d-vect_feat.shape[-1]),
                              'constant', 0)
        coupling_pred_raw = self.decode_edge(v_pad, e_pad)
        
        # print(torch.max(passthrough_coupling_types_encoded.long()  + 2),
        #       torch.min(passthrough_coupling_types_encoded.long()  + 2),
        # )
        cte = self.coupling_types_embed(passthrough_coupling_types_encoded.long()  + 2)

        if self.final_use_softmax == True:
            cte_sm = torch.softmax(cte, dim=-1)
        elif self.final_use_softmax == 'sigmoid':
            cte_sm = torch.sigmoid(cte)
        else:
            cte_sm = cte

        coupling_pred = torch.stack([l(F.relu(coupling_pred_raw),
                                       cte_sm) for l in self.ull])

        
        coupling_pred_mu, coupling_pred_std = bootstrap_perm_compute(coupling_pred, input_idx,
                                                                     self.mixture_num_obs_per,
                                                                     training=self.training)

        #print(coupling_pred_raw.shape,
        #      cte.shape, cte_sm.shape, coupling_pred.shape)

        ret = {'mu' : x_1, 'std' : std,
               'coupling_mu' : coupling_pred_mu, 
               'coupling_std' : coupling_pred_std, 
               #'coupling_truth' : coupling,
               'coupling_types' : passthrough_coupling_types_encoded.long(), 
               'coupling_mask' : coupling_mask}
        if also_return_g_features:
            ret['g_features'] = g_squeeze
            
        return ret




class CouplingLoss(nn.Module):
    def __init__(self, shift_norm='l2', shift_scale=1.0, 
                 shift_loss_weight = 1.0,
                 coupling_field = 'coupling_mu',
                 coupling_loss_weight = 1.0, loss_name=None, 
                 coupling_loss_name = 'nn.MSELoss',
                 coupling_loss_config = {}):
        super(CouplingLoss, self).__init__()

        self.shift_loss = NoUncertainLoss(shift_norm, shift_scale)
        self.coupling_loss = eval(coupling_loss_name)(reduction='none', 
                                               **coupling_loss_config)
        self.shift_loss_weight = shift_loss_weight
        self.coupling_loss_weight = coupling_loss_weight
        self.coupling_field = coupling_field

    def __call__(self, pred,
                 vert_truth, 
                 vert_truth_mask,
                 edge_truth,
                 edge_truth_mask, 
                 ## ADD OTHJERS
                 vert_mask):

        y = pred[self.coupling_field]
        #assert not torch.isnan(y).any()
        # assert not torch.isnan(pred_mask).any()
        # for k, v in pred.items():
        #     if  torch.isnan(v).any():
        #         raise Exception(f"k={k}") 

        # if torch.sum(pred_mask) > 0:
        #     l_shift = self.shift_loss(pred, y, pred_mask, input_mask)
        #     assert not torch.isnan(l_shift).any()
        # else:
        #     l_shift = torch.tensor(0.0).to(y.device)


        BATCH_N = y.shape[0]
        

        #MAX_N = y.shape[1]

        coupling_mask = edge_truth_mask[:, :, :, 0]

        coupling_pred = pred[self.coupling_field][:, :, :, 0]
        
        coupling_truth = edge_truth[:, :, :, 0]
        #print("coupling_pred.shape=", coupling_pred.shape,
        #     "coupling_truth.shape=", coupling_truth.shape)
        l_coupling = self.coupling_loss(coupling_pred, coupling_truth)
        assert not torch.isnan(l_coupling).any()

        l_coupling = l_coupling[coupling_mask >0].mean()
        #l_coupling = l_coupling.mean()
                              
        #loss = self.shift_loss_weight* l_shift + self.coupling_loss_weight * l_coupling
        return {#'loss' : loss, 
                'loss' : l_coupling, 
                #'loss_shift' : l_shift
        }
    
    
class WeightedCouplingLoss(nn.Module):
    def __init__(self, shift_norm='l2', shift_scale=1.0, 
                 shift_loss_weight = 1.0,
                 coupling_field = 'coupling_mu',
                 coupling_loss_weight = 1.0,
                 loss_name=None,
                 coupling_type_weights = None,
                 coupling_loss_name = 'nn.MSELoss',
                 coupling_loss_config = {}):
        super(WeightedCouplingLoss, self).__init__()

        self.shift_loss = NoUncertainLoss(shift_norm, shift_scale)
        self.coupling_loss = eval(coupling_loss_name)(reduction='none', 
                                               **coupling_loss_config)
        self.shift_loss_weight = shift_loss_weight
        self.coupling_loss_weight = coupling_loss_weight
        self.coupling_field = coupling_field
        self.coupling_type_weights = coupling_type_weights 

    def __call__(self, pred,
                 vert_truth, 
                 vert_truth_mask,
                 edge_truth,
                 edge_truth_mask, 
                 ## ADD OTHJERS
                 vert_mask):

        y = pred[self.coupling_field]

        coupling_types = pred['coupling_types']

            
        
        #assert not torch.isnan(y).any()
        # assert not torch.isnan(pred_mask).any()
        # for k, v in pred.items():
        #     if  torch.isnan(v).any():
        #         raise Exception(f"k={k}") 

        # if torch.sum(pred_mask) > 0:
        #     l_shift = self.shift_loss(pred, y, pred_mask, input_mask)
        #     assert not torch.isnan(l_shift).any()
        # else:
        #     l_shift = torch.tensor(0.0).to(y.device)


        BATCH_N = y.shape[0]
        

        #MAX_N = y.shape[1]

        coupling_mask = edge_truth_mask[:, :, :, 0]

        coupling_pred = pred[self.coupling_field][:, :, :, 0]
        
        coupling_truth = edge_truth[:, :, :, 0]
        #print("coupling_pred.shape=", coupling_pred.shape,
        #     "coupling_truth.shape=", coupling_truth.shape)
        l_coupling = self.coupling_loss(coupling_pred, coupling_truth)
        assert not torch.isnan(l_coupling).any()

        if self.coupling_type_weights is not None:
            coupling_type_weights  = torch.Tensor(np.array(self.coupling_type_weights, dtype=np.float32)).to(y.device)
            coupling_weights = coupling_type_weights[coupling_types + 2]
            l_coupling = l_coupling * coupling_weights 
        
        l_coupling = l_coupling[coupling_mask >0].mean()
        #l_coupling = l_coupling.mean()
                              
        #loss = self.shift_loss_weight* l_shift + self.coupling_loss_weight * l_coupling
        return {#'loss' : loss, 
                'loss' : l_coupling, 
                #'loss_shift' : l_shift
        }
