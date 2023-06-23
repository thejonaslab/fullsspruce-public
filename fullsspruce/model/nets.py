
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from fullsspruce.model.net_util import *
from fullsspruce import util
from fullsspruce.model import seminets

class GraphVertConfigBootstrapWithMultiMax(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None, 
                 int_d = None, layer_n = None, 
                 mixture_n = 5,
                 mixture_num_obs_per=1,
                 resnet=True, 
                 gml_class = 'GraphMatLayers',
                 gml_config = {}, 
                 init_noise=1e-5,
                 init_bias = 0.0, agg_func=None, GS=1, OUT_DIM=1, 
                 input_norm='batch', out_std= False, 
                 resnet_out = False, resnet_blocks = (3, ), 
                 resnet_d = 128,
                 resnet_norm = 'layer',
                 resnet_dropout = 0.0, 
                 inner_norm=None, 
                 out_std_exp = False, 
                 force_lin_init=False, 
                 use_random_subsets=True):
        
        """
        GraphVertConfigBootstrap with multiple max outs
        """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_out_n=", g_feature_out_n)

        super( GraphVertConfigBootstrapWithMultiMax, self).__init__()
        self.gml = eval(gml_class)(g_feature_n, g_feature_out_n, 
                                   resnet=resnet, noise=init_noise,
                                   agg_func=parse_agg_func(agg_func), 
                                   norm = inner_norm, 
                                   GS=GS,
                                   **gml_config)

        if input_norm == 'batch':
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == 'layer':
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        self.resnet_out = resnet_out 
        if not resnet_out:
            self.mix_out = nn.ModuleList([nn.Linear(g_feature_out_n[-1], OUT_DIM) for _ in range(mixture_n)])
        else:
            self.mix_out = nn.ModuleList([ResNetRegressionMaskedBN(g_feature_out_n[-1], 
                                                                   block_sizes = resnet_blocks, 
                                                                   INT_D = resnet_d, 
                                                                   FINAL_D=resnet_d,
                                                                   norm = resnet_norm,
                                                                   dropout = resnet_dropout, 
                                                                   OUT_DIM=OUT_DIM) for _ in range(mixture_n)])

        self.out_std = out_std
        self.out_std_exp = False

        self.use_random_subsets = use_random_subsets
        self.mixture_num_obs_per = mixture_num_obs_per
        
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

    def forward(self, adj, vect_feat, input_mask, input_idx, adj_oh,
                return_g_features = False, also_return_g_features = False,
                **kwargs):

        G = adj
        
        BATCH_N, MAX_N, _ = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, 
                                             vect_feat, 
                                             input_mask)
        
        G_features = self.gml(G, vect_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])
        
        if self.resnet_out:
            x_1 = [m(g_squeeze_flat, input_mask.reshape(-1)).reshape(BATCH_N, MAX_N, -1) for m in self.mix_out]
        else:
            x_1 = [m(g_squeeze) for m in self.mix_out]

        x_1 = torch.stack(x_1)

        x_1, std = bootstrap_perm_compute(x_1, input_idx,
                                          self.mixture_num_obs_per,
                                          training=self.training)

        ret = {'shift_mu' : x_1, 'shift_std' : std}
        if also_return_g_features:
            ret['g_features'] = g_squeeze
        return ret

class GraphDecodeBootstrapWithMultiMax(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None, 
                 int_d = None, layer_n = None, 
                 mixture_n = 5,
                 mixture_num_obs_per=1,
                 resnet=True, 
                 gml_class = 'GraphMatLayers',
                 gml_config = {},
                 decode_class = 'seminets.DecodeWithVertices',
                 decode_config = {}, 
                 init_noise=1e-5,
                 init_bias = 0.0, agg_func=None, GS=1, GS_mat= None, OUT_DIM=1, 
                 input_norm='batch', input_norm_e='batch', out_std= False, 
                 resnet_out = False, resnet_blocks = (3, ), 
                 resnet_d = 128,
                 resnet_norm = 'layer',
                 resnet_dropout = 0.0, 
                 inner_norm=None, 
                 embed_edge = False,
                 embed_vert = False,
                 out_std_exp = False, 
                 force_lin_init=False, 
                 use_random_subsets=True):
        """
        Using Decode for vertices with bootstraph and multiple max outs.
        """
        self.int_d = int_d
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_out_n=", g_feature_out_n)

        super( GraphDecodeBootstrapWithMultiMax, self).__init__()

        if GS_mat is None:
            GS_mat = GS
        self.GS_mat = GS_mat

        self.decoder = eval(decode_class)(**decode_config)

        mix_in = g_feature_out_n[-1]
        if not gml_class is None:
            self.gml = eval(gml_class)(g_feature_n, g_feature_out_n, 
                                   resnet=resnet, noise=init_noise,
                                   agg_func=parse_agg_func(agg_func), 
                                   norm = inner_norm, 
                                   GS=GS_mat,
                                   **gml_config)
            mix_in *= 2
        else:
            self.gml = None

        if input_norm == 'batch':
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == 'layer':
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        if input_norm_e == 'batch':
            self.input_e_norm = MaskedBatchNorm1d(GS)
        elif input_norm_e == 'layer':
            self.input_e_norm = nn.LayerNorm(GS)
        else:
            self.input_e_norm = None

        self.resnet_out = resnet_out 
        if not resnet_out:
            self.mix_out = nn.ModuleList([nn.Linear(mix_in, OUT_DIM) for _ in range(mixture_n)])
        else:
            self.mix_out = nn.ModuleList([ResNetRegressionMaskedBN(mix_in, 
                                                                   block_sizes = resnet_blocks, 
                                                                   INT_D = resnet_d, 
                                                                   FINAL_D=resnet_d,
                                                                   norm = resnet_norm,
                                                                   dropout = resnet_dropout, 
                                                                   OUT_DIM=OUT_DIM) for _ in range(mixture_n)])

        self.out_std = out_std
        self.out_std_exp = False

        self.e_embed = nn.Linear(GS, int_d)
        self.embed_edge = embed_edge
        
        self.v_embed = nn.Linear(g_feature_n, int_d)
        self.embed_vert = embed_vert

        self.use_random_subsets = use_random_subsets
        self.mixture_num_obs_per = mixture_num_obs_per
        
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

    def forward(self, adj, vect_feat, input_mask, input_idx, adj_oh,
                return_g_features = False, also_return_g_features = False,
                **kwargs):

        G = adj
        G_mat = adj[:, :self.GS_mat, :, :]
        
        BATCH_N, MAX_N, _ = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, 
                                             vect_feat, 
                                             input_mask)
        
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

        G_bip_features, _ = self.decoder(v_pad, e_pad)
        if not self.gml is None:
            G_mat_features = self.gml(G_mat, vect_feat, input_mask)
            G_features = torch.cat([G_bip_features, G_mat_features], dim=-1)
        else:
            G_features = G_bip_features

        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])    

        if self.resnet_out:
            x_1 = [m(g_squeeze_flat, input_mask.reshape(-1)).reshape(BATCH_N, MAX_N, -1) for m in self.mix_out]
        else:
            x_1 = [m(g_squeeze) for m in self.mix_out]

        x_1 = torch.stack(x_1)

        x_1, std = bootstrap_perm_compute(x_1, input_idx,
                                          self.mixture_num_obs_per,
                                          training=self.training)

        ret = {'shift_mu' : x_1, 'shift_std' : std}
        if also_return_g_features:
            ret['g_features'] = g_squeeze
        return ret

    
class DropoutEmbedExp(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None, 
                 int_d = None, layer_n = None, 
                 mixture_n = 5,
                 mixture_num_obs_per=1,
                 resnet=True, 
                 gml_class = 'GraphMatLayers',
                 gml_config = {}, 
                 init_noise=1e-5,
                 init_bias = 0.0, agg_func=None, GS=1, OUT_DIM=1, 
                 input_norm='batch', out_std= False, 
                 resnet_out = False, resnet_blocks = (3, ), 
                 resnet_d = 128,
                 resnet_norm = 'layer',
                 resnet_dropout = 0.0, 
                 inner_norm=None, 
                 out_std_exp = False, 
                 force_lin_init=False, 
                 use_random_subsets=True,
                 input_vert_dropout_p = 0.0,
                 input_edge_dropout_p = 0.0, 
                 embed_edges = False, 
    ):
        
        """
        
        """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_out_n=", g_feature_out_n)

        super( DropoutEmbedExp, self).__init__()
        self.gml = eval(gml_class)(g_feature_n, g_feature_out_n, 
                                   resnet=resnet, noise=init_noise,
                                   agg_func=parse_agg_func(agg_func), 
                                   norm = inner_norm, 
                                   GS=GS,
                                   **gml_config)

        if input_norm == 'batch':
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == 'layer':
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        self.resnet_out = resnet_out 
        if not resnet_out:
            self.mix_out = nn.ModuleList([nn.Linear(g_feature_out_n[-1], OUT_DIM) for _ in range(mixture_n)])
        else:
            self.mix_out = nn.ModuleList([ResNetRegressionMaskedBN(g_feature_out_n[-1], 
                                                                   block_sizes = resnet_blocks, 
                                                                   INT_D = resnet_d, 
                                                                   FINAL_D=resnet_d,
                                                                   norm = resnet_norm,
                                                                   dropout = resnet_dropout, 
                                                                   OUT_DIM=OUT_DIM) for _ in range(mixture_n)])

            
        self.input_vert_dropout = nn.Dropout(input_vert_dropout_p)
        self.input_edge_dropout = nn.Dropout(input_edge_dropout_p)
        
        self.out_std = out_std
        self.out_std_exp = False

        self.use_random_subsets = use_random_subsets
        self.mixture_num_obs_per = mixture_num_obs_per
        if embed_edges:
            self.edge_lin = nn.Linear(GS, GS)
        else:
            self.edge_lin = nn.Identity(GS)
            
        
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

    def forward(self, adj, vect_feat, input_mask, input_idx, adj_oh,
                return_g_features = False, also_return_g_features = False,
                **kwargs):
        G = self.edge_lin(adj.permute(0, 2, 3, 1)).permute(0, 3, 1,2)
        
        
        BATCH_N, MAX_N, _ = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, 
                                             vect_feat, 
                                             input_mask)

        vect_feat = vect_feat * self.input_vert_dropout(input_mask).unsqueeze(-1)
        G = self.input_edge_dropout(G)
        
        G_features = self.gml(G, vect_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])
        
        if self.resnet_out:
            x_1 = [m(g_squeeze_flat, input_mask.reshape(-1)).reshape(BATCH_N, MAX_N, -1) for m in self.mix_out]
        else:
            x_1 = [m(g_squeeze) for m in self.mix_out]

        x_1 = torch.stack(x_1)

        x_1, std = bootstrap_perm_compute(x_1, input_idx,
                                          self.mixture_num_obs_per,
                                          training=self.training)
        

        ret = {'mu' : x_1, 'std' : std}
        if also_return_g_features:
            ret['g_features'] = g_squeeze
        return ret
