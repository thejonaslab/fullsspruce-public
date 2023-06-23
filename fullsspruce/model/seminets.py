import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from fullsspruce.model.net_util import * 


class DecodeEdge(nn.Module):
    def __init__(self, D, output_feat, otherstuff=None):
        super( DecodeEdge, self).__init__()

        self.l1 = nn.Linear(D, D)
        self.l2 = nn.Linear(D, D)
        self.l3 = nn.Linear(D, D)
        self.l4 = nn.Linear(D, output_feat)
        
    def forward(self, v):
        v1 = F.relu(self.l1(v))

        
        e = F.relu(v1.unsqueeze(1) +  v1.unsqueeze(2))
        
        e  = F.relu(self.l2(e))
        e_v = torch.max(e, dim=2)[0]
        e_v = torch.relu(self.l3(e_v))
        e2 = F.relu(e_v.unsqueeze(1) + e_v.unsqueeze(2))
        
        out  = self.l4(e + e2 )
        return out 
        


class ReconLoss(nn.Module):
    def __init__(self, pred_norm='l2', pred_scale=1.0, 
                 pred_loss_weight = 1.0, 
                 recon_loss_weight = 1.0, loss_name=None, 
                 recon_loss_name = 'nn.BCEWithLogitsLoss',
                 recon_loss_config = {}):
        super(ReconLoss, self).__init__()

        self.pred_loss = NoUncertainLoss(pred_norm, pred_scale)
        self.recon_loss = eval(recon_loss_name)(reduction='none', 
                                               **recon_loss_config)
        self.pred_loss_weight = pred_loss_weight
        self.recon_loss_weight = recon_loss_weight

    def __call__(self, pred, y, pred_mask, input_mask):
        assert not torch.isnan(y).any()
        assert not torch.isnan(pred_mask).any()
        for k, v in pred.items():
            if  torch.isnan(v).any():
                raise Exception(f"k={k}") 

        if torch.sum(pred_mask) > 0:
            l_pred = self.pred_loss(pred, y, pred_mask, input_mask)
            assert not torch.isnan(l_pred).any()
        else:
            l_pred = torch.tensor(0.0).to(y.device)


        BATCH_N = y.shape[0]
        
        input_mask_2d = input_mask.unsqueeze(1) * input_mask.unsqueeze(-1)

        #g_in = pred['recon_feature'] #  .reshape(BATCH_N, -1)
        recon_features_edge = pred['recon_features_edge']
        MAX_N = recon_features_edge.shape[1]

        decoded_features_edge = pred['decoded_features_edge']#.reshape(BATCH_N, -1)
        l_recon = self.recon_loss(decoded_features_edge, recon_features_edge)
        assert not torch.isnan(l_recon).any()

        l_recon = l_recon[input_mask_2d.unsqueeze(-1).expand(BATCH_N, MAX_N, MAX_N, l_recon.shape[-1])>0].mean()
        #l_recon = l_recon.mean()
        loss = 0
        if self.pred_loss_weight > 0.0:
            loss = loss + self.pred_loss_weight* l_pred
        else:
            l_pred = torch.Tensor([0.0])
        if self.recon_loss_weight > 0.0:
            loss = loss + self.recon_loss_weight * l_recon

        return {'loss' : loss, 
                'loss_recon' : l_recon, 
                'loss_pred' : l_pred}


class DistReconLoss(nn.Module):
    def __init__(self, pred_norm='l2', pred_scale=1.0, 
                 pred_loss_weight = 1.0, 
                 recon_loss_weight = 1.0, loss_name=None, 
                 recon_loss_name = 'nn.BCEWithLogitsLoss',
                 recon_loss_config = {}):
        super(DistReconLoss, self).__init__()

        self.pred_loss = NoUncertainLoss(pred_norm, pred_scale)
        self.recon_loss = eval(recon_loss_name)(reduction='none', 
                                               **recon_loss_config)
        self.pred_loss_weight = pred_loss_weight
        self.recon_loss_weight = recon_loss_weight

    def __call__(self, pred, y, pred_mask, input_mask):
        assert not torch.isnan(y).any()
        assert not torch.isnan(pred_mask).any()
        for k, v in pred.items():
            if  torch.isnan(v).any():
                raise Exception(f"k={k}") 

        if torch.sum(pred_mask) > 0:
            l_pred = self.pred_loss(pred, y, pred_mask, input_mask)
            assert not torch.isnan(l_pred).any()
        else:
            l_pred = torch.tensor(0.0).to(y.device)


        BATCHinput_N = y.shape[0]
        
        input_mask_2d = input_mask.unsqueeze(1) * input_mask.unsqueeze(-1)

        dist_mat_in = pred['dist_mat'] 
        MAX_N = dist_mat_in.shape[1]

        dist_decode = pred['dist_decode']#.reshape(BATCH_N, -1)
        l_recon = self.recon_loss(dist_decode, dist_mat_in)
        assert not torch.isnan(l_recon).any()

        l_recon = l_recon[input_mask_2d.unsqueeze(-1).expand(BATCH_N, MAX_N, MAX_N, dist_decode.shape[-1])>0].mean()
        #l_recon = l_recon.mean()

        loss = 0
        if self.pred_loss_weight > 0.0:
            print("including pred loss")
            loss = loss + self.pred_loss_weight* l_pred
        else:
            l_pred = 0.0
        if self.recon_loss_weigth > 0.0:
            loss = loss + self.recon_loss_weight * l_recon
            
        outloss=  {'loss' : loss, 
                'loss_recon' : l_recon, 
                'loss_pred' : l_pred}

        return outloss


class Vpack:
    def __init__(self, BATCH_N, MAX_N, F, mask):
        """
        
        """
        self.BATCH_N = BATCH_N
        self.F = F
        self.MAX_N = MAX_N 
        self.mask = mask
        
    def zero(self, V):
        mask = self.mask.reshape(-1).unsqueeze(-1)
        return (V.reshape(-1, self.F) * mask).reshape(V.shape)
    
    def pack(self, V):
        # V_flat = V.reshape(-1, self.F)
        # mask = (self.mask>0).reshape(-1)
        # return V_flat[mask]
        V_flat = V.reshape(-1, self.F)
        #mask = (self.mask>0).reshape(-1)
        return V_flat # [mask]
    
    def unpack(self, V):
        # output = torch.zeros((self.BATCH_N *self.MAX_N, V.shape[-1]), device=V.device)
        # mask = (self.mask>0).reshape(-1)
        # output[mask] = V
        output = V
        return output.reshape(self.BATCH_N, self.MAX_N, V.shape[-1])
    
    
class Epack:
    def __init__(self, BATCH_N, MAX_N, F, mask):
        """
        
        """
        self.BATCH_N = BATCH_N
        self.F = F
        self.MAX_N = MAX_N 
        self.mask = mask
        
    def zero(self, E):
        mask = self.mask.reshape(-1).unsqueeze(-1)
        return (E.reshape(-1, self.F) * mask).reshape(E.shape)
    
    def pack(self, E):
        # E_flat = E.reshape(-1, self.F)
        # mask = (self.mask>0).reshape(-1)
        # return E_flat[mask]
        E_flat = E.reshape(-1, self.F)
        return E_flat
    
    def unpack(self, E):
        # output = torch.zeros((self.BATCH_N * self.MAX_N * self.MAX_N, E.shape[-1]), 
        #                      device=E.device)
        # mask = (self.mask>0).reshape(-1)
        # output[mask] = E
        # return output.reshape(self.BATCH_N, self.MAX_N, self.MAX_N, E.shape[-1])
        output = E
        return output.reshape(self.BATCH_N, self.MAX_N, self.MAX_N, E.shape[-1])
    
    
class Decode14(nn.Module):
    """
    Now with norm
    """
    def __init__(self, D, output_feat,
                 input_dropout_p=0.0,
                 steps= 4,
                 int_e_dropout_p = 0.0 ,
                 int_v_dropout_p = 0.0 ,
                 e_to_v_func = 'mean',
                 e_norm = None,
                 v_norm = None,
                 v_combine = 'prod', 
                 raw_e_out = False,
                 v_combine_scale = None, 
                 out_transform=None):
        super( Decode14, self).__init__()

        self.e_cell = nn.ModuleList([nn.GRUCell(D, D) for _ in range(steps)])
        self.v_cell = nn.ModuleList([nn.GRUCell(D, D) for _ in range(steps)])

        self.out_transform = out_transform

        self.input_dropout_p = input_dropout_p
        if input_dropout_p > 0:
            self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.steps = steps

        self.e_dropout = nn.Dropout(p=int_e_dropout_p)
        self.v_dropout = nn.Dropout(p=int_v_dropout_p)
        self.e_to_v_func = e_to_v_func

        
        if e_norm == 'batch':
            self.e_norm_l = nn.ModuleList([nn.BatchNorm1d(D) for _ in range(steps)])
        elif e_norm == 'layer':
            self.e_norm_l = nn.ModuleList([nn.LayerNorm(D) for _ in range(steps)])
        else:
            self.e_norm_l = nn.ModuleList([nn.Identity() for _ in range(steps)])

        if v_norm == 'batch':
            self.v_norm_l = nn.ModuleList([nn.BatchNorm1d(D) for _ in range(steps)])
        elif v_norm == 'layer':
            self.v_norm_l = nn.ModuleList([nn.LayerNorm(D) for _ in range(steps)])
        else:
            self.v_norm_l = nn.ModuleList([nn.Identity() for _ in range(steps)])
            
        self.v_combine = v_combine
        self.v_combine_scale = v_combine_scale
        self.raw_e_out = raw_e_out
        
    def forward(self, v, e_in = None):

        BATCH_N = v.shape[0]
        MAX_N = v.shape[1]
        D = v.shape[2]
        
        vp = Vpack(BATCH_N, MAX_N, D,
                   torch.ones(v.shape[0], v.shape[1]).to(v.device))
        ep = Epack(BATCH_N, MAX_N, D,
                   torch.ones(v.shape[0], v.shape[1], v.shape[1]).to(v.device))

        if self.input_dropout_p  > 0.0:
            v = self.input_dropout(v)
            e_in = self.input_dropout(e_in)


        v_in_f = vp.pack(v)
        if e_in is None:
            e_in_f = ep.pack(torch.zeros(v.shape[0], v.shape[1], v.shape[1],
                                         v.shape[2]).to(v.device))
        else:
            e_in_f = ep.pack(e_in)
        
        v_h = v_in_f
        e_h = e_in_f
        
        e_v = ep.pack(goodmax(e_in, dim=1))
        #print("e_v.shape=", e_v.shape)
        
        for i in range(self.steps):
            v_h = self.v_cell[i](e_v, v_h)

            v_h = self.v_norm_l[i](v_h)

            v_h = self.v_dropout(v_h)
                
            v_h_up = vp.unpack(v_h)
            if self.v_combine == 'prod':
                v_e = v_h_up.unsqueeze(1)  * v_h_up.unsqueeze(2)
            elif self.v_combine == 'sum':
                v_e = v_h_up.unsqueeze(1)  + v_h_up.unsqueeze(2)
            if self.v_combine_scale == 'rootd':
                v_e = v_e / np.sqrt(v_h_up.shape[-1])
            elif self.v_combine_scale == 'd':
                v_e = v_e / v_h_up.shape[-1]
            elif self.v_combine_scale == 'norm1':
                v_e = v_e / (torch.sum(v_e, dim=-1) + 1e-4).unsqueeze(-1)
                
            v_e_p = ep.pack(v_e)
            e_h = self.e_cell[i](v_e_p, e_h)

            e_h = self.e_norm_l[i](e_h)
            
            e_h = self.e_dropout(e_h)
            e_h_up = ep.unpack(e_h)
            if self.e_to_v_func == 'mean':
                e_v_up = torch.mean(e_h_up, dim=1)
            elif self.e_to_v_func == 'max':
                e_v_up = goodmax(e_h_up, dim=1)

                

            e_v = vp.pack(e_v_up)

        if self.raw_e_out:
            return e_h_up
        
        out = self.out_l(e_h_up)
        
        if self.out_transform == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.out_transform == 'relu':
            out = F.relu(out)
        return out 


class DecodeWithVertices(nn.Module):
    """
    Returns learned features for vertices as well
    """
    def __init__(self, D, output_feat,
                 input_dropout_p=0.0,
                 steps= 4,
                 int_e_dropout_p = 0.0 ,
                 int_v_dropout_p = 0.0 ,
                 e_to_v_func = 'mean',
                 e_norm = None,
                 v_norm = None,
                 v_combine = 'prod', 
                 raw_out = False,
                 v_combine_scale = None, 
                 out_transform=None):
        super( DecodeWithVertices, self).__init__()

        self.e_cell = nn.ModuleList([nn.GRUCell(D, D) for _ in range(steps)])
        self.v_cell = nn.ModuleList([nn.GRUCell(D, D) for _ in range(steps)])

        self.out_transform = out_transform
        self.out_v = nn.Linear(D, output_feat)
        self.out_e = nn.Linear(D, output_feat)

        self.input_dropout_p = input_dropout_p
        if input_dropout_p > 0:
            self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.steps = steps

        self.e_dropout = nn.Dropout(p=int_e_dropout_p)
        self.v_dropout = nn.Dropout(p=int_v_dropout_p)
        self.e_to_v_func = e_to_v_func

        
        if e_norm == 'batch':
            self.e_norm_l = nn.ModuleList([nn.BatchNorm1d(D) for _ in range(steps)])
        elif e_norm == 'layer':
            self.e_norm_l = nn.ModuleList([nn.LayerNorm(D) for _ in range(steps)])
        else:
            self.e_norm_l = nn.ModuleList([nn.Identity() for _ in range(steps)])

        if v_norm == 'batch':
            self.v_norm_l = nn.ModuleList([nn.BatchNorm1d(D) for _ in range(steps)])
        elif v_norm == 'layer':
            self.v_norm_l = nn.ModuleList([nn.LayerNorm(D) for _ in range(steps)])
        else:
            self.v_norm_l = nn.ModuleList([nn.Identity() for _ in range(steps)])
            
        self.v_combine = v_combine
        self.v_combine_scale = v_combine_scale
        self.raw_out = raw_out
        
    def forward(self, v, e_in = None):

        BATCH_N = v.shape[0]
        MAX_N = v.shape[1]
        D = v.shape[2]
        
        vp = Vpack(BATCH_N, MAX_N, D,
                   torch.ones(v.shape[0], v.shape[1]).to(v.device))
        ep = Epack(BATCH_N, MAX_N, D,
                   torch.ones(v.shape[0], v.shape[1], v.shape[1]).to(v.device))

        if self.input_dropout_p  > 0.0:
            v = self.input_dropout(v)
            e_in = self.input_dropout(e_in)


        v_in_f = vp.pack(v)
        if e_in is None:
            e_in_f = ep.pack(torch.zeros(v.shape[0], v.shape[1], v.shape[1],
                                         v.shape[2]).to(v.device))
        else:
            e_in_f = ep.pack(e_in)
        
        v_h = v_in_f
        e_h = e_in_f
        
        e_v = ep.pack(goodmax(e_in, dim=1))
        #print("e_v.shape=", e_v.shape)
        
        for i in range(self.steps):
            v_h = self.v_cell[i](e_v, v_h)

            v_h = self.v_norm_l[i](v_h)

            v_h = self.v_dropout(v_h)
                
            v_h_up = vp.unpack(v_h)
            if self.v_combine == 'prod':
                v_e = v_h_up.unsqueeze(1)  * v_h_up.unsqueeze(2)
            elif self.v_combine == 'sum':
                v_e = v_h_up.unsqueeze(1)  + v_h_up.unsqueeze(2)
            if self.v_combine_scale == 'rootd':
                v_e = v_e / np.sqrt(v_h_up.shape[-1])
            elif self.v_combine_scale == 'd':
                v_e = v_e / v_h_up.shape[-1]
            elif self.v_combine_scale == 'norm1':
                v_e = v_e / (torch.sum(v_e, dim=-1) + 1e-4).unsqueeze(-1)
                
            v_e_p = ep.pack(v_e)
            e_h = self.e_cell[i](v_e_p, e_h)

            e_h = self.e_norm_l[i](e_h)
            
            e_h = self.e_dropout(e_h)
            e_h_up = ep.unpack(e_h)
            if self.e_to_v_func == 'mean':
                e_v_up = torch.mean(e_h_up, dim=1)
            elif self.e_to_v_func == 'max':
                e_v_up = goodmax(e_h_up, dim=1)

                

            e_v = vp.pack(e_v_up)

        if self.raw_out:
            return v_h_up, e_h_up
        
        out_e = self.out_e(e_h_up)
        out_v = self.out_v(v_h_up)
        
        if self.out_transform == 'sigmoid':
            out_e = torch.sigmoid(out_e)
            out_v = torch.sigmoid(out_v)
        elif self.out_transform == 'relu':
            out_e = F.relu(out_e)
            out_v = F.relu(out_v)
        return out_v, out_e
