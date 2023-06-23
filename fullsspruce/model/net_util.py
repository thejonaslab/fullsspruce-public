import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, init_std=1e-6, 
                 output_dim = None):
        #print("Creating resnet with input_dim={} hidden_dim={} depth={}".format(input_dim, hidden_dim, depth))
        print("depth=", depth)
        assert(depth >= 0)
        super(ResNet, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        if output_dim is None:
            output_dim = hidden_dim

        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim, init_std) for i in range(depth)])

    def forward(self, input):
        input = input.view(input.size(0), -1)
        x = self.linear_in(input)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.linear_out(x)

class MaskedBatchNorm1d(nn.Module):
    def __init__(self, feature_n):
        """
        Batchnorm1d that skips some rows in the batch 
        """

        super(MaskedBatchNorm1d, self).__init__()
        self.feature_n = feature_n
        self.bn = nn.BatchNorm1d(feature_n)

    def forward(self, x, mask):
        assert x.shape[0] == mask.shape[0]
        assert mask.dim() == 1
        
        bin_mask = mask > 0
        y_i = self.bn(x[bin_mask])
        y = torch.zeros(x.shape, device=x.device)
        y[bin_mask] = y_i
        return y

class MaskedLayerNorm1d(nn.Module):
    def __init__(self, feature_n):
        """
        LayerNorm that skips some rows in the batch 
        """

        super(MaskedLayerNorm1d, self).__init__()
        self.feature_n = feature_n
        self.bn = nn.LayerNorm(feature_n)

    def forward(self, x, mask):
        assert x.shape[0] == mask.shape[0]
        assert mask.dim() == 1
        
        bin_mask = mask > 0
        y_i = self.bn(x[bin_mask])
        y = torch.zeros(x.shape, device=x.device)
        y[bin_mask] = y_i
        return y

class ResidualBlock(nn.Module):
    def __init__(self, dim, noise=1e-6):
        super(ResidualBlock, self).__init__()
        self.noise = noise
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim, bias = False)
        self.l1.bias.data.uniform_(-self.noise,self.noise)
        self.l1.weight.data.uniform_(-self.noise,self.noise) #?!
        self.l2.weight.data.uniform_(-self.noise,self.noise)

    def forward(self, x):
        return x + self.l2(F.relu(self.l1(x)))

class SumLayers(nn.Module):
    """
    Fully-connected layers that sum elements in a set
    """
    def __init__(self, input_D, input_max, filter_n, layer_count):
        super(SumLayers, self).__init__()
        self.fc1 = nn.Linear(input_D, filter_n)
        self.relu1 = nn.ReLU()
        self.fc_blocks = nn.ModuleList([nn.Sequential(nn.Linear(filter_n, filter_n), nn.ReLU()) for _ in range(layer_count-1)])
        
        
    def forward(self, X, present):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        
        xt = X  # .transpose(1, 2)
        x = self.fc1(xt) 
        x = self.relu1(x)
        for fcre in self.fc_blocks:
            x = fcre(x)
        
        x = (present.unsqueeze(-1) * x)
        
        return x.sum(1)

class ResNetRegression(nn.Module):
    def __init__(self, D, block_sizes, INT_D, FINAL_D, 
                 use_batch_norm=False, OUT_DIM=1):
        super(ResNetRegression, self).__init__()

        layers = [nn.Linear(D, INT_D)]

        for block_size in block_sizes:
            layers.append(ResNet(INT_D, INT_D, block_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(INT_D))
        layers.append(nn.Linear(INT_D, FINAL_D))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(FINAL_D, OUT_DIM))
                
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X) 

class ResNetRegressionMaskedBN(nn.Module):
    def __init__(self, D, block_sizes, INT_D, FINAL_D, 
                 OUT_DIM=1, norm='batch', dropout=0.0):
        super(ResNetRegressionMaskedBN, self).__init__()

        layers = [nn.Linear(D, INT_D)]
        usemask = [False]
        for block_size in block_sizes:
            layers.append(ResNet(INT_D, INT_D, block_size))
            usemask.append(False)

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
                usemask.append(False)
            if norm == 'layer':
                layers.append(MaskedLayerNorm1d(INT_D))
                usemask.append(True)
            elif norm == 'batch':
                layers.append(MaskedBatchNorm1d(INT_D))
                usemask.append(True)
        layers.append(nn.Linear(INT_D, OUT_DIM))
        usemask.append(False)
            
        self.layers = nn.ModuleList(layers)
        self.usemask = usemask

    def forward(self, x, mask):
        for l, use_mask in zip(self.layers, self.usemask):
            if use_mask:
                x = l(x, mask)
            else:
                x = l(x)
        return x

def goodmax(x, dim):
    return torch.max(x, dim=dim)[0]

class GraphMatLayer(nn.Module):
    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 dropout=0.0, use_bias=True):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M
        
        if GS != 1 then there will be a per-graph-channel 
        linear layer
        """
        super(GraphMatLayer, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        self.dropout = dropout
        self.dropout_layers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            if use_bias:
                l.bias.data.normal_(0.0, self.noise)
            l.weight.data.normal_(0.0, self.noise) #?!
            self.linlayers.append(l)
            if dropout > 0.0:
                self.dropout_layers.append(nn.Dropout(p=dropout))
            
        #self.r = nn.PReLU()
        self.r = nn.ReLU()
        self.agg_func = agg_func
 
    def forward(self, G, x):
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout > 0:
                y = self.dropout_layers[i](y)
            return y

        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)])
        # this is per-batch-element
        xout = torch.stack([torch.matmul(G[i], multi_x[:, i]) for i in range(x.shape[0])])

        x = self.r(xout)
        if self.agg_func is not None:
            x = self.agg_func(x, dim=1)
        return x
        

class GraphMatLayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast', 
                 layer_config = {}):
        super(GraphMatLayers, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet

        LayerClass = eval(layer_class)
        for li in range(len(output_features_n)):
            if li == 0:
                gl = LayerClass(input_feature_n, output_features_n[0],
                                noise=noise, agg_func=agg_func, GS=GS, 
                                use_bias=not norm or force_use_bias, 
                                **layer_config)
            else:
                gl = LayerClass(output_features_n[li-1], 
                                output_features_n[li], 
                                noise=noise, agg_func=agg_func, GS=GS, 
                                use_bias=not norm or force_use_bias, 
                                **layer_config)
            
            self.gl.append(gl)

        self.norm = norm
        if self.norm is not None:
            if self.norm == 'batch':
                Nlayer = MaskedBatchNorm1d
            elif self.norm == 'layer':
                Nlayer = MaskedLayerNorm1d
            self.bn = nn.ModuleList([Nlayer(f) for f in output_features_n])
            
        
    def forward(self, G, x, input_mask=None):
        for gi, gl in enumerate(self.gl):
            x2 = gl(G, x)
            if self.norm:
                x2 = self.bn[gi](x2.reshape(-1, x2.shape[-1]), 
                                 input_mask.reshape(-1)).reshape(x2.shape)

            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
        

        return x

class GraphMatHighwayLayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 noise=1e-5, agg_func=None):
        super(GraphMatHighwayLayers, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet

        for li in range(len(output_features_n)):
            if li == 0:
                gl = GraphMatLayer(input_feature_n, output_features_n[0],
                                   noise=noise, agg_func=agg_func, GS=GS)
            else:
                gl = GraphMatLayer(output_features_n[li-1], 
                                   output_features_n[li], 
                                   noise=noise, agg_func=agg_func, GS=GS)
            
            self.gl.append(gl)

    def forward(self, G, x):
        highway_out = []
        for gl in self.gl:
            x2 = gl(G, x)
            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
            highway_out.append(x2)

        return x, torch.stack(highway_out, -1)

def batch_diagonal_extract(x):
    BATCH_N, M, _, N = x.shape

    return torch.stack([x[:, i, i, :] for i in range(M)], dim=1)

def parse_agg_func(agg_func):
    if isinstance(agg_func, str):
        if agg_func == 'goodmax':
            return goodmax
        elif agg_func == 'sum':
            return torch.sum
        elif agg_func == 'mean':
            return torch.mean
        else:
            raise NotImplementedError()
    return agg_func

class TukeyBiweight(nn.Module):
    """
    implementation of tukey's biweight loss

    """

    def __init__(self, c):
        self.c = c

    def __call__(self, true, pred):
        c = self.c
        
        
        r = true-pred
        r_abs = torch.abs(r)
        check = (r_abs <= c).float()
        
        sub_th = (1 - (1-(r / c)**2)**3)
        other = 1.0
        #print(true.shape, pred.shape, sub_th
        result = (sub_th * check + 1.0 * (1-check))
        return torch.mean(result * c**2/6.0)

def getLossFromNorm(norm, reduct=None):
    if norm == 'l2':
        if not reduct is None:
            return nn.MSELoss(reduction=reduct)
        else:
            return nn.MSELoss()
    elif norm == 'huber' : 
        if not reduct is None:
            return nn.SmoothL1Loss(reduction=reduct)
        else:
            return nn.SmoothL1Loss()
    elif 'tukeybw' in norm:
        c = float(norm.split('-')[1])
        return TukeyBiweight(c)   
    else:
        return None


class NoUncertainLoss(nn.Module):
    """
    """
    def __init__(self, norm='l2', scale=1.0, **kwargs):
        super(NoUncertainLoss, self).__init__()
        self.loss = getLossFromNorm(norm)
            
        self.scale = scale

    def __call__(self, res, vert_pred, vert_pred_mask,
                 edge_pred, edge_pred_mask, 
                 vert_mask):

        mu = res['shift_mu']
        mask = vert_pred_mask
        
        # assert torch.sum(mask) > 0
        loss = torch.tensor(0.0, device=vert_pred.device)
        if torch.sum(mask) > 0:
            y_masked = vert_pred[mask>0].reshape(-1, 1) * self.scale
            mu_masked = mu[mask>0].reshape(-1, 1) * self.scale
            loss = self.loss(y_masked, mu_masked)

        return loss # self.loss(y_masked, mu_masked)

class MSELogNormalLoss(nn.Module):
    def __init__(self, use_std_term = True, 
                 use_log1p=True, std_regularize=0.0, 
                 std_pow = 2.0):
        super(MSELogNormalLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow

    def __call__(self, y, mu, std):
        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize

        std_term = -0.5 * log(2*np.pi * std**2 ) 
        log_pdf = - (y-mu)**2/(2.0 * std **self.std_pow)
        if self.use_std_term :
            log_pdf += std_term 

        return -log_pdf.mean()


def log_normal_nolog(y, mu, std):
    element_wise =  -(y - mu)**2 / (2*std**2)  - std
    return element_wise

def log_student_t(y, mu, std, v=1.0):
    return -torch.log(1.0 + (y-mu)**2/(v * std)) - std

def log_normal(y, mu, std):
    element_wise =  -(y - mu)**2 / (2*std**2)  - torch.log(std)
    return element_wise

class MSECustomLoss(nn.Module):
    def __init__(self, use_std_term = True, 
                 use_log1p=True, std_regularize=0.0, 
                 std_pow = 2.0):
        super(MSECustomLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow

    def __call__(self, y, mu, std):
        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize

        # std_term = -0.5 * log(2*np.pi * std**self.std_pow ) 
        # log_pdf = - (y-mu)**2/(2.0 * std **self.std_pow)

        # if self.use_std_term :
        #     log_pdf += std_term 

        # return -log_pdf.mean()
        return -log_normal(y, mu, std).mean()

class MaskedMSELoss(nn.Module):
    """
    Masked mean squared error
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()
        
    def __call__(self, y, x, mask):
        x_masked = x[mask>0].reshape(-1, 1)
        y_masked = y[mask>0].reshape(-1, 1)
        return self.mseloss(x_masked, y_masked)

class MaskedMSSELoss(nn.Module):
    """
    Masked mean squared error
    """

    def __init__(self):
        super(MaskedMSSELoss, self).__init__()
        
    def __call__(self, y, x, mask):
        x_masked = x[mask>0].reshape(-1, 1)
        y_masked = y[mask>0].reshape(-1, 1)
        return ((x_masked - y_masked)**4).mean()



class MaskedMSEScaledLoss(nn.Module):
    """
    Masked mean squared error
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()
        
    def __call__(self, y, x, mask):
        x_masked = x[mask>0].reshape(-1, 1)
        y_masked = y[mask>0].reshape(-1, 1)
        return self.mseloss(x_masked, y_masked)


class NormUncertainLoss(nn.Module):
    """
    Masked uncertainty loss
    """
    def __init__(self, 
                 mu_scale = torch.Tensor([1.0]), 
                 std_scale = torch.Tensor([1.0]), 
                 use_std_term = True, 
                 use_log1p=False, std_regularize=0.0, 
                 std_pow = 2.0, **kwargs):
        super(NormUncertainLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow
        self.mu_scale = mu_scale
        self.std_scale = std_scale

    def __call__(self, pred, y,  mask):
        ### NOTE pred is a tuple! 
        mu, std = pred['mu'], pred['std']

        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize


        y_scaled = y / self.mu_scale
        mu_scaled = mu / self.mu_scale
        std_scaled = std / self.std_scale 

        y_scaled_masked = y_scaled[mask>0].reshape(-1, 1)
        mu_scaled_masked = mu_scaled[mask>0].reshape(-1, 1)
        std_scaled_masked = std_scaled[mask>0].reshape(-1, 1)
        # return -log_normal_nolog(y_scaled_masked, 
        #                          mu_scaled_masked, 
        #                          std_scaled_masked).mean()
        return -log_normal_nolog(y_scaled_masked, 
                              mu_scaled_masked, 
                              std_scaled_masked).mean()


class UncertainLoss(nn.Module):
    """
    simple uncertain loss
    """
    def __init__(self, 
                 mu_scale = 1.0, 
                 std_scale = 1.0, 
                 norm = 'l2', 
                 std_regularize = 0.1, 
                 std_pow = 2.0, 
                 std_weight = 1.0, 
                 use_reg_log = False, **kwargs):

        super(UncertainLoss, self).__init__()
        self.mu_scale = mu_scale
        self.std_scale = std_scale
        self.std_regularize = std_regularize
        self.norm = norm

        if norm == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        elif norm == 'huber' : 
            self.loss = nn.SmoothL1Loss(reduction='none')

        self.std_pow = std_pow
        self.std_weight = std_weight
        self.use_reg_log = use_reg_log

    def __call__(self, pred, y,  mask, vert_mask):

        mu, std = pred['mu'], pred['std']

        
        std = std + self.std_regularize


        y_scaled = y / self.mu_scale
        mu_scaled = mu / self.mu_scale
        std_scaled = std / self.std_scale 

        y_scaled_masked = y_scaled[mask>0].reshape(-1, 1)
        mu_scaled_masked = mu_scaled[mask>0].reshape(-1, 1)
        std_scaled_masked = std_scaled[mask>0].reshape(-1, 1)

        sm = std_scaled_masked**self.std_pow
        
        sml = std_scaled_masked 
        if self.use_reg_log:
            sml = torch.log(sml)

        l = self.loss(y_scaled_masked, mu_scaled_masked) / (sm) +  self.std_weight * sml
        return torch.mean(l)

def disagreeLoss(ab_y, ab_mu, ab_mask, exp_y, exp_mu, exp_mask, 
                    loss, dis_loss, 
                    regularize_by = 'batch', 
                    lambda_a = 1.0, 
                    lambda_e = 1.0, 
                    scale=1.0):
    """
    Calculate disagreement loss for a batch by calculating disagreement
    between ground truths, scaling predictions and ground truth accordingly,
    then getting the error between predictions and ground truth. Goal is
    to scale down prediction errors when making predictions where the ground
    truth varies more greatly.

    Inputs:
    ab_y: ground truth for ab initio channel
    ab_mu: prediction on ab initio channel 
    ab_mask: mask for ab initio channel
    exp_y: ground truth for experimental channel
    exp_mu: prediction on experimental channel
    exp_mask: mask for experimental channel
    loss: loss function to use between predictions and ground truth
    dis_loss: loss function to use between ground truth measures
    regularize_by: granularity to apply disagreement to (batch, molecule, atom)
    lambda_a: weighting to give to ab initio channel error
    lambda_e: weighting to give to experimental channel error
    scale: weighting to give to all elements  
    """
    BATCH_N, MOL_N, PRED_N = ab_y.shape

    y_ab_masked = ab_y[ab_mask>0].reshape(-1, 1) * scale
    mu_ab_masked = ab_mu[ab_mask>0].reshape(-1, 1) * scale
    loss_ab = lambda_a*loss(y_ab_masked, mu_ab_masked)

    if regularize_by == 'molecule':
        # Regularize per molecule by rescaling each molecule
        # Regularize between each mol (1 mol per batch) by comparing
        # the ground truths where experimental data exists
        ab_only, exp_only = [], []
        # for i in range(BATCH_N):
        #     if torch.sum(ab_mask[i,:,:]) == 0:
        #         exp_only += [i]
        #     elif torch.sum(exp_mask[i,:,:]) == 0:
        #         ab_only += [i]
        # scales = torch.tensor([lambda_e/(1 + dis_loss(exp_y[i,:,:][exp_mask[i,:,:] > 0], ab_y[i,:,:][exp_mask[i,:,:] > 0]))\
        #                             for i in range(BATCH_N)])
        losses = dis_loss(exp_y, ab_y) * exp_mask
        scales = lambda_e/(1 + losses.sum(dim=(1,2))/exp_mask.sum(dim=(1,2)))
        scales = torch.where(torch.sum(ab_mask,dim=(1,2))==0,torch.ones(BATCH_N).to(ab_mask.device),scales)
        scales = scales.repeat_interleave(MOL_N).reshape(BATCH_N,MOL_N,PRED_N)
        scales = scales.to(exp_y.device)
        y_exp_masked = (exp_y[exp_mask>0].reshape(-1, 1)*scales[exp_mask>0].reshape(-1, 1)) * scale
        mu_exp_masked = (exp_mu[exp_mask>0].reshape(-1, 1)*scales[exp_mask>0].reshape(-1, 1)) * scale
        loss_exp = loss(y_exp_masked, mu_exp_masked)
    elif regularize_by == 'atom':
        # Regularize per atom by rescaling each atom
        ab_only, exp_only = [], []
        # for i in range(BATCH_N):
        #     for j in range(MOL_N):
        #         if torch.sum(ab_mask[i,j,:]) == 0:
        #             exp_only += [(i,j)]
        #         elif torch.sum(exp_mask[i,j,:]) == 0:
        #             ab_only += [(i,j)]
        # scales = torch.tensor([[lambda_e/(1 + dis_loss(exp_y[i,j,:][exp_mask[i,j,:] > 0], ab_y[i,j,:][exp_mask[i,j,:] > 0]))\
        #                             for j in range(MOL_N)]\
        #                             for i in range(BATCH_N)])
        losses = dis_loss(exp_y, ab_y) * exp_mask
        scales = lambda_e/(1 + losses.sum(dim=2)/exp_mask.sum(dim=2))
        scales = torch.where(torch.sum(ab_mask,dim=2)==0,torch.ones((BATCH_N,MOL_N)).to(ab_mask.device),scales)
        scales = scales.reshape(BATCH_N,MOL_N,PRED_N)
        scales = scales.to(exp_y.device)
        y_exp_masked = (exp_y[exp_mask>0].reshape(-1, 1)*scales[exp_mask>0].reshape(-1, 1)) * scale
        mu_exp_masked = (exp_mu[exp_mask>0].reshape(-1, 1)*scales[exp_mask>0].reshape(-1, 1)) * scale
        loss_exp = loss(y_exp_masked, mu_exp_masked)
    else:
        # Regularize per batch by dividing the total loss by the regularization term
        y_exp_masked = exp_y[exp_mask>0].reshape(-1, 1) * scale
        mu_exp_masked = exp_mu[exp_mask>0].reshape(-1, 1) * scale
        y_ab_masked_by_exp = ab_y[exp_mask>0].reshape(-1, 1) * scale
        loss_exp = loss(y_exp_masked, mu_exp_masked)
        loss_comp = 1 + torch.mean(dis_loss(y_exp_masked, y_ab_masked_by_exp))
        loss_exp = (lambda_e*loss_exp)/loss_comp
    return loss_exp + loss_ab

class DisagreementLoss(nn.Module):
    """
    Loss function for disagreement regularization. 
    There should be at least two output channels.
    """
    def __init__(self, norm='l2', scale=1.0, disagree_norm='l2', regularize_by='molecule', lambda_a=1.0, lambda_e=1.0, **kwargs):
        super(DisagreementLoss, self).__init__()
        self.loss = getLossFromNorm(norm)
        self.disagree_loss = getLossFromNorm(disagree_norm,reduct='none')
        self.regularize_by = regularize_by
        self.lambda_a = lambda_a
        self.lambda_e = lambda_e
        self.scale = scale

    def __call__(self, res, vert_pred, vert_pred_mask,
                 edge_pred, edge_pred_mask, 
                 vert_mask):

        mu = res['shift_mu']
        ab_mu = mu[:, :, :1]
        exp_mu = mu[:, :, 1:2]
        mask = vert_pred_mask
        ab_mask = mask[:, :, :1]
        exp_mask = mask[:, :, 1:2]
        y = vert_pred
        ab_y = y[:, :, :1]
        exp_y = y[:, :, 1:2]

        # assert (torch.sum(ab_mask) > 0 or torch.sum(exp_mask) > 0)
        loss = torch.tensor(0.0, device=y.device)
        if torch.sum(ab_mask) == 0 and not torch.sum(exp_mask) == 0:
            # If no ab data available
            y_exp_masked = exp_y[exp_mask>0].reshape(-1, 1) * self.scale
            mu_exp_masked = exp_mu[exp_mask>0].reshape(-1, 1) * self.scale
            loss = self.loss(y_exp_masked, mu_exp_masked)
        elif torch.sum(exp_mask) == 0 and not torch.sum(ab_mask) == 0:
            # If no exp data available
            y_ab_masked = ab_y[ab_mask>0].reshape(-1, 1) * self.scale
            mu_ab_masked = ab_mu[ab_mask>0].reshape(-1, 1) * self.scale
            loss = self.loss(y_ab_masked, mu_ab_masked)
        elif not torch.sum(exp_mask) == 0 and not torch.sum(ab_mask) == 0:
            # If both available
            # y_ab_masked = ab_y[ab_mask>0].reshape(-1, 1) * self.scale
            # mu_ab_masked = ab_mu[ab_mask>0].reshape(-1, 1) * self.scale
            # y_exp_masked = exp_y[exp_mask>0].reshape(-1, 1) * self.scale
            # mu_exp_masked = exp_mu[exp_mask>0].reshape(-1, 1) * self.scale
            # y_ab_masked_by_exp = ab_y[exp_mask>0].reshape(-1, 1) * self.scale
            # loss_exp = self.loss(y_exp_masked, mu_exp_masked)
            # loss_ab = self.loss(y_ab_masked, mu_ab_masked)
            # loss_comp = self.disagree_loss(y_exp_masked, y_ab_masked_by_exp)
            # w_e = 1/(loss_comp)
            loss = disagreeLoss(ab_y, ab_mu, ab_mask, exp_y, exp_mu, exp_mask, self.loss, self.disagree_loss, self.regularize_by, self.lambda_a, self.lambda_e, self.scale)

        return loss # Return regularized loss

class DisagreementLossOneToTwo(nn.Module):
    """
    Loss function for disagreement regularization. 
    There should be one output channel and multiple ground truths.
    """
    def __init__(self, norm='l2', scale=1.0, disagree_norm='l2', regularize_by='molecule', lambda_a=1.0, lambda_e=1.0, **kwargs):
        super(DisagreementLossOneToTwo, self).__init__()
        self.loss = getLossFromNorm(norm)
        self.disagree_loss = getLossFromNorm(disagree_norm,reduct='none')
        self.regularize_by = regularize_by
        self.lambda_a = lambda_a
        self.lambda_e = lambda_e
        self.scale = scale

    def __call__(self, res, vert_pred, vert_pred_mask,
                 edge_pred, edge_pred_mask, 
                 vert_mask):

        mu = res['shift_mu']
        mask = vert_pred_mask
        ab_mask = mask[:, :, :1]
        exp_mask = mask[:, :, 1:2]
        y = vert_pred
        ab_y = y[:, :, :1]
        exp_y = y[:, :, 1:2]
        # assert (torch.sum(ab_mask) > 0 or torch.sum(exp_mask) > 0)
        # loss_exp, loss_ab, w_e = 0, 0, 1.0
        loss = torch.tensor(0.0,device=y.device)
        if torch.sum(ab_mask) == 0 and not torch.sum(exp_mask) == 0:
            # If no ab data available
            y_exp_masked = exp_y[exp_mask>0].reshape(-1, 1) * self.scale
            mu_exp_masked = mu[exp_mask>0].reshape(-1, 1) * self.scale
            loss = self.loss(y_exp_masked, mu_exp_masked)
        elif torch.sum(exp_mask) == 0 and not torch.sum(ab_mask) == 0:
            # If no exp data available
            y_ab_masked = ab_y[ab_mask>0].reshape(-1, 1) * self.scale
            mu_ab_masked = mu[ab_mask>0].reshape(-1, 1) * self.scale
            loss = self.loss(y_ab_masked, mu_ab_masked)
        elif not torch.sum(exp_mask) == 0 and not torch.sum(ab_mask) == 0:
            # If both available
            loss  = disagreeLoss(ab_y, mu, ab_mask, exp_y, mu, exp_mask, self.loss, self.disagree_loss, self.regularize_by, self.lambda_a, self.lambda_e, self.scale)

        return loss # Return regularized loss

class MultiShiftLoss(nn.Module):
    """
    Loss function for a model predicting multiple types of shifts on different channels.
    Options for the channels to also use disagreement regularization.
    """
    def __init__(self, losses, **kwargs):
        """
        Arguments
            Losses: A list of dictionaries defining parameters to set up the loss
                for each shift/channel.
        """
        super(MultiShiftLoss, self).__init__()
        self.losses = []
        self.combine = []
        for l in losses:
            self.losses += [(eval(l['loss_name'])(**l['loss_params']), l['channels'])]
            self.combine += [l['loss_weight']]

    def __call__(self, res, vert_pred, vert_pred_mask,
                 edge_pred, edge_pred_mask, vert_mask):

        mu = res['shift_mu']
        calculated_losses = []
        for loss, channels in self.losses:
            res = {'shift_mu': torch.stack([mu[:,:,i] for i in channels], dim=2)}
            mask = torch.stack([vert_pred_mask[:,:,i] for i in channels], dim=2)
            y = torch.stack([vert_pred[:,:,i] for i in channels], dim=2)
            calculated_losses += [loss(res, y, mask, edge_pred, edge_pred_mask, vert_mask)]
        
        # Loss is linearly combination of calculated losses with weights from arguments
        loss = 0
        for w, l in zip(self.combine, calculated_losses):
            loss += w*l
        return loss

class SimpleLoss(nn.Module):
    """
    """
    def __init__(self, norm='l2', scale=1.0, **kwargs):
        super(SimpleLoss, self).__init__()
        if norm == 'l2':
            self.loss = nn.MSELoss()
        elif norm == 'huber' : 
            self.loss = nn.SmoothL1Loss()
        elif 'tukeybw' in norm:
            c = float(norm.split('-')[1])
            self.loss = TukeyBiweight(c)
            
        self.scale = scale

    def __call__(self, pred,
                 vert_pred, 
                 vert_pred_mask,
                 edge_pred,
                 edge_pred_mask, 
                 ## ADD OTHJERS
                 vert_mask):

        mu = pred['mu'] ## FIXME FOR VERT
        
        assert torch.sum(vert_pred_mask) > 0

        
        y_masked = vert_pred[vert_pred_mask>0].reshape(-1, 1) * self.scale
        mu_masked = mu[vert_pred_mask>0].reshape(-1, 1) * self.scale

        return self.loss(y_masked, mu_masked)

class GraphMatLayerFast(nn.Module):
    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 dropout=False, use_bias=False, 
                 ):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M
        
        if GS != 1 then there will be a per-graph-channel 
        linear layer
        """
        super(GraphMatLayerFast, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            if self.noise == 0:
                if use_bias:
                    l.bias.data.normal_(0.0, 1e-4)
                torch.nn.init.xavier_uniform_(l.weight)
            else:
                if use_bias:
                    l.bias.data.normal_(0.0, self.noise)
                l.weight.data.normal_(0.0, self.noise) #?!
            self.linlayers.append(l)
            
        #self.r = nn.PReLU()
        self.r = nn.LeakyReLU()
        self.agg_func = agg_func
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            return y

        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        xout = torch.einsum("ijkl,jilm->jikm", [G, multi_x])
        xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout
        

class GraphMatLayerFastSCM(nn.Module):
    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 nonlin='relu'
                 ):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M
        
        if GS != 1 then there will be a per-graph-channel 
        linear layer
        """
        super(GraphMatLayerFastSCM, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P)
            if self.noise == 0:
                l.bias.data.normal_(0.0, 1e-4)
                torch.nn.init.xavier_uniform_(l.weight)
            else:
                l.bias.data.normal_(0.0, self.noise)
                l.weight.data.normal_(0.0, self.noise) #?!
            self.linlayers.append(l)
            
        #self.r = nn.PReLU()
        if nonlin == 'relu':
            self.r = nn.ReLU()
        elif nonlin == 'prelu':
            self.r = nn.PReLU()
        elif nonlin == 'selu' :
            self.r = nn.SELU()
        else:
            raise ValueError(nonlin)
        self.agg_func = agg_func
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            return y

        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        xout = torch.einsum("ijkl,jilm->jikm", [G, multi_x])

        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return self.r(xout)

class GraphMatLayerFastPow(nn.Module):
    def __init__(self, C, P, GS=1,  
                 mat_pow = 1, 
                 mat_diag = False,
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 nonlin = None, 
                 dropout = 0.0, 
                 norm_by_neighbors=False, 
                 ):
        """

        """
        super(GraphMatLayerFastPow, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            self.linlayers.append(l)
        self.dropout_rate = dropout

        if self.dropout_rate > 0:
            self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(GS)])

        #self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.nonlin == 'sigmoid':
            self.r = nn.Sigmoid()
        elif self.nonlin == 'tanh':
            self.r = nn.Tanh()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f'unknown nonlin {nonlin}')
            
        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout_rate > 0.0:
                y = self.dropout_layers[i](y)
            return y
        Gprod = G
        for mp in range(self.mat_pow -1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod
            
        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])

        if self.norm_by_neighbors:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout


class GraphMatLayerFastPowSwap(nn.Module):
    def __init__(self, C, P, GS=1,  
                 mat_pow = 1, 
                 mat_diag = False,
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 nonlin = None, 
                 dropout = 0.0, 
                 norm_by_neighbors=False, 
                 ):
        """

        """
        super(GraphMatLayerFastPowSwap, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            self.linlayers.append(l)
        self.dropout_rate = dropout

        if self.dropout_rate > 0:
            self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(GS)])

        #self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f'unknown nonlin {nonlin}')
            
        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout_rate > 0.0:
                y = self.dropout_layers[i](y)
            return y
        Gprod = G
        for mp in range(self.mat_pow -1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod
            
        # multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        # xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])
        # print("x.shape=", x.shape, "multi_x.shape=", multi_x.shape, 
        #       "Gprod.shape=", Gprod.shape, "xout.shape=", xout.shape)
        
        x_adj = torch.einsum("ijkl,ilm->jikm", [Gprod, x])
        xout = torch.stack([apply_ll(i, x_adj[i]) for i in range(self.GS)])
        # print("\nx.shape=", x.shape, 
        #       "x_adj.shape=", x_adj.shape, 
        #       "Gprod.shape=", Gprod.shape, 
        #       "xout.shape=", xout.shape)
        
                              
        if self.norm_by_neighbors:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout

    
class GraphMatLayerFastPowSingleLayer(nn.Module):
    def __init__(self, C, P, GS=1,  
                 mat_pow = 1, 
                 mat_diag = False,
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 nonlin = None, 
                 dropout = 0.0, 
                 norm_by_neighbors=False, 
                 ):
        """

        """
        super(GraphMatLayerFastPowSingleLayer, self).__init__()

        self.GS = GS
        self.noise=noise

        self.l = nn.Linear(C, P, bias=use_bias)
        self.dropout_rate = dropout

        # if self.dropout_rate > 0:
        #     self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(GS)])

        #self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f'unknown nonlin {nonlin}')
            
        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(x):
            y = self.l(x)
            if self.dropout_rate > 0.0:
                y = self.dropout_layers(y)
            return y
        Gprod = G
        for mp in range(self.mat_pow -1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod
            
        # multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        # xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])
        # print("x.shape=", x.shape, "multi_x.shape=", multi_x.shape, 
        #       "Gprod.shape=", Gprod.shape, "xout.shape=", xout.shape)
        
        x_adj = torch.einsum("ijkl,ilm->jikm", [Gprod, x])
        xout = torch.stack([apply_ll(x_adj[i]) for i in range(self.GS)])
        # print("\nx.shape=", x.shape, 
        #       "x_adj.shape=", x_adj.shape, 
        #       "Gprod.shape=", Gprod.shape, 
        #       "xout.shape=", xout.shape)
        
                              
        if self.norm_by_neighbors:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout

class GraphMatLayerExpression(nn.Module):
    def __init__(self, C, P, GS=1,  
                 terms = [{'power': 1, 'diag' : False}], 
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 nonlin = 'leakyrelu', 
                 per_nonlin = None,
                 dropout = 0.0,
                 cross_term_agg_func = 'sum', 
                 norm_by_neighbors=False, 
                 ):
        """
        Terms: [{'power': 3, 'diag': False}]
        
        """
    
        super(GraphMatLayerExpression, self).__init__()

        self.pow_ops = nn.ModuleList()
        for t in terms:
            l = GraphMatLayerFastPow(C, P, GS, 
                                     mat_pow = t.get('power', 1), 
                                     mat_diag = t.get('diag', False), 
                                     noise = noise, 
                                     use_bias = use_bias, 
                                     nonlin = t.get('nonlin', per_nonlin), 
                                     norm_by_neighbors=norm_by_neighbors, 
                                     dropout = dropout)
            self.pow_ops.append(l)
            
        self.nonlin = nonlin
        if self.nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.nonlin == 'relu':
            self.r = nn.ReLU()
        elif self.nonlin == 'sigmoid':
            self.r = nn.Sigmoid()
        elif self.nonlin == 'tanh':
            self.r = nn.Tanh()
            
        self.agg_func = agg_func
        self.cross_term_agg_func = cross_term_agg_func
        self.norm_by_neighbors = norm_by_neighbors

    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape    
        
        terms_stack = torch.stack([l(G, x) for l in self.pow_ops], dim=-1)

        if self.cross_term_agg_func == 'sum':
            xout = torch.sum(terms_stack, dim=-1)
        elif self.cross_term_agg_func == 'max':
            xout = torch.max(terms_stack, dim=-1)[0]
        elif self.cross_term_agg_func == 'prod':
            xout = torch.prod(terms_stack, dim=-1)
        else:
            raise ValueError(f"unknown cross term agg func {self.cross_term_agg_func}")
        
        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout


def apply_masked_1d_norm(norm, x, mask):
    """
    Apply one of these norms and do the reshaping
    """
    F_N = x.shape[-1]
    x_flat = x.reshape(-1, F_N)
    mask_flat = mask.reshape(-1)
    out_flat = norm(x_flat, mask_flat)
    out = out_flat.reshape(*x.shape)
    return out

class GraphMatPerBondType(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast', 
                 layer_config = {}):
        super(GraphMatPerBondType, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet
        self.GS = GS
        self.agg_func = agg_func
        
        LayerClass = eval(layer_class)
        for li in range(len(output_features_n)):
            per_chan_l = nn.ModuleList()
            for c_i in range(GS):
                
                if li == 0:
                    gl = LayerClass(input_feature_n, output_features_n[0],
                                    noise=noise, agg_func=None, GS=1, 
                                    use_bias=not norm or force_use_bias, 
                                    **layer_config)
                else:
                    gl = LayerClass(output_features_n[li-1], 
                                    output_features_n[li], 
                                    noise=noise, agg_func=None, GS=1, 
                                    use_bias=not norm or force_use_bias, 
                                    **layer_config)
                per_chan_l.append(gl)
            self.gl.append(per_chan_l)

        self.norm = norm
        if self.norm is not None:
            if self.norm == 'batch':
                Nlayer = MaskedBatchNorm1d
            elif self.norm == 'layer':
                Nlayer = MaskedLayerNorm1d
            
            self.bn = nn.ModuleList([nn.ModuleList([Nlayer(f) for _ in range(GS)]) for f in output_features_n])
            
    def forward(self, G, x, input_mask=None):
        x_per_chan = [x] * self.GS
        for gi, gl in enumerate(self.gl):
            for c_i in range(self.GS):
                x2 = gl[c_i](G[:, c_i:c_i+1], x_per_chan[c_i]).squeeze()
                if self.norm:
                    x2 = self.bn[gi][c_i](x2.reshape(-1, x2.shape[-1]), 
                                          input_mask.reshape(-1)).reshape(x2.shape)

                if self.resnet and gi > 0:
                    x_per_chan[c_i] = x_per_chan[c_i] + x2
                else:
                    x_per_chan[c_i] = x2
                    


        x_agg = torch.stack(x_per_chan, 1)
        x_out = self.agg_func(x_agg, 1)
        
        return x_out



class GraphMatPerBondTypeDebug(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast', 
                 layer_config = {}):
        super(GraphMatPerBondTypeDebug, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet
        self.GS = GS
        self.agg_func = agg_func
        
        LayerClass = eval(layer_class)
        for li in range(len(output_features_n)):
            per_chan_l = nn.ModuleList()
            for c_i in range(GS):
                
                if li == 0:
                    gl = LayerClass(input_feature_n, output_features_n[0],
                                    noise=noise, agg_func=None, GS=1, 
                                    use_bias=not norm or force_use_bias, 
                                    **layer_config)
                else:
                    gl = LayerClass(output_features_n[li-1], 
                                    output_features_n[li], 
                                    noise=noise, agg_func=None, GS=1, 
                                    use_bias=not norm or force_use_bias, 
                                    **layer_config)
                per_chan_l.append(gl)
            self.gl.append(per_chan_l)

        self.norm = norm
        if self.norm is not None:
            if self.norm == 'batch':
                Nlayer = MaskedBatchNorm1d
            elif self.norm == 'layer':
                Nlayer = MaskedLayerNorm1d
            
            self.bn = nn.ModuleList([nn.ModuleList([Nlayer(f) for _ in range(GS)]) for f in output_features_n])
            
        self.final_l = nn.Linear(GS * output_features_n[-1],
                                 output_features_n[-1])
        
    def forward(self, G, x, input_mask=None):
        x_per_chan = [x] * self.GS
        for gi, gl in enumerate(self.gl):
            for c_i in range(self.GS):
                x2 = gl[c_i](G[:, c_i:c_i+1], x_per_chan[c_i]).squeeze()
                if self.norm:
                    x2 = self.bn[gi][c_i](x2.reshape(-1, x2.shape[-1]), 
                                          input_mask.reshape(-1)).reshape(x2.shape)

                if self.resnet and gi > 0:
                    x_per_chan[c_i] = x_per_chan[c_i] + x2
                else:
                    x_per_chan[c_i] = x2
                    


        x_agg = torch.cat(x_per_chan, -1)
        
        x_out = F.relu(self.final_l(x_agg))
        
        return x_out
    

class GraphMatPerBondTypeDebug2(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast', 
                 layer_config = {}):
        super(GraphMatPerBondTypeDebug2, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet
        self.GS = GS
        self.agg_func = agg_func
        
        LayerClass = eval(layer_class)

        self.cross_chan_lin = nn.ModuleList()
        for li in range(len(output_features_n)):
            per_chan_l = nn.ModuleList()
            for c_i in range(GS):
                
                if li == 0:
                    gl = LayerClass(input_feature_n, output_features_n[0],
                                    noise=noise, agg_func=None, GS=1, 
                                    use_bias=not norm or force_use_bias, 
                                    **layer_config)
                else:
                    gl = LayerClass(output_features_n[li-1], 
                                    output_features_n[li], 
                                    noise=noise, agg_func=None, GS=1, 
                                    use_bias=not norm or force_use_bias, 
                                    **layer_config)
                per_chan_l.append(gl)
            self.gl.append(per_chan_l)
            self.cross_chan_lin.append(nn.Linear(GS * output_features_n[li],
                                              output_features_n[li]))

        self.norm = norm
        if self.norm is not None:
            if self.norm == 'batch':
                Nlayer = MaskedBatchNorm1d
            elif self.norm == 'layer':
                Nlayer = MaskedLayerNorm1d
            
            self.bn = nn.ModuleList([nn.ModuleList([Nlayer(f) for _ in range(GS)]) for f in output_features_n])
            
        self.final_l = nn.Linear(GS * output_features_n[-1],
                                 output_features_n[-1])
        
    def forward(self, G, x, input_mask=None):
        x_per_chan = [x] * self.GS
        for gi, gl in enumerate(self.gl):
            x_per_chan_latest = []
            for c_i in range(self.GS):
                x2 = gl[c_i](G[:, c_i:c_i+1], x_per_chan[c_i]).squeeze()
                if self.norm:
                    x2 = self.bn[gi][c_i](x2.reshape(-1, x2.shape[-1]), 
                                          input_mask.reshape(-1)).reshape(x2.shape)
                x_per_chan_latest.append(x2)
                
            x_agg = torch.cat(x_per_chan_latest, -1)

            weight = self.cross_chan_lin[gi](x_agg)
            for c_i in range(self.GS):
                if self.resnet and gi > 0:
                    x_per_chan[c_i] = x_per_chan[c_i] + x_per_chan_latest[c_i] * torch.sigmoid(weight)
                else:
                    x_per_chan[c_i] = x2
                    


        x_agg = torch.cat(x_per_chan, -1)
        
        x_out = F.relu(self.final_l(x_agg))
        
        return x_out
    
def bootstrap_compute(x_1, input_idx, var_eps=1e-5, training=True):
    """
    shape is MIX_N, BATCH_SIZE, ....
    """
    MIX_N = x_1.shape[0]
    BATCH_N = x_1.shape[1]
    
    if training:
        x_zeros = np.zeros(x_1.shape)
        rand_ints = (input_idx % MIX_N).cpu().numpy()
        #print(rand_ints)
        for i, v in enumerate(rand_ints):
            x_zeros[v, i, :, :] = 1
        x_1_sub = torch.Tensor(x_zeros).to(x_1.device) * x_1
        x_1_sub = x_1_sub.sum(dim=0)
    else:
        x_1_sub = x_1.mean(dim=0)
    # x_1_sub = torch.stack([x_1[v, v_i] for v_i, v in enumerate(idx)])
    if MIX_N > 1:
        std = torch.sqrt(torch.var(x_1, dim=0) + var_eps)
    else:
        std = torch.ones_like(x_1_sub) * var_eps
    return x_1_sub, std



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
                x_zeros[rs[j], i, :, :] = 1
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



class PermMinLoss(nn.Module):
    """
    """
    def __init__(self, norm='l2', scale=1.0, **kwargs):
        super(PermMinLoss, self).__init__()
        if norm == 'l2':
            self.loss = nn.MSELoss()
        elif norm == 'huber' : 
            self.loss = nn.SmoothL1Loss()

        self.scale = scale

    def __call__(self, pred, y,  mask, vert_mask):
        
        mu = pred['mu']
        assert mu.shape[2] == 1
        mu = mu.squeeze(-1)

        # pickle.dump({'mu' : mu.cpu().detach(),
        #              'y' : y.squeeze(-1).cpu().detach(), 
        #              'mask' : mask.squeeze(-1).cpu().detach()},
        #             open("/tmp/test.debug", 'wb'))
        y_sorted, mask_sorted = util.min_assign(mu.cpu().detach(), 
                                                y.squeeze(-1).cpu().detach(), 
                                                mask.squeeze(-1).cpu().detach())
        y_sorted = y_sorted.to(y.device)
        mask_sorted = mask_sorted.to(mask.device)
        assert torch.sum(mask) > 0
        assert torch.sum(mask_sorted) > 0
        y_masked = y_sorted[mask_sorted>0].reshape(-1, 1) * self.scale
        mu_masked = mu[mask_sorted>0].reshape(-1, 1) * self.scale
        # print()
        # print("y_masked=", y_masked[:10].cpu().detach().numpy().flatten())
        # print("mu_masked=", mu_masked[:10].cpu().detach().numpy().flatten())

        l = self.loss(y_masked, mu_masked)
        if torch.isnan(l).any():
            print("loss is ", l,
                  y_masked, mu_masked)

        return l

class GraphMatLayerExpressionWNorm(nn.Module):
    def __init__(self, C, P, GS=1,  
                 terms = [{'power': 1, 'diag' : False}], 
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 post_agg_nonlin = None,
                 post_agg_norm = None,
                 per_nonlin = None,
                 dropout = 0.0,
                 cross_term_agg_func = 'sum', 
                 norm_by_neighbors=False, 
                 ):
        """
        """
    
        super(GraphMatLayerExpressionWNorm, self).__init__()

        self.pow_ops = nn.ModuleList()
        for t in terms:
            l = GraphMatLayerFastPow(C, P, GS, 
                                     mat_pow = t.get('power', 1), 
                                     mat_diag = t.get('diag', False), 
                                     noise = noise, 
                                     use_bias = use_bias, 
                                     nonlin = t.get('nonlin', per_nonlin), 
                                     norm_by_neighbors=norm_by_neighbors, 
                                     dropout = dropout)
            self.pow_ops.append(l)
            
        self.post_agg_nonlin = post_agg_nonlin
        if self.post_agg_nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.post_agg_nonlin == 'relu':
            self.r = nn.ReLU()
        elif self.post_agg_nonlin == 'sigmoid':
            self.r = nn.Sigmoid()
        elif self.post_agg_nonlin == 'tanh':
            self.r = nn.Tanh()
            
        self.agg_func = agg_func
        self.cross_term_agg_func = cross_term_agg_func
        self.norm_by_neighbors = norm_by_neighbors
        self.post_agg_norm = post_agg_norm
        if post_agg_norm == 'layer':
            self.pa_norm = nn.LayerNorm(P)

        elif post_agg_norm == 'batch':
            self.pa_norm = nn.BatchNorm1d(P)
            
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape    
        
        terms_stack = torch.stack([l(G, x) for l in self.pow_ops], dim=-1)

        if self.cross_term_agg_func == 'sum':
            xout = torch.sum(terms_stack, dim=-1)
        elif self.cross_term_agg_func == 'max':
            xout = torch.max(terms_stack, dim=-1)[0]
        elif self.cross_term_agg_func == 'prod':
            xout = torch.prod(terms_stack, dim=-1)
        else:
            raise ValueError(f"unknown cross term agg func {self.cross_term_agg_func}")
        
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)

        if self.post_agg_nonlin is not None:
            xout = self.r(xout)
        if self.post_agg_norm is not None:
            xout = self.pa_norm(xout.reshape(-1, xout.shape[-1])).reshape(xout.shape)
            
        return xout


class GraphMatLayerFastPow2(nn.Module):
    def __init__(self, C, P, GS=1,  
                 mat_pow = 1, 
                 mat_diag = False,
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 nonlin = None, 
                 dropout = 0.0, 
                 norm_by_neighbors=False, 
                 ):
        """
        Two layer MLP 

        """
        super(GraphMatLayerFastPow2, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers1 = nn.ModuleList()
        self.linlayers2 = nn.ModuleList()
        
        for ll in range(GS):
            l = nn.Linear(C, P)
            self.linlayers1.append(l)
            l = nn.Linear(P, P)
            self.linlayers2.append(l)
        self.dropout_rate = dropout

        if self.dropout_rate > 0:
            self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(GS)])

        #self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.nonlin == 'sigmoid':
            self.r = nn.Sigmoid()
        elif self.nonlin == 'tanh':
            self.r = nn.Tanh()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f'unknown nonlin {nonlin}')
            
        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(i, x):
            y = F.relu(self.linlayers1[i](x))
            y = self.linlayers2[i](y)
            
            if self.dropout_rate > 0.0:
                y = self.dropout_layers[i](y)
            return y
        Gprod = G
        for mp in range(self.mat_pow -1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod
        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        #print("Gprod.shape=", Gprod.shape, "multi_x.shape=", multi_x.shape)
        xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])

        if self.norm_by_neighbors != False:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            if self.norm_by_neighbors == 'sqrt':
                xout = xout / torch.sqrt(G_neighbors.unsqueeze(-1))
                
            else:
                xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout



class GraphMatLayerExpressionWNorm2(nn.Module):
    def __init__(self, C, P, GS=1,  
                 terms = [{'power': 1, 'diag' : False}], 
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 post_agg_nonlin = None,
                 post_agg_norm = None,
                 per_nonlin = None,
                 dropout = 0.0,
                 cross_term_agg_func = 'sum', 
                 norm_by_neighbors=False, 
                 ):
        """
        """
    
        super(GraphMatLayerExpressionWNorm2, self).__init__()

        self.pow_ops = nn.ModuleList()
        for t in terms:
            l = GraphMatLayerFastPow2(C, P, GS, 
                                     mat_pow = t.get('power', 1), 
                                     mat_diag = t.get('diag', False), 
                                     noise = noise, 
                                     use_bias = use_bias, 
                                     nonlin = t.get('nonlin', per_nonlin), 
                                     norm_by_neighbors=norm_by_neighbors, 
                                     dropout = dropout)
            self.pow_ops.append(l)
            
        self.post_agg_nonlin = post_agg_nonlin
        if self.post_agg_nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.post_agg_nonlin == 'relu':
            self.r = nn.ReLU()
        elif self.post_agg_nonlin == 'sigmoid':
            self.r = nn.Sigmoid()
        elif self.post_agg_nonlin == 'tanh':
            self.r = nn.Tanh()
            
        self.agg_func = agg_func
        self.cross_term_agg_func = cross_term_agg_func
        self.norm_by_neighbors = norm_by_neighbors
        self.post_agg_norm = post_agg_norm
        if post_agg_norm == 'layer':
            self.pa_norm = nn.LayerNorm(P)

        elif post_agg_norm == 'batch':
            self.pa_norm = nn.BatchNorm1d(P)
            
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape    
        
        terms_stack = torch.stack([l(G, x) for l in self.pow_ops], dim=-1)

        if self.cross_term_agg_func == 'sum':
            xout = torch.sum(terms_stack, dim=-1)
        elif self.cross_term_agg_func == 'max':
            xout = torch.max(terms_stack, dim=-1)[0]
        elif self.cross_term_agg_func == 'prod':
            xout = torch.prod(terms_stack, dim=-1)
        else:
            raise ValueError(f"unknown cross term agg func {self.cross_term_agg_func}")
        
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)

        if self.post_agg_nonlin is not None:
            xout = self.r(xout)
        if self.post_agg_norm is not None:
            xout = self.pa_norm(xout.reshape(-1, xout.shape[-1])).reshape(xout.shape)
            
        return xout


def create_nonlin(nonlin):
    if nonlin == 'leakyrelu':
        r = nn.LeakyReLU()
    elif nonlin == 'sigmoid':
        r = nn.Sigmoid()
    elif nonlin == 'tanh':
        r = nn.Tanh()
    elif nonlin == 'relu':
        r = nn.ReLU()
    elif nonlin == 'identity':
        r = nn.Identity()
    else:
        raise ValueError(f'unknown nonlin {nonlin}')
    
    return r

class GCNLDLayer(nn.Module):
    def __init__(self, C, P, GS=1,  
                 terms = [{'power': 1, 'diag' : False}], 
                 mlp_config = {'layer_n': 1,
                               'nonlin' : 'leakyrelu'}, 
                 chanagg = 'pre',
                 dropout = 0.0,
                 learn_w = True,
                 norm_by_degree = False, 
                 **kwargs
                 ):
        """
        """
        super(GCNLDLayer, self).__init__()
        self.terms = terms
        self.C = C
        self.P = P
        if learn_w:
            self.scalar_weights = nn.Parameter(torch.zeros(len(terms)))
        else:
            self.scalar_weights = torch.zeros(len(terms))
        
        self.chanagg = chanagg
        self.norm_by_degree = norm_by_degree
        
         
        if self.chanagg == 'cat':
            self.out_lin = MLP(input_d = C * GS,
                               output_d = P,
                               d = P, 
                               **mlp_config)
        else:
            self.out_lin = MLP(input_d = C,
                               output_d = P,
                               d = P, 
                               **mlp_config)


        self.dropout_p = dropout
        if self.dropout_p > 0:
            
            self.dropout = nn.Dropout(p=dropout)
        
    def mpow(self, G, k):
        Gprod = G
        for i in range(k-1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        return Gprod

    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape    
        
        # first compute each power
        Gdiag = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device)

        G_terms = torch.zeros_like(G)
        for ti, t in enumerate(self.terms):
            G_pow = self.mpow(G, t['power'])
            if t.get('diag', False):
                G_pow = G_pow * Gdiag
            G_terms = G_terms + G_pow * torch.sigmoid(self.scalar_weights[ti])

        Xp = G_terms @ x.unsqueeze(1)

        # normalization
        G_norm = torch.clamp(G.sum(dim=-1), min=1)
        if self.norm_by_degree:
            Xp = Xp / G_norm.unsqueeze(-1)
        
        
        if self.chanagg  == 'cat':
            a = Xp.permute(0, 2, 3, 1)
            Xp = a.reshape(a.shape[0], a.shape[1], -1)
        X = self.out_lin(Xp)
        if self.dropout_p > 0:
            X = self.dropout(X)
    
        if self.chanagg == 'goodmax':
            X = goodmax(X, 1)
        
        return X
    


class MLP(nn.Module):
    def __init__(self, layer_n=1, d=128,
                 input_d = None,
                 output_d = None,
                 nonlin='relu',
                 final_nonlin=True,
                 use_bias=True):
        super(MLP, self).__init__()

        ml = []
        for i in range(layer_n):
            in_d = d
            out_d = d
            if i == 0 and input_d is not None:
                in_d = input_d
            if (i == (layer_n -1)) and output_d is not None:
                out_d = output_d
            
            linlayer = nn.Linear(in_d, out_d, use_bias)

            ml.append(linlayer)
            nonlin_layer = create_nonlin(nonlin)
            if i == (layer_n -1) and not final_nonlin:
                pass
            else:
                ml.append(nonlin_layer)
        self.ml = nn.Sequential(*ml)
    def forward(self, x):
        return self.ml(x)
            
    
class GCNLDLinPerChanLayer(nn.Module):
    def __init__(self, C, P, GS=1,  
                 terms = [{'power': 1, 'diag' : False}], 
                 nonlin = 'leakyrelu', 
                 chanagg = 'pre',
                 dropout = 0.0,
                 learn_w = True,
                 norm_by_degree = 'degree', 
                 w_transform = 'sigmoid', 
                 mlp_config = {'layer_n': 1,
                               'nonlin' : 'leakyrelu'}, 
                 **kwargs
                 ):
        """
        """
        super(GCNLDLinPerChanLayer, self).__init__()
        self.terms = terms
        self.C = C
        self.P = P
        if learn_w:
            self.scalar_weights = nn.Parameter(torch.zeros(len(terms)))
        else:
            self.scalar_weights = torch.zeros(len(terms))
        
        self.chanagg = chanagg
        
        self.out_lin = nn.ModuleList([MLP(input_d = C,
                                          d = P, 
                                          output_d = P,
                                          **mlp_config) for _ in range(GS)])
        self.w_transform = w_transform
        self.dropout_p = dropout
        if self.dropout_p > 0:
            
            self.dropout = nn.Dropout(p=dropout)

        self.norm_by_degree = norm_by_degree
        
    def mpow(self, G, k):
        Gprod = G
        for i in range(k-1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        return Gprod

    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape    
        
        # first compute each power
        Gdiag = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device)

        G_terms = torch.zeros_like(G)
        for ti, t in enumerate(self.terms):

            if self.w_transform == 'sigmoid':
                w = torch.sigmoid(self.scalar_weights[ti])
            elif self.w_transform == 'tanh':
                w = torch.tanh(self.scalar_weights[ti])
                
            G_pow = self.mpow(G, t['power'])
            if t.get('diag', False):
                G_pow = G_pow * Gdiag
            G_terms = G_terms + G_pow * w


        # normalization
        if self.norm_by_degree == 'degree':
            
            G_norm = torch.clamp(G.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)
        elif self.norm_by_degree == 'total':

            G_norm = torch.clamp(G_terms.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)

        Xp = G_terms @ x.unsqueeze(1)


        

        XP0 = Xp.permute(1, 0, 2, 3)
        
        X = [l(x) for l, x in zip(self.out_lin, XP0)]
        X = torch.stack(X)
        if self.dropout_p > 0:
            X = self.dropout(X)
            
        if self.chanagg == 'goodmax':
            X = goodmax(X, 0)
        
        return X
    
    
class GCNLDLinPerChanLayerDEBUG(nn.Module):
    def __init__(self, C, P, GS=1,  
                 terms = [{'power': 1, 'diag' : False}], 
                 nonlin = 'leakyrelu', 
                 chanagg = 'pre',
                 dropout = 0.0,
                 learn_w = True,
                 norm_by_degree = 'degree', 
                 w_transform = 'sigmoid', 
                 mlp_config = {'layer_n': 1,
                               'nonlin' : 'leakyrelu'}, 
                 **kwargs
                 ):
        """
        """
        super(GCNLDLinPerChanLayerDEBUG, self).__init__()
        self.terms = terms
        self.C = C
        self.P = P
        if learn_w:
            self.scalar_weights = nn.Parameter(torch.zeros(len(terms)))
        else:
            self.scalar_weights = torch.zeros(len(terms))
        
        self.chanagg = chanagg
        
        self.out_lin = nn.ModuleList([MLP(input_d = C,
                                          d = P, 
                                          output_d = P,
                                          **mlp_config) for _ in range(GS)])
        self.w_transform = w_transform
        self.dropout_p = dropout
        if self.dropout_p > 0:
            
            self.dropout = nn.Dropout(p=dropout)

        self.norm_by_degree = norm_by_degree

        
    def mpow(self, G, k):
        Gprod = G
        for i in range(k-1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        return Gprod

    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape    

        G_embed = self.chan_embed(G.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # first compute each power
        Gdiag = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device)

        G_terms = torch.zeros_like(G)
        for ti, t in enumerate(self.terms):

            if self.w_transform == 'sigmoid':
                w = torch.sigmoid(self.scalar_weights[ti])
            elif self.w_transform == 'tanh':
                w = torch.tanh(self.scalar_weights[ti])
                
            G_pow = self.mpow(G, t['power'])
            if t.get('diag', False):
                G_pow = G_pow * Gdiag
            G_terms = G_terms + G_pow * w


        # normalization
        if self.norm_by_degree == 'degree':
            
            G_norm = torch.clamp(G.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)
        elif self.norm_by_degree == 'total':

            G_norm = torch.clamp(G_terms.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)


        X = [l(x) for l in self.out_lin]
        X = torch.stack(X, 1)
        #print("X.shape=", X.shape, "G_terms.shape=", G_terms.shape)
        #X = torch.clamp(G_terms, max=1) @ X
        X = G_terms @ X
        
        # Xp = G_terms @ x.unsqueeze(1)

        # XP0 = Xp.permute(1, 0, 2, 3)
        # X = [l(x) for l, x in zip(self.out_lin, XP0)]

        #print("Xout.shape=", X.shape)
        #lkhasdlsaj
        if self.dropout_p > 0:
            X = self.dropout(X)
            
        if self.chanagg == 'goodmax':
            X = goodmax(X, 1)
        elif self.chanagg == 'sum':
            X = torch.sum(X, 1)
        elif self.chanagg == 'mean':
            X = torch.mean(X, 1)
        
        return X
    

class GCNLDLinPerChanLayerEdgeEmbed(nn.Module):
    def __init__(self, C, P, GS=1,  
                 terms = [{'power': 1, 'diag' : False}], 
                 nonlin = 'leakyrelu', 
                 chanagg = 'pre',
                 dropout = 0.0,
                 learn_w = True,
                 embed_dim_multiple = 1,
                 embed_transform = None, 
                 norm_by_degree = 'degree', 
                 w_transform = 'sigmoid', 
                 mlp_config = {'layer_n': 1,
                               'nonlin' : 'leakyrelu'}, 
                 **kwargs
                 ):
        """
        """
        super(GCNLDLinPerChanLayerEdgeEmbed, self).__init__()
        self.terms = terms
        self.C = C
        self.P = P
        if learn_w:
            self.scalar_weights = nn.Parameter(torch.zeros(len(terms)))
        else:
            self.scalar_weights = torch.zeros(len(terms))
        
        self.chanagg = chanagg


        self.chan_embed = nn.Linear(GS, GS*embed_dim_multiple)
        
        self.out_lin = nn.ModuleList([MLP(input_d = C,
                                          d = P, 
                                          output_d = P,
                                          **mlp_config) for _ in range(GS*embed_dim_multiple)])
        self.w_transform = w_transform
        self.dropout_p = dropout
        if self.dropout_p > 0:
            
            self.dropout = nn.Dropout(p=dropout)

        self.norm_by_degree = norm_by_degree
        self.embed_transform = embed_transform

        
        
    def mpow(self, G, k):
        Gprod = G
        for i in range(k-1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        return Gprod

    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape    

        G_embed = self.chan_embed(G.permute(0, 2, 3, 1))
        if self.embed_transform == 'sigmoid':
            G_embed = torch.sigmoid(G_embed)
        elif self.embed_transform == 'softmax':
            G_embed = torch.softmax(G_embed, -1)
        
        G = G_embed.permute(0, 3, 1, 2)
        # first compute each power
        Gdiag = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device)

        G_terms = torch.zeros_like(G)
        for ti, t in enumerate(self.terms):

            if self.w_transform == 'sigmoid':
                w = torch.sigmoid(self.scalar_weights[ti])
            elif self.w_transform == 'tanh':
                w = torch.tanh(self.scalar_weights[ti])
                
            G_pow = self.mpow(G, t['power'])
            if t.get('diag', False):
                G_pow = G_pow * Gdiag
            G_terms = G_terms + G_pow * w


        # normalization
        if self.norm_by_degree == 'degree':
            
            G_norm = torch.clamp(G.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)
        elif self.norm_by_degree == 'total':

            G_norm = torch.clamp(G_terms.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)



        X = [l(x) for l in self.out_lin]
        X = torch.stack(X, 1)

        if self.dropout_p > 0:
            X = self.dropout(X)
        
        #print("X.shape=", X.shape, "G_terms.shape=", G_terms.shape)
        #X = torch.clamp(G_terms, max=1) @ X
        X = G_terms @ X
        
        # Xp = G_terms @ x.unsqueeze(1)

        # XP0 = Xp.permute(1, 0, 2, 3)
        # X = [l(x) for l, x in zip(self.out_lin, XP0)]

        #print("Xout.shape=", X.shape)
        #lkhasdlsaj
        # if self.dropout_p > 0:
        #     X = self.dropout(X)
            
        if self.chanagg == 'goodmax':
            X = goodmax(X, 1)
        elif self.chanagg == 'sum':
            X = torch.sum(X, 1)
        elif self.chanagg == 'mean':
            X = torch.mean(X, 1)
        
        return X
    



class GCNLDLinPerChanLayerAttn(nn.Module):
    def __init__(self, C, P, GS=1,  
                 terms = [{'power': 1, 'diag' : False}], 
                 nonlin = 'leakyrelu', 
                 chanagg = 'pre',
                 dropout = 0.0,
                 #learn_w = True,
                 norm_by_degree = 'degree', 
                 #w_transform = 'sigmoid', 
                 mlp_config = {'layer_n': 1,
                               'nonlin' : 'leakyrelu'}, 
                 **kwargs
                 ):
        """
        """
        super(GCNLDLinPerChanLayerAttn, self).__init__()
        self.terms = terms
        self.C = C
        self.P = P
        # if learn_w:
        #     self.scalar_weights = nn.Parameter(torch.zeros(len(terms)))
        # else:
        #     self.scalar_weights = torch.zeros(len(terms))
        
        self.chanagg = chanagg
        
        self.out_lin = nn.ModuleList([MLP(input_d = C,
                                          d = P, 
                                          output_d = P,
                                          **mlp_config) for _ in range(GS)])
        #self.w_transform = w_transform
        self.dropout_p = dropout
        if self.dropout_p > 0:
            
            self.dropout = nn.Dropout(p=dropout)

        self.norm_by_degree = norm_by_degree

        self.term_attn = MLP(input_d=self.C,
                                d = 128,
                                layer_n = 3, 
                                output_d = len(terms),
                                final_nonlin = False)
        
    def mpow(self, G, k):
        Gprod = G
        for i in range(k-1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        return Gprod

    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape    
        
        # first compute each power
        Gdiag = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device)

        G_terms = []
        for ti, t in enumerate(self.terms):

            # if self.w_transform == 'sigmoid':
            #     w = torch.sigmoid(self.scalar_weights[ti])
            # elif self.w_transform == 'tanh':
            #     w = torch.tanh(self.scalar_weights[ti])
                
            G_pow = self.mpow(G, t['power'])
            if t.get('diag', False):
                G_pow = G_pow * Gdiag
            G_terms.append(G_pow)

            


        # normalization
        if self.norm_by_degree == 'degree':
            
            G_norm = torch.clamp(G.sum(dim=-1), min=1)
            G_terms = [G_term / G_norm.unsqueeze(-1) for G_term in G_terms]
        elif self.norm_by_degree == 'total':

            G_norm = torch.clamp(G_terms.sum(dim=-1), min=1)
            G_terms = [G_term / G_norm.unsqueeze(-1) for G_term in G_terms]

        


        X = [l(x) for l in self.out_lin]
        X = torch.stack(X, 1)

        if self.dropout_p > 0:
            X = self.dropout(X)
            
        
        attention = torch.softmax(self.term_attn(x), -1)
        #attention = torch.sigmoid(self.term_attn(x))
        #print("X.shape=", X.shape, "G_terms.shape=", G_terms.shape)
        Xterms = torch.stack([G_term @ X for G_term in G_terms], -1)
        attention = attention.unsqueeze(1).unsqueeze(3)
        #print("Xterms.shape=", Xterms.shape,
        #      "attention.shape=", attention.shape)
        X = (Xterms * attention).sum(dim=-1)
                
        # Xp = G_terms @ x.unsqueeze(1)

        # XP0 = Xp.permute(1, 0, 2, 3)
        # X = [l(x) for l, x in zip(self.out_lin, XP0)]

        #print("Xout.shape=", X.shape)
        #lkhasdlsaj
        if self.chanagg == 'goodmax':
            X = goodmax(X, 1)
        
        return X

class GraphMatLayersDebug(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast',
                 intra_layer_dropout_p = 0.0, 
                 layer_config = {}):
        super(GraphMatLayersDebug, self).__init__()
        
        self.gl = nn.ModuleList()
        self.dr = nn.ModuleList()
        self.resnet = resnet

        LayerClass = eval(layer_class)
        for li in range(len(output_features_n)):
            if li == 0:
                gl = LayerClass(input_feature_n, output_features_n[0],
                                noise=noise, agg_func=agg_func, GS=GS, 
                                use_bias=not norm or force_use_bias, 
                                **layer_config)
            else:
                gl = LayerClass(output_features_n[li-1], 
                                output_features_n[li], 
                                noise=noise, agg_func=agg_func, GS=GS, 
                                use_bias=not norm or force_use_bias, 
                                **layer_config)
            
            self.gl.append(gl)
            if intra_layer_dropout_p > 0:
                dr = nn.Dropout(intra_layer_dropout_p)
            else:
                dr = nn.Identity()
            self.dr.append(dr)

        self.norm = norm
        if self.norm is not None:
            if self.norm == 'batch':
                Nlayer = MaskedBatchNorm1d
            elif self.norm == 'layer':
                Nlayer = MaskedLayerNorm1d
            self.bn = nn.ModuleList([Nlayer(f) for f in output_features_n])
            
        
    def forward(self, G, x, input_mask=None):
        for gi, gl in enumerate(self.gl):
            x2 = gl(G, x)
            if self.norm:
                x2 = self.bn[gi](x2.reshape(-1, x2.shape[-1]), 
                                 input_mask.reshape(-1)).reshape(x2.shape)

            x2 = x2 * self.dr[gi](input_mask).unsqueeze(-1)
                    
            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
        

        return x

    
class GraphMatLayerExpressionWNormAfter2(nn.Module):
    def __init__(self, C, P, GS=1,  
                 terms = [{'power': 1, 'diag' : False}], 
                 noise=1e-6, agg_func=None, 
                 use_bias=False, 
                 post_agg_nonlin = None,
                 post_agg_norm = None,
                 per_nonlin = None,
                 dropout = 0.0,
                 cross_term_agg_func = 'sum', 
                 norm_by_neighbors=False, 
                 ):
        """
        """
    
        super(GraphMatLayerExpressionWNormAfter2, self).__init__()

        self.pow_ops = nn.ModuleList()
        for t in terms:
            l = GraphMatLayerFastPow2(C, P, GS, 
                                     mat_pow = t.get('power', 1), 
                                     mat_diag = t.get('diag', False), 
                                     noise = noise, 
                                     use_bias = use_bias, 
                                     nonlin = t.get('nonlin', per_nonlin), 
                                     norm_by_neighbors=norm_by_neighbors, 
                                     dropout = dropout)
            self.pow_ops.append(l)
            
        self.post_agg_nonlin = post_agg_nonlin
        if self.post_agg_nonlin == 'leakyrelu':
            self.r = nn.LeakyReLU()
        elif self.post_agg_nonlin == 'relu':
            self.r = nn.ReLU()
        elif self.post_agg_nonlin == 'sigmoid':
            self.r = nn.Sigmoid()
        elif self.post_agg_nonlin == 'tanh':
            self.r = nn.Tanh()
            
        self.agg_func = agg_func
        self.cross_term_agg_func = cross_term_agg_func
        self.norm_by_neighbors = norm_by_neighbors
        self.post_agg_norm = post_agg_norm
        if post_agg_norm == 'layer':
            self.pa_norm = nn.LayerNorm(P)

        elif post_agg_norm == 'batch':
            self.pa_norm = nn.BatchNorm1d(P)
            
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape    
        
        terms_stack = torch.stack([l(G, x) for l in self.pow_ops], dim=-1)

        if self.cross_term_agg_func == 'sum':
            xout = torch.sum(terms_stack, dim=-1)
        elif self.cross_term_agg_func == 'max':
            xout = torch.max(terms_stack, dim=-1)[0]
        elif self.cross_term_agg_func == 'prod':
            xout = torch.prod(terms_stack, dim=-1)
        else:
            raise ValueError(f"unknown cross term agg func {self.cross_term_agg_func}")
        
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)

        if self.post_agg_norm is not None:
            xout = self.pa_norm(xout.reshape(-1, xout.shape[-1])).reshape(xout.shape)
        if self.post_agg_nonlin is not None:
            xout = self.r(xout)
            
        return xout
