import torch

import pickle
import copy

import torch
import torch.autograd
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import os
import shutil
from fullsspruce import util
from fullsspruce.model import nets
from fullsspruce.model import coupling

default_atomicno = [1, 6, 7, 8, 9, 15, 16, 17]

### Create datasets and data loaders

default_feat_vect_args = dict(feat_atomicno_onehot=default_atomicno, 
                              feat_pos=False, feat_atomicno=True,
                              feat_valence=True, aromatic=True, 
                              hybridization=True, 
                              partial_charge=False, formal_charge=True,  # WE SHOULD REALLY USE THIS 
                              r_covalent=False,
                              total_valence_onehot=True, 
                              mmff_atom_types_onehot =False, 
                              r_vanderwals=False, 
                              default_valence=True, rings=True)

default_feat_edge_args = dict(feat_distances = False, 
                             feat_r_pow = None)

default_feat_bond_args = dict()

default_feat_mol_geom_args = dict()

default_split_weights = [1, 1.5, 2, 3]

default_adj_args = dict(edge_weighted=False, 
                        norm_adj=True, add_identity=True, 
                        split_weights=default_split_weights)


default_mol_args = dict() # possible_solvents= ['CDCl3', 'DMSO-d6', 'D2O', 'CCl4'])

default_dist_mat_args = dict()

DEFAULT_DATA_HPARAMS = {'feat_vect_args' : default_feat_vect_args, 
                        'feat_edge_args' : default_feat_edge_args, 
                        'feat_bond_args': default_feat_bond_args,
                        'feat_mol_geom_args' : default_feat_mol_geom_args, 
                        'adj_args' : default_adj_args,
                        'mol_args' : default_mol_args,
                        'dist_mat_args' : default_dist_mat_args,
                        'coupling_args' : {'compute_coupling' : False}}


class CVSplit:
    def __init__(self, how, **args):
        self.how = how
        self.args = args

    def get_phase(self, mol, fp):
        if self.how == 'morgan_fingerprint_mod':
            mod = self.args['mod']
            test = self.args['test']

            if (fp['morgan_fingerprint'] % mod) in test:
                return 'test'
            else:
                return 'train'
        elif self.how == 'morgan_fingerprint_list':
            train = pickle.load(open(self.args['train'], 'rb'))

            if fp['morgan_fingerprint'] in train:
                return 'train'
            else:
                return 'test'
        elif self.how == 'smiles_list':
            train = pickle.load(open(self.args['train'], 'rb'))

            if fp['smiles'] in train:
                return 'train'
            else:
                return 'test' 
        else:
            raise ValueError(f"unknown method {self.how}")

def make_dataset(dataset_config, hparams,
                 pred_config, dataset_class,
                 MAX_N,
                 cv_splitter,
                 train_sample=0,
                 passthrough_config = {}):
    """
    """


    
    filename = dataset_config['filename']
    phase = dataset_config.get('phase', 'train')
    dataset_spect_assign = dataset_config.get("spect_assign", True) 
    frac_per_epoch = dataset_config.get('frac_per_epoch', 1.0)
    d = pickle.load(open(filename, 'rb'))
    if dataset_config.get('subsample_to', 0) > 0:
        print("SUBSAMPLE TO", dataset_config.get('subsample_to', 0))
        if len(d) > dataset_config['subsample_to']:
            d = d.sample(dataset_config['subsample_to'],
                         random_state = dataset_config.get('subsample_seed', 0))
    filter_max_n = dataset_config.get('filter_max_n', 0)
    spect_dict_field = dataset_config.get('spect_dict_field', 'spect_dict')
    print("THE SPECT DICT IS", spect_dict_field)
    filter_bond_max_n = dataset_config.get('filter_bond_max_n', 0)


    if filter_max_n > 0:
        d['atom_n'] = d.rdmol.apply(lambda m: m.GetNumAtoms())

        print("filtering for atom max_n <=", filter_max_n, " from", len(d))
        d = d[d.atom_n <= filter_max_n]
        print("after filter length=", len(d))

    if filter_bond_max_n > 0:
        d['bond_n'] = d.rdmol.apply(lambda m: m.GetNumBonds())

        print("filtering for bond max_n <=", filter_bond_max_n, " from", len(d))
        d = d[d.bond_n <= filter_bond_max_n]
        print("after filter length=", len(d))




    # ids = []

    for extra_feature_config in dataset_config.get('extra_features', []):
        # add these to the data frame
        extra_feat_filename = extra_feature_config['filename']
        
        if 'field' in extra_feature_config: # assume it's a dictionary
            feat_dict = pickle.load(open(extra_feat_filename, 'rb'))
            # ids += [set(feat_dict.keys())]
            s = pd.Series(feat_dict)
            s.name = extra_feature_config['field']
            before_join_size = len(d)
            print(d.molecule_id.dtype, s.index.dtype)
            d = d.join(s, on='molecule_id')
            join_size = len(d)
            d = d.dropna()
            drop_size = len(d)
            print(f"field {extra_feature_config['field']} : from {before_join_size} -> {join_size} -> {drop_size}")
        else:
            raise NotImplementedError("don't yet support dataframes")
        
    d_phase = d.apply(lambda row : cv_splitter.get_phase(row.rdmol, 
                                                        {'morgan_fingerprint': row.morgan4_crc32,
                                                        'smiles': row.smiles}), 
                      axis=1)
    
    df = d[d_phase == phase]
    datasets = {}
    other_args = hparams.get('other_args', {})

    records = df.to_dict('records')
    metafile = dataset_config.get('metafile', None)
    # if not metafile is None:
        # records_metafile = pickle.load(open(metafile, 'rb'))
    # if not ids == []:
    #     seen_ids = set.intersection(*ids)
    #     records = [r for r in records if r['molecule_id'] in seen_ids]# records_metafile['ids']]

    ds_c = eval(dataset_class)
    ds = ds_c(#df.rdmol.tolist(), 
                #spect_data,
                records, 
                MAX_N, #num_tgt_nucs, 
                # hparams.get('metafile',None),
                hparams['feat_vect_args'], 
                hparams['feat_edge_args'], 
                hparams['feat_mol_geom_args'], 
                hparams['feat_bond_args'],
                hparams['adj_args'],
                hparams['mol_args'],
                hparams['dist_mat_args'], 
                hparams['coupling_args'],
                pred_config = pred_config,
                passthrough_config = passthrough_config, 
                #extra_npy_filenames = dataset_extra_data,
                frac_per_epoch = frac_per_epoch,
                spect_assign = dataset_spect_assign,
                **other_args
                )

    print(f"{phase} has {len(df)} records")
        
    phase_data = {'mol' : df.rdmol,
                #   'spect' : spect_data,
                  'df' : df}
    return ds, phase_data

def create_checkpoint_func(CHECKPOINT_DIR,
                           exp_output_name,
                           rank,
                           every_n, latest_n,
                           train_sampler=None, test_sampler=None):
    """
        Checkpointing func, only save model state on the rank 0 thread
    """
    def checkpoint_func(epoch_i, net, optimizer, scheduler=None):

        full = False
        if epoch_i % every_n == 0:
            full = True
        elif epoch_i % latest_n != 0:
            return {}

        t1 = time.time()
        
        base_filename = os.path.join(CHECKPOINT_DIR,
                                     f"{exp_output_name}.{epoch_i:08d}")

        latest_filename = os.path.join(CHECKPOINT_DIR, f"{exp_output_name}.latest")
            
        # save the module directly (outside of DDP wrapper)
        if rank == 0 and full:
            torch.save(net.module, base_filename)

        # save common model-and-training-based state
        
        if rank == 0:
            common_checkpoint_state = {'epoch_i' : epoch_i}
            common_checkpoint_state['model_state_dict'] = net.state_dict()
            common_checkpoint_state['optimizer_state_dict'] = optimizer.state_dict()
            
            if scheduler is not None:
                common_checkpoint_state['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(common_checkpoint_state, f"{latest_filename}.common.checkpoint")
            
            if full:
                shutil.copyfile(f"{latest_filename}.common.checkpoint",
                            f"{base_filename}.common.checkpoint")


        # now save other worker-specific state            
        rank_specific_checkpoint_state = {'epoch_i' : epoch_i, 
                                          'sampler_state' :{}}
        if train_sampler is not None:
            rank_specific_checkpoint_state['sampler_state']['train'] = train_sampler.get_state()

        if test_sampler is not None:
            rank_specific_checkpoint_state['sampler_state']['test'] = test_sampler.get_state()            
        
        torch.save(rank_specific_checkpoint_state, f"{latest_filename}.{rank}.checkpoint")
        if full:
            shutil.copyfile(f"{latest_filename}.{rank}.checkpoint",
                        f"{base_filename}.{rank}.checkpoint")
        
        t2 = time.time()
        return {'savetime' : t2-t1}
    
    return checkpoint_func


def run_epoch(net, device, optimizer, criterion, dl, 
              pred_only = False, 
              return_pred = False, desc="train", 
              print_shapes=False, progress_bar=True, 
              writer=None, epoch_i=None, res_skip_keys= [],
              clip_grad_value = None, scheduler=None):
    t1_total= time.time()

    ### DEBUGGING we should clean this up
    MAX_N = 64

    if not pred_only:
        net.train()
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        if optimizer is not None:
            optimizer.zero_grad()
        torch.set_grad_enabled(False)

    accum_pred = []
    extra_loss_fields = {}

    running_loss = 0.0
    total_points = 0
    total_compute_time = 0.0
    if progress_bar:
        iterator =  tqdm(enumerate(dl), total=len(dl), desc=desc, leave=False)
    else:
        iterator = enumerate(dl)

    input_row_count = 0
    for _, batch in iterator:
        
        t1 = time.time()
        if print_shapes:
            for k, v in batch.items():
                print("{}.shape={}".format(k, v.shape))
        if not pred_only:
            optimizer.zero_grad()

        batch_t = {k : v.to(device) for k, v in batch.items()}
        #with torch.autograd.detect_anomaly():
        # for k, v in batch_t.items():
        #     assert not torch.isnan(v).any()

        res = net(**batch_t)
        vert_pred_batch_t = batch_t['vert_pred']
        vert_pred_mask_batch_t = batch_t['vert_pred_mask']
        edge_pred_batch_t = batch_t['edge_pred']
        edge_pred_mask_batch_t = batch_t['edge_pred_mask']

        input_mask_t = batch_t['input_mask']
        input_idx_t = batch_t['input_idx']
        
        return_pred_t1 = time.time()
        if return_pred:
            accum_pred_val = {}
            if isinstance(res, dict):
                for k, v in res.items():
                    if k not in res_skip_keys:
                        if isinstance(res[k], torch.Tensor):
                            accum_pred_val[k] = res[k].cpu().detach().numpy()
            else:
                
                accum_pred_val['res'] = res.cpu().detach().numpy()
            accum_pred_val['vert_pred_mask'] = vert_pred_mask_batch_t.cpu().detach().numpy()
            accum_pred_val['vert_pred'] = vert_pred_batch_t.cpu().detach().numpy()
            accum_pred_val['edge_pred_mask'] = edge_pred_mask_batch_t.cpu().detach().numpy()
            accum_pred_val['edge_pred'] = edge_pred_batch_t.cpu().detach().numpy()
            accum_pred_val['input_idx'] = input_idx_t.cpu().detach().numpy().reshape(-1, 1)
            accum_pred_val['input_mask'] = input_mask_t.cpu().detach().numpy()

            # extra fields
            for k, v in batch.items():
                if k.startswith("passthrough_"):
                    accum_pred_val[k] = v.cpu().detach().numpy()
            
            accum_pred.append(accum_pred_val)
        return_pred_t2 = time.time()
        loss_dict = {}
        if criterion is None:
            loss = 0.0
        else:
            loss = criterion(res,
                             vert_pred_batch_t,
                             vert_pred_mask_batch_t,
                             edge_pred_batch_t,
                             edge_pred_mask_batch_t, 
                             ## EDGE HERE
                             
                             input_mask_t)
            if isinstance(loss, dict):
                loss_dict = loss
                loss = loss_dict['loss']

        if not pred_only:
            loss.backward()
            # for n, p in net.named_parameters():
            #     if 'weight' in n:
            #         writer.add_scalar(f"grads/{n}", torch.max(torch.abs(p.grad)), epoch_i)

            if clip_grad_value is not None:
                nn.utils.clip_grad_value_(net.parameters(), clip_grad_value)
            
            optimizer.step()

        train_points = batch['input_mask'].shape[0]
        if criterion is not None:
            running_loss += loss.item() * train_points
            for k, v in loss_dict.items():
                if k not in extra_loss_fields:
                    extra_loss_fields[k] = v.item() * train_points
                else: 
                    extra_loss_fields[k] += v.item() * train_points
            
        total_points +=  train_points


        t2 = time.time()
        total_compute_time += (t2-t1)

        input_row_count += batch['adj'].shape[0]

        if scheduler is not None:
            scheduler.step()
    t2_total = time.time()
    
    #print('running_loss=', running_loss)
    total_points = max(total_points, 1)
    res =  {'timing' : 0.0, 
            'running_loss' : running_loss, 
            'total_points' : total_points, 
            'mean_loss' : running_loss / total_points,
            'runtime' : t2_total-t1_total, 
            'compute_time' : total_compute_time, 
            'run_efficiency' : total_compute_time / (t2_total-t1_total), 
            'pts_per_sec' : input_row_count / (t2_total-t1_total), 
            }


    for elf, v in extra_loss_fields.items():
        #print(f"extra loss fields {elf} = {v}")
        res[f'loss_total_{elf}'] = v
        res[f'loss_mean_{elf}'] = v/total_points

    if return_pred:
        keys = accum_pred[0].keys()
        for k in keys:
            accum_pred_v = np.vstack([a[k] for a in accum_pred])
            res[f'pred_{k}'] = accum_pred_v
            
    return res


VALIDATE_EVERY = 10

def generic_runner(net, optimizer, scheduler, criterion, 
                   dl_train, dl_test, start_epoch,
                   MAX_EPOCHS=1000, 
                   use_std=False, 
                   writer=None, validate_funcs = None, 
                   checkpoint_func = None, prog_bar=True,
                   clip_grad_value = None,
                   device=None,):
    

    # loss_scale = torch.Tensor(loss_scale)
    # std_scale = torch.Tensor(std_scale)

    res_skip_keys = ['g_in', 'g_decode']

    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        per_batch_scheduler = scheduler
    else:
        per_batch_scheduler = None

    if prog_bar:
        iterator = tqdm(range(start_epoch, MAX_EPOCHS))
    else:
        iterator = range(start_epoch, MAX_EPOCHS)
    for epoch_i in iterator:

        net.train()
        train_res = run_epoch(net, device, optimizer, criterion, dl_train, 
                              pred_only = False, 
                              return_pred=True, progress_bar=prog_bar,
                              desc='train', writer=writer, epoch_i=epoch_i, 
                              res_skip_keys = res_skip_keys,
                              clip_grad_value = clip_grad_value,
                              scheduler=per_batch_scheduler, 
        )

        [v(train_res, "train_", epoch_i) for v in validate_funcs]

        if epoch_i % VALIDATE_EVERY == 0:
            net.eval()
            test_res = run_epoch(net, device, optimizer, criterion, dl_test, 
                                 pred_only = True, 
                                 progress_bar=prog_bar, 
                                 return_pred=True, desc='validate', 
                                 res_skip_keys=res_skip_keys)
            [v(test_res, "validate_", epoch_i) for v in validate_funcs]

            if writer is not None:
                writer.add_scalar("train_loss-test_loss",
                                  train_res['mean_loss'] - test_res['mean_loss'],
                                  epoch_i)

        if checkpoint_func is not None:
            checkpoint_func(epoch_i = epoch_i, net =net, optimizer=optimizer, scheduler=scheduler)
            
        if scheduler is not None and (per_batch_scheduler is None):
            scheduler.step()


def create_shift_uncertain_validate_func(config, writer):
    def val_func(input_res, prefix, epoch_i): # val, mask, truth):
        mu = input_res['pred_shift_mu']
        std = input_res['pred_shift_std'] 
        pred_mask = input_res['pred_vert_pred_mask']
        truth = input_res['pred_vert_pred']
        mean_loss = input_res['mean_loss']
        #print("validate_func mu.shape=", mu.shape, "Truth.shape=", truth.shape)
        res = {'mean_loss' : mean_loss, 
               'run_epoch_time' : input_res['runtime'], 
               'run_efficinecy' : input_res['run_efficiency'], 
               'run_pts_per_sec' : input_res['pts_per_sec']}

        # extra losses
        for k, v in input_res.items():
            if 'loss_total_' in k:
                res[k] = v
            if 'loss_mean_' in k:
                res[k] = v


        for ni, n in enumerate(config['fields']):
            delta = (mu[:, :, ni] - truth[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
            if len(delta) == 0:
                continue
            masked_std = (std[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
            res[f"{n}/delta_std"] = np.std(delta)
            res[f"{n}/delta_max"] = np.max(np.abs(delta))
            res[f"{n}/delta_mean_abs"] = np.mean(np.abs(delta))
            res[f"{n}/delta_abs_90"] = np.percentile(np.abs(delta), 90)
            res[f"{n}/std/mean"] = np.mean(masked_std)
            res[f"{n}/std/min"] = np.min(masked_std)
            res[f"{n}/std/max"] = np.max(masked_std)
            delta = np.nan_to_num(delta)
            masked_std = np.nan_to_num(masked_std)

            writer.add_histogram(f"{prefix}{n}_delta_abs", 
                                 np.abs(delta), epoch_i)
            writer.add_histogram(f"{prefix}{n}_delta_abs_dB", 
                                 np.log10(np.abs(delta)+1e-6), epoch_i)


            writer.add_histogram(f"{n}_std", 
                                 masked_std, epoch_i)
            sorted_delta_abs = np.abs(delta)[np.argsort(masked_std)]
            
            for frac in [10, 50, 90]:
                res[f"{n}/sorted_delta_abs_{frac}"] = np.mean(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])
                res[f"{n}/sorted_delta_abs_{frac}_max"] = np.max(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])
            
        exception = False

        for metric_name, metric_val in res.items():
            #print(f"{metric_name} is {metric_val}")
            
            if not np.isfinite(metric_val):
                exception = True
                #print(f"{metric_name} is {metric_val}")
            writer.add_scalar("{}{}".format(prefix, metric_name), 
                              metric_val, epoch_i)
        if exception:
            raise ValueError(f"{prefix}{metric_name} found some nans")


    return val_func


def create_coupling_validate_func(config, writer, every_n=1):
    coupling_type_list = config.get('coupling_lut', {})
    coupling_type_lut = {-1 : 'other'}
    for si, s in enumerate(coupling_type_list):
        coupling_type_lut[si] = f"{s[1]}J{s[0]}"
    coupling_index  = config['coupling_index'] 
    def val_func(input_res, prefix, epoch_i):
        
        #print("input_res.keys() =", input_res.keys())
        #print( input_res['pred_edge_pred'].shape)
    
        coupling_pred = input_res['pred_coupling_pred'][:, :, :, coupling_index]
        coupling_truth = input_res['pred_edge_pred'][:, :, :, coupling_index]
        coupling_mask = input_res['pred_edge_pred_mask'][:, :, :, coupling_index]

        #coupling_truth = 
        
        BATCH_N, MAX_N, _ = coupling_pred.shape

        coupling_types = input_res['pred_passthrough_coupling_types_encoded']


        
        delta = coupling_pred - coupling_truth

        delta_present = delta[coupling_mask > 0]
        delta_types = coupling_types[coupling_mask > 0]

        metrics = {'coupling_delta_abs' : np.mean(np.abs(delta_present)),
                    'coupling_delta_sq' : np.mean(delta_present**2),
                    'coupling_n' : np.sum(coupling_mask), 
                    'run_epoch_time' : input_res['runtime'], 
                    'run_efficinecy' : input_res['run_efficiency'], 
                    'run_pts_per_sec' : input_res['pts_per_sec']
                   }

        # break errors into luts
        different_coupling_types = {k : list() for k in coupling_type_lut.keys()}
        for ct, v in zip(delta_types, delta_present):
            different_coupling_types[ct].append(np.abs(v))

        for k, v in coupling_type_lut.items():
            #print('adding metric', f"coupling_{v}_delta_abs")
            if len(different_coupling_types[k]) > 0:
                metrics[f"coupling_{v}_delta_abs"] = np.mean(different_coupling_types[k])
            else:
                pass
            
            #print("Warning, only",
            #len(different_coupling_types[k]),
            #          "entries for", k, v)
            #print("done")
    
        exception = False
        for metric_name, metric_val in metrics.items():
            if not np.isfinite(metric_val):
                exception = True
                print(f"{metric_name} is {metric_val}")

            writer.add_scalar("{}{}".format(prefix, metric_name), 
                              metric_val, epoch_i)
        if exception:
            raise ValueError(f"{prefix}{metric_name} found some nans")

                
        # mu = input_res['pred_mu']
        # std = input_res['pred_std'] 
        # pred_mask = input_res['pred_mask']
        # truth = input_res['pred_truth']
        # mean_loss = input_res['mean_loss']
        # #print("validate_func mu.shape=", mu.shape, "Truth.shape=", truth.shape)
        # res = {'mean_loss' : mean_loss, 
        #        'run_epoch_time' : input_res['runtime'], 
        #        'run_efficinecy' : input_res['run_efficiency'], 
        #        'run_pts_per_sec' : input_res['pts_per_sec']}

        # # extra losses
        # for k, v in input_res.items():
        #     if 'loss_total_' in k:
        #         res[k] = v
        #     if 'loss_mean_' in k:
        #         res[k] = v


        # for ni, n in enumerate(tgt_nucs):
        #     delta = (mu[:, :, ni] - truth[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
        #     if len(delta) == 0:
        #         continue
        #     masked_std = (std[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
        #     res[f"{n}/delta_std"] = np.std(delta)
        #     res[f"{n}/delta_max"] = np.max(np.abs(delta))
        #     res[f"{n}/delta_mean_abs"] = np.mean(np.abs(delta))
        #     res[f"{n}/delta_abs_90"] = np.percentile(np.abs(delta), 90)
        #     res[f"{n}/std/mean"] = np.mean(masked_std)
        #     res[f"{n}/std/min"] = np.min(masked_std)
        #     res[f"{n}/std/max"] = np.max(masked_std)
        #     delta = np.nan_to_num(delta)
        #     masked_std = np.nan_to_num(masked_std)

        #     writer.add_histogram(f"{prefix}{n}_delta_abs", 
        #                          np.abs(delta), epoch_i)
        #     writer.add_histogram(f"{prefix}{n}_delta_abs_dB", 
        #                          np.log10(np.abs(delta)+1e-6), epoch_i)


        #     writer.add_histogram(f"{n}_std", 
        #                          masked_std, epoch_i)
        #     sorted_delta_abs = np.abs(delta)[np.argsort(masked_std)]
            
        #     for frac in [10, 50, 90]:
        #         res[f"{n}/sorted_delta_abs_{frac}"] = np.mean(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])
        #         res[f"{n}/sorted_delta_abs_{frac}_max"] = np.max(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])
            
        # exception = False

        # for metric_name, metric_val in res.items():
        #     #print(f"{metric_name} is {metric_val}")
            
        #     if not np.isfinite(metric_val):
        #         exception = True
        #         #print(f"{metric_name} is {metric_val}")
        #     writer.add_scalar("{}{}".format(prefix, metric_name), 
        #                       metric_val, epoch_i)
        # if exception:
        #     raise ValueError(f"{prefix}{metric_name} found some nans")


    return val_func


def create_coupling_uncertain_validate_func(config, writer):
    coupling_type_list = config.get('coupling_lut', {})
    coupling_type_lut = {-1 : 'other'}
    run_every_n = config.get('run_every', 10)
    for si, s in enumerate(coupling_type_list):
        coupling_type_lut[si] = f"{s[1]}J{s[0]}"
    coupling_index  = config['coupling_index'] 
    def val_func(input_res, prefix, epoch_i):

        if epoch_i % run_every_n != 0:
            return
            
        #print("input_res.keys() =", input_res.keys())
        #print( input_res['pred_edge_pred'].shape)
    
        coupling_mu = input_res['pred_coupling_mu'][:, :, :, coupling_index]
        coupling_std = input_res['pred_coupling_std'][:, :, :, coupling_index]
        
        coupling_truth = input_res['pred_edge_pred'][:, :, :, coupling_index]
        coupling_mask = input_res['pred_edge_pred_mask'][:, :, :, coupling_index]

        
        BATCH_N, MAX_N, _ = coupling_mu.shape

        coupling_types = input_res['pred_passthrough_coupling_types_encoded']


        
        delta = coupling_mu - coupling_truth

        delta_present = delta[coupling_mask > 0]
        std_present = coupling_std[coupling_mask > 0]
        
        delta_types = coupling_types[coupling_mask > 0]

        metrics = {'coupling_delta_abs' : np.mean(np.abs(delta_present)),
                   'coupling_delta_sq' : np.mean(delta_present**2),
                   'coupling_n' : np.sum(coupling_mask), 
                   'run_epoch_time' : input_res['runtime'], 
                    'run_efficinecy' : input_res['run_efficiency'], 
                    'run_pts_per_sec' : input_res['pts_per_sec']
                   }

        # break errors into luts
        different_coupling_types_delta = {k : list() for k in coupling_type_lut.keys()}
        different_coupling_types_std = {k : list() for k in coupling_type_lut.keys()}
        for ct, delta, std in zip(delta_types, delta_present, std_present):
            different_coupling_types_delta[ct].append(np.abs(delta))
            different_coupling_types_std[ct].append(std)

        for k, v in coupling_type_lut.items():
            #print('adding metric', f"coupling_{v}_delta_abs")
            if len(different_coupling_types_delta[k]) > 0:
                deltas = np.array(different_coupling_types_delta[k])
                stds = np.array(different_coupling_types_std[k])

                base_metric= f"coupling_{v}"
                metrics[f"{base_metric}/delta_abs"] = np.mean(deltas)


                sorted_delta_abs = np.abs(deltas)[np.argsort(stds)]
            
                for frac in [10, 50, 90]:
                    metrics[f"{base_metric}/sorted_delta_abs_{frac}"] = np.mean(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])
                    metrics[f"{base_metric}/sorted_delta_abs_{frac}_max"] = np.max(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])
                
            else:
                pass
            
            #print("Warning, only",
            #len(different_coupling_types[k]),
            #          "entries for", k, v)
            #print("done")
    
        exception = False
        for metric_name, metric_val in metrics.items():
            if not np.isfinite(metric_val):
                exception = True
                print(f"{metric_name} is {metric_val}")

            writer.add_scalar("{}{}".format(prefix, metric_name), 
                              metric_val, epoch_i)
        if exception:
            raise ValueError(f"{prefix}{metric_name} found some nans")

                
        # mu = input_res['pred_mu']
        # std = input_res['pred_std'] 
        # pred_mask = input_res['pred_mask']
        # truth = input_res['pred_truth']
        # mean_loss = input_res['mean_loss']
        # #print("validate_func mu.shape=", mu.shape, "Truth.shape=", truth.shape)
        # res = {'mean_loss' : mean_loss, 
        #        'run_epoch_time' : input_res['runtime'], 
        #        'run_efficinecy' : input_res['run_efficiency'], 
        #        'run_pts_per_sec' : input_res['pts_per_sec']}

        # # extra losses
        # for k, v in input_res.items():
        #     if 'loss_total_' in k:
        #         res[k] = v
        #     if 'loss_mean_' in k:
        #         res[k] = v


        # for ni, n in enumerate(tgt_nucs):
        #     delta = (mu[:, :, ni] - truth[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
        #     if len(delta) == 0:
        #         continue
        #     masked_std = (std[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
        #     res[f"{n}/delta_std"] = np.std(delta)
        #     res[f"{n}/delta_max"] = np.max(np.abs(delta))
        #     res[f"{n}/delta_mean_abs"] = np.mean(np.abs(delta))
        #     res[f"{n}/delta_abs_90"] = np.percentile(np.abs(delta), 90)
        #     res[f"{n}/std/mean"] = np.mean(masked_std)
        #     res[f"{n}/std/min"] = np.min(masked_std)
        #     res[f"{n}/std/max"] = np.max(masked_std)
        #     delta = np.nan_to_num(delta)
        #     masked_std = np.nan_to_num(masked_std)

        #     writer.add_histogram(f"{prefix}{n}_delta_abs", 
        #                          np.abs(delta), epoch_i)
        #     writer.add_histogram(f"{prefix}{n}_delta_abs_dB", 
        #                          np.log10(np.abs(delta)+1e-6), epoch_i)


        #     writer.add_histogram(f"{n}_std", 
        #                          masked_std, epoch_i)
        #     sorted_delta_abs = np.abs(delta)[np.argsort(masked_std)]
            
        #     for frac in [10, 50, 90]:
        #         res[f"{n}/sorted_delta_abs_{frac}"] = np.mean(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])
        #         res[f"{n}/sorted_delta_abs_{frac}_max"] = np.max(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])
            
        # exception = False

        # for metric_name, metric_val in res.items():
        #     #print(f"{metric_name} is {metric_val}")
            
        #     if not np.isfinite(metric_val):
        #         exception = True
        #         #print(f"{metric_name} is {metric_val}")
        #     writer.add_scalar("{}{}".format(prefix, metric_name), 
        #                       metric_val, epoch_i)
        # if exception:
        #     raise ValueError(f"{prefix}{metric_name} found some nans")


    return val_func


def create_optimizer(opt_params, net_params):
    opt_direct_params = {}
    optimizer_name = opt_params.get('optimizer', 'adam') 
    if optimizer_name == 'adam':
        for p in ['lr', 'amsgrad', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adam(net_params, **opt_direct_params)
    elif optimizer_name == 'adamw':
        for p in ['lr', 'amsgrad', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.AdamW(net_params, **opt_direct_params)
    elif optimizer_name == 'adamax':
        for p in ['lr', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adamax(net_params, **opt_direct_params)
        
    elif optimizer_name == 'adagrad':
        for p in ['lr', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adagrad(net_params, **opt_direct_params)
        
    elif optimizer_name == 'rmsprop':
        for p in ['lr', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.RMSprop(net_params, **opt_direct_params)
        
    elif optimizer_name == 'sgd':
        for p in ['lr', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.SGD(net_params, **opt_direct_params)

    return optimizer


def create_loss(loss_params, device):
    loss_name = loss_params['loss_name']

    std_regularize = loss_params.get('std_regularize', 0.01)
    mu_scale = torch.Tensor(loss_params.get('mu_scale', [1.0])).to(device)
    std_scale = torch.Tensor(loss_params.get('std_scale', [1.0])).to(device)

    if loss_name == 'NormUncertainLoss':
        criterion = nets.NormUncertainLoss(mu_scale, 
                                           std_scale,
                                           std_regularize = std_regularize)
    elif loss_name == 'UncertainLoss':
        criterion = nets.UncertainLoss(mu_scale, 
                                       std_scale,
                                       norm = loss_params['norm'], 
                                       std_regularize = std_regularize, 
                                       std_pow = loss_params['std_pow'], 
                                       use_reg_log = loss_params['use_reg_log'],
                                       std_weight = loss_params['std_weight'])

    elif loss_name == "NoUncertainLoss":
        
        criterion = nets.NoUncertainLoss(**loss_params)
    elif loss_name == 'MultiShiftLoss':
        criterion = nets.MultiShiftLoss(**loss_params)
    elif loss_name == "DisagreementLoss":
        
        criterion = nets.DisagreementLoss(**loss_params)
    elif loss_name == "DisagreementLossOneToTwo":
        
        criterion = nets.DisagreementLossOneToTwo(**loss_params)
    elif loss_name == "SimpleLoss":
        
        criterion = nets.SimpleLoss(**loss_params)

    elif loss_name == "PermMinLoss":
        
        criterion = nets.PermMinLoss(**loss_params)
    elif loss_name == "ReconLoss":
        
        criterion = seminets.ReconLoss(**loss_params)
    elif loss_name == "CouplingLoss":
        
        criterion = coupling.CouplingLoss(**loss_params)
    elif loss_name == "WeightedCouplingLoss":
        
        criterion = coupling.WeightedCouplingLoss(**loss_params)
    elif loss_name == "DistReconLoss":
        
        criterion = seminets.DistReconLoss(**loss_params)
    else:
        raise ValueError(loss_name)


    return criterion



def create_coupling_validate_loss(writer, every_n=1):
    def val_func(input_res, prefix, epoch_i):
        if epoch_i % every_n != 0:
            return
        writer.add_scalar("{}{}".format(prefix, 'mean_loss'),
                          input_res['mean_loss'], epoch_i)
        
    return val_func


class SubsetSampler(torch.utils.data.Sampler):
    """
    A dataset sampler for sampling from a subset
    of data each epoch. 
    
    epoch_size: the size of the epoch 
    ds_size: the total size of the dataset
    
    this way you can always have epochs of the same size
    to compare training perf across different datasets. 
    """
    def __init__(self, epoch_size, ds_size, shuffle=False,
                 world_size=1, rank=0, seed=None, logging_name= None):
        self.epoch_size = epoch_size
        self.ds_size = ds_size
        self.pos = 0
        self.shuffle = shuffle
        
        rank_subset_size = ds_size // world_size
        self.rank_subset_size = rank_subset_size

        self.world_size = world_size
        self.rank = rank

        if seed is None:
            seed = rank
        self.bit_generator = np.random.PCG64(seed)
        self.rng = np.random.default_rng(self.bit_generator)

        if shuffle:
            
            self.rank_subset_idx = self.rng.permutation(ds_size)[rank*rank_subset_size:(rank+1)*rank_subset_size]
        else:
            self.rank_subset_idx = self.rng.arange(ds_size)[rank*rank_subset_size:(rank+1)*rank_subset_size]            
                
        self.compute_idx()
        self.logging_name = logging_name
        if logging_name is not None:
            self.logging_fid = open(f"{logging_name}.samples", 'a')

    def get_state(self):
        """
        Returns the state to reinitialize the sampler
        """

        return {'pos' : self.pos,
                'epoch_size' : self.epoch_size,
                'idx' : self.idx, 
                'ds_size' : self.ds_size,
                'bg_state' : self.bit_generator.state}


    def set_state(self, state_dict):
        assert self.epoch_size == state_dict['epoch_size']
        assert self.ds_size == state_dict['ds_size']

        self.pos = state_dict['pos']
        self.bit_generator.state = state_dict['bg_state']
        self.idx = state_dict['idx']

        
    def compute_idx(self):
        # print('recomputing idx, rank=', self.rank)
              
        if self.shuffle:
            self.idx = self.rng.permutation(self.rank_subset_idx)
        else:
            self.idx = self.rank_subset_idx

    def __len__(self):
        return self.epoch_size // self.world_size 
    
    def __iter__(self):
        for i in range(self.__len__()):
            y = self.idx[self.pos]
            if self.logging_name is not None:
                self.logging_fid.write(f"{y}\n")
                self.logging_fid.flush()
            yield y
            self.pos = (self.pos + 1) % len(self.idx)

            if self.pos == 0:
                self.compute_idx()
                
