"""
Pipeline for generating predictions on code

"""

import numpy as np
import pandas as pd
import torch
import time
import pickle
from fullsspruce.featurize.netdataio import * 
import os

import torch.distributed as dist

from fullsspruce import metrics
from fullsspruce.geom_util import geometryFeaturizerGenerator, DEFAULT_GAUSS_BINS

from tqdm import tqdm
from fullsspruce import netutil
from ruffus import * 
from glob import glob
import copy
from fullsspruce import predwrap

import rdkit
from rdkit import Chem

PRED_DIR = "forward.preds"

td = lambda x : os.path.join(PRED_DIR, x)

EXPERIMENTS = {
}    

EXPERIMENTS['proton_GNN'] = {
    'model': 'checkpoints/fs_def_1H_5_27.proton_GNN.537094279807',
    'checkpoints' : [100, 500],
    'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}], 
    'pred_fields' : ['pred_shift_mu', 'pred_shift_std'],
    'batch_size' : 32, 
    'nuc' : '1H',
    'dataset' : "processed_dbs/shifts.nmrshiftdb.128_128_HCONFSPCl_wcharge_1H.dataset.pickle"
}

EXPERIMENTS['carbon_GNN'] = {
    'model': 'checkpoints/fs_def_13C_5_27.carbon_GNN.537094139799',
    'checkpoints' : [450],
    'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}], 
    'pred_fields' : ['pred_shift_mu', 'pred_shift_std'],
    'batch_size' : 32, 
    'nuc' : '13C',
    'dataset' : "processed_dbs/shifts.nmrshiftdb.128_128_HCONFSPCl_wcharge_13C.dataset.pickle"
}

EXPERIMENTS['proton_GNN_with_decode'] = {
    'model': 'checkpoints/decode_1H_ETKDG.proton_GNN_with_decode.610150146029',
    'checkpoints' : [500],
    'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}], 
    'pred_fields' : ['pred_shift_mu', 'pred_shift_std'],
    'batch_size' : 32, 
    'max_atom_n' : 32,
    'nuc' : '1H',
    'extra_features' : [{
        'filename': 'distance_features/nmrshiftdb_ETKDG_opt_50_20_distances_means.pickle',
        'field' : 'mean_distance_mat'
    }, {
        'filename': 'distance_features/nmrshiftdb_ETKDG_opt_50_20_angles_mat.pickle',
        'field' : 'mean_angle_mat'
        
    }, {
        'filename': 'distance_features/nmrshiftdb_ETKDG_opt_50_20_gauss_bins_default.pickle',
        'field' : 'conf_gauss_bins'
    }],
    'dataset' : "processed_dbs/shifts.nmrshiftdb.128_128_HCONFSPCl_wcharge_1H.dataset.pickle"
}

EXPERIMENTS['carbon_GNN_with_decode'] = {
    'model': 'checkpoints/decode_13C_ETKDG.carbon_GNN_with_decode.610149635867',
    'checkpoints' : [500],
    'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}], 
    'pred_fields' : ['pred_shift_mu', 'pred_shift_std'],
    'batch_size' : 32, 
    'max_atom_n' : 32,
    'nuc' : '13C',
    'extra_features' : [{
        'filename': 'distance_features/nmrshiftdb_ETKDG_opt_50_20_distances_means.pickle',
        'field' : 'mean_distance_mat'
    }, {
        'filename': 'distance_features/nmrshiftdb_ETKDG_opt_50_20_angles_mat.pickle',
        'field' : 'mean_angle_mat'
        
    }, {
        'filename': 'distance_features/nmrshiftdb_ETKDG_opt_50_20_gauss_bins_default.pickle',
        'field' : 'conf_gauss_bins'
    }],
    'dataset' : "processed_dbs/shifts.nmrshiftdb.128_128_HCONFSPCl_wcharge_13C.dataset.pickle"
}

##

COUPLING_FIELD_MAPS = {'scalar' :  {'coupling_mu' : ('rename', 'pred_coupling_pred'), 
                                   'coupling_std' : ('const', 1.0)},
                       'uncertain' : {'coupling_mu' : ('rename', 'pred_coupling_mu'), 
                                      'coupling_std' : ('rename', 'pred_coupling_std')}
                       }

COUPLING_EXPERIMENTS = {}

COUPLING_EXPERIMENTS['coupling_ETKDG_default'] = {'model' : f"checkpoints/default_coupling_extended.coupling_ETKDG_default.732205914631",
                                                                     'checkpoints' : [500],
                                                                     'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}],
                                                                     'pred_fields' : ['pred_coupling_mu', 'pred_coupling_std'], 
                                                                     'batch_size' : 64,
                                                                     'max_atom_n' : 64,
                                                                     'coupling_field_map' : COUPLING_FIELD_MAPS['uncertain'], 
                                                                     'dataset' : "processed_dbs/couplings.128_128_HCONFSPCl_wcharge_ch3avg.dataset.pickle", 
                                                                     
                                                                     'extra_features' : [{
                                                                         'filename': 'distance_features/nmrshiftdb_ETKDG_10_distances_means.pickle',
                                                                         'field' : 'mean_distance_mat'
                                                                     },
                                                                     {
                                                                         'filename': 'distance_features/nmrshiftdb_ETKDG_10_angles_mat.pickle',
                                                                         'field' : 'mean_angle_mat'    
                                                                     },
                                                                     {
                                                                         'filename': 'distance_features/nmrshiftdb_ETKDG_10_gauss_bins_default.pickle',
                                                                         'field' : 'conf_gauss_bins'
                                                                     }]
}

COUPLING_EXPERIMENTS['coupling_PT_default'] = {'model' : f"checkpoints/coupling_PT_default.coupling_PT_default.612289161188",
                                                                     'checkpoints' : [500],
                                                                     'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}],
                                                                     'pred_fields' : ['pred_coupling_mu', 'pred_coupling_std'], 
                                                                     'batch_size' : 64,
                                                                     'max_atom_n' : 64,
                                                                     'coupling_field_map' : COUPLING_FIELD_MAPS['uncertain'], 
                                                                     'dataset' : "processed_dbs/couplings.128_128_HCONFSPCl_wcharge_ch3avg.dataset.pickle", 
                                                                     
                                                                     'extra_features' : [{
                                                                         'filename': 'distance_features/nmrshiftdb_PT_distances_means.pickle',
                                                                         'field' : 'mean_distance_mat'
                                                                     },
                                                                     {
                                                                         'filename': 'distance_features/nmrshiftdb_PT_angles_means.pickle',
                                                                         'field' : 'mean_angle_mat'    
                                                                     },
                                                                     {
                                                                         'filename': 'distance_features/nmrshiftdb_PT_gauss_bins_default.pickle',
                                                                         'field' : 'conf_gauss_bins'
                                                                     }]
}

##

def params():
    
    for exp_name, exp_config in EXPERIMENTS.items():
        
        outfiles = td(f"{exp_name}.meta.pickle"), td(f"{exp_name}.feather")
        yield None, outfiles, exp_config

def coupling_params():
    
    for exp_name, exp_config in COUPLING_EXPERIMENTS.items():
        
        outfiles = td(f"{exp_name}.coupling.meta.pickle"), td(f"{exp_name}.coupling.feather")
        yield None, outfiles, exp_config

DEVICE = torch.device('cpu')
# DEVICE_ID = 0
if torch.cuda.is_available():
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '11223'

    # dist.init_process_group("nccl", rank=0, world_size=1)
    DEVICE = torch.device('cuda')                 

@mkdir(PRED_DIR)
@files(params)
def train_test_predict_shifts(infile, outfiles, config):
    """
    New pipeline version for predicting couplings

    """
    meta_outfile, data_outfile = outfiles

    model_filename = config['model'] 
    dataset_filename = config['dataset']
    print("config=", config)
    print("loading dataset", dataset_filename)

    all_df = pickle.load(open(config['dataset'], 'rb'))
    for extra_feature_config in config.get('extra_features', []):
        # add these to the data frame
        if 'filename' in extra_feature_config: # assume it's a dictionary
            feat_dict = pickle.load(open(extra_feature_config['filename'], 'rb'))
            s = pd.Series(feat_dict)
            s.name = extra_feature_config['field']
            all_df = all_df.join(s, on='molecule_id')
            all_df = all_df.dropna()
        elif 'geom_func' in extra_feature_config:
            # Geom Func assumed to be of type GeometryGenerator (see geom_util.py)
            features, meta = extra_feature_config['geom_func'](list(all_df['rdmol']))
            for f in extra_feature_config['fields']:
                all_df[f] = [np.nan if mol_f is np.nan else mol_f[f] for mol_f in features]
            all_df = all_df.dropna()
        else:
            raise NotImplementedError("Unsupported method for obtaining extra features.")
        
    if 'debug_max' in config:
        all_df = all_df.sample(config['debug_max'])


    atoms = np.concatenate([[a.GetSymbol() for a in m.GetAtoms()] for m in all_df.rdmol])

    print("unique atoms", np.unique(atoms))

    meta_filename = f"{model_filename}.meta"

    meta = pickle.load(open(meta_filename, 'rb'))

    tgt_df = all_df

    if 'max_atom_n' in config:
        tgt_df = tgt_df[tgt_df.rdmol.apply(lambda m : m.GetNumAtoms() <= config['max_atom_n'])]

    metafile = meta.get('metafile')
    if not metafile is None:
        records_meta = pickle.load(open(metafile, 'rb'))
        tgt_df = tgt_df[tgt_df.molecule_id in records_meta['ids']]

    whole_records = tgt_df.to_dict('records')
    
    meta_pred_config = meta['pred_config']
    pred_channel = config.get('pred_channel', 0)
    vert = meta_pred_config['vert']
    eval_truth = config.get('eval_truth', vert[pred_channel])
    
    # create truth dict
    data_field = eval_truth['data_field']
    index = eval_truth['index']

    true_vals = []
    for _,(_, row) in enumerate(tgt_df.iterrows()):
        if pd.isna(row[data_field]):
            continue
        f_dict = row[data_field][index]
        mol = Chem.Mol(row['rdmol'])
        for atom_idx, v in f_dict.items():
            atom = mol.GetAtomWithIdx(atom_idx)
            partner = max([x.GetAtomicNum() for x in atom.GetNeighbors()])
            true_vals.append({'mol_id' : row['molecule_id'], 
                              'atom_idx' : atom_idx, 
                               'value' :  v,
                               'bond_partner': partner})
    true_val_df = pd.DataFrame(true_vals).set_index(['mol_id', 'atom_idx'])
    true_val_df

    metadata_res = []
    for checkpoint_i in config['checkpoints']:

        checkpoint_filename = f"{model_filename}.{checkpoint_i:08d}" #".model"
        print("running", checkpoint_filename)
        model = predwrap.PredModel(meta_filename, 
                                   checkpoint_filename, 
                                   DEVICE)

        t1 = time.time()
        vert_pred_df, edge_pred_df  = model.pred(whole_records, 
                                pred_fields=['pred_shift_mu', 'pred_shift_std'],#, 'pred_shift_samples'],
                                prog_bar=True, 
                                BATCH_SIZE=config.get("batch_size", 32))
        
        t2 = time.time()
        print("calculated", len(tgt_df), "mols in ", t2-t1, "sec")
        print(vert_pred_df.dtypes)


        t1_manual = time.time()
        manual_pivot_out = {}
        print(vert_pred_df.columns)
        for _, row in tqdm(vert_pred_df.iterrows(), desc='manual_pivot',
                               total=len(vert_pred_df)):
            key = (row.mol_id, row.atom_idx)
            if key not in manual_pivot_out:
                manual_pivot_out[key] = {}
            # elif row.field in manual_pivot_out[key]:
            #     print('overwrite:', row.field, key)
            #     quit
            if row.pred_chan == pred_channel:
                manual_pivot_out[key][row.field] = row.val
        shift_pred_df = pd.DataFrame.from_dict(manual_pivot_out, orient='index').reset_index()
        shift_pred_df = shift_pred_df.rename(columns = {'level_0': 'mol_id',
                                        'level_1': 'atom_idx'})
        shift_pred_df = shift_pred_df.rename(columns = {'pred_shift_mu' : 'shift_mu',
                                                        'pred_shift_std' : 'shift_std',
                                                        # 'pred_shift_samples' : 'shift_samples', 
        })

        print(shift_pred_df.columns) 

        t2_manual = time.time()
        
        print(f"manual took {t2_manual - t1_manual:3.2f} sec")
        print(shift_pred_df.dtypes)
        
        t1_join = time.time()
        results_df = shift_pred_df.join(true_val_df, on = ['mol_id', 'atom_idx'])
        t2_join = time.time()
        print(f"join took {t2_join - t1_join:3.2f} sec")
        
        tgt_dict_molid = tgt_df.set_index('molecule_id').to_dict()
        results_df['morgan4_crc32'] = results_df['mol_id'].map(tgt_dict_molid['morgan4_crc32'])
        results_df['smiles'] = results_df['mol_id'].map(tgt_dict_molid['smiles'])
        results_df = results_df.dropna().reset_index(drop=True)
        results_df['nuc'] = config['nuc']

        results_df['epoch_i'] = checkpoint_i

        data_epoch_outfile = data_outfile + f".{checkpoint_i}"
        results_df.to_feather(data_epoch_outfile)
        metadata_res.append({'time' : t2-t1, 
                             'epoch_filename' : data_epoch_outfile, 
                             'epoch' : checkpoint_i,
                             'mol' : len(tgt_df)})

    metadata_df = pd.DataFrame(metadata_res)


    pickle.dump({'model_filename' : model_filename, 
                 'dataset_filename' : dataset_filename, 
                 'config' : config, 
                 'meta' : metadata_df, },
                open(meta_outfile, 'wb'))
    # data outfile done
    pickle.dump({'data' : True}, 
                open(data_outfile, 'wb'))
                 
@mkdir(PRED_DIR)
@files(coupling_params)
def train_test_predict_coupling(infile, outfiles, config):
    """a
    pipeline for predicting coupling

    """
    meta_outfile, data_outfile = outfiles


    model_filename = config['model'] 
    dataset_filename = config['dataset']
    print("config=", config)
    print("loading dataset", dataset_filename)

    all_df = pickle.load(open(config['dataset'], 'rb'))
    for extra_feature_config in config.get('extra_features', []):
        # add these to the data frame
        if 'filename' in extra_feature_config: # assume it's a dictionary
            feat_dict = pickle.load(open(extra_feature_config['filename'], 'rb'))
            s = pd.Series(feat_dict)
            s.name = extra_feature_config['field']
            all_df = all_df.join(s, on='molecule_id')
            all_df = all_df.dropna()
        elif 'geom_func' in extra_feature_config:
            # Geom Func assumed to be of type GeometryGenerator (see geom_util.py)
            features, meta = extra_feature_config['geom_func'](list(all_df['rdmol']))
            for f in extra_feature_config['fields']:
                all_df[f] = [np.nan if mol_f is np.nan else mol_f[f] for mol_f in features]
            all_df = all_df.dropna()
        else:
            raise NotImplementedError("Unsupported method for obtaining extra features.")
        
    if 'debug_max' in config:
        all_df = all_df.sample(config['debug_max'])


    atoms = np.concatenate([[a.GetSymbol() for a in m.GetAtoms()] for m in all_df.rdmol])

    print("unique atoms", np.unique(atoms))

    meta_filename = f"{model_filename}.meta"

    meta = pickle.load(open(meta_filename, 'rb'))

    print("Loaded meta")

    coupling_field_map = config.get('coupling_field_map', {})

    pred_fields = config.get('pred_fields', ['pred_coupling_mu', 'pred_coupling_std'])

    tgt_df = all_df
    if 'max_atom_n' in config:
        tgt_df = tgt_df[tgt_df.rdmol.apply(lambda m : m.GetNumAtoms() <= config['max_atom_n'])]

    if 'elements_only' in config:
        eo = set(config['elements_only'])
        def is_sub(m):
            elts = set([a.GetSymbol() for a in m.GetAtoms()])
            return elts.issubset(eo)
        tgt_df = tgt_df[tgt_df.rdmol.apply(is_sub)]

    print("Filtered dataset")

    whole_records = tgt_df.to_dict('records')

    print("Converted dataset to dictionary")


    ###################################################
    # load true values
    ###################################################

    meta_pred_config = meta['pred_config']

    edge = meta_pred_config['edge']
    data_field = edge[0]['data_field']

    print("Loading true values, tgt_df length:", len(tgt_df))

    true_vals = []
    for _,(_, row) in tqdm(enumerate(tgt_df.iterrows())):
        coup_dict = row[data_field]
        coupling_types = row['coupling_types']
        for _, (atomidx_1, atomidx_2, _, _, c_t, c_d) in coupling_types.iterrows():
            if (atomidx_1, atomidx_2) in coup_dict:
                true_vals.append({'mol_id' : row['molecule_id'], 
                                'atomidx_1' : atomidx_1,
                                'atomidx_2' : atomidx_2,
                                'coupling_elts' : c_t,
                                'coupling_dist' : c_d,                              
                                'value' :  coup_dict[(atomidx_1,atomidx_2)]})
                                
    true_val_df = pd.DataFrame(true_vals).set_index(['mol_id', 'atomidx_1', 'atomidx_2'])

    print("Got true values")
    ###################################################
    # do the actual prediction
    ###################################################

    metadata_res = []
    for checkpoint_i in config['checkpoints']:
        checkpoint_filename = f"{model_filename}.{checkpoint_i:08d}"
        print("running", checkpoint_filename)
        model = predwrap.PredModel(meta_filename, 
                                   checkpoint_filename, 
                                   DEVICE)

        #tgt_df = all_df.copy()

        t1 = time.time()
        vert_pred_df, edge_pred_df  = model.pred(whole_records, 
                                                 pred_fields=pred_fields, 
                                                 prog_bar=True, 
                                                 BATCH_SIZE=config.get("batch_size", 32))
        t2 = time.time()
        print("calculated", len(tgt_df), "mols in ", t2-t1, "sec")

        coupling_pred_df = pd.pivot_table(edge_pred_df, index=['mol_id', 'atomidx_1', 'atomidx_2'], 
                       columns=['field'], values='val').reset_index()

        results_df = coupling_pred_df.join(true_val_df, on = ['mol_id',  'atomidx_1', 'atomidx_2'])
        tgt_dict_molid = tgt_df.set_index('molecule_id').to_dict()
        results_df['morgan4_crc32'] = results_df['mol_id'].map(tgt_dict_molid['morgan4_crc32'])
        results_df['smiles'] = results_df['mol_id'].map(tgt_dict_molid['smiles'])
        results_df = results_df.dropna().reset_index(drop=True)

        results_df['epoch_i'] = checkpoint_i

        # field cleanups
        for out_field_name, field_rename in coupling_field_map.items():
            if field_rename[0] == 'rename':
                results_df = results_df.rename(columns = {field_rename[1] : out_field_name})
            elif field_rename[0] == 'const':
                results_df[out_field_name] = field_rename[1]
            else:
                raise ValueError(f"unknown field rename action {field_rename}")
        
        data_epoch_outfile = data_outfile + f".{checkpoint_i}"
        results_df.to_feather(data_epoch_outfile)
        metadata_res.append({'time' : t2-t1, 
                             'epoch_filename' : data_epoch_outfile, 
                             'epoch' : checkpoint_i,
                             'mol' : len(tgt_df)})

    metadata_df = pd.DataFrame(metadata_res)


    pickle.dump({'model_filename' : model_filename, 
                 'dataset_filename' : dataset_filename, 
                 'config' : config, 
                 'meta' : metadata_df, },
                open(meta_outfile, 'wb'))
    # data outfile done
    pickle.dump({'data' : True}, 
                open(data_outfile, 'wb'))
                 

STATS_CONFIGS = {'coupling' : {'pred_field' : 'coupling_mu',
                               'pred_field_std' : 'coupling_std',
                               'groupby' : ['coupling_elts', 'coupling_dist'],
                               'stats_groups' : [('CH', 1),
                                                 ('HH', 2), 
                                                 ('HH', 3), 
                                                 ('HH', 4),
                                                 ('HH', None),
                                                 ('CC', None),
                                                 (None, None), 
                                                 ]},

                 'shift' : {'pred_field' : 'shift_mu',
                            'pred_field_std' : 'shift_std',
                            'groupby' : ['nuc', 'bond_partner'],
                            'stats_groups': [('1H', 6),
                                             ('1H', 7),
                                             ('1H', 8),
                                             ('1H', None),
                                             ('13C', None)],
                 }

}
                 

    
@transform([train_test_predict_shifts,
            train_test_predict_coupling],
           suffix(".meta.pickle"), 
           ".summary.pickle")
def compute_summary_stats(infile, outfile):
    meta_filename, _ = infile
    a = pickle.load(open(meta_filename, 'rb'))
    meta_df = a['meta']
    exp_config = a['config']
    print(infile[0])
    if 'coupling' in infile[0]:
        stats_config = STATS_CONFIGS['coupling']
    else:
        stats_config = STATS_CONFIGS['shift']
        
    cv_sets = exp_config['cv_sets']
    
    all_pred_df = []
    for set_idx, cv_i_set_params in enumerate(cv_sets):
        cv_func = netutil.CVSplit(**cv_i_set_params)

        for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
            pred_df = pd.read_feather(row.epoch_filename)
            print(pred_df.dtypes)
            pred_df['epoch'] = row.epoch
            pred_df['delta'] = pred_df[stats_config['pred_field']] - pred_df.value
            pred_df['delta_abs'] = np.abs(pred_df.delta)
            pred_df['cv_set_i'] = set_idx
            print("pred_df.dtype", pred_df.dtypes)
            pred_df['phase'] =  pred_df.apply(lambda pred_row : \
                                              cv_func.get_phase(None, 
                                                                {'morgan_fingerprint': pred_row.morgan4_crc32,
                                                                'smiles': pred_row.smiles}), 
                                              axis=1)
            print("Phase Value Counts:\n", pred_df['phase'].value_counts())

            all_pred_df.append(pred_df)
    all_pred_df = pd.concat(all_pred_df)

    groupby_fields = stats_config['groupby'] + ['cv_set_i', 'phase', 'epoch']
        
    pred_metrics = all_pred_df.groupby(groupby_fields).apply(metrics.compute_stats, 
                                                             mol_id_field='mol_id',
                                                             tgt_field = stats_config['pred_field'])

    feather_filename = outfile.replace(".pickle", ".feather")
    all_pred_df.reset_index(drop=True).to_feather(feather_filename)

    pickle.dump({'pred_metrics_df' : pred_metrics, 
                 'all_pred_df_filename' : feather_filename, 
                 'infile' : infile,
                 'stats_config': stats_config, 
                 'exp_config' : exp_config}, 
                open(outfile, 'wb'))


@transform(compute_summary_stats, 
           suffix(".summary.pickle"), 
           ".summary.txt")
def summary_textfile(infile, outfile):
    d = pickle.load(open(infile, 'rb'))
    pred_metrics_df = d['pred_metrics_df']

    with open(outfile, 'w') as fid:
        fid.write(pred_metrics_df.to_string())
        fid.write("\n")
    print("wrote", outfile)

@transform(compute_summary_stats, 
           suffix(".summary.pickle"), 
           ".per_conf_stats.pickle")
def per_confidence_stats(infile, outfile):
    """
    Generate the states broken down by conf
    """
    TGT_THOLDS = [0.1, 0.2, 0.5, 0.9, 0.95, 1.0]

    d = pickle.load(open(infile, 'rb'))
    stats_config = d['stats_config']

    
    all_pred_infile = d['all_pred_df_filename']
    df = pd.read_feather(all_pred_infile)
    all_mdf = []

    # stats_subsets
    for stats_group in stats_config['stats_groups']:
        df_sub = df
        for field_name, val in zip(stats_config['groupby'], stats_group):
            if val is not None:
                df_sub = df_sub[df_sub[field_name] == val]

        for (phase, epoch), g in df_sub.groupby(['phase', 'epoch_i']):
            BIN_N=400
            m, _, frac_data = metrics.sorted_bin_stats(np.array(g[stats_config['pred_field_std']]), 
                                                   np.array(g.delta_abs), BIN_N)

            idx = np.searchsorted(frac_data, TGT_THOLDS)
            mdf = pd.DataFrame({'mae' : m[idx], 'frac_data' : TGT_THOLDS})
            mdf['phase'] = phase
            mdf['epoch'] = epoch
            mdf['stats_group'] = str(stats_group)
            all_mdf.append(mdf)

    mdf = pd.concat(all_mdf)

    pickle.dump({'infile' : infile, 
                 'all_pred_df_filename' : all_pred_infile, 
                 "df" : mdf
                #  "min" : _min, 
                #  "max": _max
                 }, open(outfile, 'wb'))
    
    r = pd.pivot_table(mdf, index=['stats_group', 'phase', 'epoch'], columns=['frac_data'])

    with open(outfile.replace(".pickle", ".txt"), 'w') as fid:
        fid.write(r.to_string())
        fid.write('\n')

@transform([train_test_predict_shifts,
            train_test_predict_coupling],
           suffix(".meta.pickle"), 
           ".test_split.pickle")
def get_train_test_split(infile, outfile):
    meta_filename, _ = infile
    a = pickle.load(open(meta_filename, 'rb'))
    meta_df = a['meta']
    config = a['config']

    print("Writing out train/test split for", config['model'], "on", config['dataset'])
    all_df = pickle.load(open(config['dataset'], 'rb'))

    meta_filename = f"{config['model']}.meta"

    meta = pickle.load(open(meta_filename, 'rb'))

    tgt_df = all_df

    if 'max_atom_n' in config:
        tgt_df = tgt_df[tgt_df.rdmol.apply(lambda m : m.GetNumAtoms() <= config['max_atom_n'])]

    if 'elements_only' in config:
        eo = set(config['elements_only'])
        def is_sub(m):
            elts = set([a.GetSymbol() for a in m.GetAtoms()])
            return elts.issubset(eo)
        tgt_df = tgt_df[tgt_df.rdmol.apply(is_sub)]

    metafile = meta.get('metafile')
    if not metafile is None:
        records_meta = pickle.load(open(metafile, 'rb'))
        tgt_df = tgt_df[tgt_df.molecule_id in records_meta['ids']]

    cv_sets = config['cv_sets']
    
    tt_splits = []
    for set_idx, cv_i_set_params in enumerate(cv_sets):
        cv_func = netutil.CVSplit(**cv_i_set_params)
        split = {}
        for _, row in tgt_df.iterrows():
            p = cv_func.get_phase(None, {'morgan_fingerprint': row.morgan4_crc32,
                                            'smiles': row.smiles})
            split[p] = split.get(p, []) + [(row.molecule_id, Chem.MolToSmiles(row.rdmol, isomericSmiles=True))]
    
        tt_splits += [split]

    pickle.dump(tt_splits, open(outfile, 'wb'))



if __name__ == "__main__":
    pipeline_run([train_test_predict_shifts,
                  train_test_predict_coupling, 
                  compute_summary_stats, 
                  summary_textfile, per_confidence_stats,
                  get_train_test_split])
    
