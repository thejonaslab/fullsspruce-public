import pickle
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from tqdm import tqdm
tqdm.pandas()
import os
import yaml
from ruffus import *
import copy

import scipy
from scipy.special import logsumexp

from fullsspruce.predictor import Predictor
from fullsspruce.geom_util import geometryFeaturizerGenerator, DEFAULT_GAUSS_BINS
from fullsspruce.etkdg import get_viable_stereoisomers
from fullsspruce.metrics import calculate_DP4

import rdkit
from rdkit import Chem

EXPERIMENTS = {
    'GDB_NMRshiftDBmodel': {
        'model_1H': {
            'meta': 'fullsspruce/checkpoints/fs_def_1H_5_27.proton_GNN.537094279807.meta',
            'checkpoint': 'fullsspruce/checkpoints/fs_def_1H_5_27.proton_GNN.537094279807.00000500'
        },
        'model_13C': {
            'meta': 'fullsspruce/checkpoints/fs_def_13C_5_27.carbon_GNN.537094139799.meta',
            'checkpoint': 'fullsspruce/checkpoints/fs_def_13C_5_27.carbon_GNN.537094139799.00000450'
        },
        'featurizer': ('geom_util', {
            'geoms': [],
            'func': "empty"
        }),
        'dataset_1H': 'processed_dbs/shifts.gdb17-stereo-dft.b3lyp-631gd.0-48a-e2-c16.m6311_smd_cdcl.48_48_HCONFSPCl_wcharge_1H_eqHavg.dataset.pickle',
        'dataset_13C': 'processed_dbs/shifts.gdb17-stereo-dft.b3lyp-631gd.0-48a-e2-c16.m6311_smd_cdcl.48_48_HCONFSPCl_wcharge_13C_ch3avg.dataset.pickle',
        'which_mols': 'single_copy'
    },
}

OUTPUT_DIR = "dp4_analyses/" 
td = lambda x : os.path.join(OUTPUT_DIR, x)

def params():
    for exp_name, config in EXPERIMENTS.items():
        infiles = [config['dataset_1H'], config['dataset_13C']]
        
        outfiles = [td(f"{exp_name}.meta.pickle"),
                    td(f"{exp_name}.preds.pickle")]

        yield infiles, outfiles, exp_name, config

@mkdir(OUTPUT_DIR)
@files(params)
def get_preds(infiles, outfiles, exp_name, config):
    """
    For each Mol that we want to analyze, get the 1H and 13C predictions for each 
    of its conformers.
    """
    meta_outfile, data_outfile = outfiles

    geo = {}
    if 'featurizer' in config:
        g, feats = config['featurizer']
        if g == 'geom_util':
            geo = geometryFeaturizerGenerator(**feats)
        else:
            geo = feats

    predictor = Predictor(models = {'1H': config['model_1H'], '13C': config['model_13C']},
                          geometry = geo,
                          use_gpu = True,
                          prog_bar = True,
                          batch_size = config.get('batch_size', 128)) 

    print('Loading and Merging Datasets:')
    proton_df = pickle.load(open(config['dataset_1H'], 'rb'))
    carbon_df = pickle.load(open(config['dataset_13C'], 'rb'))

    subsample = config.get('subsample', max(len(proton_df), len(carbon_df)))

    if config['which_mols'] == 'single_copy':
        proton_df = proton_df.drop_duplicates(subset=['smiles'])
        carbon_df = carbon_df.drop_duplicates(subset=['smiles'])
        all_df = proton_df.merge(carbon_df, on=['molecule_id', 'smiles', 'morgan4_crc32'], suffixes = ['_1H', '_13C'])
        all_df['isomer'] = 0
        new_rows = []
        t = len(all_df)
        for _, row in tqdm(all_df.iloc[:subsample].iterrows(), total=subsample):
            # m = Chem.MolFromSmiles(Chem.MolToSmiles(row['rdmol_1H'], isomericSmiles=False))
            m = row['rdmol_1H']
            Chem.AssignStereochemistry(m, force=True, flagPossibleStereoCenters=True)
            for ind, m_i in enumerate(get_viable_stereoisomers(m, opts={'unique': True, 'onlyUnassigned': False}, num=7, max_embed_attempts=20, ignoreFirst=True)):
                m_i = Chem.AddHs(m_i)
                # for i in range(m_i.GetNumAtoms()):
                #     print(i, m_i.GetAtomWithIdx(i).GetAtomicNum())
                new_row = copy.deepcopy(row)
                new_row['isomer'] = ind + 1
                new_row['rdmol_1H'] = m_i
                new_row['rdmol_13C'] = m_i
                new_rows += [new_row]
        all_df = pd.concat([all_df[:subsample], pd.DataFrame(new_rows)])
    else:
        print('Unrecognized stereoisomer selection method.')
        quit()
            
    print('Predicting Stereoisomers')

    preds, meta = predictor.predict(list(all_df.rdmol_1H), properties = ['1H', '13C'])
    all_df['preds'] = preds

    print("Removing rows which did not make all predictions.")
    all_df['valid'] = all_df.progress_apply(lambda x: np.nan if (x['preds'] is None or x['preds']['1H'] == [] or x['preds']['13C'] == []) else 1, axis=1)
    all_df = all_df.dropna()

    all_df = all_df.drop(columns=['rdmol_1H', 'rdmol_13C', 'valid'])
    pickle.dump(all_df, open(data_outfile, 'wb'))
    meta['data_filename'] = data_outfile

    pickle.dump({'config': config,
                 'meta': meta},
                 open(meta_outfile, 'wb'))

def assign_DP4(row):
    """
    Take in a row from a dataframe and assign its DP4 probability
    """
    comp, pred = {}, {}
    comp['1H'] = row['spect_dict_1H'][0]
    comp['13C'] = row['spect_dict_13C'][0]
    pred['1H'] = {}
    for d in row['preds']['1H']:
        pred['1H'][d['atom_idx']] = d['pred_mu']
    pred['13C'] = {}
    for d in row['preds']['13C']:
        pred['13C'][d['atom_idx']] = d['pred_mu']
    # print("Comp:", comp)
    # print("Pred:", pred)
    return calculate_DP4(comp, pred)

@transform([get_preds],
            suffix(".meta.pickle"),
            ".dp4.pickle")
def get_dp4(infile, outfile):
    meta_filename, data_filename = infile
    a = pickle.load(open(meta_filename, 'rb'))
    meta = a['meta']
    exp_config = a['config']
    all_df = pickle.load(open(data_filename, 'rb'))
    print("Getting DP4 assignments.")
    all_df['DP4_ind'] = all_df.progress_apply(assign_DP4, axis=1)
    all_df['DP4'] = all_df['DP4_ind'].apply(lambda x: x[0] + x[1])

    feather_filename = outfile.replace('.pickle', ".feather")

    all_df = all_df.drop(columns=['spect_dict_1H', 'spect_dict_13C', 'preds'])

    all_df = all_df.reset_index()
    all_df.to_feather(feather_filename)
    
    pickle.dump({'all_df_filename': feather_filename,
                 'infile': infile,
                 'config': exp_config},
                 open(outfile, 'wb'))

@transform(get_dp4,
            suffix('.dp4.pickle'),
            ".summary.feather")
def get_stats(infile, outfile):
    d = pickle.load(open(infile, 'rb'))
    all_df = pd.read_feather(d['all_df_filename'])
    per_mol_stats, per_mol_1H_stats, per_mol_13C_stats = [], [], []

    print("Calculating DP4 probabilities.")

    for (m_id, smiles, morgan4_crc32), g in all_df.groupby(['molecule_id', 'smiles', 'morgan4_crc32']):
        denom = logsumexp(g.DP4)
        sp = list(np.exp(g.DP4 - denom))
        g['Stereo_Prob'] = sp
        dp4_ind = pd.DataFrame(list(g['DP4_ind']), columns=['1H', '13C'])
        denom_p = logsumexp(dp4_ind['1H'])
        sp_p = list(np.exp(dp4_ind['1H'] - denom_p))
        g['Stereo_Prob_1H'] = sp_p
        denom_c = logsumexp(dp4_ind['13C'])
        sp_c = list(np.exp(dp4_ind['13C'] - denom_c))
        g['Stereo_Prob_13C'] = sp_c

        stats = pd.Series({'molecule_id': m_id, 'smiles': smiles})
        for i, p in enumerate(sp):
            stats['prob_' + str(i)] = p
        # stats['probabilities'] = g.Stereo_Prob
        stats['correct'] = (np.argmax(list(g.DP4)) == 0)
        stats['improvement'] = (g.Stereo_Prob.iloc[0]*np.count_nonzero(sp))
        stats['N'] = np.argmax(list(g.DP4))
        stats['probability_of_correct_assignment'] = g.Stereo_Prob.iloc[0]
        stats['highest_incorrect'] = np.max(g.Stereo_Prob.iloc[1:])
        try:
            stats['second_highest'] = np.partition(list(g.Stereo_Prob), -2)[-2]
        except ValueError:
            stats['second_highest'] = None

        # Proton
        stats_p = pd.Series({'molecule_id': m_id, 'smiles': smiles})
        for i, p in enumerate(sp_p):
            stats_p['prob_' + str(i)] = p
        # stats_p['probabilities'] = g.Stereo_Prob_1H
        stats_p['correct'] = (np.argmax(list(sp_p)) == 0)
        stats_p['improvement'] = (g.Stereo_Prob_1H.iloc[0]*np.count_nonzero(sp_p))
        stats_p['N'] = np.argmax(list(sp_p))
        stats_p['probability_of_correct_assignment'] = g.Stereo_Prob_1H.iloc[0]
        stats_p['highest_incorrect'] = np.max(g.Stereo_Prob_1H.iloc[1:])
        try:
            stats_p['second_highest'] = np.partition(list(g.Stereo_Prob_1H), -2)[-2]
        except ValueError:
            stats_p['second_highest'] = None

        # Carbon
        stats_c = pd.Series({'molecule_id': m_id, 'smiles': smiles})
        for i, p in enumerate(sp_c):
            stats_c['prob_' + str(i)] = p
        # stats_c['probabilities'] = g.Stereo_Prob_13C
        stats_c['correct'] = (np.argmax(list(sp_c)) == 0)
        stats_c['improvement'] = (g.Stereo_Prob_13C.iloc[0]*np.count_nonzero(sp_c))
        stats_c['N'] = np.argmax(list(sp_c))
        stats_c['probability_of_correct_assignment'] = g.Stereo_Prob_13C.iloc[0]
        stats_c['highest_incorrect'] = np.max(g.Stereo_Prob_13C.iloc[1:])
        try:
            stats_c['second_highest'] = np.partition(list(g.Stereo_Prob_13C), -2)[-2]
        except ValueError:
            stats_c['second_highest'] = None

        per_mol_stats += [stats]
        per_mol_1H_stats += [stats_p]
        per_mol_13C_stats += [stats_c]

    per_mol_stats = pd.DataFrame(per_mol_stats)
    per_mol_stats = per_mol_stats.reset_index()
    per_mol_1H_stats = pd.DataFrame(per_mol_1H_stats)
    per_mol_1H_stats = per_mol_1H_stats.reset_index()
    per_mol_13C_stats = pd.DataFrame(per_mol_13C_stats)
    per_mol_13C_stats = per_mol_13C_stats.reset_index()

    all_stats = [per_mol_stats, per_mol_1H_stats, per_mol_13C_stats]

    r = pd.DataFrame({'Percent Correct': [np.sum(ms.correct)/len(ms) for ms in all_stats],
                      'Total Correct': [np.sum(ms.correct) for ms in all_stats],
                      'Total Evaluated': [len(ms) for ms in all_stats],
                      'Mean Correct Probability': [np.nanmean(ms.probability_of_correct_assignment) for ms in all_stats],
                      'Median Correct Probability': [np.nanmedian(ms.probability_of_correct_assignment) for ms in all_stats],
                      'Improvement Factor': [np.nanmean(ms.improvement) for ms in all_stats],
                      'Top-2': [np.sum(ms.N <= 2) for ms in all_stats],
                      'Top-3': [np.sum(ms.N <= 3) for ms in all_stats]}, 
                      index=['DP4', '1H', '13C'])

    with open(outfile.replace(".feather", ".txt"), 'w') as fid:
        print('writing')
        fid.write(r.to_string())
        fid.write('\n')
        print(outfile.replace(".pickle", ".txt"))

    per_mol_stats.to_feather(outfile)
    per_mol_1H_stats.to_feather(outfile.replace(".feather", ".1H.feather"))
    per_mol_13C_stats.to_feather(outfile.replace(".feather", ".13C.feather"))

if __name__ == '__main__':
    pipeline_run([get_preds, get_dp4, get_stats], checksum_level=0)