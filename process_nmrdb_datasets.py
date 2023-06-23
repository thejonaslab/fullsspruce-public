"""

Takes a spect db and turns it into a pickle'd dataframe which is a dataset

The nmr db format used by specdata and nmrabinito are great but can be slow
to query and are intended as "formats of record". So we use
this script to convert them into pickled "datasets" suitable for training. 

"""


import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors as rdMD
import rdkit.Chem.rdForceFieldHelpers
import pickle
from fullsspruce import util
import zlib
import scipy.special

import sqlalchemy
from sqlalchemy import sql, func
from ruffus import *


### first filter through all the molecules that have the indicated
### kind of spectra

### then select on those spectra

### emit the resulting feather files with smiles, canonical spectra conditioning, spectra array

HCONF = ['H', 'C', 'O', 'N', 'F']
HCONFSPCl = ['H', 'C', 'O', 'N', 'F', 'S', 'P', 'Cl']

DEFAULT_COUPLING_CONFIG = {                  
    'max_atom_n' : 64, 
    'max_heavy_atom_n' : 64, 
    'elements' : HCONF, 
    'allow_radicals' : False, 
    'allow_atom_formal_charge' : False, 
    'max_ring_size' : 14, 
    'min_ring_size' : 2, 
    'allow_unknown_hybridization' : False, 
    # 'spectra_nuc' : ['13C', '1H'],
    'allow_mol_formal_charge' : False
}

COUPLING_DATASETS = {

    '128_128_HCONFSPCl_wcharge' : {'db_filename' :'/jonaslab/data/nmr/nmrab/coupling.nmrshiftdb.b3lyp-631gd.32a-e2-c16.coupling_g_fconly_DEBUG.db', 
                                   'source' : ['coupling.nmrshiftdb.b3lyp-631gd.32a-e2-c16.coupling_g_fconly_DEBUG'], 
                                   #  'spectra_nuc' : ['1H'], 
                                   'max_atom_n' : 128,
                                   'max_heavy_atom_n' : 128, 
                                   'elements' : HCONFSPCl,
                                   'allow_atom_formal_charge' : True,
                                   'allow_mol_formal_charge' : True}, 
    '128_128_HCONFSPCl_wcharge_ch3avg' : {'db_filename' :'/jonaslab/data/nmr/nmrab/coupling.nmrshiftdb.b3lyp-631gd.32a-e2-c16.coupling_g_fconly_DEBUG.db',
                                          'source' : ['coupling.nmrshiftdb.b3lyp-631gd.32a-e2-c16.coupling_g_fconly_DEBUG'], 
                                          #  'spectra_nuc' : ['1H'], 
                                          'max_atom_n' : 128, 
                                          'max_heavy_atom_n' : 128, 
                                          'elements' : HCONFSPCl,
                                          'methyl_coupling_avg' : True,
                                          'allow_atom_formal_charge' : True,
                                          'allow_mol_formal_charge' : True}, 
    'bally_rablen.128_128_HCONFSPCl_wcharge_ch3avg' : {'db_filename' :'/jonaslab/data/nmr/spectdata/bally_rablen.2011/bally_rablen_2011.db',
                                        #   'source' : ['coupling.nmrshiftdb.b3lyp-631gd.32a-e2-c16.coupling_g_fconly_DEBUG'], 
                                          #  'spectra_nuc' : ['1H'], 
                                          'max_atom_n' : 128, 
                                          'max_heavy_atom_n' : 128, 
                                          'elements' : HCONFSPCl,
                                          'methyl_coupling_avg' : True,
                                          'allow_atom_formal_charge' : True,
                                          'allow_mol_formal_charge' : True,
                                          'molecule_id_column': 'spectrum_id'}, 
    'kutateladze.exp_only.128_128_HCONFSPCl_wcharge_ch3avg' : {'db_filename' :'/jonaslab/data/nmr/spectdata/kutateladze2015/kutateladze2015.db',
                                          'source' : ['experimental'], 
                                          'spectra_source': ['experimental'],
                                          #  'spectra_nuc' : ['1H'], 
                                          'max_atom_n' : 128, 
                                          'max_heavy_atom_n' : 128, 
                                          'elements' : HCONFSPCl,
                                          'methyl_coupling_avg' : True,
                                          'allow_atom_formal_charge' : True,
                                          'allow_mol_formal_charge' : True,
                                          'molecule_id_column': 'spectrum_id'},
    'kutateladze.their_calc_only.128_128_HCONFSPCl_wcharge_ch3avg' : {'db_filename' :'/jonaslab/data/nmr/spectdata/kutateladze2015/kutateladze2015.db',
                                          'source' : ['their_calculation'], 
                                          'spectra_source': ['their_calculation'],
                                          #  'spectra_nuc' : ['1H'], 
                                          'max_atom_n' : 128, 
                                          'max_heavy_atom_n' : 128, 
                                          'elements' : HCONFSPCl,
                                          'methyl_coupling_avg' : True,
                                          'allow_atom_formal_charge' : True,
                                          'allow_mol_formal_charge' : True,
                                          'molecule_id_column': 'spectrum_id'}, 
    'gissmo.128_128_HCONFSPCl_wcharge_ch3avg' : {'db_filename' :'/jonaslab/data/nmr/spectdata/gissmo/gissmo.db',
                                        #   'source' : ['coupling.nmrshiftdb.b3lyp-631gd.32a-e2-c16.coupling_g_fconly_DEBUG'], 
                                          #  'spectra_nuc' : ['1H'], 
                                          'max_atom_n' : 128, 
                                          'max_heavy_atom_n' : 128, 
                                          'elements' : HCONFSPCl,
                                          'methyl_coupling_avg' : True,
                                          'allow_atom_formal_charge' : True,
                                          'allow_mol_formal_charge' : True,
                                          'molecule_id_column': 'spectrum_id'}, 
}






SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))


OUTPUT_DIR = "/jonaslab/data/nmr/processed_dbs/" 
td = lambda x : os.path.join(OUTPUT_DIR, x)

def params_couplings():
    for exp_name, ec in COUPLING_DATASETS.items():
        config = DEFAULT_COUPLING_CONFIG.copy()
        config.update(ec)

        yield (config['db_filename'], 
               (td("couplings.{}.mol.feather".format(exp_name)), 
                td("couplings.{}.spect.feather".format(exp_name)),  
                td("couplings.{}.meta.pickle".format(exp_name))),
               config)

@mkdir(OUTPUT_DIR)
@files(params_couplings)
def preprocess_data_coupling(db_infile, outfile, config):            

    DB_URL = f"sqlite+pysqlite:///{db_infile}"
    
    mol_outfile, coupling_outfile, meta_outfile = outfile

    engine = sqlalchemy.create_engine(DB_URL)
    conn = engine.connect()

    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)

    molecules = meta.tables['molecules']
    spectra = meta.tables['spectra']
    peaks = meta.tables['peaks']
    couplings = meta.tables['couplings']


    stmt = sql.select([molecules.c.id, molecules.c.source_id, molecules.c.morgan4_crc32, molecules.c.bmol])\
           .where(molecules.c.id == spectra.c.mol_id)
    ## filter by source
    if 'source' in config:
        stmt = stmt.where(spectra.c.source.in_(config['source']))

    # stmt = stmt.where(spectra.c.nucleus.in_(config['spectra_nuc']))

    stmt = stmt.distinct()
    print(str(stmt))

    output_mol_df, skip_reason_df = util.filter_mols(conn.execute(stmt), 
                                                     config, 
                                                     other_attributes = ['source_id', 'bmol', 'morgan4_crc32'],
                                                     mol_from_binary=True)
    if 'subsample_to' in config:
        output_mol_df = output_mol_df.sample(config['subsample_to'], random_state = 1234)
    
    print(output_mol_df.head())
    output_mol_df = output_mol_df.reset_index()
    output_mol_df.to_feather(mol_outfile)

    # now we select the spectra
    # stmt = sql.select([peaks.c.id, peaks.c.atom_idx, peaks.c.multiplicity, 
    #                    peaks.c.value, peaks.c.spectrum_id, spectra.c.mol_id])\
    #                   .where(peaks.c.spectrum_id == spectra.c.id) \
    #                   .where(spectra.c.nucleus.in_(config['spectra_nuc']))\
    #                   .where(spectra.c.molecule_id.in_(output_mol_df.molecule_id))
    # peak_df = pd.read_sql(stmt, engine)
    # peak_df.to_feather(spect_outfile)
    #skip_reason_df = pd.DataFrame(skip_reason)

    stmt = sql.select([couplings, spectra.c.mol_id]).where(couplings.c.spectrum_id == spectra.c.id)\
                                            .where(spectra.c.mol_id.in_(output_mol_df.molecule_id))

    if 'spectra_source' in config:
        stmt = stmt.where(spectra.c.source.in_(config['spectra_source']))

    J_df = pd.read_sql(stmt, engine).reset_index()
    J_df.to_feather(coupling_outfile)

                       
    print(skip_reason_df.reason.value_counts())

    pickle.dump({'skip_reason_df' : skip_reason_df, 
                 'config' : config},
                open(meta_outfile, 'wb'))


@transform(preprocess_data_coupling, 
           suffix(".mol.feather"), 
           ".dataset.pickle")
def create_clean_dataset_coupling(infiles, outfile):
    # shifts_infile, spect_infile, coupling_infile, meta_infile = infiles
    mol_infile, coupling_infile, meta_infile = infiles

    meta_input = pickle.load(open(meta_infile, 'rb'))

    methyl_coupling_avg = meta_input['config'].get('methyl_coupling_avg', False)
    m_id_col = meta_input['config'].get('molecule_id_column', 'source_id')

    mol_df = pd.read_feather(mol_infile).set_index('molecule_id')

    print("mol_df.dtypes=")
    print(mol_df.dtypes)
    print("len(mol_df)=", len(mol_df))

    # spect_df = pd.read_feather(spect_infile)
    coupling_df = pd.read_feather(coupling_infile)
    print("len(coupling_df)=", len(coupling_df))

    def smaller_first(a, b):
        if a < b:
            return a, b
        else:
            return b, a

    results = []
    for (molecule_id, spectrum_id), g in \
        tqdm(coupling_df.groupby(['mol_id', 'spectrum_id']),
             total=len(coupling_df[['mol_id', 'spectrum_id']].drop_duplicates())):
        mol_row = mol_df.loc[molecule_id]

        mol = Chem.Mol(zlib.decompress(mol_row.bmol))
        N = mol.GetNumAtoms()

        # spect_dict = [{row['atom_idx'] : row['value'] for _, row in g.iterrows()}]

        # coupling_sub_df = coupling_df[(coupling_df.spectrum_id == spectrum_id) & (coupling_df.molecule_id == molecule_id)]
        coupling_dict = {}
        for _, r in g.iterrows():
            if r.atom_idx1 in range(N) and r.atom_idx2 in range(N):
                coupling_dict[smaller_first(r.atom_idx1, r.atom_idx2)] = r.value
        
        if methyl_coupling_avg:
            try:
                coupling_dict = util.methyl_average_coupling(mol, coupling_dict)
            except Exception as e:
                print('Mol', int(mol_row[m_id_col]), 'skipped due to', e)
                continue
        results.append({'molecule_id' : molecule_id, #int(mol_row[m_id_col]), 
                        'rdmol' : mol, 
    #                     'spect_dict': spect_dict,
                        'coupling_dict' : coupling_dict, 
                        'coupling_types': get_coupling_types(mol),
                        'smiles' : mol_row.simple_smiles, 
                        'morgan4_crc32' : mol_row.morgan4_crc32,
                        'spectrum_id' : spectrum_id})
    results_df = pd.DataFrame(results)
    print(results_df.morgan4_crc32.value_counts())

    print("dataset has", len(results_df), "rows")
    pickle.dump(results_df, open(outfile,'wb'))


def get_coupling_types(m):
    N = m.GetNumAtoms()
    out = []
    for i in range(N):
        for j in range(i+1, N):
            p = Chem.rdmolops.GetShortestPath(m, i, j)
            bonds_between = len(p) - 1
            symb_1 = m.GetAtomWithIdx(i).GetSymbol()
            symb_2 = m.GetAtomWithIdx(j).GetSymbol()
            symb_sorted = sorted([symb_1, symb_2])

            out.append({'atomidx_1' : i, 
                        'atomidx_2' : j, 
                        'atomelt_1' : symb_1, 
                        'atomelt_2' : symb_2, 
                        'coupling_type' : f"{symb_sorted[0]}{symb_sorted[1]}", 
                        'coupling_dist': bonds_between, 
                       })
    return pd.DataFrame(out)


SHIFT_DATASETS = {
    'nmrshiftdb.128_128_HCONFSPCl_wcharge_13C' : {'db_filename':"/jonaslab/data/nmr/spectdata/nmrshiftdb/nmrshiftdb.db", 
                                                  'source' : ['nmrshiftdb'], 
                                                  'spectra_nuc' : ['13C'], 
                                                  'max_atom_n' : 128, 
                                                  'max_hevy_atom_n' : 128,
                                                  'elements' : HCONFSPCl,
                                                  'allow_atom_formal_charge' : True,
                                                  'allow_mol_formal_charge' : True}, 
    'nmrshiftdb.128_128_HCONFSPCl_wcharge_1H' : {'db_filename':"/jonaslab/data/nmr/spectdata/nmrshiftdb/nmrshiftdb.db",
                                                 'source' : ['nmrshiftdb'], 
                                                 'spectra_nuc' : ['1H'], 
                                                 'max_atom_n' : 128, 
                                                 'max_hevy_atom_n' : 128,
                                                 'elements' : HCONFSPCl,
                                                 'allow_atom_formal_charge' : True,
                                                 'allow_mol_formal_charge' : True}, 
    'nmrshiftdb_ab.b3lyp-631gd.32a-e2-c16.g_m6311_cdcl.32_32_HCONFSPCl_wcharge_13C' :
    {'db_filename':"/jonaslab/data/nmr/nmrab/iso.nmrshiftdb.b3lyp-631gd.32a-e2-c16.g_m6311_cdcl.db",
     'spectra_nuc' : ['13C'], 
     'max_atom_n' : 32,
     'mol_id_field' : 'source_id',      
     
     'max_hevy_atom_n' : 32,
     'elements' : HCONFSPCl,
     'allow_atom_formal_charge' : True,
     'allow_mol_formal_charge' : True}, 
    'nmrshiftdb_ab.b3lyp-631gd.32a-e2-c16.g_m6311_cdcl.32_32_HCONFSPCl_wcharge_1H' :
    {'db_filename':"/jonaslab/data/nmr/nmrab/iso.nmrshiftdb.b3lyp-631gd.32a-e2-c16.g_m6311_cdcl.db",
     'spectra_nuc' : ['1H'], 
     'max_atom_n' : 32, 
     'max_hevy_atom_n' : 32,
     'mol_id_field' : 'source_id',      
     'elements' : HCONFSPCl,
     'allow_atom_formal_charge' : True,
     'allow_mol_formal_charge' : True}, 

    'nmrshiftdb_ab.b3lyp-631gd.32a-e2-c16.g_m6311_cdcl.32_32_HCONFSPCl_wcharge_13C_ch3avg' :
       {'db_filename':"/jonaslab/data/nmr/nmrab/iso.nmrshiftdb.b3lyp-631gd.32a-e2-c16.g_m6311_cdcl.db",
        'spectra_nuc' : ['13C'], 
        'max_atom_n' : 32, 
        'max_hevy_atom_n' : 32,
        'elements' : HCONFSPCl,
        'mol_id_field' : 'source_id', 
        'methyl_shift_avg' : True,     
        'allow_atom_formal_charge' : True,
        'allow_mol_formal_charge' : True}, 
    'nmrshiftdb_ab.b3lyp-631gd.32a-e2-c16.g_m6311_cdcl.32_32_HCONFSPCl_wcharge_1H_ch3avg' :
       {'db_filename':"/jonaslab/data/nmr/nmrab/iso.nmrshiftdb.b3lyp-631gd.32a-e2-c16.g_m6311_cdcl.db",
        'spectra_nuc' : ['1H'], 
        'max_atom_n' : 32, 
        'max_hevy_atom_n' : 32,
        'mol_id_field' : 'source_id', 
        'elements' : HCONFSPCl,
        'methyl_shift_avg' : True,     
        'allow_atom_formal_charge' : True,
        'allow_mol_formal_charge' : True}, 

    # 'gdb17-stereo-dft.b3lyp-631gd.0-48a-e2-c16.m6311_smd_cdcl.75pct.48_48_HCONFSPCl_wcharge_13C_ch3avg' :
    #    {'db_filename':"/jonaslab/data/nmr/nmrab/iso.gdb17-stereo-dft.b3lyp-631gd.0-48a-e2-c16.m6311_smd_cdcl.75pct.db",
    #     'spectra_nuc' : ['13C'], 
    #     'max_atom_n' : 48, 
    #     'max_hevy_atom_n' : 48,
    #     'elements' : HCONFSPCl,
    #     'sanitize_mol' : True, 
    #     'methyl_shift_avg' : True,     
    #     'allow_atom_formal_charge' : True,
    #     'allow_mol_formal_charge' : True},
    
    # 'gdb17-stereo-dft.b3lyp-631gd.0-48a-e2-c16.m6311_smd_cdcl.75pct.48_48_HCONFSPCl_wcharge_1H_ch3avg' :
    #    {'db_filename':"/jonaslab/data/nmr/nmrab/iso.gdb17-stereo-dft.b3lyp-631gd.0-48a-e2-c16.m6311_smd_cdcl.75pct.db",
    #     'spectra_nuc' : ['1H'], 
    #     'max_atom_n' : 48, 
    #     'max_hevy_atom_n' : 48,
    #     'elements' : HCONFSPCl,
    #     'sanitize_mol' : True, 
    #     'methyl_shift_avg' : True,
    #     'allow_atom_formal_charge' : True,
    #     'allow_mol_formal_charge' : True}, 

    
    # 'gdb17-stereo-dft.b3lyp-631gd.0-48a-e2-c16.m6311_smd_cdcl.48_48_HCONFSPCl_wcharge_13C_ch3avg' :
    #    {'db_filename':"/jonaslab/data/nmr/nmrab/iso.gdb17-stereo-dft.b3lyp-631gd.0-48a-e2-c16.m6311_smd_cdcl.db",
    #     'spectra_nuc' : ['13C'], 
    #     'max_atom_n' : 48, 
    #     'max_hevy_atom_n' : 48,
    #     'elements' : HCONFSPCl,
    #     'sanitize_mol' : True,
    #     'mol_id_field' : 'source_id', 
    #     'methyl_shift_avg' : True,     
    #     'allow_atom_formal_charge' : True,
    #     'allow_mol_formal_charge' : True},
    
    # 'gdb17-stereo-dft.b3lyp-631gd.0-48a-e2-c16.m6311_smd_cdcl.48_48_HCONFSPCl_wcharge_1H_ch3avg' :
    #    {'db_filename':"/jonaslab/data/nmr/nmrab/iso.gdb17-stereo-dft.b3lyp-631gd.0-48a-e2-c16.m6311_smd_cdcl.db",
    #     'spectra_nuc' : ['1H'], 
    #     'max_atom_n' : 48, 
    #     'max_hevy_atom_n' : 48,
    #     'elements' : HCONFSPCl,
    #     'sanitize_mol' : True, 
    #     'mol_id_field' : 'source_id', 
    #     'methyl_shift_avg' : True,
    #     'allow_atom_formal_charge' : True,
    #     'allow_mol_formal_charge' : True}, 
    
    # 'exp5K.128_128_HCONFSPCl_wcharge_13C' : {'db_filename':"/jonaslab/data/nmr/spectdata/cascade-8k/exp5k.dft.db",  
    #                                               'spectra_nuc' : ['13C'], 
    #                                               'max_atom_n' : 128, 
    #                                               'max_hevy_atom_n' : 128,
    #                                               'elements' : HCONFSPCl,
    #                                               'allow_atom_formal_charge' : True,
    #                                               'allow_mol_formal_charge' : True}, 

    # 'bmrb_metabolomics.128_128_HCONFSPCl_wcharge_13C' : {'db_filename':"/jonaslab/data/nmr/spectdata/bmrb-metabolomics/bmrb_metabolomics.db",  
    #                                               'spectra_nuc' : ['13C'], 
    #                                               'max_atom_n' : 128, 
    #                                               'max_hevy_atom_n' : 128,
    #                                               'elements' : HCONFSPCl,
    #                                             #   'allow_unknown_hybridization': True,
    #                                               'allow_atom_formal_charge' : True,
    #                                               'allow_mol_formal_charge' : True}, 

    # 'bmrb_metabolomics.128_128_HCONFSPCl_wcharge_1H' : {'db_filename':"/jonaslab/data/nmr/spectdata/bmrb-metabolomics/bmrb_metabolomics.db",  
    #                                               'spectra_nuc' : ['1H'], 
    #                                               'max_atom_n' : 128, 
    #                                               'max_hevy_atom_n' : 128,
    #                                               'elements' : HCONFSPCl,
    #                                             #   'allow_unknown_hybridization': True,
    #                                               'allow_atom_formal_charge' : True,
    #                                               'allow_mol_formal_charge' : True}, 
}


DEFAULT_SHIFT_CONFIG = {                  
    'max_atom_n' : 64, 
    'max_heavy_atom_n' : 64, 
    'elements' : HCONF, 
    'allow_radicals' : False, 
    'allow_atom_formal_charge' : False, 
    'max_ring_size' : 14, 
    'min_ring_size' : 2, 
    'allow_unknown_hybridization' : False, 
    'spectra_nuc' : ['13C', '1H'],
    'allow_mol_formal_charge' : False
}


def params_shifts():
    for exp_name, ec in SHIFT_DATASETS.items():
        config = DEFAULT_SHIFT_CONFIG.copy()
        config.update(ec)

        yield (config['db_filename'], 
               (td("shifts.{}.mol.feather".format(exp_name)), 
                td("shifts.{}.spect.feather".format(exp_name)),  
                td("shifts.{}.meta.pickle".format(exp_name))),
               config)

@mkdir(OUTPUT_DIR)
@files(params_shifts)
def preprocess_data_shifts(db_infile, outfile, config):            

    DB_URL = f"sqlite+pysqlite:///{db_infile}"
    mol_outfile, spect_outfile, meta_outfile = outfile
    ### construct the query

    assert os.path.exists(db_infile)
    engine = sqlalchemy.create_engine(DB_URL)
    conn = engine.connect()
    meta = sqlalchemy.MetaData()
    
    meta.reflect(bind=engine)

    molecules = meta.tables['molecules']
    spectra = meta.tables['spectra']
    peaks = meta.tables['peaks']
    couplings = meta.tables['couplings']


    stmt = sql.select([molecules.c.id, molecules.c.bmol,
                       molecules.c.source_id, 
                       molecules.c.smiles,
                       molecules.c.morgan4_crc32])\
            .where(molecules.c.id == spectra.c.mol_id).where(molecules.c.smiles != '')
    ## filter by source
    if 'source' in config:
        stmt = stmt.where(spectra.c.source.in_(config['source']))

    stmt = stmt.where(spectra.c.nucleus.in_(config['spectra_nuc']))
    
    stmt = stmt.distinct()
    print(str(stmt))

    output_mol_df, skip_reason_df = util.filter_mols(conn.execute(stmt), 
                                                     config, 
                                                     other_attributes = ['bmol', 'morgan4_crc32', 'source_id'],
                                                     mol_from_binary=True,
                                                     sanitize = config.get("sanitize_mol", False))
    print(skip_reason_df.value_counts())

    assert len(output_mol_df) > 0 
    print('returned', len(output_mol_df), len(skip_reason_df))
    if 'subsample_to' in config:
        output_mol_df = output_mol_df.sample(config['subsample_to'], random_state = 1234)
    
    print(output_mol_df.head())
    output_mol_df = output_mol_df.reset_index()
    output_mol_df.to_feather(mol_outfile)

    # now we select the spectra
    stmt = sql.select([peaks.c.id, peaks.c.atom_idx, peaks.c.multiplicity, 
                       peaks.c.value, peaks.c.spectrum_id, spectra.c.mol_id])\
                      .where(peaks.c.spectrum_id == spectra.c.id)\
                      .where(spectra.c.nucleus.in_(config['spectra_nuc']))\
                      .where(spectra.c.mol_id.in_(output_mol_df.molecule_id))

    print('sql stmt is', stmt)
    peak_df = pd.read_sql(stmt, engine)
    assert len(peak_df) > 0 
    peak_df.to_feather(spect_outfile)
    #skip_reason_df = pd.DataFrame(skip_reason)

    print(skip_reason_df.reason.value_counts())

    pickle.dump({'skip_reason_df' : skip_reason_df, 
                 'config' : config},
                open(meta_outfile, 'wb'))


@transform(preprocess_data_shifts, 
           suffix(".mol.feather"), 
           ".dataset.pickle")
def create_clean_dataset_shifts(infiles, outfile):
    shifts_infile, spect_infile, meta_infile = infiles

    mol_df = pd.read_feather(shifts_infile).set_index('molecule_id')
    spect_df = pd.read_feather(spect_infile)

    meta_input = pickle.load(open(meta_infile, 'rb'))

    methyl_shift_avg = meta_input['config'].get('methyl_shift_avg', False)

    mol_id_field = meta_input['config'].get("mol_id_field", 'molecule_id')

    results = []
    for (molecule_id, spectrum_id), g in spect_df.groupby(['mol_id', 'spectrum_id']):
        # mol_rows = output_mol_df.loc[output_mol_df['molecule_id'] == molecule_id]
        # assert len(mol_rows) == 1
        # mol_row = mol_rows.iloc[0]
        mol_row = mol_df.loc[molecule_id]
        spect_dict = [{row['atom_idx'] : row['value'] for _, row in g.iterrows()}]
        mol = Chem.Mol(zlib.decompress(mol_row.bmol))

        if mol_id_field != 'molecule_id':
            molecule_id = mol_row[mol_id_field]
            
        if methyl_shift_avg:
            for c, hs in util.get_methyl_hydrogens(mol):
                avg_val = []
                avg_tgts = []
                for h in hs:
                    if h in spect_dict[0]:
                        avg_tgts.append(h)
                        avg_val.append(spect_dict[0][h])
                for a in avg_tgts:
                    spect_dict[0][a] = np.mean(avg_val)
            
        results.append({'molecule_id' : int(molecule_id), 
                        'rdmol' : mol, 
                        'spect_dict': spect_dict, 
                        'smiles' : mol_row.simple_smiles, 
                        'morgan4_crc32' : mol_row.morgan4_crc32, # util.morgan4_crc32(mol)
                        'spectrum_id' : spectrum_id})
    results_df = pd.DataFrame(results)
    print(results_df.morgan4_crc32.value_counts())

    print("dataset has", len(results_df), "rows")
    pickle.dump(results_df, open(outfile,'wb'))




if __name__ == "__main__":
    pipeline_run([preprocess_data_coupling, create_clean_dataset_coupling,
                  preprocess_data_shifts, create_clean_dataset_shifts],
                 checksum_level=0, multiprocess=8)
