import numpy as np
import pandas as pd
from numpy.random import default_rng
from tqdm import tqdm
import os
from fullsspruce import netutil
import rdkit
import math
import copy

from rdkit import Chem
from rdkit.Chem import AllChem

import tinygraph.io.rdkit
import tinygraph as tg

import yaml
import pickle
from ruffus import *
from glob import glob
import zlib
from scipy.stats import norm

from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker
from fullsspruce import util
from fullsspruce import geom_util

from fullsspruce.featurize import molecule_features
from fullsspruce.geom_util import get_mask_values, getBoundsMatrixCorrected, geometryFeaturizerGenerator, best_VM_model

# from vonmises.predictor import *
from fullsspruce import etkdg

from nonconformity import model
from nonconformity.generate_confs import load_whole_db, dbconfset_to_rdmol

import sys

DATASETS = {
    # 'nmrshiftdb_rdkit_distances': {'files': '/jonaslab/data/nmr/spectdata/nmrshiftdb/*.db',
    #                                 'feature_generator': 'rdkit',
    #                                 'feature': 'mean_distance_mat'},
    # 'nmrshiftdb_VM_distances_test': {'files': '/jonaslab/data/nmr/spectdata/nmrshiftdb/*.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat'],
    #                                     'func': 'VM',
    #                                     'exp_config': {
    #                                         'model': best_VM_model, 
    #                                         'max_bonds': 4,
    #                                         'mean_mask_choice': 'rdkit'
    #                                     } 
    #                                 }},
    # 'nmrshiftdb_VM_angles_test': {'files': '/jonaslab/data/nmr/spectdata/nmrshiftdb/*.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'feature': 'VM_angle',
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat'],
    #                                     'func': 'VM',
    #                                     'exp_config': {
    #                                         'model': best_VM_model
    #                                     }
    #                                 }
    # },
    # 'nmrshiftdb_PT_distances_means': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                     'feature_generator': 'geometry_generator',
    #                                     'f_config': {
    #                                         'geoms': ['mean_distance_mat'],
    #                                         'func': 'confs',
    #                                         'prog_bar': True
    #                                     }
    # },
    # 'nmrshiftdb_ETKDG_distances_means': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                     'feature_generator': 'geometry_generator',
    #                                     'f_config': {
    #                                         'geoms': ['mean_distance_mat'],
    #                                         'func': 'ETKDG',
    #                                         'exp_config': {
    #                                             'num_confs': 50,
    #                                             'max_attempts': 50
    #                                         },
    #                                         'prog_bar': True
    #                                     }
    # },
    # 'nmrshiftdb_ETKDG_distances_means_max_15': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                     'feature_generator': 'geometry_generator',
    #                                     'f_config': {
    #                                         'geoms': ['mean_distance_mat'],
    #                                         'func': 'ETKDG',
    #                                         'exp_config': {
    #                                             'num_confs': 50,
    #                                             'max_attempts': 50,
    #                                             'max_bonds': 4, 
    #                                             'mean_mask_choice': 'rdkit'
    #                                         },
    #                                         'prog_bar': True
    #                                     }
    # },
    # 'nmrshiftdb_PT_distances_means_max_15': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                             'feature_generator': 'geometry_generator',
    #                                             'f_config': {
    #                                                 'geoms': ['mean_distance_mat'],
    #                                                 'func': 'confs',
    #                                                 'exp_config': {
    #                                                     'max_bonds': 4,
    #                                                     'mean_mask_choice': 'rdkit' 
    #                                                 },
    #                                                 'prog_bar': True
    #                                             }
    # },
    # 'nmrshiftdb_PT_distances_means_max_18': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                             'feature_generator': 'geometry_generator',
    #                                             'f_config': {
    #                                                 'geoms': ['mean_distance_mat'],
    #                                                 'func': 'confs',
    #                                                 'exp_config': {
    #                                                     'max_bonds': 7,
    #                                                     'mean_mask_choice': 'rdkit' 
    #                                                 },
    #                                                 'prog_bar': True
    #                                             }
    # },                                  
    # 'nmrshiftdb_PT_gauss_bins_default': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                         'feature_generator': 'geometry_generator',
    #                                         'f_config': {
    #                                             'geoms': ['conf_gauss_bins'],
    #                                             'func': 'confs',
    #                                             'exp_config': {
    #                                                 'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                             },
    #                                             'prog_bar': True
    #                                         }
    # },
    # 'nmrshiftdb_ETKDG_gauss_bins_default': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                         'feature_generator': 'geometry_generator',
    #                                         'f_config': {
    #                                             'geoms': ['conf_gauss_bins'],
    #                                             'func': 'ETKDG',
    #                                             'exp_config': {
    #                                                 'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                                 'num_confs': 50,
    #                                                 'max_attempts': 50
    #                                             },
    #                                             'prog_bar': True
    #                                         }
    # },
    # 'nmrshiftdb_ETKDG_gauss_bins_default_15': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                         'feature_generator': 'geometry_generator',
    #                                         'f_config': {
    #                                             'geoms': ['conf_gauss_bins'],
    #                                             'func': 'ETKDG',
    #                                             'exp_config': {
    #                                                 'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                                 'num_confs': 50,
    #                                                 'max_attempts': 50,
    #                                                 'max_bonds': 4
    #                                             },
    #                                             'prog_bar': True
    #                                         }
    # },
    # 'nmrshiftdb_PT_gauss_bins_default_15': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                             'feature_generator': 'geometry_generator',
    #                                             'f_config': {
    #                                                 'geoms': ['conf_gauss_bins'],
    #                                                 'func': 'confs',
    #                                                 'exp_config': {
    #                                                     'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                                     'max_bonds': 4
    #                                                 },
    #                                                 'prog_bar': True
    #                                             }
    # },
    # 'nmrshiftdb_PT_gauss_bins_default_18': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                             'feature_generator': 'geometry_generator',
    #                                             'f_config': {
    #                                                 'geoms': ['conf_gauss_bins'],
    #                                                 'func': 'confs',
    #                                                 'exp_config': {
    #                                                     'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                                     'max_bonds': 7
    #                                                 },
    #                                                 'prog_bar': True
    #                                             }
    # },
    # 'nmrshiftdb_PT_angles_means': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                     'feature_generator': 'geometry_generator',
    #                                     'f_config': {
    #                                         'geoms': ['mean_angle_mat'],
    #                                         'func': 'confs',
    #                                         'prog_bar': True
    #                                     }
    # },
    # 'nmrshiftdb_ETKDG_angles_means': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                     'feature_generator': 'geometry_generator',
    #                                     'f_config': {
    #                                         'geoms': ['mean_angle_mat'],
    #                                         'func': 'ETKDG',
    #                                         'exp_config': {
    #                                             'num_confs': 50,
    #                                             'max_attempts': 50
    #                                         },
    #                                         'prog_bar': True
    #                                     }
    # },
    # 'nmrshiftdb_ETKDG_10_augment_5': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                     'feature_generator': 'geometry_generator',
    #                                     'feature_map': {
    #                                         'mean_distance_mat': '_distances_means',
    #                                         'mean_angle_mat': '_angles_mat',
    #                                         'conf_gauss_bins': '_gauss_bins_default',
    #                                         'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                     },
    #                                     'conf_choice': 'augmented', 
    #                                     'copies': 5,
    #                                     'f_config': {
    #                                         'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                         'func': 'ETKDG',
    #                                         'exp_config': {
    #                                             'num_confs': 10,
    #                                             'max_attempts': 20,
    #                                             'g_params': geom_util.DEFAULT_GAUSS_BINS
    #                                         },
    #                                         'prog_bar': True
    #                                     }
    # },
    # # 'nmrshiftdb_ETKDG_500': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    # #                                     'feature_generator': 'geometry_generator',
    # #                                     'feature_map': {
    # #                                         'mean_distance_mat': '_distances_means',
    # #                                         'mean_angle_mat': '_angles_mat',
    # #                                         'conf_gauss_bins': '_gauss_bins_default',
    # #                                         'mean_dihedral_angle_mat': '_dihedrals_means'
    # #                                     },
    # #                                     'f_config': {
    # #                                         'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    # #                                         'func': 'ETKDG',
    # #                                         'exp_config': {
    # #                                             'g_params': geom_util.DEFAULT_GAUSS_BINS,
    # #                                             'num_confs': 500,
    # #                                             'max_attempts': 500
    # #                                         },
    # #                                         'prog_bar': True
    # #                                     }
    # # },
    # 'nmrshiftdb_ETKDG_opt_50_20': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_10_enrich': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_10.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'enrich': {
    #                                     'num_confs': 5
    #                                 },
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_10': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_10.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_dihedrals_means': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'f_config': {
    #                                     'geoms': ['mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_PT_dihedrals_means': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                     'feature_generator': 'geometry_generator',
    #                                     'f_config': {
    #                                         'geoms': ['mean_dihedral_angle_mat'],
    #                                         'func': 'confs',
    #                                         'prog_bar': True
    #                                     }
    # },
    # 'nmrshiftdb_ETKDG_dihedrals_means': {'files' : '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
    #                                     'feature_generator': 'geometry_generator',
    #                                     'f_config': {
    #                                         'geoms': ['mean_dihedral_angle_mat'],
    #                                         'func': 'ETKDG',
    #                                         'exp_config': {
    #                                             'num_confs': 50,
    #                                             'max_attempts': 50
    #                                         },
    #                                         'prog_bar': True
    #                                     }
    # }
    # 'gdb17-stereo-0_PT_distances_means': {'files' : '/data/ericj/moldata-conf/gdb17-stereo-pt-conf/working.data/0.*.moldata.db',
    #                                         'mol_id_field' : 'source_id', 
    #                                         'feature_generator': 'geometry_generator',
    #                                         'f_config': {
    #                                             'geoms': ['mean_distance_mat'],
    #                                             'func': 'confs',
    #                                             'prog_bar': True
    #                                         }
    # },
    # 'gdb17-stereo-0_PT_gauss_bins_default': {'files' : '/data/ericj/moldata-conf/gdb17-stereo-pt-conf/working.data/0.*.moldata.db',
    #                                             'feature_generator': 'geometry_generator',
    #                                             'mol_id_field' : 'source_id', 
    #                                             'f_config': {
    #                                                 'geoms': ['conf_gauss_bins'],
    #                                                 'func': 'confs',
    #                                                 'exp_config': {
    #                                                     'g_params': geom_util.DEFAULT_GAUSS_BINS
    #                                                 },
    #                                                 'prog_bar': True
    #                                             }
    # },
    # 'gdb17-stereo-0_PT_gauss_bins_scaled': {'files' : '/data/ericj/moldata-conf/gdb17-stereo-pt-conf/working.data/0.*.moldata.db',
    #                                             'feature_generator': 'geometry_generator',
    #                                             'mol_id_field' : 'source_id', 
    #                                             'f_config': {
    #                                                 'geoms': ['conf_gauss_bins'],
    #                                                 'func': 'confs',
    #                                                 'exp_config': {
    #                                                     'g_params': molecule_features.FEAT_R_GAUSSIAN_FILTERS['scaled']
    #                                                 },
    #                                                 'prog_bar': True
    #                                             }
    # },
    # 'gdb17-stereo-0_PT_gauss_bins_scaled_narrow': {'files' : '/data/ericj/moldata-conf/gdb17-stereo-pt-conf/working.data/0.*.moldata.db',
    #                                                     'feature_generator': 'geometry_generator',
    #                                                     'mol_id_field' : 'source_id',
    #                                                     'f_config': {
    #                                                         'geoms': ['conf_gauss_bins'],
    #                                                         'func': 'confs',
    #                                                         'exp_config': {
    #                                                             'g_params': molecule_features.FEAT_R_GAUSSIAN_FILTERS['scaled_narrow']
    #                                                         },
    #                                                         'prog_bar': True
    #                                                     }
    # },
    # 'nmrshiftdb_ETKDG_50_split_10': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'conf_choice': 'split',
    #                                 'orig_conf_n': 50,
    #                                 'copies': 5,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'bally_rablen_ETKDG_50': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.bally_rablen_2011_etkdg_clean_opt_50_20.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'kutateladze2015_ETKDG_50': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.kutateladze2015_etkdg_clean_opt_50_20.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'gissmo_ETKDG_50': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.gissmo_pt_etkdg_clean_opt_50_20.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample40': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 40,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample30': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 30,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample20': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 20,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample10': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 10,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample7': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 7,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample4': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 4,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample3': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 3,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample2': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 2,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample1': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 1,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_enrich': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'enrich':
    #                                 {
    #                                     'num_confs': 5
    #                                 },
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample40_enrich': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'enrich':
    #                                 {
    #                                     'num_confs': 5
    #                                 },
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 40,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample30_enrich': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'enrich':
    #                                 {
    #                                     'num_confs': 5
    #                                 },
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 30,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample20_enrich': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'enrich':
    #                                 {
    #                                     'num_confs': 5
    #                                 },
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 20,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample10_enrich': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'enrich':
    #                                 {
    #                                     'num_confs': 5
    #                                 },
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 10,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample7_enrich': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'enrich':
    #                                 {
    #                                     'num_confs': 7
    #                                 },
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 7,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample4_enrich': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'enrich':
    #                                 {
    #                                     'num_confs': 10
    #                                 },
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 4,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample3_enrich': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'enrich':
    #                                 {
    #                                     'num_confs': 20
    #                                 },
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 3,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample2_enrich': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'enrich':
    #                                 {
    #                                     'num_confs': 25
    #                                 },
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 2,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'nmrshiftdb_ETKDG_opt_50_20_boltzmann_sample1_enrich': {'files': '/jonaslab/data/moldata-conf/small-misc/conf_search.nmrshiftdb_etkdg_opt_50_20.??.moldata.db',
    #                                 'feature_generator': 'geometry_generator',
    #                                 'enrich':
    #                                 {
    #                                     'num_confs': 25
    #                                 },
    #                                 'conf_choice': 'sample_existing',
    #                                 'orig_conf_n': 50,
    #                                 'samples': 1,
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'gdb17_ETKDG_opt_50_20': {'files': '/jonaslab/data/moldata-conf/GDB-17-stereo-dft-conf/*.moldata.db',
    #                                 'mol_id_field': 'source_id',
    #                                 # 'mol_from_binary': False,
    #                                 'feature_generator': 'geometry_generator',
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'ETKDG',
    #                                     'exp_config': {
    #                                         'num_confs': 50,
    #                                         'max_attempts': 20,
    #                                         'optimize': True,
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },
    # 'gdb17_PT': {'files': '/jonaslab/data/moldata-conf/GDB-17-stereo-pt-conf/*.db',
    #                                 'mol_id_field': 'source_id',
    #                                 'read_nonconformity_db': True, 
    #                                 'mol_from_binary': False,
    #                                 'feature_generator': 'geometry_generator',
    #                                 'feature_map': {
    #                                     'mean_distance_mat': '_distances_means',
    #                                     'mean_angle_mat': '_angles_mat',
    #                                     'conf_gauss_bins': '_gauss_bins_default',
    #                                     'mean_dihedral_angle_mat': '_dihedrals_means'
    #                                 },
    #                                 'f_config': {
    #                                     'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
    #                                     'func': 'confs',
    #                                     'exp_config': {
    #                                         # 'num_confs': 50,
    #                                         # 'max_attempts': 20,
    #                                         # 'optimize': True,
    #                                         'g_params': geom_util.DEFAULT_GAUSS_BINS,
    #                                         # 'boltzmann_weight': True
    #                                     },
    #                                     'prog_bar': True
    #                                 }
    # },

    'benzoics_ETKDG_opt_50_20_boltzmann': {'file': 'benzoic_acids.feather',
                                    'feature_generator': 'geometry_generator',
                                    'feature_map': {
                                        'mean_distance_mat': '_distances_means',
                                        'mean_angle_mat': '_angles_mat',
                                        'conf_gauss_bins': '_gauss_bins_default',
                                        'mean_dihedral_angle_mat': '_dihedrals_means'
                                    },
                                    'mol_id_field': 'molecule_id',
                                    'f_config': {
                                        'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
                                        'func': 'ETKDG',
                                        'exp_config': {
                                            'g_params': geom_util.DEFAULT_GAUSS_BINS,
                                            'num_confs': 50,
                                            'max_attempts': 20,
                                            'optimize': True,
                                            'boltzmann_weight': True
                                        },
                                        'prog_bar': True
                                    }
    },

    'nmrshiftdb_ETKDG_opt_10_20_boltzmann': {'files': '/jonaslab/data/moldata-conf/nmrshiftdb-pt-conf/??.moldata.db',
                                    'feature_generator': 'geometry_generator',
                                    'feature_map': {
                                        'mean_distance_mat': '_distances_means',
                                        'mean_angle_mat': '_angles_mat',
                                        'conf_gauss_bins': '_gauss_bins_default',
                                        'mean_dihedral_angle_mat': '_dihedrals_means'
                                    },
                                    'f_config': {
                                        'geoms': ['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins', 'mean_dihedral_angle_mat'],
                                        'func': 'ETKDG',
                                        'exp_config': {
                                            'g_params': geom_util.DEFAULT_GAUSS_BINS,
                                            'num_confs': 10,
                                            'max_attempts': 20,
                                            'optimize': True,
                                            'boltzmann_weight': True
                                        },
                                        'prog_bar': True
                                    }
    },
}

OUTPUT_DIR = "distance_features/" 
td = lambda x : os.path.join(OUTPUT_DIR, x)

def append_suffix_pickle(path, suff):
    return path[:-7] + suff + path[-7:]

def params():
    for exp_name, exp_config in DATASETS.items():
        if 'files' in exp_config:
            infiles = glob(exp_config['files'])
        else:
            infiles = exp_config['file']
        
        outfiles = td(f"{exp_name}.pickle")

        yield infiles, outfiles, exp_name, exp_config

@mkdir(OUTPUT_DIR)
@files(params)
def create_features(infiles, outfiles, exp_name, exp_config):
    print("Creating", exp_name)
    feat_out = outfiles
    # skipped = []
    repeated = []
    ids = []
    # dist_out, meta_out = outfiles
    all_mols = {} # molid to rdkit Mol

    mol_id_field = exp_config.get('mol_id_field', 'id')

    if infiles.__class__ == list:
        for db_filename in tqdm(infiles, desc='loading dbs'):
            tqdm.write(f"Loading mols from db: {db_filename}")
            if exp_config.get('read_nonconformity_db', False):
                engine = create_engine(f"sqlite:///{db_filename}")
                Session = sessionmaker()
                Session.configure(bind=engine)
                session = Session()
                mol_q = session.query(model.Molecule)
                i = 0
                m = mol_q[i]
                with tqdm(total=20000) as pbar:
                    while True:
                        pbar.update(1)
                        for cs in m.confsets:
                            rdmol = dbconfset_to_rdmol(cs, m.rdmol)
                            all_mols[m.source_id] = rdmol
                        try:
                            i += 1
                            m = mol_q[i]
                        except: 
                            break
                pbar.close()
            else:
                sql_engine = create_engine(f"sqlite+pysqlite:///{db_filename}")
                # molecules_table = Table("molecules", MetaData(), autoload_with=sql_engine)
                with sql_engine.connect() as conn:
                    rows = conn.execute('select * from molecules')
                    for row in rows:
                        identifier = int(row[mol_id_field])
                        if exp_config.get('mol_from_binary', True):
                            m = Chem.Mol(zlib.decompress(row['bmol']))
                        else:
                            m = Chem.Mol(row['rdmol'])
                        if 'enrich' in exp_config:
                            m = util.enrich_confs_methyl_permute(m, exp_config['enrich']['num_confs'], random_seed=exp_config['enrich'].get('random_seed', 0))
                        all_mols[identifier] = m
    else:
        m_df = pd.read_feather(infiles)
        for i, row in m_df.iterrows():
            all_mols[row[mol_id_field]] = Chem.AddHs(Chem.MolFromSmiles(row['SMILES']))

    if exp_config['feature_generator'] == 'rdkit':
        features = {}
        m_list =  list(all_mols.values())
        for m in m_list:
            # Initialize RingInfo (required for BoundsMatrix)
            Chem.rdmolops.FastFindRings(m)
        
        if exp_config['feature'] == 'mean_distance_mat':
            f_list = [getBoundsMatrixCorrected(mol) for mol in tqdm(m_list)]
            for i, mid in enumerate(all_mols.keys()):
                features[mid] = f_list[i]
            if len(repeated) > 0:
                print(repeated)
                raise Exception(f"warning, there were {len(repeated)} ids!")
            pickle.dump(features, open(feat_out, 'wb'))
        else:
           raise Exception(f"Unrecognized featurization method for rdkit: {exp_config['feature']}.") 

    elif exp_config['feature_generator'] == 'geometry_generator':
        choice = exp_config.get('conf_choice', None)
        if choice == 'split':
            n = int(math.ceil(math.log10(exp_config['copies'])))
            mol_ids = [b*(10**n)+i for b in all_mols.keys() for i in range(exp_config['copies'])]
            
            splits = np.arange(exp_config['orig_conf_n'])
            rng = default_rng(exp_config.get('seed',0))
            rng.shuffle(splits)
            s = exp_config['orig_conf_n']//exp_config['copies']

            temps = []
            for i in range(exp_config['copies']):
                generator = geometryFeaturizerGenerator(**exp_config['f_config'])
                temp_f, _ = generator(all_mols.values(),conf_idxs=splits[s*i:s*(i+1)])
                temps += [temp_f]
            f = [val for tup in zip(*temps) for val in tup]
        elif choice == 'augmented':
            n = int(math.ceil(math.log10(exp_config['copies'])))
            mol_ids = [b*(10**n)+i for b in all_mols.keys() for i in range(exp_config['copies'])]

            temps = []
            for i in range(exp_config['copies']):
                c = copy.deepcopy(exp_config['f_config'])
                c['exp_config']['seed'] = i
                generator = geometryFeaturizerGenerator(**c)
                temp_f, _ = generator(all_mols.values())
                temps += [temp_f]
            f = [val for tup in zip(*temps) for val in tup]
        if choice == 'sample_existing':
            rng = default_rng(exp_config.get('seed',0))
            idxs = rng.choice(exp_config['orig_conf_n'], exp_config['samples'], replace=False, shuffle=False)
            if 'enrich' in exp_config:
                en_confs = exp_config['enrich']['num_confs']
                for j in range(exp_config['samples']):
                    idxs = np.append(idxs, [exp_config['orig_conf_n'] + (idxs[j]*en_confs) + k for k in range(en_confs)])
            mol_ids = all_mols.keys()
            generator = geometryFeaturizerGenerator(**exp_config['f_config'])
            f, _ = generator(list(all_mols.values()), conf_idxs=idxs)
        else:
            mol_ids = all_mols.keys()
            generator = geometryFeaturizerGenerator(**exp_config['f_config'])
            f, _ = generator(list(all_mols.values()))
        c = 0
        for x in f:
            if x is None or x[exp_config['f_config']['geoms'][0]] is None:
                c += 1
        print(c, "molecules were unable to be featurized.")
        if 'feature_map' in exp_config:
            all_f = {}
            for feature, suffix in exp_config['feature_map'].items():
                features = {}
                for i, mid in enumerate(mol_ids):
                    if f[i] is None:
                        features[mid] = np.nan
                    else:
                        features[mid] = f[i][feature]
                all_f[suffix] = features
                pickle.dump(features, open(append_suffix_pickle(feat_out, suffix), 'wb'))
            pickle.dump(all_f, open(feat_out, 'wb'))
        else:
            f_list = [np.nan if m is None else m[exp_config['f_config']['geoms'][0]] for m in f]
            features = {}
            for i, mid in enumerate(mol_ids):
                features[mid] = f_list[i]
            if len(repeated) > 0:
                print(repeated)
                raise Exception(f"warning, there were {len(repeated)} ids!")
            pickle.dump(features, open(feat_out, 'wb'))
    else:
        raise Exception(f"Unrecognized featurization generator {exp_config['feature_generator']}.")

    
    # pickle.dump({'ids': ids}, open(meta_out, 'wb'))

if __name__ == "__main__":
    pipeline_run([create_features], checksum_level=0)
