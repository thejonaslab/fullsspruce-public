import numpy as np
import pandas as pd
from tqdm import  tqdm
from rdkit import Chem
import pickle
import os

from glob import glob
import json 

import time
from fullsspruce import util

from rdkit.Chem import AllChem

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import  tqdm
from fullsspruce.featurize.netdataio import * 
from fullsspruce import netutil
import sys
import click
import resource

import yaml

import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size, is_local):

    if is_local:
        print("setup, RANK=", rank, "WORLD_SIZE=", world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'SLURM_JOB_ID' in os.environ:
            os.environ['MASTER_PORT'] = str((int(os.environ['SLURM_JOB_ID']) % 10000 ) + 2000)
        else:
            os.environ['MASTER_PORT'] = '11224'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        my_hostname = socket.gethostname()
        my_jobid = os.environ.get('SLURM_JOB_ID', 0)
        dist_job_metadata_filename = f"dist_job_meta.{my_jobid}"
        print(f"setup id: {my_jobid} rank: {rank} my hostname: {my_hostname}")
        
        if rank == 0:
            control_meta = {'control_hostname' : my_hostname,
                             'control_port'  : 12355}
            with open(dist_job_metadata_filename, 'wb') as fid:
                pickle.dump(control_meta, 
                            fid)
            time.sleep(3)
        else:
            time.sleep(5) #### FIXME we really need to fix this
            with open(dist_job_metadata_filename, 'rb') as fid:
                control_meta = pickle.load(fid)
        os.environ['MASTER_ADDR'] = control_meta['control_hostname']
        os.environ['MASTER_PORT'] = str(control_meta['control_port'])

        print(f"id: {my_jobid} rank: {rank} my hostname: {my_hostname} control hostname: {control_meta['control_hostname']}")
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        print(f"init_process_group completed for rank {rank}")
        
def cleanup():
    dist.destroy_process_group()

    
CHECKPOINT_DIR = "checkpoints/" 
CHECKPOINT_EVERY = 50
LATEST_EVERY = 15


# def train(rank, world_size, CHECKPOINT_BASENAME,
#           exp_config, MODEL_NAME,
#           exp_config_filename=None,
#           load_from_checkpoint=False):
def train(rank, device_id, exp_config,
          exp_output_name, world_size,
          is_local, load_from_checkpoint=False):

    setup(rank, world_size, is_local)
    
    if device_id == -1:
        device_id = rank # for local setup

    device = torch.device(device_id)
    print(f"Device {device} has device_id={device_id}")
    torch.cuda.set_device(device_id)
    
    np.random.seed(exp_config['seed'])


    MAX_N = exp_config['tgt_max_n']
    start_epoch = -1

    DATALOADER_NUM_WORKERS= exp_config.get('DATALOADER_NUM_WORKERS',
                                             0)
    print(f"Running with DATALOADER_NUM_WORKERS={DATALOADER_NUM_WORKERS}")
    BATCH_SIZE = exp_config['batch_size']
    dataset_hparams_update = exp_config['dataset_hparams']
    dataset_hparams = netutil.DEFAULT_DATA_HPARAMS
    util.recursive_update(dataset_hparams, 
                          dataset_hparams_update)

    exp_data = exp_config['exp_data']
    cv_func = netutil.CVSplit(**exp_data['cv_split'])
    pred_config = exp_config['pred_config']
    dataset_class = exp_data.get('ds_class', 'MoleculeDatasetMulti')
    print("DS:", dataset_class)


    checkpoint_filename = ""
    if load_from_checkpoint:
        print("loading checkpoint", checkpoint_filename, flush=True)

        rank_checkpoint_filename = os.path.join(CHECKPOINT_DIR, exp_output_name + f".latest.{rank}.checkpoint")
        rank_checkpoint_data = torch.load(rank_checkpoint_filename, map_location=lambda storage, loc: storage)

        common_checkpoint_filename = os.path.join(CHECKPOINT_DIR, exp_output_name + f".latest.common.checkpoint")
        common_checkpoint_data = torch.load(common_checkpoint_filename, map_location=lambda storage, loc: storage)
        print("loading checkpoint", checkpoint_filename, "done", flush=True)


    passthrough_config =  exp_config.get('passthroughs', {})
    datasets = {}
    for ds_config_i, dataset_config in enumerate(exp_data['data']):
        ds, phase_data = netutil.make_dataset(dataset_config, dataset_hparams,
                                              pred_config, dataset_class,
                                              MAX_N, cv_func,
                                              passthrough_config = passthrough_config)
        phase = dataset_config['phase']
        if phase not in datasets:
            datasets[phase] = []
        datasets[phase].append(ds)

        # pickle.dump(phase_data,
        #             open(CHECKPOINT_BASENAME + f".data.{ds_config_i}.{phase}.data", 'wb'))
        
    ds_train = datasets['train'][0] if len(datasets['train']) == 1 else torch.utils.data.ConcatDataset(datasets['train'])
    ds_test = datasets['test'][0] if len(datasets['test']) == 1 else torch.utils.data.ConcatDataset(datasets['test'])

    print("we are training with", len(ds_train))
    print("we are testing with", len(ds_test))

    dataloader_name = exp_config.get("dataloader_func",
                                 'torch.utils.data.DataLoader')

    dataloader_creator = eval(dataloader_name)

    epoch_size = exp_config.get('epoch_size', 8192)
    
    train_sampler = netutil.SubsetSampler(epoch_size, len(ds_train), shuffle=True,
                                          world_size=world_size, rank=rank,
                                          seed = exp_config.get('seed', 0))

        
    dl_train = dataloader_creator(ds_train, batch_size=BATCH_SIZE, 
                                  sampler=train_sampler, 
                                  pin_memory=False,
                                  num_workers=DATALOADER_NUM_WORKERS,
                                  persistent_workers= DATALOADER_NUM_WORKERS >0)
    dl_test = dataloader_creator(ds_test, batch_size=BATCH_SIZE, 
                                 shuffle=True,pin_memory=False,
                                 num_workers=DATALOADER_NUM_WORKERS,
                                 persistent_workers=DATALOADER_NUM_WORKERS >0)
    

    net_params = exp_config['net_params']
    net_name = exp_config['net_name']

    net_params['g_feature_n'] = ds_test[0]['vect_feat'].shape[-1]
    net_params['GS'] = ds_test[0]['adj'].shape[0]

    torch.manual_seed(exp_config['seed'])
    net = eval(net_name)(**net_params)

    torch.manual_seed(exp_config['seed'] + os.getpid())
    
    if exp_config.get("load_checkpoint", None) is not None:
        print("LOADING CHECKPOINT", exp_config['load_checkpoint'])
        module = torch.load(exp_config['load_checkpoint'])
        net.load_state_dict(module.state_dict())
        # net.load_state_dict(torch.load(exp_config['load_checkpoint']))
    print(f"moving net to {device}")
    net = net.to(device, non_blocking=True)
    print("creating DDP")
    net = DDP(net, device_ids=[device_id], output_device=device_id)
    print("Created DDP with device_ids=[", device_id, "]")
    
    loss_params = exp_config['loss_params']
    criterion = netutil.create_loss(loss_params, device)
    
    opt_params = exp_config['opt_params']
    optimizer = netutil.create_optimizer(opt_params, net.parameters())
        
    scheduler_name = opt_params.get('scheduler_name', 'steplr')
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=opt_params['scheduler_step_size'],
                                                    gamma=opt_params['scheduler_gamma'])

    # checkpoint_filename = CHECKPOINT_BASENAME + ".{epoch_i:08d}"
    print("checkpoint:", checkpoint_filename)
    checkpoint_every_n = exp_config.get('checkpoint_every', CHECKPOINT_EVERY)
    latest_every = exp_config.get('latest_every', LATEST_EVERY)
    
    checkpoint_func = netutil.create_checkpoint_func(CHECKPOINT_DIR,
                                             exp_output_name, rank,
                                             checkpoint_every_n, latest_every,
                                             train_sampler)

    #checkpoint_func = netutil.create_checkpoint_func(checkpoint_every, latest_every, CHECKPOINT_BASENAME)#checkpoint_filename)

    
    TENSORBOARD_DIR = exp_config.get('tblogdir', f"tensorboard.logs")
    if rank ==0 :
        writer = SummaryWriter(f"{TENSORBOARD_DIR}/{exp_output_name}")
    else:
        writer = None
        
    validate_config = exp_config.get('validate_config', {})
    validate_funcs = []
    for k, v in validate_config.items():
        if k == 'shift_uncertain_validate':
            validate_func = netutil.create_shift_uncertain_validate_func(v, writer)
        elif k == 'coupling_validate':
            validate_func = netutil.create_coupling_validate_func(v, writer)
                                                                  
        elif k == 'coupling_uncertain_validate':
            validate_func = netutil.create_coupling_uncertain_validate_func(v, writer)

        elif k == 'coupling_validate_loss':
            validate_func = netutil.create_coupling_validate_loss(writer)
        else:
            raise ValueErorr(f"unknown function {k}")
        validate_funcs.append(validate_func)
        

    metadata = {'dataset_hparams' : dataset_hparams, 
                'net_params' : net_params, 
                'opt_params' : opt_params, 
                'exp_data' : exp_data, 
                #'meta_infile' : meta_infile, 
                'exp_config' : exp_config, 
                'validate_config': validate_config,
                'passthrough_config' : passthrough_config, 
                'max_n' : MAX_N,
                'pred_config' : pred_config, 
                'net_name' : net_name, 
                'batch_size' : BATCH_SIZE, 
                'loss_params' : loss_params,
                'ds_class': dataset_class}

    if not load_from_checkpoint and rank == 0:
        print(json.dumps(metadata, indent=4))

        pickle.dump(metadata,
                    open(os.path.join(CHECKPOINT_DIR, \
                                      f"{exp_output_name}.meta"), 'wb'))
        
    
    if load_from_checkpoint :
        net.load_state_dict(common_checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(common_checkpoint_data['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(common_checkpoint_data['scheduler_state_dict'])

        train_sampler.set_state(rank_checkpoint_data['sampler_state']['train'])
        
        start_epoch = common_checkpoint_data['epoch_i']

        print("loaded checkpoint, starting from epoch ", start_epoch+1)


    
    if rank == 0 :
        print(exp_output_name)
    else:
        checkpoint_func = None
        validate_funcs = []
        

    netutil.generic_runner(net, optimizer, scheduler, criterion, 
                           dl_train, dl_test, start_epoch + 1,
                           MAX_EPOCHS=exp_config['max_epochs'], 
                           writer=writer, 
                           validate_funcs= validate_funcs, 
                           checkpoint_func= checkpoint_func,
                           prog_bar = rank == 0, 
                           clip_grad_value=exp_config.get('clip_grad_value', None),
                           device=torch.device(f'cuda:{rank}'))

@click.command()
@click.argument('exp_config_filename', type=str)
@click.argument('exp_extra_name', type=str, default="")
@click.option('-c', '--checkpoint-name', type=str, default="")
@click.option('-nc', '--no-checkpoint', type=bool, default=False)
@click.option('-ws', '--world-size', type=int, default=1)
@click.option('-l', '--local', type=bool, default=False, is_flag=True)
def run_train(exp_config_filename, exp_extra_name, checkpoint_name="",
              no_checkpoint=False, world_size=1, 
              local=True):


    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096*20, rlimit[1]))
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)

    print("rlimit is", rlimit)
    exp_config = yaml.load(open(exp_config_filename, 'r'), Loader=yaml.FullLoader)
    exp_config_name = os.path.splitext(os.path.basename(exp_config_filename))[0]

    if checkpoint_name == "":
        checkpoint_name = "{:08d}{:04d}".format(int(time.time()) % 100000000,
                                                    os.getpid() % 10000)
    if exp_extra_name != "":
        exp_output_name = exp_extra_name + "." + exp_config_name + "." + checkpoint_name
    else:
        exp_output_name = exp_config_name + "." + checkpoint_name
        

    saved_yaml_filename = os.path.join(CHECKPOINT_DIR, 
                                       f"{exp_output_name}.yaml")
                                       

    if os.path.exists(saved_yaml_filename):
        load_from_checkpoint = not no_checkpoint
        if not no_checkpoint:
            exp_config = yaml.load(open(saved_yaml_filename, 'r'), Loader=yaml.FullLoader)
    else:
        load_from_checkpoint = False
        ### We write an exact copy of the yaml so we can diff 
        with open(saved_yaml_filename, 'w') as fid:
            fid.write(open(exp_config_filename, 'r').read())
    
    print("load_from_checkpoint=", load_from_checkpoint)

    if local:
        print("MANUALLY SPAWNING")
        mp.spawn(train, args=(-1, exp_config, exp_output_name,
                              world_size, local, 
                              load_from_checkpoint, 
        ),
                nprocs=world_size, join=True)
    else:

        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        train(rank, local_rank, exp_config, exp_output_name, world_size,
              local, load_from_checkpoint)
    
    # train(exp_config, exp_output_name, load_from_checkpoint)

    

if __name__ == "__main__":
    run_train()

