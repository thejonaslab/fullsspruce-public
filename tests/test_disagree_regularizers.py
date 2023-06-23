import torch
import pytest 
import torch.nn as nn
from fullsspruce.model import nets

def test_disagree_loss_small():
    """
    Test the DisagreementLoss classes with small inputs.
    """
    # Required for all losses but not used to calculuate this loss
    edge_pred, edge_pred_mask, vert_mask = torch.zeros(2,2,2,0), torch.zeros(2,2,2,0), torch.zeros(2,2,2) 

    # Set up various losses to use
    small_loss = nets.DisagreementLoss(regularize_by='batch')
    small_loss_weighted = nets.DisagreementLoss(lambda_e = 2.0, regularize_by='batch')
    small_loss_per_mol = nets.DisagreementLoss()
    small_loss_per_atom = nets.DisagreementLoss(regularize_by='atom')
    # Set up ground truth and predicted values with small differences so they can be calculated by hand
    ground = torch.tensor([[[1.0,1.0],[0.0,0.0]],[[1.0,0.9],[0.0,0.2]]])
    pred = torch.tensor([[[1.1,1.1],[0.2,0.2]],[[1.0,1.0],[0.0,-0.1]]])
    # All values seen by loss function
    mask = torch.tensor([[[1,1],[1,1]],[[1,1],[1,1]]])

    # Test per batch regularization with known values
    torch.testing.assert_close(torch.tensor((.05 + (.15/1.0125))/4), 
                small_loss({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))
    # Test same values, but with lambda_e = 2.0
    torch.testing.assert_close(torch.tensor((.05 + (.3/1.0125))/4), 
                small_loss_weighted({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))
    # Test per molecule regularization with known values
    torch.testing.assert_close(torch.tensor((.05 + .05 + (.1/(1.025**2)))/4), 
                small_loss_per_mol({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))
    # Test per atom regularization with known values
    torch.testing.assert_close(torch.tensor((.05 + .05 + (.01/(1.01**2)) + (.09/(1.04**2)))/4), 
                small_loss_per_atom({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))

    disagreeLossOneToTwo = nets.DisagreementLossOneToTwo(regularize_by='batch')
    disagreeLossOneToTwo_weighted = nets.DisagreementLossOneToTwo(lambda_e = 2.0,regularize_by='batch')
    disagreeLossOneToTwo_by_mol = nets.DisagreementLossOneToTwo()
    disagreeLossOneToTwo_by_atom = nets.DisagreementLossOneToTwo(regularize_by='atom')

    pred = torch.tensor([[[1.1],[0.2]],[[1.0],[0.0]]])

    # Test one pred channel to two ground truths for the same set of losses with known values
    torch.testing.assert_close(torch.tensor((.05 + (.1/1.0125))/4), 
                disagreeLossOneToTwo({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))
    torch.testing.assert_close(torch.tensor((.05 + (.2/1.0125))/4), 
                disagreeLossOneToTwo_weighted({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))
    torch.testing.assert_close(torch.tensor((.05 + .05 + (.05/(1.025**2)))/4), 
                disagreeLossOneToTwo_by_mol({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))
    torch.testing.assert_close(torch.tensor((.05 + .05 + (.01/(1.01**2)) + (.04/(1.04**2)))/4), 
                disagreeLossOneToTwo_by_atom({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))


def test_disagree_loss_random():
    """
    Test the DisagreementLoss classes with random inputs.
    """
    edge_pred, edge_pred_mask, vert_mask = torch.zeros(128,128,2,0), torch.zeros(128,128,2,0), torch.zeros(128,128,2) 
    
    mseLoss = nn.MSELoss()
    disagreeLoss = nets.DisagreementLoss(regularize_by='batch')
    disagreeLossOneToTwo = nets.DisagreementLossOneToTwo(regularize_by='batch')

    # ground = torch.tensor([[[1.0,1.0],[0.0,0.0]],[[1.0,0.9],[0.0,0.1]]])
    # pred = torch.tensor([[[1.1,1.1],[0.2,0.2]],[[1.0,1.0],[0.0,-0.1]]])
    ground = torch.rand((128,128,2))
    pred = torch.rand((128,128,2))
    mask_ab = torch.zeros((128,128,2))
    mask_ab[:,:,:1] = 1
    mask_exp = torch.zeros((128,128,2))
    mask_exp[:,:,1:2] = 1

    # Test batch regularization properly ignoring when ab or exp is completely masked out
    torch.testing.assert_close(mseLoss(ground[:,:,:1], pred[:,:,:1]), 
                disagreeLoss({'shift_mu': pred}, ground, mask_ab, edge_pred, edge_pred_mask, vert_mask))
    torch.testing.assert_close(mseLoss(ground[:,:,1:2], pred[:,:,1:2]), 
                disagreeLoss({'shift_mu': pred}, ground, mask_exp, edge_pred, edge_pred_mask, vert_mask))

    pred_one = torch.rand((128,128,1))
    # Again but with one to two loss
    torch.testing.assert_close(mseLoss(ground[:,:,:1], pred_one), 
                disagreeLossOneToTwo({'shift_mu': pred_one}, ground, mask_ab, edge_pred, edge_pred_mask, vert_mask))
    torch.testing.assert_close(mseLoss(ground[:,:,1:2], pred_one), 
                disagreeLossOneToTwo({'shift_mu': pred_one}, ground, mask_exp, edge_pred, edge_pred_mask, vert_mask))

def test_disagree_different_atoms():
    """
    Test for disagreement loss using molecules with different numbers of atoms
    """
    # Required for all losses but not used to calculuate this loss
    edge_pred, edge_pred_mask, vert_mask = torch.zeros(2,2,2,0), torch.zeros(2,2,2,0), torch.zeros(2,2,2) 

    # Set up loss functions
    small_loss_per_mol = nets.DisagreementLoss()
    small_loss_per_atom = nets.DisagreementLoss(regularize_by='atom')

    # Set up ground truth and predicted values with small differences so they can be calculated by hand
    ground = torch.tensor([[[1.0,1.0],[0.0,0.0],[3.0,3.0]],[[1.0,0.9],[0.0,0.2],[3.4,3.5]]])
    pred = torch.tensor([[[1.1,1.1],[0.2,0.2],[3.0,3.0]],[[1.0,1.0],[0.0,-0.1],[4.0,3.0]]])
    # Mask out last atom in second molecule to give them different numbers of atoms
    mask = torch.tensor([[[1,1],[1,1],[1,1]],[[1,1],[1,1],[0,0]]])

    # Test per molecule regularization with known values
    torch.testing.assert_close(torch.tensor((.05 + .05 + (.1/(1.025**2)))/5), 
                small_loss_per_mol({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))
    # Test per atom regularization with known values
    torch.testing.assert_close(torch.tensor((.05 + .05 + (.01/(1.01**2)) + (.09/(1.04**2)))/5), 
                small_loss_per_atom({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))

def test_multi_shift_loss():
    """
    Test for MultiShiftLoss, which uses disagreement regularization.
    """
    # Required for all losses but not used to calculuate this loss
    edge_pred, edge_pred_mask, vert_mask = torch.zeros(2,2,2,0), torch.zeros(2,2,2,0), torch.zeros(2,2,2) 

    # Set up loss functions
    losses = [{'loss_name': 'DisagreementLoss',
                'loss_params': {
                    'regularize_by': 'atom'
                },
                'channels': [2*i, (2*i) + 1],
                'loss_weight': w
            }  for i, w in enumerate([1,10])]
    two_shift_disagree_loss = nets.MultiShiftLoss(losses)

    # Set up ground truth and predicted values with small differences so they can be calculated by hand
    ground = torch.concat([torch.tensor([[[1.0,1.0],[0.0,0.0],[3.0,3.0]],[[1.0,0.9],[0.0,0.2],[3.4,3.5]]]) for _ in range(2)],axis=2)
    pred = torch.concat([torch.tensor([[[1.1,1.1],[0.2,0.2],[3.0,3.0]],[[1.0,1.0],[0.0,-0.1],[4.0,3.0]]]) for _ in range(2)], axis=2)
    # Mask out last atom in second molecule to give them different numbers of atoms
    mask = torch.stack([torch.tensor([[1,1,1],[1,1,0]]) for _ in range(4)], axis=2)

    # Test per atom regularization with known values
    torch.testing.assert_close(torch.tensor((.05 + .05 + (.01/(1.01**2)) + (.09/(1.04**2)))*(11/5)), 
                two_shift_disagree_loss({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))

    # Run test with the first channel fully masked out.

    # Set up ground truth and predicted values with small differences so they can be calculated by hand
    ground = torch.concat([torch.tensor([[[1.0,1.0],[0.0,0.0],[3.0,3.0]],[[1.0,0.9],[0.0,0.2],[3.4,3.5]]]) for _ in range(2)],axis=2)
    pred = torch.concat([torch.tensor([[[1.1,1.1],[0.2,0.2],[3.0,3.0]],[[1.0,1.0],[0.0,-0.1],[4.0,3.0]]]) for _ in range(2)], axis=2)
    # Mask out last atom in second molecule to give them different numbers of atoms
    mask = torch.stack([torch.zeros((2,3)), torch.zeros((2,3)), torch.tensor([[1,1,1],[1,1,0]]), torch.tensor([[1,1,1],[1,1,0]])], axis=2)

    # Test per atom regularization with known values
    torch.testing.assert_close(torch.tensor((.05 + .05 + (.01/(1.01**2)) + (.09/(1.04**2)))*2), 
                two_shift_disagree_loss({'shift_mu': pred}, ground, mask, edge_pred, edge_pred_mask, vert_mask))
