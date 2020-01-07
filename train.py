from __future__ import print_function
"""
File modified from:
	https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import os
import sys
import cv2
import tqdm
import pathlib
import argparse
import numpy as np
import scipy.io as sio

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import torchvision.utils as vutils
from torchvision import transforms
from tensorboardX import SummaryWriter

import utils.consensus_loss
import utils.folder

from model import ResNet, Bottleneck

def resnet50(weights_path, device):

    state_dict_ = torch.load(weights_path, map_location=device)
    state_dict_model = state_dict_['state_dict']

    modified_state_dict = {}
    for key in state_dict_model.keys():
        mod_key = key[7:]
        modified_state_dict.update({mod_key: state_dict_model[key]})

    model = ResNet(Bottleneck, [3, 4, 6, 3], modified_state_dict)
    model.load_state_dict(modified_state_dict, strict=False)

    return model


def eval_pass_collect_stats(args, model, device, target_test_loader):
    # Run a bunch of forward passes to collect the target statistics before evaluating on the test set
    model.train(mode=True)
    with torch.no_grad():
        for i in range(10):
            print("Pass {} ...".format(i))
            for data, _ in target_test_loader:
                # dont care about source statistics after its trained.
                data = torch.cat((data, data, data), dim=0)
                data = data.to(device)
                output = model(data)


def train_infinite_collect_stats(args, model, device, source_train_loader,
                                 target_train_loader, optimizer, lambda_mec_loss,
                                 target_test_loader):
    writer = SummaryWriter()
    # source_iter = iter(source_train_loader)
    target_iter = iter(target_train_loader)
    
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[6000], gamma=0.1)
    best_test_loss = 1e6
    best_acc = 0
    
    print('-'*88)

    for epoch in range(args.num_iters):
        
        model.train()    
        for i , data in enumerate(tqdm.tqdm(source_train_loader)):
            source_data, source_y = data

            try:
                target_data, target_data_dup, _ = next(target_iter)
            except:
                target_iter = iter(target_train_loader)
                target_data, target_data_dup, _ = next(target_iter)
            # concat the source and target mini-batches
            data = torch.cat((source_data, target_data, target_data_dup), dim=0)
            data, source_y = data.to(device), source_y.to(device)

            optimizer.zero_grad()
            output = model(data)
            source_output, target_output, target_output_dup = torch.split(
                output, split_size_or_sections=output.shape[0] // 3, dim=0)

            mec_criterion = utils.consensus_loss.MinEntropyConsensusLoss(
                num_classes=args.num_classes, device=device)

            cls_loss = F.nll_loss(F.log_softmax(source_output), source_y)
            mec_loss = lambda_mec_loss * \
                mec_criterion(target_output, target_output_dup)

            loss = cls_loss + mec_loss

        
            loss.backward()
            optimizer.step()
        

        writer.add_scalar('train/cls_loss', cls_loss.item(), epoch)
        writer.add_scalar('train/mec_loss', mec_loss.item(), epoch)
        writer.add_scalar('train/loss', loss, epoch)
        
        exp_lr_scheduler.step()

        if epoch % args.log_interval == 0:
            print('Train Epoch: [{}/{}]\tClassification Loss: {:.6f} \t MEC Loss: {:.6f}'.format(
                epoch, args.num_iters, cls_loss.item(), mec_loss.item()
            ))

        test_loss, test_acc = test(args, model, device, target_test_loader)
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)

        if test_acc >  best_acc:
            weight_name = f'model_{epoch}_{test_loss:.2f}_{test_acc:.2f}.pth'
            PATH = pathlib.Path.cwd() / 'weights' / weight_name
            best_acc = test_acc
            print(f'Epoch: {epoch}. New best acc loss: {best_acc}')
            torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'cls_loss': cls_loss.item(),
				'mec_loss': mec_loss.item(),
				'test_loss': test_loss
			}, PATH)
		# if (i + 1) % args.check_acc_step == 0:
			# pass
    
    print("Training is complete...")
    print("Running a bunch of forward passes to estimate the population statistics of target...")
    eval_pass_collect_stats(args, model, device, target_test_loader)
    print("Finally computing the precision on the test set...")
    test(args, model, device, target_test_loader)

def correct_count(model, device, data_loader):
    model.eval()
    correct_cnt = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)


def test(args, model, device, target_test_loader):
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output, dim=1),
                                    target, size_average=False).item()
            pred = F.softmax(output, dim=1).max(1, keepdim=True)[
                1]  # get the index of max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(target_test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))

    return test_loss, 100. * correct / len(target_test_loader.dataset)

def _random_affine_augmentation(x):
    M = np.float32([[1 + np.random.normal(0.0, 0.1), np.random.normal(0.0, 0.1), 0],
                    [np.random.normal(0.0, 0.1), 1 + np.random.normal(0.0, 0.1), 0]])
    rows, cols = x.shape[1:3]
    dst = cv2.warpAffine(np.transpose(x.numpy(), [1, 2, 0]), M, (cols, rows))
    dst = np.transpose(dst, [2, 0, 1])
    return torch.from_numpy(dst)


def _gaussian_blur(x, sigma=0.1):
    ksize = int(sigma + 0.5) * 8 + 1
    dst = cv2.GaussianBlur(x.numpy(), (ksize, ksize), sigma)
    return torch.from_numpy(dst)


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch DWT-MEC OfficeHome')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--source_batch_size', type=int, default=8,
                        help='input source batch size for training (default: 20)')
    parser.add_argument('--target_batch_size', type=int, default=8,
                        help='input target batch size for training (default: 20)')
    parser.add_argument('--test_batch_size', type=int, default=10,
                        help='input batch size for testing (default: 10)')
    parser.add_argument('--s_dset_path', type=str,
                        default='../data/OfficeHomeDataset_10072016/Art', help="The source dataset path")
    parser.add_argument('--t_dset_path', type=str,
                        default='../data/OfficeHomeDataset_10072016/Clipart', help="The target dataset path")
    parser.add_argument('--resnet_path', type=str,
                        default='../data/models/model_best_gr_4.pth.tar', help="The pre-trained model path")
    parser.add_argument('--img_resize', type=int,
                        default=256, help='size of the input image')
    parser.add_argument('--img_crop_size', type=int,
                        default=224, help='size of the cropped image')
    parser.add_argument('--num_iters', type=int, default=10000,
                        help='number of iterations to train (default: 10000)')
    parser.add_argument('--check_acc_step', type=int, default=1,
                        help='number of iterations steps to check validation accuracy (default: 10)')
    parser.add_argument('--lr_change_step', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num_classes', type=int, default=65,
                        help='number of classes in the dataset')
    parser.add_argument('--sgd_momentum', type=float,
                        default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--running_momentum', type=float, default=0.1,
                        help='Running momentum for domain statistics(default: 0.1)')
    parser.add_argument('--lambda_mec_loss', type=float, default=0.1,
                        help='Value of lambda for the entropy loss (default: 0.1)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    args = parser.parse_args()

    # set the seed
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # transformation on the source data during training and test data during test
    data_transform = transforms.Compose([
        transforms.Resize((args.img_resize, args.img_resize)
                          ),  # spatial size of vgg-f input
        transforms.RandomCrop((args.img_crop_size, args.img_crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    # transformation on the target data
    data_transform_dup = transforms.Compose([
        transforms.Resize((args.img_resize, args.img_resize)),
        transforms.RandomCrop((args.img_crop_size, args.img_crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: _random_affine_augmentation(x)),
        transforms.Lambda(lambda x: _gaussian_blur(x)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    # train data sets
    source_dataset = utils.folder.ImageFolder(root=args.s_dset_path,
                                        transform=data_transform)
    target_dataset = utils.folder.ImageFolder(root=args.t_dset_path,
                                        transform=data_transform,
                                        transform_aug=data_transform_dup)

    # test data sets
    target_dataset_test = utils.folder.ImageFolder(root=args.t_dset_path,
                                             transform=data_transform)

    # '''''''''''' Train loaders ''''''''''''''' #
    source_trainloader = torch.utils.data.DataLoader(source_dataset,
                                                     batch_size=args.source_batch_size,
                                                     shuffle=True,
                                                     num_workers=args.num_workers,
                                                     drop_last=True)

    target_trainloader = torch.utils.data.DataLoader(target_dataset,
                                                     batch_size=args.source_batch_size,
                                                     shuffle=True,
                                                     num_workers=args.num_workers,
                                                     drop_last=True)

    # '''''''''''' Test loader ''''''''''''''' #
    target_testloader = torch.utils.data.DataLoader(target_dataset_test,
                                                    batch_size=args.test_batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers)

    model = resnet50(args.resnet_path, device).to(device)

    final_layer_params = []
    rest_of_the_net_params = []

    for name, param in model.named_parameters():
        if name.startswith('fc_out'):
            final_layer_params.append(param)
        else:
            rest_of_the_net_params.append(param)

    optimizer = optim.SGD([
        {'params': rest_of_the_net_params},
        {'params': final_layer_params, 'lr': args.lr}
    ], lr=args.lr * 0.1, momentum=0.9, weight_decay=5e-4)

    train_infinite_collect_stats(args=args,
                                 model=model,
                                 device=device,
                                 source_train_loader=source_trainloader,
                                 target_train_loader=target_trainloader,
                                 optimizer=optimizer,
                                 lambda_mec_loss=args.lambda_mec_loss,
                                 target_test_loader=target_testloader)


if __name__ == '__main__':
    main()
