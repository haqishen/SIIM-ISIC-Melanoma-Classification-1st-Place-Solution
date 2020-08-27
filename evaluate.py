import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import PIL.Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2
import apex
from apex import amp
from dataset import get_df, get_transforms, MelanomaDataset
from models import Effnet_Melanoma, Resnest_Melanoma, Seresnext_Melanoma
from train import get_trans


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/')
    parser.add_argument('--data-folder', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--oof-dir', type=str, default='./oofs')
    parser.add_argument('--eval', type=str, choices=['best', 'best_20', 'final'], default="best")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args


def val_epoch(model, loader, mel_idx, is_ext=None, n_test=1, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS, PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
        auc_20 = roc_auc_score((TARGETS[is_ext == 0] == mel_idx).astype(float), PROBS[is_ext == 0, mel_idx])
        return val_loss, acc, auc, auc_20



def main():

    df, df_test, meta_features, n_meta_features, mel_idx = get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir,
        args.data_folder,
        args.use_meta
    )

    transforms_train, transforms_val = get_transforms(args.image_size)

    LOGITS = []
    PROBS = []
    dfs = []
    for fold in range(5):

        df_valid = df[df['fold'] == fold]
        if args.DEBUG:
            df_valid = pd.concat([
                df_valid[df_valid['target'] == mel_idx].sample(args.batch_size * 3),
                df_valid[df_valid['target'] != mel_idx].sample(args.batch_size * 3)
            ])

        dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
        elif args.eval == 'best_20':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_20_fold{fold}.pth')
        if args.eval == 'final':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

        model = ModelClass(
            args.enet_type,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim
        )
        model = model.to(device)

        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        this_LOGITS, this_PROBS = val_epoch(model, valid_loader, mel_idx, is_ext=df_valid['is_ext'].values, n_test=8, get_output=True)
        LOGITS.append(this_LOGITS)
        PROBS.append(this_PROBS)
        dfs.append(df_valid)

    dfs = pd.concat(dfs).reset_index(drop=True)
    dfs['pred'] = np.concatenate(PROBS).squeeze()[:, mel_idx]

    auc_all_raw = roc_auc_score(dfs['target'] == mel_idx, dfs['pred'])

    dfs2 = dfs.copy()
    for i in range(5):
        dfs2.loc[dfs2['fold'] == i, 'pred'] = dfs2.loc[dfs2['fold'] == i, 'pred'].rank(pct=True)
    auc_all_rank = roc_auc_score(dfs2['target'] == mel_idx, dfs2['pred'])

    dfs3 = dfs[dfs.is_ext == 0].copy().reset_index(drop=True)
    auc_20_raw = roc_auc_score(dfs3['target'] == mel_idx, dfs3['pred'])

    for i in range(5):
        dfs3.loc[dfs3['fold'] == i, 'pred'] = dfs3.loc[dfs3['fold'] == i, 'pred'].rank(pct=True)
    auc_20_rank = roc_auc_score(dfs3['target'] == mel_idx, dfs3['pred'])

    content = f'Eval {args.eval}:\nauc_all_raw : {auc_all_raw:.5f}\nauc_all_rank : {auc_all_rank:.5f}\nauc_20_raw : {auc_20_raw:.5f}\nauc_20_rank : {auc_20_rank:.5f}\n'
    print(content)
    with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
        appender.write(content + '\n')

    np.save(os.path.join(args.oof_dir, f'{args.kernel_type}_{args.eval}_oof.npy'), dfs['pred'].values)


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.oof_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnest101':
        ModelClass = Resnest_Melanoma
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_Melanoma
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Melanoma
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    main()
