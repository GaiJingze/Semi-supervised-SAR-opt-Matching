import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
from matplotlib import pyplot as plt

from model import SAROptMatcher
from Sen12_dataset import Sen12_dataset
from QXSLAB_dataset import QXSLAB_dataset
from supervised_loss import cross_entropy_with_nmining, cross_entropy_loss
from model_utils import get_predicted_matching_pos

def save_train_log(epoch, record_batch_idx, avg_l2_dist, avg_loss, avg_loss_ce_deep, avg_loss_ce_shallow, avg_loss_mi, log_file='./train_log.json'):
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logs = []
    else:
        logs = []

    log_entry = {
        'epoch': epoch,
        'record_batch_idx': record_batch_idx,
        'avg_l2_dist': avg_l2_dist,
        'avg_loss': avg_loss,
        'avg_loss_ce_deep': avg_loss_ce_deep,
        'avg_loss_ce_shallow': avg_loss_ce_shallow,
        'avg_loss_mi': avg_loss_mi
    }
    logs.append(log_entry)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)

def train_batch_labeled(model, data_batch, device):
    opt_img = data_batch['img_1'].to(device)
    sar_img = data_batch['img_2'].to(device)
    gt = data_batch['gt'].to(device)
    
    deep_opt_feature, shallow_opt_feature = model.backbone(opt_img)
    deep_sar_feature, shallow_sar_feature = model.backbone(sar_img)
    
    deep_opt_feature, deep_sar_feature, loss_mi_deep = model.deep_feature_enhancer(deep_opt_feature, deep_sar_feature)
    shallow_opt_feature, shallow_sar_feature, loss_mi_shallow = model.shallow_feature_enhancer(shallow_opt_feature, shallow_sar_feature)
    
    _, _, H_sar_d, W_sar_d = deep_sar_feature.shape
    _, _, H_sar_s, W_sar_s = shallow_sar_feature.shape

    # FFT-based Cross-Correlation
    deep_fft_corr = model.fft(deep_opt_feature, deep_sar_feature)
    deep_fft_corr = model.norm_layer(deep_opt_feature, deep_sar_feature, deep_fft_corr)
    deep_fft_corr = deep_fft_corr[:, :, H_sar_d-1 : - (H_sar_d-1), W_sar_d-1 : - (W_sar_d-1)]

    shallow_fft_corr = model.fft(shallow_opt_feature, shallow_sar_feature)
    shallow_fft_corr = model.norm_layer(shallow_opt_feature, shallow_sar_feature, shallow_fft_corr)
    shallow_fft_corr = shallow_fft_corr[:, :, H_sar_s-1 : - (H_sar_s-1), W_sar_s-1 : - (W_sar_s-1)]

    deep_fft_corr = deep_fft_corr.mean(dim=1, keepdim=True)
    deep_fft_corr_upscaled = F.interpolate(deep_fft_corr, size=shallow_fft_corr.shape[-2:], mode='nearest')
    shallow_fft_corr = shallow_fft_corr.mean(dim=1, keepdim=True)

    deep_fft_corr = deep_fft_corr.squeeze(1)
    deep_fft_corr_upscaled = deep_fft_corr_upscaled.squeeze(1)
    shallow_fft_corr = shallow_fft_corr.squeeze(1)
    
    _, H_deep_fft, W_deep_fft = deep_fft_corr.shape
    _, H_shallow_fft, W_shallow_fft = shallow_fft_corr.shape

    # supervised cross-entropy losses
    loss_ce_shallow = cross_entropy_with_nmining(shallow_fft_corr, gt)
    scales = torch.tensor([W_deep_fft/W_shallow_fft, H_deep_fft/H_shallow_fft], device=device)
    gt_rescaled = (gt.float() * scales).floor().long()
    loss_ce_deep = cross_entropy_loss(deep_fft_corr, gt_rescaled)

    # L2 distance for logging
    fft_for_prediction = deep_fft_corr_upscaled * shallow_fft_corr
    prediction = get_predicted_matching_pos(fft_for_prediction)
    l2_dist = torch.norm(prediction.float() - gt.float(), dim=1).mean()

    loss_mi = loss_mi_deep + loss_mi_shallow
    return loss_ce_deep, loss_ce_shallow, loss_mi, l2_dist

def train_batch_unlabeled(model, data_batch, device):
    B = data_batch['img_1'].shape[0]
    opt_img = data_batch['img_1'].to(device)
    sar_img = data_batch['img_2'].to(device)
    
    deep_opt_feature, shallow_opt_feature = model.backbone(opt_img)
    deep_sar_feature, shallow_sar_feature = model.backbone(sar_img)
    
    deep_opt_feature, deep_sar_feature, loss_mi_deep = model.deep_feature_enhancer(deep_opt_feature, deep_sar_feature)
    shallow_opt_feature, shallow_sar_feature, loss_mi_shallow = model.shallow_feature_enhancer(shallow_opt_feature, shallow_sar_feature)
    
    _, _, H_sar_d, W_sar_d = deep_sar_feature.shape
    _, _, H_sar_s, W_sar_s = shallow_sar_feature.shape

    deep_fft_corr = model.fft(deep_opt_feature, deep_sar_feature)
    deep_fft_corr = model.norm_layer(deep_opt_feature, deep_sar_feature, deep_fft_corr)
    deep_fft_corr = deep_fft_corr[:, :, H_sar_d-1 : - (H_sar_d-1), W_sar_d-1 : - (W_sar_d-1)]

    shallow_fft_corr = model.fft(shallow_opt_feature, shallow_sar_feature)
    shallow_fft_corr = model.norm_layer(shallow_opt_feature, shallow_sar_feature, shallow_fft_corr)
    shallow_fft_corr = shallow_fft_corr[:, :, H_sar_s-1 : - (H_sar_s-1), W_sar_s-1 : - (W_sar_s-1)]

    deep_fft_corr = deep_fft_corr.mean(dim=1, keepdim=True)
    # smoother upsampling
    deep_fft_corr_upscaled = F.interpolate(deep_fft_corr, size=shallow_fft_corr.shape[-2:], mode='bilinear', align_corners=False)
    shallow_fft_corr = shallow_fft_corr.mean(dim=1, keepdim=True)

    deep_fft_corr = deep_fft_corr.squeeze(1)
    deep_fft_corr_upscaled = deep_fft_corr_upscaled.squeeze(1)
    shallow_fft_corr = shallow_fft_corr.squeeze(1)
    
    temperature = 1/80
    pgt_temperature = 1/200

    pseudo_gt = deep_fft_corr_upscaled * shallow_fft_corr
    pseudo_gt_flatten = F.softmax(pseudo_gt.view(B, -1) / pgt_temperature, dim=1).detach()
    shallow_fft_corr_flatten = shallow_fft_corr.view(B, -1) / temperature

    loss_pce = F.cross_entropy(shallow_fft_corr_flatten, pseudo_gt_flatten)
    loss_mi = loss_mi_deep + loss_mi_shallow
    return loss_pce, loss_mi

def train_ss(model, labeled_dataset, unlabeled_dataset,
             labeled_batch_size=1, unlabeled_batch_size=15,
             num_epochs=5, learning_rate=5e-5, weight_decay=0.1,
             checkpoint_path=None, model_path="sar_opt_matcher", record_log = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled_loader = DataLoader(labeled_dataset, batch_size=labeled_batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=True)
    
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start_epoch = 0

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if checkpoint_path.endswith('ss.pth'):
            start_epoch = ckpt.get('epoch', 0)
            print(f"Resuming from epoch {start_epoch}")

    record_loss = record_loss_ce_deep = record_loss_ce_shallow = record_loss_mi = record_l2_dist = 0.0
    record_interval = 500
    record_batch_idx = 0

    for epoch in range(start_epoch, num_epochs):
        unlabeled_iter = iter(unlabeled_loader)
        progress_bar = tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for labeled_batch in progress_bar:
            # get next unlabeled, reset if exhausted
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            optimizer.zero_grad()

            # supervised
            loss_ce_deep, loss_ce_shallow, loss_mi, l2_dist = train_batch_labeled(model, labeled_batch, device)
            loss_labeled = loss_ce_deep + loss_ce_shallow + loss_mi

            # unsupervised
            loss_pce, loss_umi = train_batch_unlabeled(model, unlabeled_batch, device)
            loss_unlabeled = loss_pce + loss_umi

            loss = loss_labeled + loss_unlabeled
            loss.backward()
            optimizer.step()

            if record_log:
                # accumulate for logging
                record_loss += loss.item()
                record_l2_dist += l2_dist.item()
                record_loss_ce_deep += loss_ce_deep.item()
                record_loss_ce_shallow += loss_ce_shallow.item()
                record_loss_mi += loss_mi.item()
                record_batch_idx += 1

                if record_batch_idx % record_interval == 0:
                    avg_loss = record_loss / record_interval
                    avg_l2 = record_l2_dist / record_interval
                    avg_l_cd = record_loss_ce_deep / record_interval
                    avg_l_cs = record_loss_ce_shallow / record_interval
                    avg_l_mi = record_loss_mi / record_interval
                    save_train_log(epoch, record_batch_idx, avg_l2, avg_loss, avg_l_cd, avg_l_cs, avg_l_mi)
                    record_loss = record_l2_dist = record_loss_ce_deep = record_loss_ce_shallow = record_loss_mi = 0.0

            progress_bar.set_postfix({
                'l_ce_s': f"{loss_ce_shallow:.5f}",
                'l_pce': f"{loss_pce:.5f}",
                'l_mi': f"{loss_mi:.5f}",
                'l2': f"{l2_dist:.5f}"
            })

        # save epoch checkpoint
        ckpt = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        os.makedirs("train_checkpoints", exist_ok=True)
        torch.save(ckpt, f"./train_checkpoints/{model_path}_epoch_{epoch+1}.pth")

    os.makedirs("model_checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"./model_checkpoints/{model_path}.pth")

def train_fs(model, dataset,
             batch_size=16, num_epochs=5,
             learning_rate=5e-5, weight_decay=0.1,
             checkpoint_path=None, model_path="sar_opt_matcher_fs"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start_epoch = 0

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")

    record_loss = record_loss_ce_deep = record_loss_ce_shallow = record_loss_mi = record_l2_dist = 0.0
    record_interval = 10000
    record_batch_idx = 0

    for epoch in range(start_epoch, num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            loss_ce_deep, loss_ce_shallow, loss_mi, l2_dist = train_batch_labeled(model, batch, device)
            loss = loss_ce_deep + loss_ce_shallow + loss_mi
            loss.backward()
            optimizer.step()

            record_loss += loss.item()
            record_l2_dist += l2_dist.item()
            record_loss_ce_deep += loss_ce_deep.item()
            record_loss_ce_shallow += loss_ce_shallow.item()
            record_loss_mi += loss_mi.item()
            record_batch_idx += 1

            if record_batch_idx % record_interval == 0:
                avg_loss = record_loss / record_interval
                avg_l2 = record_l2_dist / record_interval
                avg_l_cd = record_loss_ce_deep / record_interval
                avg_l_cs = record_loss_ce_shallow / record_interval
                avg_l_mi = record_loss_mi / record_interval
                save_train_log(epoch, record_batch_idx, avg_l2, avg_loss, avg_l_cd, avg_l_cs, avg_l_mi)
                record_loss = record_l2_dist = record_loss_ce_deep = record_loss_ce_shallow = record_loss_mi = 0.0

            progress_bar.set_postfix({
                'l_ce_d': f"{loss_ce_deep:.5f}",
                'l_ce_s': f"{loss_ce_shallow:.5f}",
                'l_mi': f"{loss_mi:.5f}",
                'l2': f"{l2_dist:.5f}"
            })

        os.makedirs("train_checkpoints", exist_ok=True)
        ckpt = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(ckpt, f"./train_checkpoints/{model_path}_epoch_{epoch+1}.pth")

    os.makedirs("model_checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"./model_checkpoints/{model_path}.pth")


if __name__ == "__main__":

    labeled_dataset = Sen12_dataset((256, 256), (192, 192), data_range=range(0, 16000))
    unlabeled_dataset = Sen12_dataset((256, 256), (192, 192), data_range=range(16000, 256000))

    #labeled_dataset = QXSLAB_dataset((256, 256), (192, 192), data_range=range(0, 20000))
    #unlabeled_dataset = QXSLAB_dataset((256, 256), (192, 192), data_range=range(1000, 16000))

    model = SAROptMatcher()

    train_ss(model, labeled_dataset, unlabeled_dataset, num_epochs=10, learning_rate = 0.00001, model_path="sar_opt_matcher")