# -*- coding: utf-8 -*-
import os
import math
import random
import numpy as np
from typing import Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class EEGAugmenter:
    def __init__(self, noise_std=0.01, shift_max=0.1, scale_range=(0.9, 1.1),
                 dropout_prob=0.1, mixup_alpha=0.2, p_augment=1):
        """
        参数：
            noise_std: 高斯噪声标准差
            shift_max: 时间平移比例（例如 0.1 = 最多 ±10% 时间偏移）
            scale_range: 振幅缩放范围
            dropout_prob: 通道随机失活概率
            mixup_alpha: mixup参数（Beta分布）
            p_augment: 每个样本被增广的总体概率
        """
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.scale_range = scale_range
        self.dropout_prob = dropout_prob
        self.mixup_alpha = mixup_alpha
        self.p_augment = p_augment

    def add_noise(self, x):
        noise = np.random.randn(*x.shape) * self.noise_std
        return x + noise

    def time_shift(self, x):
        T = x.shape[-1]
        shift = np.random.randint(-int(T*self.shift_max), int(T*self.shift_max))
        return np.roll(x, shift, axis=-1)

    def scale_amplitude(self, x):
        scale = np.random.uniform(*self.scale_range)
        return x * scale

    def channel_dropout(self, x):
        mask = np.random.binomial(1, 1 - self.dropout_prob, size=x.shape[-2])
        return x * mask[:, None]

    def mixup(self, x1, y1, x2, y2):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        x_mix = lam * x1 + (1 - lam) * x2
        y_mix = lam * y1 + (1 - lam) * y2
        return x_mix, y_mix

    def __call__(self, X, y=None):
        """对 batch (N,C,T) 或单样本 (C,T) 增广，每次执行所有增强"""
        X_aug = X.copy()
        for i in range(len(X_aug)):
            if np.random.rand() < self.p_augment:
                x = X_aug[i]
                # 顺序执行全部增强
                x = self.add_noise(x)
                x = self.time_shift(x)
                x = self.scale_amplitude(x)
                x = self.channel_dropout(x)
                X_aug[i] = x
        return X_aug
# ========== 0. 可复现 ==========
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ========== 1. EEGNet 定义 ==========
# 参考: Lawhern et al., EEGNet: A Compact CNN for EEG-based BCI (2018)
class EEGNet(nn.Module):
    """
    输入: (N, 1, C, T)
    参数:
        C: 通道数
        T: 单trial时间点数
        n_classes: 类别数
        F1: 第一层滤波器数量
        D: depthwise 的深度乘子
        F2: 第二层分离卷积滤波器数量(= F1*D)
        kernel_length: 时间卷积核长
        drop_rate: dropout比例
        pool1, pool2: 两次池化的时间维窗口
        norm_eps: BN小常数
    """
    def __init__(self, C:int, T:int, n_classes:int,
                 F1:int=8, D:int=2, F2:int=None,
                 kernel_length:int=64, drop_rate:float=0.5,
                 pool1:int=4, pool2:int=8, norm_eps:float=1e-5):
        super().__init__()
        if F2 is None:
            F2 = F1 * D

        self.block1 = nn.Sequential(
            # Temporal Convolution (沿时间轴, 不混通道)
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length//2), bias=False),
            nn.BatchNorm2d(F1, eps=norm_eps)
        )
        # Depthwise Convolution (跨通道, 不混时间)
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1*D, (C, 1), groups=F1, bias=False),  # depthwise across channels
            nn.BatchNorm2d(F1*D, eps=norm_eps),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout(drop_rate)
        )
        # Separable Convolution = pointwise(1x1) + temporal conv
        self.separable = nn.Sequential(
            nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False),  # temporal
            nn.BatchNorm2d(F2, eps=norm_eps),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout(drop_rate)
        )

        # 计算分类层输入维度
        with torch.no_grad():
            x = torch.zeros(1, 1, C, T)
            x = self.block1(x)
            x = self.depthwise(x)
            x = self.separable(x)
            flat_dim = x.numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, n_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.classifier(x)
        return x


# ========== 2. 数据集封装 ==========
class EEGDataset(Dataset):
    """
    X: np.ndarray, shape (N, 1, C, T) 或 (N, C, T) 也可，会自动扩维到 (N,1,C,T)
    y: np.ndarray, shape (N,)
    """
    def __init__(self, X:np.ndarray, y:np.ndarray, dtype=torch.float32):
        assert X.ndim in (3,4)
        if X.ndim == 3:
            X = X[:, None, :, :]  # (N, C, T) -> (N,1,C,T)
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ========== 3. 训练/验证工具 ==========
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
        self.stop = False

    def step(self, metric):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True
        return self.stop

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, running_loss = 0, 0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += X.size(0)
    return running_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, running_loss = 0, 0, 0.0
    all_pred, all_true = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        running_loss += loss.item() * X.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += X.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
    acc = correct / total
    y_pred = np.concatenate(all_pred); y_true = np.concatenate(all_true)
    return running_loss/total, acc, y_true, y_pred


# ========== 4. 组装可运行示例 ==========
"""
将你的 EEG epoch 数据转成 (N, C, T) 或 (N, 1, C, T)：
- 若你已有 rawData 每类为 (n_epochs, n_channels, n_times)，可以拼起来：
    X = np.concatenate([rawData[p] for p in pinyins], axis=0)  # (N, C, T)
    y = np.concatenate([np.full(len(rawData[p]), idx) for idx,p in enumerate(pinyins)])
- 或用你自己的数据加载逻辑，保证形状一致即可。
这里给一个随机数据 demo，替换成你的数据即可。
"""
def build_dummy_data(N=256, C=59, T=1200, n_classes=8):
    X = np.random.randn(N, C, T).astype(np.float32) * 1e-6  # 模拟微伏级
    y = np.random.randint(0, n_classes, size=N)
    return X, y

def main_train_kfold(
    C:int=59, T:int=121, n_classes:int=8,
    batch_size:int=32, epochs:int=100,
    lr:float=1e-3, weight_decay:float=1e-4,
    F1:int=8, D:int=2, kernel_length:int=128,
    drop_rate:float=0.4, pool1:int=4, pool2:int=8,
    early_patience:int=15, n_splits:int=10,
    save_dir:str="./models"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----------------------------
    # 载入数据
    # ----------------------------
    data_dict = np.load("speech_decoding_epochs-59-channels.npy", allow_pickle=True).item()
    for k in data_dict:
        data_dict[k] = np.array(data_dict[k])

    # 构造 X, y
    X_list, y_list = [], []
    for label, arr in data_dict.items():
        n_samples = arr.shape[0]
        X_list.append(arr)
        y_list.append(np.full(n_samples, label))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list)

    mapping = {"mā":0, "má":1, "mǎ":2, "mà":3, "mī":4, "mí":5, "mǐ":6, "mì":7}
    y = np.vectorize(lambda x: mapping[x])(y)

    # 标准化（z-score）
    eps = 1e-8
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + eps
    X = (X - mean) / std

    # ----------------------------
    # 五折交叉验证
    # ----------------------------
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n========== Fold {fold+1}/{n_splits} ==========")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 增广
        augmenter = EEGAugmenter(noise_std=0.005, shift_max=0.05, scale_range=(0.9,1.1))
        X_train_aug = augmenter(X_train)

        ds_train = EEGDataset(X_train_aug, y_train)
        ds_val   = EEGDataset(X_val, y_val)
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
        dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

        # 模型定义
        model = EEGNet(C=C, T=T, n_classes=n_classes,
                       F1=F1, D=D, kernel_length=kernel_length,
                       drop_rate=drop_rate, pool1=pool1, pool2=pool2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        stopper = EarlyStopping(patience=early_patience)

        save_path = f"{save_dir}/fold_{fold+1}_best.pt"
        best_acc = 0.0

        # ----------------------------
        # 训练
        # ----------------------------
        for ep in range(1, epochs+1):
            tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, device)
            val_loss, val_acc, y_true, y_pred = evaluate(model, dl_val, device)
            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)

            print(f"[Fold {fold+1} | Epoch {ep:03d}/{epochs}] "
                  f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"Val loss {val_loss:.4f} acc {val_acc:.4f} (best {best_acc:.4f})")

            if stopper.step(val_acc):
                print("Early stopping triggered.")
                break

        # ----------------------------
        # 验证
        # ----------------------------
        model.load_state_dict(torch.load(save_path, map_location=device))
        _, val_acc, y_true, y_pred = evaluate(model, dl_val, device)
        fold_accs.append(val_acc)

        print(f"\n✅ Fold {fold+1} best accuracy: {val_acc:.4f}")
        print(classification_report(y_true, y_pred, digits=3))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    # ----------------------------
    # 汇总
    # ----------------------------
    print("\n========== Cross-validation summary ==========")
    print(f"Mean acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")

    return fold_accs

if __name__ == "__main__":
    # 如果你已经有真实数据 X, y（形状见上），把它们传进 main_train 即可
    # 例如：X.shape=(N,C,T), y.shape=(N,)
    train_accs = main_train_kfold()
