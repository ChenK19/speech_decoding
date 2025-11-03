import numpy as np
import matplotlib.pyplot as plt
import mne
import torch
from torch import nn
from example_offline_read_bdf_data import readbdfdata

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
# -----------------------------
# 假设已有：
# 1. 训练好的 EEGNet 模型: model
# 2. 原始数据: raw (mne.io.Raw)
# -----------------------------

# 提取通道名，只保留 EEG 通道

filename = ['data.bdf', 'evt.bdf']
path_name = "/localdata1/liuqi/data/" + '20251030eeg-language/ck3'
raw = readbdfdata(filename, [path_name])
channel_names = [ch for ch in raw.info["ch_names"] 
                 if ch not in ["ECG", "HEOR", "HEOL", "VEOU", "VEOL"]]
print(f"使用 {len(channel_names)} 个 EEG 通道：", channel_names)

model = EEGNet(C=59, T=121, n_classes=8,
    F1=8, D=2, kernel_length=128,
    drop_rate=0.4, pool1=4, pool2=8,).to("cuda")
model.load_state_dict(torch.load("models/fold_1_best.pt"))
# ============================
# 1️⃣ 提取 Depthwise 层权重
# ============================
# 通常 EEGNet.depthwise 是 nn.Sequential 中的第 0 层 Conv2d
depthwise_weights = model.depthwise[0].weight.detach().cpu().numpy()  # shape: (F1*D, 1, C, 1)
F1D, _, C, _ = depthwise_weights.shape
print("Depthwise shape:", depthwise_weights.shape)

# 求每个通道的平均绝对权重（代表通道重要性）
channel_importance = np.mean(np.abs(depthwise_weights), axis=(0, 1, 3))
channel_importance = channel_importance[:len(channel_names)]  # 对齐

# ============================
# 2️⃣ 创建 MNE info 对象
# ============================
sfreq = raw.info["sfreq"]
info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")

# 设置标准电极布局（推荐 10-20 或 10-05）
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage)

# ============================
# 3️⃣ 绘制通道贡献度 Topomap
# ============================
plt.figure(figsize=(6, 5))
# 创建图和坐标轴
fig, ax = plt.subplots(figsize=(6, 5))

im, _ = mne.viz.plot_topomap(
    channel_importance, info,
    cmap="plasma",
    contours=0,
    res=256,
    size=4,
    axes=ax,

    show=False,        # 关键：不要自动 show
)

# 添加标题和 colorbar
ax.set_title("EEG Channel Importance (Depthwise Conv Weights)", fontsize=13)
cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
cbar.set_label("Importance (|weight|)", fontsize=11)

plt.title("EEG Channel Importance (Depthwise Conv Weights)", fontsize=13)
plt.savefig("output_pic/channel_contribution.png", bbox_inches="tight")

# ============================
# 4️⃣ 可选：打印前10个最重要通道
# ============================
sorted_idx = np.argsort(channel_importance)[::-1]
print("\n前 10 个最重要的通道：")
for i in range(10):
    ch = channel_names[sorted_idx[i]]
    val = channel_importance[sorted_idx[i]]
    print(f"{i+1:02d}. {ch:>4s}  ->  {val:.4f}")
