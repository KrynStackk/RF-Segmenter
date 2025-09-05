import os
import glob
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics import Accuracy, JaccardIndex
from tqdm import tqdm
from einops import repeat
from timm.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

CLASSES = ["Noise", "LTE", "NR", "Radar"]
COLORMAP = [[0, 0, 0], [80, 80, 80], [160, 160, 160], [255, 255, 255]]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


class SearchDataset(Dataset):
    def __init__(self, root="/kaggle/input/radarcomm", image_set="train", transform=None):
        self.images = sorted(glob.glob(os.path.join(root, image_set, 'input', '*.png')))
        self.masks = sorted(glob.glob(os.path.join(root, image_set, 'label', '*.png')))
        self.transform = transform
        self.classes = len(COLORMAP)

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        h, w = mask.shape[:2]
        seg = np.zeros((h, w), dtype=np.uint8)
        for i, color in enumerate(COLORMAP):
            seg[np.all(mask == color, axis=-1)] = i
        return seg

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)
        if self.transform is not None:
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()
        return image, mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
n_workers = os.cpu_count()

train_dataset = SearchDataset(image_set="train", transform=train_transform)
val_dataset = SearchDataset(image_set="val", transform=train_transform)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)


class SSCA(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False,
                 device=None, dtype=None, c_increase=32, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = self.d_model
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.conv_c_in = nn.Sequential(
            nn.Conv2d(2, c_increase, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_increase),
            nn.GELU(),
        )
        self.conv_c_out = nn.Sequential(
            nn.Conv2d(c_increase, 1, kernel_size=1, bias=False),
            nn.GELU(),
        )
        xc_proj_weight = [nn.Linear(c_increase, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight for _ in range(2)]
        self.xc_proj_weight = nn.Parameter(torch.stack(xc_proj_weight, dim=0))
        dtc_projs = [self.dt_init(self.dt_rank, c_increase, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(2)]
        self.dtc_projs_weight = nn.Parameter(torch.stack([dtc_proj.weight for dtc_proj in dtc_projs], dim=0))
        self.dtc_projs_bias = nn.Parameter(torch.stack([dtc_proj.bias for dtc_proj in dtc_projs], dim=0))
        self.Ac_logs = self.A_log_init(self.d_state, c_increase, copies=2, merge=True)
        self.Dcs = self.D_init(c_increase, copies=2, merge=True)
        self.out_norm_c = nn.LayerNorm(self.d_model)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core_channel(self, xc: torch.Tensor):
        B, C, H, W = xc.shape
        avg_pool_feats = self.gap(xc)
        max_pool_feats = self.gmp(xc)
        xsc = torch.cat([avg_pool_feats, max_pool_feats], dim=1)
        xsc = xsc.view(B, 2, C, 1)
        xsc = self.conv_c_in(xsc)
        xsc = xsc.squeeze(-1)
        B, D_conv_c_in, L = xsc.shape
        xsc = torch.stack([xsc, torch.flip(xsc, dims=[-1])], dim=1)
        xc_dbl = torch.einsum("b k d l, k c d -> b k c l", xsc, self.xc_proj_weight)
        dts, Bs, Cs = torch.split(xc_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dtc_projs_weight).contiguous()
        xsc = xsc.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Ds = self.Dcs.float().view(-1)
        As = -torch.exp(self.Ac_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dtc_projs_bias.float().view(-1)
        out_y = selective_scan_fn(
            xsc, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False
        ).view(B, 2, -1, L)
        y = out_y[:, 0].float() + torch.flip(out_y[:, 1], dims=[-1]).float()
        y = y.unsqueeze(-1)
        y = self.conv_c_out(y)
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm_c(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        attn = self.forward_core_channel(x)
        x_attn = x * attn
        return x + x_attn


class FusionAttention(nn.Module):
    def __init__(self, channels, num_branches=2, reduction_ratio=8):
        super().__init__()
        self.num_branches = num_branches
        self.channels_per_branch = channels // num_branches
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )

    def forward(self, branch_outputs):
        x_merged = torch.cat(branch_outputs, dim=1)
        a_weights = self.mlp(x_merged)
        a_weights = a_weights.view(-1, self.num_branches, self.channels_per_branch, 1, 1)
        a_weights = F.softmax(a_weights, dim=1)
        fused = []
        for i in range(self.num_branches):
            w = a_weights[:, i, :, :, :]
            fused.append(branch_outputs[i] * w.expand_as(branch_outputs[i]))
        return torch.cat(fused, dim=1)


class DepthwiseAsymmetricConv(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv_a = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, padding), groups=channels, dilation=(1, dilation)),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=channels, dilation=(dilation, 1)),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv_a(x), self.conv_b(x)


class ADAC(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4, stride=1):
        super().__init__()
        hidden = in_channels * rate
        part = hidden // 2
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, stride=stride, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )
        self.normal = DepthwiseAsymmetricConv(part, kernel_size=3, dilation=1)
        self.dilated = DepthwiseAsymmetricConv(part, kernel_size=3, dilation=2)
        self.fuse = FusionAttention(hidden, num_branches=2)
        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.is_identity = (in_channels == out_channels and stride == 1)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.conv_in(x)
        p1, p2 = torch.chunk(x, 2, dim=1)
        p1 = sum(self.normal(p1))
        p2 = sum(self.dilated(p2))
        x = self.fuse([p1, p2])
        x = self.conv_out(x)
        if self.is_identity:
            x = x + residual
        return self.act(x)


class ADACs(nn.Module):
    def __init__(self, in_channels, out_channels, rate=2, stride=1, nblock=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        cur_in = in_channels
        for i in range(nblock):
            s = stride if i == 0 else 1
            self.blocks.append(ADAC(cur_in, out_channels, rate=rate, stride=s))
            cur_in = out_channels

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        if C % self.groups != 0:
            raise ValueError("C must be divisible by groups")
        g = self.groups
        x = x.view(N, g, C // g, H, W).transpose(1, 2).contiguous().view(N, C, H, W)
        return x


class UPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rate=2):
        super().__init__()
        groups = 1 if rate == 1 else math.gcd(in_channels, out_channels)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=rate, stride=rate, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class MSAF(nn.Module):
    def __init__(self, c_int=96, c_hidden_smn=160, c_out=96, num_features=4, smn_shuffle_groups=4, ffc_shuffle_groups=4):
        super().__init__()
        self.c_int = c_int
        self.num_features = num_features
        self.Up1 = UPBlock(64, c_int, rate=1)
        self.Up2 = UPBlock(96, c_int, rate=2)
        self.Up3 = UPBlock(160, c_int, rate=4)
        self.Up4 = UPBlock(256, c_int, rate=8)
        conv1_in = self.num_features * c_int
        smn_layers = [
            nn.Conv2d(conv1_in, c_hidden_smn, 1, bias=False),
            nn.BatchNorm2d(c_hidden_smn),
            nn.ReLU(inplace=True)
        ]
        if c_hidden_smn % smn_shuffle_groups == 0:
            smn_layers.append(ChannelShuffle(smn_shuffle_groups))
        smn_layers.append(nn.Conv2d(c_hidden_smn, self.num_features * self.num_features * c_int, 1))
        self.smn = nn.Sequential(*smn_layers)
        ffc_conv1_in = self.num_features * c_int
        ffc_shuffle_optional = []
        if c_out % ffc_shuffle_groups == 0:
            ffc_shuffle_optional.append(ChannelShuffle(ffc_shuffle_groups))
        self.final_fusion_conv = nn.Sequential(
            nn.Conv2d(ffc_conv1_in, c_out, 1, bias=False),
            *ffc_shuffle_optional,
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False, groups=c_out),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, A, B, C, D):
        s1 = self.Up1(A)
        s2 = self.Up2(B)
        s3 = self.Up3(C)
        s4 = self.Up4(D)
        s_list = [s1, s2, s3, s4]
        v_list = [F.adaptive_avg_pool2d(s, (1, 1)) for s in s_list]
        v_agg = torch.cat(v_list, dim=1)
        m_flat = self.smn(v_agg)
        m_ij = torch.sigmoid(m_flat.view(s_list[0].size(0), self.num_features, self.num_features, self.c_int, 1, 1))
        s_prime = []
        for i in range(self.num_features):
            cur = torch.zeros_like(s_list[0])
            for j in range(self.num_features):
                cur = cur + s_list[j] * m_ij[:, i, j, :, :, :]
            s_prime.append(cur)
        s_cat = torch.cat(s_prime, dim=1)
        return self.final_fusion_conv(s_cat)


class RF_Segmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ADACs(32, 64, rate=2, stride=2, nblock=2),
        )
        self.stage2 = nn.Sequential(
            ADACs(64, 96, rate=2, stride=2, nblock=2),
        )
        self.stage3 = nn.Sequential(
            ADACs(96, 160, rate=2, stride=2, nblock=2),
            SSCA(d_model=160, d_state=16),
        )
        self.stage4 = nn.Sequential(
            ADACs(160, 256, rate=2, stride=2, nblock=2),
            SSCA(d_model=256, d_state=16),
        )
        self.msaf = MSAF(c_int=96, c_hidden_smn=160, c_out=96, num_features=4)
        self.ssca = SSCA(d_model=96, d_state=16)
        self.up = UPBlock(96, 4, rate=4)

    def forward(self, x):
        x = self.conv_in(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x = self.msaf(x1, x2, x3, x4)
        x = self.ssca(x)
        x = self.up(x)
        return x


model = RF_Segmenter().to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")
print(f"Number of trainable parameters: {trainable_params}")

n_eps = 60
lr = 1e-3
scheduler_step_size = 20
scheduler_gamma = 0.2
num_classes = 4

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


def evaluate(model, dataloader, criterion, device, current_num_classes):
    model.eval()
    val_loss_meter = AverageMeter()
    acc_metric = Accuracy(task="multiclass", num_classes=current_num_classes, average='micro').to(device)
    iou_metric = JaccardIndex(task="multiclass", num_classes=current_num_classes, average='macro').to(device)
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            val_loss_meter.update(loss.item(), n=y.size(0))
            preds = torch.argmax(y_hat, dim=1)
            acc_metric.update(preds, y)
            iou_metric.update(preds, y)
    return val_loss_meter.avg, acc_metric.compute().item(), iou_metric.compute().item()


start_epoch = 1
best_val_mean_iou = 0.0
best_epoch_val = 0


for ep in range(start_epoch, n_eps + 1):
    model.train()
    train_loss_meter = AverageMeter()
    acc_metric = Accuracy(task="multiclass", num_classes=num_classes, average='micro').to(device)
    iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, average='macro').to(device)
    with tqdm(trainloader, desc=f"Training Epoch {ep}/{n_eps}", unit="batch") as tepoch:
        for x, y in tepoch:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss_meter.update(loss.item(), n=y.size(0))
            preds = torch.argmax(y_hat, dim=1)
            acc_metric.update(preds, y)
            iou_metric.update(preds, y)
            tepoch.set_postfix(loss=train_loss_meter.avg, acc=acc_metric.compute().item(), iou=iou_metric.compute().item())
    train_acc = acc_metric.compute().item()
    train_iou = iou_metric.compute().item()
    val_loss, val_acc, val_iou = evaluate(model, valloader, criterion, device, num_classes)

    print(f"\nEpoch {ep}/{n_eps} | LR {scheduler.get_last_lr()[0]:.6f}")
    print(f"Train: loss={train_loss_meter.avg:.4f}, acc={train_acc:.4f}, iou={train_iou:.4f}")
    print(f"Val  : loss={val_loss:.4f}, acc={val_acc:.4f}, iou={val_iou:.4f}")
    scheduler.step()
    ckpt_path = f"epoch_{ep}_ValAcc_{val_acc:.4f}_ValIoU_{val_iou:.4f}.pt"
    torch.save({
        'epoch': ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'current_val_accuracy': val_acc,
        'current_val_mean_iou': val_iou,
        'train_loss': train_loss_meter.avg,
        'val_loss': val_loss,
        'best_val_mean_iou': best_val_mean_iou,
        'best_epoch': best_epoch_val,
    }, ckpt_path)
    if val_iou > best_val_mean_iou:
        best_val_mean_iou = val_iou
        best_epoch_val = ep
        torch.save({'epoch': ep, 'model_state_dict': model.state_dict()}, 'best_model_state.pt')
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_mean_iou': best_val_mean_iou,
            'best_epoch': best_epoch_val,
        }, 'best_checkpoint.pt')

print(f"Done. Best Val IoU: {best_val_mean_iou:.4f} @ epoch {best_epoch_val}")
