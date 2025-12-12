
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def setup_distributed():
    """initialize environment（LOCAL_RANK/ RANK/ WORLD_SIZE）"""
    os.environ['OMP_NUM_THREADS'] = '1'
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, world_size = 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def reduce_sum(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

merged_batch = torch.load(os.path.join("raw_data.pt"))

static_features = torch.cat([merged_batch['T_scalar'], merged_batch['ε̇_scalar']], dim=1)
stress_features = merged_batch['sigma_curve']

# repeat 20
repeated_static_features = static_features.unsqueeze(1).repeat(1, 20, 1)

static_mean = repeated_static_features.mean(dim=0)
static_std = repeated_static_features.std(dim=0)

norm_static_features = (repeated_static_features - static_mean) / static_std

stress_mean = stress_features.mean(dim=0)
stress_std = stress_features.std(dim=0)
norm_stress_features = (stress_features - stress_mean) / stress_std

def inverse_normalize_stress(data, mean=stress_mean, std=stress_std):
    return data * std + mean

def inverse_normalize_static(data, mean=static_mean, std=static_std):
    return data * std + mean

class FlexuralDataset(Dataset):
    def __init__(self, static_features, stress_sequences):
        """
        Args:
            cond_features: [num_samples, static_input_size]
            stress_sequences: [num_samples, seq_len, 1]
        """
        self.static = torch.FloatTensor(static_features)
        self.stress = torch.FloatTensor(stress_sequences)

    def __len__(self):
        return len(self.static)

    def __getitem__(self, idx):
        return {
            "static": self.static[idx],
            "stress": self.stress[idx]
        }



norm_static_features = torch.tensor(norm_static_features, dtype=torch.float32)

norm_stress_features = torch.tensor(norm_stress_features.unsqueeze(2), dtype=torch.float32)

print(f"  Normalized Condition Features batch shape: {norm_static_features.shape}")
print(f"  Normalized Stress Features batch shape: {norm_stress_features.shape}")



dataset = FlexuralDataset(norm_static_features, norm_stress_features)

class ConditionalAE(nn.Module):
    def __init__(self, seq_len=8, z_dim=16, hidden_dim=64, process_dim=2, stress_dim=1):
        super().__init__()
        self.seq_len = seq_len
        self.process_dim = process_dim
        self.stress_dim = stress_dim
        assert z_dim % 2 == 0, "z_dim must be even"
        self.z_dim_data = z_dim // 2
        self.z_dim_cond = z_dim - self.z_dim_data

        enc_input_dim = process_dim + stress_dim
        self.enc_rnn = nn.LSTM(enc_input_dim, hidden_size=hidden_dim, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.fc_mu_data = nn.Linear(2*hidden_dim, self.z_dim_data)
        self.cond_aggregator = nn.Sequential(
            nn.Linear(seq_len * process_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.z_dim_cond)
        )

        self.film_gen = nn.Linear(self.z_dim_cond, 2 * hidden_dim)
        dec_input_dim = z_dim + process_dim
        self.dec_cell = nn.LSTMCell(input_size=dec_input_dim, hidden_size=hidden_dim)
        self.out_lin = nn.Linear(hidden_dim, stress_dim)

        self.c_predictor = nn.Sequential(
            nn.Linear(seq_len * stress_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * process_dim)
        )

    def encode(self, x, c):
        inp = torch.cat([x, c], dim=-1)
        out, _ = self.enc_rnn(inp)
        h_last = out[:, -1, :]
        z_data = self.fc_mu_data(h_last)
        c_flat = c.view(c.size(0), -1)
        z_cond = self.cond_aggregator(c_flat)
        z = torch.cat([z_data, z_cond], dim=1)
        return z, z_data, z_cond

    def decode(self, z, c_new):
        z_data, z_cond = torch.split(z, [self.z_dim_data, self.z_dim_cond], dim=1)
        B, T, _ = c_new.shape
        film_params = self.film_gen(z_cond)
        gamma, beta = torch.split(film_params, film_params.shape[1]//2, dim=1)

        h = torch.zeros(B, self.dec_cell.hidden_size, device=z.device)
        c_state = torch.zeros_like(h)
        outputs = []

        for t in range(T):
            ct = c_new[:, t, :]
            inp = torch.cat([z, ct], dim=1)
            h, c_state = self.dec_cell(inp, (h, c_state))
            h = gamma * h + beta
            y_t = self.out_lin(h)

            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

    def forward(self, x, c, c_new):
        z, z_data, z_cond = self.encode(x, c)
        rec = self.decode(z, c)
        gen = self.decode(z, c_new)
        gen_flat = gen.view(gen.size(0), -1)
        c_recon = self.c_predictor(gen_flat).view(-1, self.seq_len, self.process_dim)
        return {'rec': rec, 'gen': gen, 'z_data': z_data, 'z_cond': z_cond, 'c_recon': c_recon}

# ------------------------------
# Train
# ------------------------------
def train_epoch(model, loader, optimizer, device, tf_ratio, cond_weight=1.0):
    model.train()
    total_rec, total_cond, total_b = 0.0, 0.0, 0
    for batch in loader:
        x = batch['stress'].to(device)
        c = batch['static'].to(device)  # norm_static_features [B,T,process_dim]
        optimizer.zero_grad()
        out = model(x, c, c_new=c)
        loss_rec  = F.mse_loss(out['rec'], x)
        loss_cond = F.mse_loss(out['c_recon'], c)
        loss = loss_rec + cond_weight * loss_cond
        loss.backward()
        optimizer.step()

        bsz = x.size(0)
        total_rec  += loss_rec.item()  * bsz
        total_cond += loss_cond.item() * bsz
        total_b    += bsz


    t_rec  = torch.tensor([total_rec], device=device)
    t_cond = torch.tensor([total_cond], device=device)
    t_b    = torch.tensor([total_b], device=device)
    reduce_sum(t_rec); reduce_sum(t_cond); reduce_sum(t_b)
    return {'rec': (t_rec / t_b).item(), 'cond': (t_cond / t_b).item()}

def val_epoch(model, loader, device):
    model.eval()
    total_sum, total_count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch['stress'].to(device)
            c = batch['static'].to(device)
            out = model(x, c, c_new=c)

            total_sum += F.mse_loss(out['rec'], x, reduction='sum').item()
            total_count += x.numel()
    t_sum   = torch.tensor([total_sum], device=device)
    t_count = torch.tensor([total_count], device=device, dtype=torch.float32)
    reduce_sum(t_sum); reduce_sum(t_count)
    return (t_sum / t_count).item()

# ------------------------------
# main workflow（DDP）
# ------------------------------
def main():
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    config = {
        "batch_size": 128,
        "epochs": 1000,
        "lr": 1e-3,
        "z_dim": 16,
        "hidden_dim": 128,
        "seq_len": 20,
        "process_dim": 2,
        "min_tf": 0.1,
        "patience": 50,
        "seed": seed,
        "ckpt_path": "pre-train_deformPTM.pth",
        "cond_weight": 1.0,
        "num_workers": 4,
        "pin_memory": True,
    }

    total_len = len(dataset)
    split = int(0.9 * total_len)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split, total_len - split],
                                                            generator=torch.Generator().manual_seed(seed))
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=False,
        persistent_workers=(config['num_workers'] > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=False,
        persistent_workers=(config['num_workers'] > 0),
    )

    model = ConditionalAE(
        seq_len=config['seq_len'],
        z_dim=config['z_dim'],
        hidden_dim=config['hidden_dim'],
        process_dim=config['process_dim'],
        stress_dim=1
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-5)

    best_val = float('inf')
    no_imp = 0
    total_steps = len(train_loader) * config['epochs']
    step = 0

    for epoch in range(1, config['epochs'] + 1):
        train_sampler.set_epoch(epoch)

        tf_ratio = max(config['min_tf'], 1 - step / max(total_steps, 1))
        train_metrics = train_epoch(model, train_loader, optimizer, device, tf_ratio,
                                    cond_weight=config.get('cond_weight', 1.0))
        val_rec = val_epoch(model, val_loader, device)
        scheduler.step()

        if is_main_process():
            print(f"[{epoch}/{config['epochs']}] rec={train_metrics['rec']:.6f} "
                  f"cond={train_metrics['cond']:.6f} | val_rec={val_rec:.6f}")

        improved = val_rec < best_val
        flag = torch.tensor([1 if improved else 0], device=device)
        reduce_sum(flag) 
        improved_global = flag.item() > 0

        if improved_global:
            best_val = val_rec
            no_imp = 0
            if is_main_process():
                torch.save(model.module.state_dict(), config['ckpt_path'])
        else:
            no_imp += 1
            if no_imp >= config['patience']:
                if is_main_process():
                    print(f"Early stopping at epoch {epoch}")
                break

        step += len(train_loader)

    cleanup_distributed()

if __name__ == '__main__':
    main()



