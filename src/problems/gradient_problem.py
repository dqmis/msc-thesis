import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def cvar_util(
    rel_matrix: torch.Tensor,
    allocations: torch.Tensor,
    group_assignments: torch.Tensor,
    k_rec: int,
    rho: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Vectorized, fully-differentiable CVaR‐style objective over groups.
    """
    # 1) Greedy top‐k sum per example
    greedy_allocs = rel_matrix.topk(k_rec, dim=1).values.sum(dim=1)  # (N,)

    eps = 1e-8
    loss_per_item = 1 - allocations / (greedy_allocs + eps)  # (N,)

    # 3) Remap group IDs to 0…G−1
    unique_groups, inverse = torch.unique(group_assignments, return_inverse=True)
    G = unique_groups.numel()

    # 4) Sum losses and counts per group via scatter_add_
    device = rel_matrix.device
    dtype = loss_per_item.dtype
    sum_losses = torch.zeros(G, device=device, dtype=dtype).scatter_add_(0, inverse, loss_per_item)
    counts = torch.zeros(G, device=device, dtype=dtype).scatter_add_(
        0, inverse, torch.ones_like(loss_per_item)
    )
    norm_losses = sum_losses / (counts + eps)

    # 5) CVaR objective
    rho_clamped = rho.clamp(min=0)
    excess = torch.relu(norm_losses - rho_clamped).sum()
    cvar_obj = rho_clamped + excess / ((1 - alpha) * G)

    return cvar_obj


class TwoLayerSelector(nn.Module):
    def __init__(self, n_users, hidden_dim, n_items):
        super().__init__()
        self.z1 = nn.Parameter(torch.empty(n_users, hidden_dim))
        self.h1 = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.z2 = nn.Parameter(torch.empty(hidden_dim, n_items))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_items))

        # rho parameter for CVaR
        self.rho = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.z1)
        nn.init.xavier_uniform_(self.h1)
        nn.init.xavier_uniform_(self.z2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        nn.init.zeros_(self.rho)

    @property
    def grad_norm(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    def forward(self):
        x = torch.relu(self.z1 @ self.h1 + self.b1)
        logits = x @ self.z2 + self.b2
        return logits


def _log_losses(loos, l_util, l_card, l_prod, l_bin, rho, tau, grad_norm, epoch) -> None:
    print(
        f"Epoch {epoch:5d} — "
        f"loss: {loos.item():.4f}, "
        f"util: {l_util.item():.2f}, "
        f"card: {l_card.item():.4f}, "
        f"prod: {l_prod.item():.4f}, "
        f"bin: {l_bin.item():.4f}, "
        f"rho: {rho.item():.4f}, "
        f"tau: {tau:.4f}, "
        f"grad: {grad_norm:.4f}"
    )


def compute_consumer_optimal_solution_cvar_grad(
    rel_matrix: np.ndarray,
    k_rec: int,
    producer_max_min_utility: float,
    gamma: float,
    group_assignments: np.ndarray,
    alpha: float,
    hidden_dim: int = 200,
    max_epochs: int = 50000,
    verbose: bool = False,
    max_patience: int = 5,
    patience_delta: float = 1e-4,
) -> tuple[np.ndarray, list[tuple[float, float, float, float, float, float]]]:
    _INITIAL_TAU = 1.0
    _FINAL_TAU = 0.001
    _ANNEAL_RATE = 0.995
    _MAX_NORM = 5

    _rel_matrix = torch.tensor(rel_matrix)
    n, m = _rel_matrix.shape
    model = TwoLayerSelector(n, hidden_dim, m)

    _producer_max_min_utility = torch.ones(m) * math.ceil(producer_max_min_utility * gamma)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100, factor=0.5)

    losses = []
    tau = 1
    best_loss = float(1e10)
    for epoch in range(1, max_epochs + 1):
        tau = max(_INITIAL_TAU * (_ANNEAL_RATE**epoch), _FINAL_TAU)

        allocations = torch.sigmoid(model() / tau)
        consumer_allocations = (allocations * _rel_matrix).sum(dim=1)

        # CVaR loss
        l_util = cvar_util(
            _rel_matrix,
            consumer_allocations,
            group_assignments=torch.tensor(group_assignments),
            k_rec=k_rec,
            rho=model.rho,
            alpha=alpha,
        )
        # k_rec constraint
        l_card = (allocations.sum(dim=1) - k_rec).pow(2).mean()
        # producer max-min utility constraint
        l_prod = torch.relu(_producer_max_min_utility - allocations.sum(dim=0)).pow(2).mean()
        # binary constraint
        l_bin = (allocations * (1 - allocations)).pow(2).mean()

        loss = l_util + l_card + l_prod + l_bin

        if epoch % 500 == 0:
            improvement = best_loss - loss
            if improvement > patience_delta:
                # real improvement: save & reset
                torch.save(model.state_dict(), "best_model.pth")
                best_loss = loss
                patience = max_patience
            else:
                # no sufficient improvement
                patience -= 1
                if patience == 0:
                    break

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), _MAX_NORM)
        opt.step()
        scheduler.step(loss)

        if (epoch % 500 == 0 or epoch == 1) and verbose:
            _log_losses(loss, l_util, l_card, l_prod, l_bin, model.rho, tau, model.grad_norm, epoch)
            losses.append((loss.item(), l_util.item(), l_card.item(), l_prod.item(), l_bin.item(), tau))

    torch.load("best_model.pth", model.state_dict())
    model_allocations = torch.sigmoid(model.eval().forward() / tau).detach().numpy()
    return model_allocations, losses


def compute_consumer_optimal_with_augmented_lagrangian_cvar(
    rel_matrix: np.ndarray,
    group_assignments: np.ndarray,
    alpha_cvar: float,
    k_rec: int,
    producer_max_min_utility: float,
    gamma: float,
    lr_primal: float = 1e-2,
    lr_dual: float = 1e-2,
    rho: float = 10.0,
    max_epochs: int = 5000,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Augmented Lagrangian primal-dual solver:
      maximize mean utility under sum_j A[i,j]=k_rec and sum_i A[i,j]>=gamma*U constraints.

    Args:
        rel_matrix: (n, m) tensor of relevances.
        allocations: (n, m) tensor for storing allocation results.
        group_assignments: (n,) tensor for consumer group assignments.
        k_rec: per-consumer recommendation count.
        producer_max_min_utility: U_max reference.
        gamma: fraction of U_max floor.
        lr_primal: LR for primal (alloc) Adam.
        lr_dual: LR for dual ascent.
        rho: penalty parameter for augmented terms.
        max_epochs: training steps.
        verbose: print diagnostics.
    Returns:
        A: (n, m) allocation matrix in [0,1].
    """

    rel_matrix = torch.tensor(rel_matrix, dtype=torch.float32)
    group_assignments = torch.tensor(group_assignments, dtype=torch.int64)

    class CVaRModule(nn.Module):
        def __init__(self, n: int, m: int):
            super().__init__()
            self.rho_cvar = nn.Parameter(torch.zeros(1))
            self.logits = nn.Parameter(torch.zeros(n, m))

    model = CVaRModule(rel_matrix.shape[0], rel_matrix.shape[1])

    n, m = rel_matrix.shape
    # primal logits
    # dual vars
    alpha = torch.zeros(n, requires_grad=False)
    beta = torch.zeros(m, requires_grad=False)
    min_prod = math.ceil(producer_max_min_utility * gamma)

    opt = optim.AdamW(model.parameters(), lr=lr_primal, weight_decay=1e-4)

    for epoch in range(1, max_epochs + 1):
        A = torch.sigmoid(model.logits / 0.1)
        consumer_allocations = (A * rel_matrix).sum(dim=1)
        # objective
        util = cvar_util(
            rel_matrix, consumer_allocations, group_assignments, k_rec, model.rho_cvar, alpha_cvar
        )
        ci = A.sum(dim=1) - k_rec
        gj = torch.relu(min_prod - A.sum(dim=0))

        # augmented Lagrangian
        loss = (
            util
            + (alpha * ci).mean()
            + (beta * gj).mean()
            + (rho / 2) * (ci.pow(2).mean())
            + (rho / 2) * (gj.pow(2).mean())
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        opt.step()

        with torch.no_grad():
            alpha += lr_dual * ci
            beta += lr_dual * gj
            beta.clamp_(min=0.0)

        if verbose and epoch % 500 == 0:
            with torch.no_grad():
                max_ci = ci.abs().max().item()
                min_prod_rec = (A.round().sum(axis=0)).min().item()
                print(
                    f"Epoch {epoch}: util={util.item():.4f}, max|ci|={max_ci:.4f}, min_prod={min_prod_rec:.4f}"
                )

    return torch.sigmoid(model.logits).detach().numpy()
