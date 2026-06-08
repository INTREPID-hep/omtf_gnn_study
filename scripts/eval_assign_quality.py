#!/usr/bin/env python
"""
Assignment quality evaluation for EdgeCompatAssign models.

Computes metrics that go beyond per-slot efficiency:
  - stub-level assignment accuracy
  - noise suppression (soft attention on noise stubs)
  - candidate purity  (of stubs assigned to slot k, fraction with track_id==k+1)
  - candidate completeness  (of true stubs for slot k, fraction hard-assigned to k)
  - soft mass on true / noise / cross-slot stubs
  - duplicate-sharing rate
  - multi-candidate assignment quality (G5/G6 windows with n_gen>=2)

Usage:
  python scripts/omtf_gmt/eval_assign_quality.py \
      --checkpoint build/omtf_gmt/checkpoints/edge_compat_assign_h64_b6a/gmt_edge_compat_assign_best.pt \
      --cache-dir  build/omtf_gmt/cache_v2_tps \
      --datasets   G1 G2 G3 G4 G5 G6 \
      --output     build/omtf_gmt/eval/assign_quality_b6a.md
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.dataset import GMTCachedDataset, collate_gmt, expand_datasets
from omtf_gmt.models.edge_compat_assign import build_edge_compat_assign

K_MAX = 3
BATCH = 512
DUP_THR = 0.20   # weight threshold for duplicate-sharing diagnostic


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class Accumulator:
    def __init__(self):
        # per-slot, over all batches
        self.n_true_stubs         = [0] * K_MAX   # total true stubs for slot k
        self.n_hard_correct       = [0] * K_MAX   # true stubs hard-assigned to correct k
        self.n_hard_assigned      = [0] * K_MAX   # all stubs hard-assigned to slot k
        self.n_hard_assigned_true = [0] * K_MAX   # stubs hard-assigned to k with correct track_id
        self.soft_mass_true       = [0.0] * K_MAX # sum of assign_weights[k, true_stubs_k]
        self.soft_mass_noise      = [0.0] * K_MAX # sum of assign_weights[k, noise_stubs]
        self.soft_mass_cross      = [0.0] * K_MAX # sum of assign_weights[k, true_stubs_k'] k'!=k
        self.soft_mass_denom      = [0] * K_MAX   # n_active_samples for slot k

        # noise stubs: max slot weight
        self.noise_max_weight_sum = 0.0
        self.n_noise_stubs        = 0

        # duplicate sharing
        self.n_shared_stubs  = 0
        self.n_valid_stubs   = 0

        # multi-candidate windows (n_gen >= 2)
        self.mc_n_true_stubs         = [0] * K_MAX
        self.mc_n_hard_correct       = [0] * K_MAX
        self.mc_n_hard_assigned_true = [0] * K_MAX
        self.mc_n_hard_assigned      = [0] * K_MAX
        self.mc_soft_mass_true       = [0.0] * K_MAX
        self.mc_soft_mass_denom      = [0] * K_MAX
        self.n_mc_windows    = 0
        self.n_total_windows = 0

    def accumulate(
        self,
        assign_weights: torch.Tensor,  # (B, K, Nmax)
        track_id: torch.Tensor,        # (B, Nmax)  int, 0..K
        valid_mask: torch.Tensor,      # (B, Nmax)  bool
        gen_pt: torch.Tensor,          # (B, K)     float
    ):
        B, K, Nmax = assign_weights.shape
        device = assign_weights.device

        # Hard assignment: each valid stub -> slot with maximum weight
        # Shape: (B, Nmax)
        hard_k = assign_weights.argmax(dim=1)

        # Determine whether each window is multi-candidate
        n_gen = (gen_pt > 0).sum(dim=1)  # (B,)
        is_mc = n_gen >= 2               # (B,) bool

        self.n_total_windows += B
        self.n_mc_windows    += int(is_mc.sum().item())

        for k in range(K):
            # Active slot: gen_pt[k] > 0
            active = gen_pt[:, k] > 0   # (B,)

            # True stubs for slot k: track_id == k+1 and valid
            true_stubs = (track_id == (k + 1)) & valid_mask   # (B, Nmax)
            noise_stubs = (track_id == 0) & valid_mask        # (B, Nmax)
            # Stubs that belong to a different real track
            other_stubs = (track_id > 0) & (track_id != (k + 1)) & valid_mask  # (B, Nmax)

            has_true = true_stubs.any(dim=1)   # (B,)
            supervise = active & has_true      # (B,)

            # ----- stub-level hard assignment accuracy -----
            # True stubs that are hard-assigned to slot k
            correctly_assigned = (hard_k == k) & true_stubs   # (B, Nmax)
            self.n_true_stubs[k]   += int(true_stubs.sum().item())
            self.n_hard_correct[k] += int(correctly_assigned.sum().item())

            # ----- candidate purity -----
            # Stubs hard-assigned to k
            assigned_to_k  = (hard_k == k) & valid_mask       # (B, Nmax)
            correct_in_k   = assigned_to_k & true_stubs        # (B, Nmax)
            self.n_hard_assigned[k]      += int(assigned_to_k.sum().item())
            self.n_hard_assigned_true[k] += int(correct_in_k.sum().item())

            # ----- soft mass distribution -----
            if supervise.any():
                aw_k = assign_weights[:, k, :]   # (B, Nmax)
                true_f  = true_stubs.float()
                noise_f = noise_stubs.float()
                other_f = other_stubs.float()

                mass_true  = (aw_k * true_f).sum(dim=1)   # (B,)
                mass_noise = (aw_k * noise_f).sum(dim=1)  # (B,)
                mass_cross = (aw_k * other_f).sum(dim=1)  # (B,)

                self.soft_mass_true[k]  += float(mass_true[supervise].sum().item())
                self.soft_mass_noise[k] += float(mass_noise[supervise].sum().item())
                self.soft_mass_cross[k] += float(mass_cross[supervise].sum().item())
                self.soft_mass_denom[k] += int(supervise.sum().item())

            # ----- multi-candidate subset -----
            sup_mc = supervise & is_mc
            if sup_mc.any():
                self.mc_n_true_stubs[k]         += int(true_stubs.sum().item())
                self.mc_n_hard_correct[k]        += int(correctly_assigned.sum().item())
                self.mc_n_hard_assigned[k]       += int(assigned_to_k.sum().item())
                self.mc_n_hard_assigned_true[k]  += int(correct_in_k.sum().item())
                aw_k2 = assign_weights[:, k, :]
                self.mc_soft_mass_true[k]  += float(
                    (aw_k2 * true_stubs.float()).sum(dim=1)[sup_mc].sum().item()
                )
                self.mc_soft_mass_denom[k] += int(sup_mc.sum().item())

        # ----- noise max-weight -----
        # For each noise stub, what is the max weight it gets across slots?
        noise_stubs_all = (track_id == 0) & valid_mask   # (B, Nmax)
        if noise_stubs_all.any():
            max_over_slots, _ = assign_weights.max(dim=1)   # (B, Nmax)
            noise_max = max_over_slots[noise_stubs_all]
            self.noise_max_weight_sum += float(noise_max.sum().item())
            self.n_noise_stubs        += int(noise_stubs_all.sum().item())

        # ----- duplicate sharing -----
        high = (assign_weights > DUP_THR) & valid_mask.unsqueeze(1)   # (B, K, Nmax)
        shared = high.sum(dim=1) > 1    # (B, Nmax)
        self.n_shared_stubs += int(shared.sum().item())
        self.n_valid_stubs  += int(valid_mask.sum().item())


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _pct(num: int, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{100.0 * num / den:.1f}%"


def _flt(num: float, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{num / den:.4f}"


def report(acc: Accumulator, datasets: list[str]) -> str:
    lines = []
    a = lines.append

    a(f"# Assignment Quality Report")
    a(f"")
    a(f"Datasets: {', '.join(datasets)}")
    a(f"Total windows: {acc.n_total_windows:,}  "
      f"(multi-candidate n_gen≥2: {acc.n_mc_windows:,})")
    a(f"")

    # ---- per-slot table ----
    a("## Per-slot assignment metrics")
    a("")
    a("| Metric | Slot 0 | Slot 1 | Slot 2 |")
    a("| --- | ---: | ---: | ---: |")

    # Stub accuracy
    acc_vals = [_pct(acc.n_hard_correct[k], acc.n_true_stubs[k]) for k in range(K_MAX)]
    a(f"| **Hard assign accuracy** (true stubs → correct slot) | {acc_vals[0]} | {acc_vals[1]} | {acc_vals[2]} |")

    # Candidate purity
    pur_vals = [_pct(acc.n_hard_assigned_true[k], acc.n_hard_assigned[k]) for k in range(K_MAX)]
    a(f"| **Candidate purity** (assigned stubs with correct track_id) | {pur_vals[0]} | {pur_vals[1]} | {pur_vals[2]} |")

    # Candidate completeness = hard accuracy (same numerator, different denom framing)
    comp_vals = [_pct(acc.n_hard_correct[k], acc.n_true_stubs[k]) for k in range(K_MAX)]
    a(f"| **Candidate completeness** (true stubs hard-assigned to correct slot) | {comp_vals[0]} | {comp_vals[1]} | {comp_vals[2]} |")

    # Soft mass on true stubs
    true_vals = [_flt(acc.soft_mass_true[k], acc.soft_mass_denom[k]) for k in range(K_MAX)]
    a(f"| **Soft mass → true stubs** (mean weight on true stubs for slot k) | {true_vals[0]} | {true_vals[1]} | {true_vals[2]} |")

    # Soft mass on noise
    noise_vals = [_flt(acc.soft_mass_noise[k], acc.soft_mass_denom[k]) for k in range(K_MAX)]
    a(f"| **Soft mass → noise stubs** | {noise_vals[0]} | {noise_vals[1]} | {noise_vals[2]} |")

    # Soft mass on cross-slot stubs
    cross_vals = [_flt(acc.soft_mass_cross[k], acc.soft_mass_denom[k]) for k in range(K_MAX)]
    a(f"| **Soft mass → cross-track stubs** | {cross_vals[0]} | {cross_vals[1]} | {cross_vals[2]} |")

    # n_true_stubs (context)
    n_true = [f"{acc.n_true_stubs[k]:,}" for k in range(K_MAX)]
    a(f"| True stubs (total) | {n_true[0]} | {n_true[1]} | {n_true[2]} |")

    a("")

    # ---- noise suppression ----
    a("## Noise stub attention")
    a("")
    a("For noise stubs (track_id=0), the maximum assignment weight across all slots:")
    if acc.n_noise_stubs > 0:
        mean_max = acc.noise_max_weight_sum / acc.n_noise_stubs
        a(f"  mean max_k(weight[k, noise_stub]) = **{mean_max:.4f}** "
          f"over {acc.n_noise_stubs:,} noise stubs")
        a(f"  (for uniform random assignment with K={K_MAX} slots: expected ≈ 0.333)")
    else:
        a("  No noise stubs found.")

    a("")

    # ---- duplicate sharing ----
    a("## Duplicate-sharing")
    a("")
    dup_pct = _pct(acc.n_shared_stubs, acc.n_valid_stubs)
    a(f"Threshold for 'high weight': >{DUP_THR:.2f}")
    a(f"Stubs with high weight from >1 slot: "
      f"**{dup_pct}** ({acc.n_shared_stubs:,} / {acc.n_valid_stubs:,} valid stubs)")

    a("")

    # ---- multi-candidate ----
    a("## Multi-candidate windows (n_gen ≥ 2)")
    a("")
    a(f"Windows with n_gen ≥ 2: {acc.n_mc_windows:,} / {acc.n_total_windows:,} "
      f"({100.0 * acc.n_mc_windows / max(1, acc.n_total_windows):.1f}%)")
    a("")
    a("| Metric | Slot 0 | Slot 1 | Slot 2 |")
    a("| --- | ---: | ---: | ---: |")

    mc_acc = [_pct(acc.mc_n_hard_correct[k], acc.mc_n_true_stubs[k]) for k in range(K_MAX)]
    a(f"| Hard assign accuracy | {mc_acc[0]} | {mc_acc[1]} | {mc_acc[2]} |")

    mc_pur = [_pct(acc.mc_n_hard_assigned_true[k], acc.mc_n_hard_assigned[k]) for k in range(K_MAX)]
    a(f"| Candidate purity | {mc_pur[0]} | {mc_pur[1]} | {mc_pur[2]} |")

    mc_soft = [_flt(acc.mc_soft_mass_true[k], acc.mc_soft_mass_denom[k]) for k in range(K_MAX)]
    a(f"| Soft mass → true stubs | {mc_soft[0]} | {mc_soft[1]} | {mc_soft[2]} |")

    a("")

    # ---- interpretation guide ----
    a("## Interpretation notes")
    a("")
    a("- **Hard assign accuracy = Candidate completeness**: true stubs that get the")
    a("  correct slot as argmax.")
    a("- **Candidate purity**: of stubs the model points slot k at, how many belong")
    a("  there. Low purity → slot is stealing stubs from noise or other tracks.")
    a("- **Soft mass → true stubs**: assignment weight placed on the correct stubs.")
    a("  Approaches 1.0 for perfect assignment.")
    a("- **Soft mass → cross-track**: weight leaked to stubs of other real tracks.")
    a("  Should be near 0 in clean single-muon windows; allowed to be higher in")
    a("  multi-candidate windows if slots are not yet well separated.")
    a("- **Noise max weight**: should be < 1/K = 0.333. Values above indicate that")
    a("  the model is attending to noise stubs rather than suppressing them.")
    a("- **Duplicate sharing**: ideally near 0. High value → diversity penalty needed.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache-dir",  required=True)
    p.add_argument("--datasets",   nargs="+", default=["G1","G2","G3","G4","G5","G6"])
    p.add_argument("--output",     default=None)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--use-assign-context", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hidden = ckpt.get("hidden", 64)
    use_ctx = args.use_assign_context

    model = build_edge_compat_assign(hidden=hidden, use_assignment_context=use_ctx)
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: EdgeCompatAssign h={hidden}  params={n_params:,}  assign_ctx={use_ctx}")
    print(f"Checkpoint: {args.checkpoint}")

    acc = Accumulator()
    physical_ds = expand_datasets(args.datasets)

    for ds_name in physical_ds:
        try:
            ds = GMTCachedDataset(args.cache_dir, ds_name)
        except (KeyError, FileNotFoundError) as e:
            print(f"  [skip] {ds_name}: {e}")
            continue

        # Use validation split (last 10%)
        n = len(ds)
        n_val = max(1, n // 10)
        val_idx = list(range(n - n_val, n))
        val_ds = torch.utils.data.Subset(ds, val_idx)

        loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            collate_fn=collate_gmt, num_workers=0)

        print(f"  {ds_name}: {len(val_ds):,} val samples", end="", flush=True)

        for batch in loader:
            stubs      = batch["stubs"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            track_id   = batch["track_id"].to(device).long()
            gen_pt     = batch["gen_pt"].to(device)

            with torch.no_grad():
                out = model(stubs, valid_mask)

            assign_weights = out["assign_weights"]   # (B, K, Nmax)

            acc.accumulate(assign_weights, track_id, valid_mask, gen_pt)

        print(" ✓")

    text = report(acc, args.datasets)
    print()
    print(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
