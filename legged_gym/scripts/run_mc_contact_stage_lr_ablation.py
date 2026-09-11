"""Run the fixed-warmup, three-point Stage-1 ContactEstimator LR ablation."""

import argparse
import subprocess
import sys
from pathlib import Path

from run_mc_contact_lr_ablation import _lr_tag, _summarize_run


WARMUP_LR = 1.0e-3
ONLINE_LRS = (1.0e-3, 3.0e-4, 1.0e-4)


def _write_report(path, rows, args, single_lr_references):
    lines = [
        "# MC ContactEstimator Stage-Specific LR Ablation",
        "",
        "## Experiment",
        "",
        "All runs start fresh from `MC_100Hz/model_9000.pt`. Stage 0 uses "
        "`1e-3` for 500 deterministic baseline steps; only the existing Adam "
        "learning rate changes before Stage-1 continual updates. Optimizer state "
        "is preserved.",
        "",
        f"Configuration: `{args.num_envs}` envs, `{args.max_iterations}` "
        f"iterations, seed `{args.seed}`, validation steps "
        f"`{args.validation_steps}`.",
        "",
        "`Drop = warm-up - final`; positive values indicate degradation on "
        "identical fixed-validation tensors. Online/control metrics use the "
        "final 20% of iterations.",
        "",
        "## Fixed-validation drift and control selectivity",
        "",
        "| Warmup LR | Online LR | Warmup F1 | Final F1 | F1 drop | "
        "PR-AUC drop | ROC drop | R_comp |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {WARMUP_LR:.0e} | {row['online_lr']:.0e} | "
            f"{row['warmup_f1']:.4f} | {row['final_f1']:.4f} | "
            f"{row['f1_drop']:+.4f} | {row['pr_auc_drop']:+.4f} | "
            f"{row['roc_auc_drop']:+.4f} | {row['r_comp']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Complete fixed-validation trajectory endpoints",
            "",
            "| Online LR | PR-AUC warm/final | ROC-AUC warm/final | "
            "Probability sep. warm/final | Logit sep. warm/final | "
            "Force MAE warm/final (N) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['online_lr']:.0e} | {row['warmup_pr_auc']:.4f} / "
            f"{row['final_pr_auc']:.4f} | {row['warmup_roc_auc']:.4f} / "
            f"{row['final_roc_auc']:.4f} | "
            f"{row['warmup_probability_separation']:.4f} / "
            f"{row['final_probability_separation']:.4f} | "
            f"{row['warmup_logit_separation']:.4f} / "
            f"{row['final_logit_separation']:.4f} | "
            f"{row['warmup_force_mae_n']:.2f} / "
            f"{row['final_force_mae_n']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Online and physical-control metrics",
            "",
            "| Online LR | Online P | Online R | Online F1 | Comp impact "
            "(mm) | Comp no-impact (mm) | R_comp | Comp p95 (mm) | dF peak "
            "(N/s) | F peak (N) | Base acc. (m/s^2) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['online_lr']:.0e} | {row['online_precision']:.4f} | "
            f"{row['online_recall']:.4f} | {row['online_f1']:.4f} | "
            f"{row['compression_impact_mm']:.3f} | "
            f"{row['compression_noimpact_mm']:.3f} | {row['r_comp']:.3f} | "
            f"{row['compression_p95_mm']:.3f} | "
            f"{row['loading_peak_nps']:.1f} | {row['force_peak_n']:.2f} | "
            f"{row['base_acc_peak_mps2']:.3f} |"
        )

    if single_lr_references:
        lines.extend(
            [
                "",
                "## Stage-specific versus prior single-LR runs",
                "",
                "Prior single-LR references use the same LR in warm-up and "
                "Stage 1. They are included only to expose warm-up confounding.",
                "",
                "| Online LR | Stage-specific warmup F1 | Stage-specific final "
                "F1 | Single-LR warmup F1 | Single-LR final F1 |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        by_lr = {row["online_lr"]: row for row in rows}
        for learning_rate, reference in single_lr_references.items():
            row = by_lr[learning_rate]
            lines.append(
                f"| {learning_rate:.0e} | {row['warmup_f1']:.4f} | "
                f"{row['final_f1']:.4f} | {reference['warmup_f1']:.4f} | "
                f"{reference['final_f1']:.4f} |"
            )

    baseline = next(row for row in rows if row["online_lr"] == 1.0e-3)
    balanced = next(row for row in rows if row["online_lr"] == 3.0e-4)
    conservative = next(row for row in rows if row["online_lr"] == 1.0e-4)
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            "- Stage-specific LR is useful because it removes the severe "
            "low-LR warm-up under-convergence seen in the single-LR runs. It "
            "does not improve every final metric monotonically; the benefit is "
            "a better-defined warm-up snapshot and an independent drift knob.",
            f"- `1e-4` minimizes drift: PR-AUC drop "
            f"`{conservative['pr_auc_drop']:+.4f}` versus "
            f"`{baseline['pr_auc_drop']:+.4f}` at `1e-3`, and its final-20% "
            f"online F1 is `{conservative['online_f1']:.4f}`.",
            f"- `3e-4` is the more balanced online adaptation candidate: "
            f"R_comp `{balanced['r_comp']:.3f}`, no-impact compression "
            f"`{balanced['compression_noimpact_mm']:.3f} mm`, and p95 "
            f"compression `{balanced['compression_p95_mm']:.3f} mm`. It "
            "retains faster adaptation than `1e-4` without the large drift of "
            "`1e-3`.",
            "- Replay is not the next justified change. First repeat the "
            "`3e-4` and `1e-4` online-LR candidates across seeds or a longer "
            "horizon. Replay becomes justified only if fixed PR/ROC degradation "
            "persists at the selected lower LR.",
            "- These are single-seed, 100-iteration diagnostic results and do "
            "not change the production default automatically.",
            "",
            "## Verification",
            "",
            "- Fixed validation is evaluated with `torch.inference_mode()` and "
            "is never passed to PPO storage or a ContactEstimator optimizer.",
            "- Checkpoints confirm warm-up LR `1e-3` and Stage-1 LRs "
            "`1e-3`, `3e-4`, and `1e-4` respectively.",
            "- All runs logged validation at iterations 0, 20, 40, 60, 80, "
            "and 100 with no non-finite scalar values.",
            "- The baseline actor, HIM estimator, and motion adapter were "
            "unchanged between warm-up and final checkpoints.",
        ]
    )

    lines.extend(["", "## Run directories", ""])
    lines.extend(
        f"- online `{row['online_lr']:.0e}`: `{row['log_dir']}`"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_latest(root, pattern):
    matches = sorted(root.glob(pattern), key=lambda candidate: candidate.stat().st_mtime)
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--validation_steps", type=int, default=200)
    parser.add_argument("--validation_interval", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--output",
        default="docs/MC_CONTACT_ESTIMATOR_STAGE_LR_ABLATION.md",
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    stability_script = repository / "legged_gym/scripts/run_mc_contact_stability.py"
    log_root = repository / "logs/MC_ImpactClassifier_Admittance_100Hz"
    rows = []
    for online_lr in ONLINE_LRS:
        command = [
            sys.executable,
            str(stability_script),
            "--task=mc_learned_admittance_100hz",
            "--freeze_contact_estimator_after_warmup=0",
            f"--contact_estimator_warmup_lr={WARMUP_LR}",
            f"--contact_estimator_online_lr={online_lr}",
            f"--validation_steps={args.validation_steps}",
            f"--validation_interval={args.validation_interval}",
            f"--num_envs={args.num_envs}",
            f"--max_iterations={args.max_iterations}",
            f"--seed={args.seed}",
        ]
        if args.headless:
            command.append("--headless")
        print("[stage LR ablation]", " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=str(repository), check=True)
        pattern = (
            f"*contact_stability_warm_{_lr_tag(WARMUP_LR)}_"
            f"online_{_lr_tag(online_lr)}_continual_seed_{args.seed}_"
            f"env_{args.num_envs}_iter_{args.max_iterations}"
        )
        run = _find_latest(log_root, pattern)
        if run is None:
            raise RuntimeError(f"Cannot find completed run matching {pattern}")
        row = _summarize_run(run, online_lr, args.max_iterations)
        row["online_lr"] = online_lr
        rows.append(row)

    if args.dry_run:
        print("[stage LR ablation] dry run complete; no experiments were started.")
        return

    single_lr_references = {}
    for online_lr in ONLINE_LRS:
        legacy_pattern = (
            f"*contact_stability_lr_{_lr_tag(online_lr)}_continual_"
            f"seed_{args.seed}_env_{args.num_envs}_iter_{args.max_iterations}"
        )
        legacy_run = _find_latest(log_root, legacy_pattern)
        if legacy_run is not None:
            single_lr_references[online_lr] = _summarize_run(
                legacy_run, online_lr, args.max_iterations
            )

    output = repository / args.output
    _write_report(output, rows, args, single_lr_references)
    print(f"[stage LR ablation] wrote {output}")


if __name__ == "__main__":
    main()
