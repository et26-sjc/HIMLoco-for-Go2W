"""Run and summarize the four-point MC ContactEstimator LR ablation.

Each child process starts from MC_100Hz/model_9000.pt. The existing stability
runner owns fixed-validation collection/evaluation; this script only orchestrates
fresh runs and summarizes their TensorBoard logs.
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_LRS = (1.0e-3, 5.0e-4, 3.0e-4, 1.0e-4)


def _lr_tag(learning_rate):
    return f"{learning_rate:.0e}".replace("-", "m").replace("+", "p")


def _load_scalars(log_dir):
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: {event.step: event.value for event in accumulator.Scalars(tag)}
        for tag in accumulator.Tags()["scalars"]
    }


def _at_step(values, step, metric_name):
    if step not in values:
        raise RuntimeError(
            f"Required TensorBoard metric {metric_name} has no value at step {step}"
        )
    return values[step]


def _last_fraction_mean(values, iterations, fraction=0.20):
    start = max(0, iterations - max(1, int(math.ceil(iterations * fraction))))
    selected = [value for step, value in values.items() if start <= step < iterations]
    if not selected:
        raise RuntimeError(
            f"No TensorBoard samples found in final window [{start}, {iterations})"
        )
    return sum(selected) / len(selected)


def _summarize_run(log_dir, learning_rate, iterations):
    scalars = _load_scalars(log_dir)

    def initial(tag):
        return _at_step(scalars.get(tag, {}), 0, tag)

    def final(tag):
        return _at_step(scalars.get(tag, {}), iterations, tag)

    def online(tag):
        return _last_fraction_mean(scalars.get(tag, {}), iterations)

    compression_impact = online("Admittance/compression_after_impact_mm")
    compression_noimpact = online("Admittance/compression_after_noimpact_mm")
    initial_f1 = initial("Validation/raw_f1")
    final_f1 = final("Validation/raw_f1")
    initial_pr_auc = initial("Validation/PR_AUC")
    final_pr_auc = final("Validation/PR_AUC")
    initial_roc_auc = initial("Validation/ROC_AUC")
    final_roc_auc = final("Validation/ROC_AUC")
    initial_probability_separation = initial(
        "Validation/raw_probability_separation"
    )
    final_probability_separation = final(
        "Validation/raw_probability_separation"
    )
    initial_logit_separation = initial("Validation/raw_logit_separation")
    final_logit_separation = final("Validation/raw_logit_separation")
    return {
        "lr": learning_rate,
        "log_dir": str(log_dir),
        "warmup_f1": initial_f1,
        "final_f1": final_f1,
        "f1_drop": initial_f1 - final_f1,
        "warmup_pr_auc": initial_pr_auc,
        "final_pr_auc": final_pr_auc,
        "pr_auc_drop": initial_pr_auc - final_pr_auc,
        "warmup_roc_auc": initial_roc_auc,
        "final_roc_auc": final_roc_auc,
        "roc_auc_drop": initial_roc_auc - final_roc_auc,
        "warmup_probability_separation": initial_probability_separation,
        "final_probability_separation": final_probability_separation,
        "probability_separation_drop": (
            initial_probability_separation - final_probability_separation
        ),
        "warmup_logit_separation": initial_logit_separation,
        "final_logit_separation": final_logit_separation,
        "logit_separation_drop": (
            initial_logit_separation - final_logit_separation
        ),
        "warmup_force_mae_n": initial("Validation/force_mae_n"),
        "final_force_mae_n": final("Validation/force_mae_n"),
        "online_precision": online("Estimator/impact_precision"),
        "online_recall": online("Estimator/impact_recall"),
        "online_f1": online("Estimator/impact_f1"),
        "compression_impact_mm": compression_impact,
        "compression_noimpact_mm": compression_noimpact,
        "r_comp": compression_impact / (compression_noimpact + 1.0e-8),
        "compression_p95_mm": online("Admittance/compression_p95_mm"),
        "force_peak_n": online("Impact/gt_3d_force_peak_mean_n"),
        "loading_peak_nps": online("Impact/gt_3d_loading_peak_mean_nps"),
        "base_acc_peak_mps2": online("Impact/gt_base_acc_peak_mean_mps2"),
    }


def _write_markdown(path, rows, args):
    lines = [
        "# MC ContactEstimator Learning-Rate Ablation",
        "",
        "All runs use fresh `MC_100Hz/model_9000.pt` initialization, continual "
        "Stage-1 ContactEstimator updates, and the same fixed validation set.",
        "",
        f"Configuration: `{args.num_envs}` envs, `{args.max_iterations}` "
        f"iterations, seed `{args.seed}`, warm-up `500`, validation steps "
        f"`{args.validation_steps}`.",
        "",
        "The online values are means over the final 20% of iterations; fixed "
        "warm-up values are measured at iteration 0 before any Stage-1 update; "
        "final values are measured at the requested final iteration. `Drop` is "
        "defined as `warm-up - final`, so a positive value means degradation.",
        "",
        "## Fixed-validation drift",
        "",
        "| LR | Warm-up F1 | Final F1 | F1 drop | Warm-up PR-AUC | "
        "Final PR-AUC | AUC drop | Warm-up ROC-AUC | Final ROC-AUC | "
        "ROC-AUC drop |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['lr']:.0e} | {row['warmup_f1']:.4f} | "
            f"{row['final_f1']:.4f} | {row['f1_drop']:+.4f} | "
            f"{row['warmup_pr_auc']:.4f} | {row['final_pr_auc']:.4f} | "
            f"{row['pr_auc_drop']:+.4f} | {row['warmup_roc_auc']:.4f} | "
            f"{row['final_roc_auc']:.4f} | {row['roc_auc_drop']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Separation and force drift",
            "",
            "| LR | Prob sep. warm/final/drop | Logit sep. warm/final/drop | "
            "Force MAE warm/final (N) |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['lr']:.0e} | {row['warmup_probability_separation']:.4f} / "
            f"{row['final_probability_separation']:.4f} / "
            f"{row['probability_separation_drop']:+.4f} | "
            f"{row['warmup_logit_separation']:.4f} / "
            f"{row['final_logit_separation']:.4f} / "
            f"{row['logit_separation_drop']:+.4f} | "
            f"{row['warmup_force_mae_n']:.2f} / "
            f"{row['final_force_mae_n']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Online and control metrics (final 20%)",
            "",
            "| LR | Online P | Online R | Online F1 | Comp impact (mm) | "
            "Comp no-impact (mm) | R_comp | Comp p95 (mm) | dF peak (N/s) | "
            "F peak (N) | Base acc. (m/s²) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['lr']:.0e} | {row['online_precision']:.4f} | "
            f"{row['online_recall']:.4f} | {row['online_f1']:.4f} | "
            f"{row['compression_impact_mm']:.3f} | "
            f"{row['compression_noimpact_mm']:.3f} | {row['r_comp']:.3f} | "
            f"{row['compression_p95_mm']:.3f} | "
            f"{row['loading_peak_nps']:.1f} | {row['force_peak_n']:.2f} | "
            f"{row['base_acc_peak_mps2']:.3f} |"
        )
    best_final_f1 = max(rows, key=lambda row: row["final_f1"])
    best_final_pr_auc = max(rows, key=lambda row: row["final_pr_auc"])
    best_online_f1 = max(rows, key=lambda row: row["online_f1"])
    lines.extend(
        [
            "",
            "## Drift interpretation",
            "",
            f"- Highest final fixed F1: `{best_final_f1['lr']:.0e}` "
            f"(`{best_final_f1['final_f1']:.4f}`).",
            f"- Highest final fixed PR-AUC: `{best_final_pr_auc['lr']:.0e}` "
            f"(`{best_final_pr_auc['final_pr_auc']:.4f}`).",
            f"- Highest final-20% online F1: `{best_online_f1['lr']:.0e}` "
            f"(`{best_online_f1['online_f1']:.4f}`).",
            "- Positive drop denotes forgetting on identical fixed tensors; "
            "negative drop denotes improvement after Stage-1 updates.",
            "- One config LR is deliberately used by both the 500-step warm-up "
            "and Stage 1. Consequently, within-run initial-to-final differences "
            "measure drift cleanly, but final values across LRs also contain "
            "different warm-up convergence quality. This ablation does not "
            "isolate a Stage-1-only LR schedule.",
            "- Selection for control must consider absolute impact/no-impact "
            "compression and quiet proxies; classifier drift alone is not a "
            "sufficient production criterion.",
        ]
    )
    lines.extend(["", "## Run directories", ""])
    lines.extend(f"- `{row['lr']:.0e}`: `{row['log_dir']}`" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        default="docs/MC_CONTACT_ESTIMATOR_LR_ABLATION.md",
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    stability_script = repository / "legged_gym/scripts/run_mc_contact_stability.py"
    log_root = repository / "logs/MC_ImpactClassifier_Admittance_100Hz"
    rows = []

    for learning_rate in DEFAULT_LRS:
        command = [
            sys.executable,
            str(stability_script),
            "--task=mc_learned_admittance_100hz",
            "--freeze_contact_estimator_after_warmup=0",
            f"--contact_estimator_lr={learning_rate}",
            f"--validation_steps={args.validation_steps}",
            f"--validation_interval={args.validation_interval}",
            f"--num_envs={args.num_envs}",
            f"--max_iterations={args.max_iterations}",
            f"--seed={args.seed}",
        ]
        if args.headless:
            command.append("--headless")
        print("[lr ablation]", " ".join(command), flush=True)
        if args.dry_run:
            continue

        subprocess.run(command, cwd=str(repository), check=True)
        pattern = (
            f"*contact_stability_warm_{_lr_tag(learning_rate)}_"
            f"online_{_lr_tag(learning_rate)}_continual_"
            f"seed_{args.seed}_env_{args.num_envs}_iter_{args.max_iterations}"
        )
        matches = sorted(log_root.glob(pattern), key=lambda path: path.stat().st_mtime)
        if not matches:
            raise RuntimeError(f"Cannot find completed run matching {pattern}")
        rows.append(_summarize_run(matches[-1], learning_rate, args.max_iterations))

    if args.dry_run:
        print("[lr ablation] dry run complete; no experiments were started.")
        return
    output = repository / args.output
    _write_markdown(output, rows, args)
    print(f"[lr ablation] wrote {output}")


if __name__ == "__main__":
    main()
