"""
Visualizer: generates matplotlib plots of experiment results.
Uses Agg backend — never blocks execution. All plots saved as PNG.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class Visualizer:
    """Generates charts from experiment result dicts. Never shows interactive windows."""

    def plot_sfri_comparison(self, results: Dict, save_path: str) -> None:
        plt = _get_plt()
        sfri_vals = [r.get("SFRI", 0) for r in results.get("runs", [])]
        if not sfri_vals:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sfri_vals, bins=20, color="steelblue", edgecolor="white")
        ax.axvline(sum(sfri_vals) / len(sfri_vals), color="red",
                   linestyle="--", label=f"Mean: {sum(sfri_vals)/len(sfri_vals):.1f}")
        ax.set_xlabel("SFRI Score")
        ax.set_ylabel("Count")
        ax.set_title("SFRI Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()

    def plot_fault_detection_rates(self, results: Dict, save_path: str) -> None:
        plt = _get_plt()
        runs = results.get("runs", [])
        if not runs:
            return
        by_method: Dict[str, List[float]] = {}
        for r in runs:
            method = r.get("detection_method", "DRL")
            by_method.setdefault(method, []).append(r.get("detection_rate", 0) * 100)
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, rates in by_method.items():
            ax.bar(method, sum(rates) / len(rates), label=method)
        ax.set_ylabel("Detection Rate (%)")
        ax.set_title("Fault Detection Rate by Method")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()

    def plot_mttd_mttr(self, results: Dict, save_path: str) -> None:
        plt = _get_plt()
        runs = results.get("runs", [])
        mttd = [r.get("MTTD", 0) for r in runs]
        mttr = [r.get("MTTR", 0) for r in runs]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(mttd, bins=15, color="orange")
        axes[0].set_title(f"MTTD (mean={sum(mttd)/max(len(mttd),1):.1f})")
        axes[0].set_xlabel("Steps")
        axes[1].hist(mttr, bins=15, color="green")
        axes[1].set_title(f"MTTR (mean={sum(mttr)/max(len(mttr),1):.1f})")
        axes[1].set_xlabel("Steps")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()

    def plot_false_positives(self, results: Dict, save_path: str) -> None:
        plt = _get_plt()
        runs = results.get("runs", [])
        fp_rates = [r.get("false_positive_rate", 0) * 100 for r in runs]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(fp_rates, bins=15, color="tomato")
        mean = sum(fp_rates) / max(len(fp_rates), 1)
        ax.axvline(mean, color="black", linestyle="--", label=f"Mean: {mean:.1f}%")
        ax.set_xlabel("False Positive Rate (%)")
        ax.set_title("False Positive Rate Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()

    def plot_reward_curve(self, episode_rewards: List[float], save_path: str) -> None:
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(episode_rewards, color="steelblue", linewidth=1.5)
        window = 10
        if len(episode_rewards) >= window:
            moving_avg = [sum(episode_rewards[i:i+window]) / window
                          for i in range(len(episode_rewards) - window + 1)]
            ax.plot(range(window-1, len(episode_rewards)), moving_avg,
                    color="red", linewidth=2, label=f"{window}-step avg")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Reward Curve")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()

    def plot_agent_comparison(self, all_agent_results: Dict, save_path: str) -> None:
        plt = _get_plt()
        metrics = ["success_rate", "SFRI", "detection_rate", "recovery_rate"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
        for i, metric in enumerate(metrics):
            for agent, results in all_agent_results.items():
                vals = [r.get(metric, 0) for r in results.get("runs", [])]
                mean = sum(vals) / max(len(vals), 1)
                axes[i].bar(agent, mean, label=agent)
            axes[i].set_title(metric)
            axes[i].set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()

    def plot_adaptation_latency(self, results: Dict, save_path: str) -> None:
        plt = _get_plt()
        runs = results.get("runs", [])
        latencies = [r.get("adaptation_latency", 0) for r in runs]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(latencies, bins=15, color="mediumpurple")
        mean = sum(latencies) / max(len(latencies), 1)
        ax.axvline(mean, color="black", linestyle="--", label=f"Mean: {mean:.1f}")
        ax.set_xlabel("Latency (steps)")
        ax.set_title("Adaptation Latency")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()

    def plot_fault_type_breakdown(self, results: Dict, save_path: str) -> None:
        plt = _get_plt()
        runs = results.get("runs", [])
        fault_counts: Dict[str, int] = {}
        for r in runs:
            for f in r.get("faults_by_type", {}).get("detected", []):
                fault_counts[f] = fault_counts.get(f, 0) + 1
        if not fault_counts:
            fault_counts = {"NO_DATA": 1}
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(list(fault_counts.keys()), list(fault_counts.values()), color="teal")
        ax.set_ylabel("Count")
        ax.set_title("Fault Type Breakdown")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()

    def generate_full_report(self, results: Dict, output_dir: str) -> str:
        """Generate all plots + summary 2×4 grid. Returns output directory path."""
        os.makedirs(output_dir, exist_ok=True)
        self.plot_sfri_comparison(results, os.path.join(output_dir, "sfri.png"))
        self.plot_fault_detection_rates(results, os.path.join(output_dir, "detection_rates.png"))
        self.plot_mttd_mttr(results, os.path.join(output_dir, "mttd_mttr.png"))
        self.plot_false_positives(results, os.path.join(output_dir, "false_positives.png"))
        self.plot_adaptation_latency(results, os.path.join(output_dir, "latency.png"))
        self.plot_fault_type_breakdown(results, os.path.join(output_dir, "fault_breakdown.png"))
        self._summary_grid(results, os.path.join(output_dir, "summary.png"))
        return output_dir

    def _summary_grid(self, results: Dict, save_path: str) -> None:
        plt = _get_plt()
        runs = results.get("runs", [])
        if not runs:
            return
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle("AETHER v3 — Experiment Summary", fontsize=14, fontweight="bold")

        def _hist(ax, vals, title, color, xlabel=""):
            ax.hist(vals, bins=15, color=color, edgecolor="white")
            mean = sum(vals) / max(len(vals), 1)
            ax.axvline(mean, color="black", linestyle="--", linewidth=1.5)
            ax.set_title(f"{title}\nμ={mean:.2f}")
            ax.set_xlabel(xlabel)

        _hist(axes[0][0], [r.get("SFRI", 0) for r in runs], "SFRI", "steelblue")
        _hist(axes[0][1], [r.get("detection_rate", 0) * 100 for r in runs], "Detection Rate %", "green")
        _hist(axes[0][2], [r.get("MTTD", 0) for r in runs], "MTTD (steps)", "orange", "steps")
        _hist(axes[0][3], [r.get("MTTR", 0) for r in runs], "MTTR (steps)", "darkorange", "steps")
        _hist(axes[1][0], [r.get("false_positive_rate", 0) * 100 for r in runs], "False Pos %", "tomato")
        _hist(axes[1][1], [r.get("adaptation_latency", 0) for r in runs], "Adapt. Latency", "mediumpurple")
        _hist(axes[1][2], [r.get("recovery_rate", 0) * 100 for r in runs], "Recovery Rate %", "teal")
        success_vals = [1 if r.get("success") else 0 for r in runs]
        axes[1][3].bar(["Success", "Failure"],
                        [sum(success_vals), len(success_vals) - sum(success_vals)],
                        color=["green", "red"])
        axes[1][3].set_title(f"Outcomes\n{sum(success_vals)/len(success_vals)*100:.1f}% success")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
