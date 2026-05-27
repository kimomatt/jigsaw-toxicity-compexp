import json
from pathlib import Path

import matplotlib.pyplot as plt


SWEEP_ROOT = Path("/workspace/compexp_sweeps")
OUTPUT_DIR = SWEEP_ROOT / "plots"

# replacing p with . to convert the string tag to a float, for example "0p35" would become 0.35
def tag_to_float(tag):
    return float(tag.replace("p", "."))

# going from the directory name to the threshold, with this example it would get 0.35 from "mean_pool_docfreq_min_0p0_max_0p35"
def run_dir_to_threshold(run_dir_name):
    # example: mean_pool_docfreq_min_0p0_max_0p35
    max_tag = run_dir_name.split("_max_")[1]
    return tag_to_float(max_tag)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    # finding the directories that correspond to the docfreq interval sweeps
    for run_dir in sorted(SWEEP_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith("mean_pool_docfreq_min_"):
            continue

        json_path = run_dir / "results" / "interval_analysis.json"
        if not json_path.exists():
            continue
        
        # when it finds a directory it extracts threshold from directory name
        threshold = run_dir_to_threshold(run_dir.name)

        # loads the json file that contains the results of the interval analysis for that run, which includes the top explanations for each neuron and interval, along with their IoU and Lift scores. 
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)


        results = payload["results"]
        # loop thru every item in results, grab neuron value, puts ids into a set, sorts into a list
        neurons = sorted({result["neuron"] for result in results})

        for neuron in neurons:
            # for each neuron, it filters the results to get only the results for that neuron, then sorts those results by the low end of the interval. This way, we can analyze how the top explanations and their corresponding IoU and Lift scores change as we vary the docfreq max threshold for each interval of activations for that neuron.
            neuron_results = [result for result in results if result["neuron"] == neuron]
            neuron_results.sort(key=lambda result: result["interval"][0])

            # goes thru all the neuron results for the neuron (there should 5 bc 5 clusters)
            # finds the metrics from the top explanation for that interval and threshold, and appends a row to the rows list with the neuron, threshold, interval index, interval low and high values, and the IoU and Lift scores for the best explanation for that interval and threshold. This way we can later plot how the best IoU and Lift scores for each interval change as we vary the docfreq max threshold.
            for interval_index, result in enumerate(neuron_results):
                top_explanations = result["top_explanations"]
                if not top_explanations:
                    continue

                best_explanation = max(
                    top_explanations,
                    key=lambda explanation: explanation["iou"],
                )

                rows.append(
                    {
                        "neuron": neuron,
                        "threshold": threshold,
                        "interval_index": interval_index,
                        "interval_low": result["interval"][0],
                        "interval_high": result["interval"][1],
                        "iou": best_explanation["iou"],
                        "lift": best_explanation["lift"],
                        "formula": best_explanation["pretty_formula"],
                    }
                )

    neurons = sorted({row["neuron"] for row in rows})

    # for each neuron, it filters the rows to get only the rows for that neuron, then gets the unique interval indices for that neuron. Then, for each metric (IoU and Lift), it creates a subplot for each interval index and plots the metric values against the docfreq max thresholds for that interval. Finally, it saves the plot for each neuron and metric combination to the output directory.
    for neuron in neurons:
        neuron_rows = [row for row in rows if row["neuron"] == neuron]
        # getting interval indices for that neuron, which should be 0-4 since there are 5 clusters/intervals, but doing it this way just in case
        interval_indices = sorted({row["interval_index"] for row in neuron_rows})

        for metric in ["iou", "lift"]:
            # 2 by 3 gird of subplots, all plots share same x-axis scale
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
            # so that u can loop thru the axes in a single loop instead of nested loops
            axes = axes.flatten()

            for interval_index in interval_indices:
                ax = axes[interval_index]
                # take rows only for this one interval index, then sort by threshold so that the points in the plot are connected in order of increasing threshold
                interval_rows = [
                    row for row in neuron_rows if row["interval_index"] == interval_index
                ]
                interval_rows.sort(key=lambda row: row["threshold"])

                x = [row["threshold"] for row in interval_rows]
                y = [row[metric] for row in interval_rows]
                interval_low = interval_rows[0]["interval_low"]
                interval_high = interval_rows[0]["interval_high"]

                ax.plot(x, y, marker="o")
                ax.set_title(
                    f"Interval {interval_index + 1}\n[{interval_low:.3f}, {interval_high:.3f}]"
                )
                ax.set_xlabel("Docfreq max threshold")
                ax.set_ylabel("IoU" if metric == "iou" else "Lift")
                ax.grid(True, alpha=0.3)

            for i in range(len(interval_indices), len(axes)):
                axes[i].axis("off")

            fig.suptitle(
                f"Neuron {neuron}: {'IoU' if metric == 'iou' else 'Lift'} by interval"
            )
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f"neuron_{neuron}_{metric}_intervals.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
