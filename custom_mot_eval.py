import sys
import os
import subprocess
import re
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# TrackEval路径
TRACKEVAL = Path('E:/Desktop/track/boxmot-master/boxmot-master/TrackEval')

def parse_mot_results(results: str) -> Dict[str, float]:
    """
    从TrackEval输出中提取HOTA, MOTA, IDF1, AssA, AssRe, IDSW, IDs 等指标。
    """
    metric_specs = {
        'HOTA':   ('HOTA:',      {'HOTA': 0, 'AssA': 2, 'AssRe': 5}),
        'MOTA':   ('CLEAR:',     {'MOTA': 0, 'IDSW': 12}),
        'IDF1':   ('Identity:',  {'IDF1': 0}),
        'IDs':    ('Count:',     {'IDs': 2}),
    }

    int_fields = {'IDSW', 'IDs'}
    metrics = {}

    for section, fields_map in metric_specs.values():
        match = re.search(fr'{re.escape(section)}.*?COMBINED\s+(.*?)\n', results, re.DOTALL)
        if match:
            fields = match.group(1).split()
            for key, idx in fields_map.items():
                if idx < len(fields):
                    value = fields[idx]
                    metrics[key] = int(value) if key in int_fields else float(value)

    return metrics

def run_trackeval(gt_folder: Path, trackers_folder: Path, seq_names: list, tracker_name: str = "my_tomato_tracking", metrics: list = ["HOTA", "CLEAR", "Identity"], num_cores: int = 4) -> str:
    """
    调用TrackEval脚本评估跟踪结果。
    """
    args = [
        sys.executable, str(TRACKEVAL / 'scripts' / 'run_mot_challenge.py'),
        "--GT_FOLDER", str(gt_folder),
        "--BENCHMARK", "",  # 自定义数据集
        "--TRACKERS_FOLDER", str(trackers_folder),
        "--TRACKERS_TO_EVAL", tracker_name,
        "--SPLIT_TO_EVAL", "val",
        "--METRICS", *metrics,
        "--USE_PARALLEL", "True",
        "--NUM_PARALLEL_CORES", str(num_cores),
        "--SKIP_SPLIT_FOL", "True",
        "--GT_LOC_FORMAT", "{gt_folder}/{seq}/gt/gt.txt",
        "--TRACKER_SUB_FOLDER", "track_results",
        "--CLASSES_TO_EVAL", "green",  # 你的类别 ID
        "--SEQ_INFO", *seq_names
    ]

    print("运行命令:", " ".join(args))

    p = subprocess.Popen(args=args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = p.communicate()

    if stderr:
        print("错误输出:\n", stderr)
    return stdout

def plot_radar_chart(metrics_dict: Dict[str, float], title: str = "MOT Metrics Radar Chart"):
    """
    绘制雷达图可视化指标。
    """
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(title)
    plt.savefig('mot_radar_chart.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="自定义MOT评估脚本")
    parser.add_argument('--gt_folder', type=str, required=True, help='GT根目录路径')
    parser.add_argument('--trackers_folder', type=str, required=True, help='跟踪结果根目录路径')
    parser.add_argument('--seq_names', type=str, nargs='+', required=True, help='序列名列表，如 tamato-seq-01')
    parser.add_argument('--tracker_name', type=str, default='my_tomato_tracking', help='跟踪器名称')
    parser.add_argument('--output_json', type=str, default='eval_results.json', help='输出JSON文件')
    args = parser.parse_args()

    if not (TRACKEVAL / 'scripts' / 'run_mot_challenge.py').exists():
        raise FileNotFoundError(f"TrackEval 脚本不存在: {TRACKEVAL / 'scripts' / 'run_mot_challenge.py'}")

    eval_output = run_trackeval(Path(args.gt_folder), Path(args.trackers_folder), args.seq_names, args.tracker_name)
    print("评估输出:\n", eval_output)

    results = parse_mot_results(eval_output)
    print("解析指标:\n", json.dumps(results, indent=4))

    with open(args.output_json, 'w') as f:
        json.dump(results, f)

    if results:
        plot_radar_chart(results)

if __name__ == "__main__":
    main()