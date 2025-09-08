import numpy as np

gt_file = r"E:\Desktop\track\boxmot-master\boxmot-master\my_eval_data\Custom\train\tamato-seq-01\gt\gt.txt"
gt_data = np.loadtxt(gt_file, delimiter=",")
gt_frames = set(gt_data[:, 0].astype(int))


tracker_file = r"E:\Desktop\track\boxmot-master\boxmot-master\runs\botsort1\botsort\tamato-seq-01.txt"
filtered_file = r"E:\Desktop\track\boxmot-master\boxmot-master\runs\botsort1\botsort\track_results\tamato-seq-01.txt"

with open(tracker_file, 'r') as f_in, open(filtered_file, 'w') as f_out:
    for line in f_in:
        frame_id = int(line.strip().split(',')[0])
        if frame_id in gt_frames:
            f_out.write(line)
