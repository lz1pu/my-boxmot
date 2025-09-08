# tracking/correct_gt.py
import argparse
import pandas as pd

def correct_gt_frames(original_gt_path, corrected_gt_path, interval):
    print(f"正在读取原始GT文件: {original_gt_path}")
    try:
        df = pd.read_csv(original_gt_path, header=None)

        # 核心逻辑：转换第一列的帧号
        # 原始帧号为 n，新的真实帧号为 1 + (n-1) * interval
        # 例如 interval=30, n=2 -> 1 + (2-1)*30 = 31 (近似为30)
        # 为了精确匹配，我们假设您的采样是均匀的
        df[0] = 1 + (df[0] - 1) * interval

        df.to_csv(corrected_gt_path, header=False, index=False)
        print(f"已将修正后的GT文件保存至: {corrected_gt_path}")

    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="修正GT文件中的稀疏采样帧号。")
    parser.add_argument('--gt-in', type=str, default=r"E:\Desktop\track\boxmot-master\boxmot-master\runs\my_tomato_tracking\img1_mot_results.txt", help='原始的、帧号为1,2,3...的GT文件路径')
    parser.add_argument('--gt-out', type=str, default=r"E:\Desktop\track\boxmot-master\boxmot-master\runs\my_tomato_tracking\filtered_results_for_eval.txt", help='修正后要保存的新GT文件路径')
    parser.add_argument('--interval', type=int, default=10, help='您采样时的帧间隔（例如每30帧采一次，就输入30）')

    args = parser.parse_args()
    correct_gt_frames(args.gt_in, args.gt_out, args.interval)