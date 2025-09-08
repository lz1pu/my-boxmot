import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_folder, target_fps):
    """
    从视频中按指定帧率提取图片帧。
    """
    video_path = Path(video_path)
    output_folder = Path(output_folder)

    # 确保输出文件夹存在
    output_folder.mkdir(parents=True, exist_ok=True)
    # 在输出文件夹内再创建一个img1文件夹，以匹配MOT数据集的结构
    img1_folder = output_folder / 'img1'
    img1_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps == 0:
        print("警告：无法读取原始视频帧率，将使用默认值 30 FPS。")
        original_fps = 30

    print(f"原始视频信息: {total_frames} 帧, {original_fps:.2f} FPS")
    print(f"目标帧率: {target_fps} FPS")

    # 计算需要跳过的帧数
    # 例如：原始30fps，目标10fps，则 skip_interval = 30/10 = 3，即每3帧取1帧。
    skip_interval = int(original_fps / target_fps)
    if skip_interval < 1:
        skip_interval = 1

    frame_idx = 0
    saved_frame_count = 0

    with tqdm(total=total_frames, desc="正在提取帧") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 判断是否是我们需要保存的帧
            if frame_idx % skip_interval == 0:
                saved_frame_count += 1
                # 文件名格式为6位数字，如 000001.jpg
                output_filename = f"{saved_frame_count:06d}.jpg"
                output_path = img1_folder / output_filename
                cv2.imwrite(str(output_path), frame)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"\n提取完成！共提取了 {saved_frame_count} 帧图片。")
    print(f"图片已保存至: {img1_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="按指定帧率从视频中提取图片。")
    parser.add_argument('--video-path', type=str, default=r"E:\Desktop\to\train\tamato-seq-01\video\IMG_2210.MOV", help='原始视频文件的路径')
    parser.add_argument('--output-folder', type=str, default=r'E:\Desktop\to\train\tamato-seq-01-30fps', help='保存提取图片的输出文件夹路径')
    parser.add_argument('--fps', type=int, default=30, help='要提取的目标帧率 (FPS)')

    args = parser.parse_args()

    extract_frames(args.video_path, args.output_folder, args.fps)