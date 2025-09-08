# tracking/rename_frames.py
import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def rename_image_files(gt_path, image_folder_path, dry_run=False):
    """
    根据gt.txt中的真实帧号，重命名图片文件夹中的文件。
    """
    gt_path = Path(gt_path)
    image_folder_path = Path(image_folder_path)

    print(f"读取标准答案 (GT) 文件: {gt_path}")
    try:
        # 1. 读取GT文件，获取所有唯一的、排序后的真实帧号
        gt_df = pd.read_csv(gt_path, header=None, usecols=[0])
        target_frame_numbers = sorted(gt_df[0].unique())
        print(f"从GT文件中解析出 {len(target_frame_numbers)} 个目标帧号。")
    except Exception as e:
        print(f"错误：读取GT文件失败: {e}")
        return

    print(f"读取图片文件夹: {image_folder_path}")
    try:
        # 2. 获取当前文件夹里所有图片的文件名，并按字母顺序排序
        current_image_files = sorted([f for f in os.listdir(image_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"在文件夹中找到 {len(current_image_files)} 张图片。")
    except Exception as e:
        print(f"错误：读取图片文件夹失败: {e}")
        return

    # 3. 安全检查：确保GT里的帧数量和图片数量完全一致
    if len(target_frame_numbers) != len(current_image_files):
        print("\n!!! 严重错误 !!!")
        print(f"GT文件中的独立帧数量 ({len(target_frame_numbers)}) 与图片文件夹中的图片数量 ({len(current_image_files)}) 不匹配！")
        print("请检查您的数据。操作已取消。")
        return

    print("\n开始重命名操作...")
    # 4. 遍历并重命名
    for i, current_filename in enumerate(tqdm(current_image_files, desc="重命名进度")):

        # 获取当前文件对应的“新”帧号
        target_frame_num = target_frame_numbers[i]

        # 获取文件扩展名 (e.g., '.jpg')
        file_extension = Path(current_filename).suffix

        # 构建新的文件名，使用6位数字并补零 (e.g., 30 -> '000030.jpg')
        new_filename = f"{target_frame_num:06d}{file_extension}"

        old_filepath = image_folder_path / current_filename
        new_filepath = image_folder_path / new_filename

        if dry_run:
            # 模拟运行模式，只打印计划，不执行操作
            print(f"[模拟运行] 计划将 '{current_filename}' 重命名为 '{new_filename}'")
        else:
            # 正式运行模式
            try:
                if old_filepath != new_filepath:
                     os.rename(old_filepath, new_filepath)
            except Exception as e:
                print(f"\n重命名 '{current_filename}' 时出错: {e}")

    if dry_run:
        print("\n模拟运行结束。没有文件被实际修改。")
    else:
        print("\n所有文件重命名成功！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="根据GT文件中的真实帧号，重命名图片文件。")
    parser.add_argument('--gt-path', type=str, default=r"E:\Desktop\to\train\tamato-seq-01\gt\gt_correct.txt", help='原始的、帧号为1,2,3...的GT文件路径')
    parser.add_argument('--image-folder', type=str, default=r'E:\Desktop\to\train\tamato-seq-01\img1', help='包含要重命名的图片的文件夹路径 (例如 .../img1)')
    parser.add_argument('--dry-run', action='store_true', help='模拟运行，只打印将要执行的操作，不实际重命名文件。')

    args = parser.parse_args()

    rename_image_files(args.gt_path, args.image_folder, args.dry_run)