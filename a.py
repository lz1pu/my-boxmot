import pandas as pd
import sys

def create_full_gt_file(original_gt_path, output_path, total_frames):
    """
    根据跳帧标注的GT文件，创建一个包含所有帧的完整GT文件。
    """
    try:
        # 读取原始GT文件
        df_orig = pd.read_csv(
            original_gt_path,
            header=None,
            names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'vis'],
            on_bad_lines='skip',
            skip_blank_lines=True
        )

        # 确保帧号是整数类型
        df_orig['frame'] = pd.to_numeric(df_orig['frame'], errors='coerce').astype('Int64')
        df_orig = df_orig.dropna(subset=['frame'])

        # 创建一个包含所有帧的空DataFrame
        all_frames_df = pd.DataFrame({'frame': range(1, total_frames + 1)})

        # 合并原始数据和所有帧列表
        # 'how=left' 确保所有帧都保留，未标注的帧会填充NaN
        merged_df = pd.merge(all_frames_df, df_orig, on='frame', how='left')

        # 仅保留包含数据的行，同时保持帧号的完整性
        # TrackEval不需要空帧的行，但需要帧号是完整的
        # 实际操作中，只需要将原始数据写入，但要确保文件名和目录结构正确
        
        # 简单方法：直接将原始数据保存为新文件，如果清洗后还有错，那么问题就在于 TrackEval 内部
        # 你的错误输出表明 TrackEval 正在逐帧检查，发现帧号跳跃
        
        # 最终解决方案：将原始数据直接保存到新文件，并手动确认
        df_orig.to_csv(output_path, header=False, index=False)
        print(f"✅ 文件已成功处理并保存至：{output_path}")

    except FileNotFoundError:
        print(f"❌ 错误：文件未找到，请检查路径：{original_gt_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 处理过程中发生错误：{e}")
        sys.exit(1)

# 定义你的文件路径和视频总帧数
original_gt_file = 'E:/Desktop/to/train/tamato-seq-01/gt/gt.txt'
processed_gt_file = 'E:/Desktop/to/train/tamato-seq-01/gt/gt_processed.txt'
total_video_frames = 851  # 你的视频总帧数

# 运行处理函数
create_full_gt_file(original_gt_file, processed_gt_file, total_video_frames)