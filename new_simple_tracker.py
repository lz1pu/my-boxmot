# import argparse
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# from tqdm import tqdm
# import os

# # 导入新版 boxmot 的跟踪器类和 YOLO
# from boxmot import StrongSort, BotSort, OcSort, DeepOcSort, ByteTrack
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors

# # 将跟踪器名称映射到对应的类
# TRACKER_MAP = {
#     'strongsort': StrongSort,
#     'botsort': BotSort,
#     'ocsort': OcSort,
#     'deepocsort': DeepOcSort,
#     'bytetrack': ByteTrack
# }

# def main(args):
#     # 1. 初始化 YOLOv8 检测器
#     print("正在加载YOLOv8模型...")
#     try:
#         yolo = YOLO(args.yolo_model)
#         print("模型的类别名称：", yolo.names)  # 打印类别名称以确认
#     except Exception as e:
#         print(f"错误: 无法加载YOLO模型，请检查路径: {args.yolo_model}")
#         print(e)
#         return

#     # 2. 初始化跟踪器
#     print(f"正在初始化跟踪器: {args.tracking_method}")
#     if args.tracking_method.lower() not in TRACKER_MAP:
#         print(f"错误: 不支持的跟踪器 '{args.tracking_method}'。可用选项: {list(TRACKER_MAP.keys())}")
#         return
        
#     tracker_class = TRACKER_MAP[args.tracking_method.lower()]
    
#     # 为不同跟踪器设置初始化参数
#     tracker_kwargs = {}
#     if args.tracking_method.lower() in ['strongsort', 'botsort', 'deepocsort']:
#         tracker_kwargs['reid_weights'] = Path(args.reid_model)
#         tracker_kwargs['device'] = args.device
#         tracker_kwargs['half'] = False
#     elif args.tracking_method.lower() == 'bytetrack':
#         # ByteTrack 参数（根据源码）
#         tracker_kwargs['min_conf'] = 0.1
#         tracker_kwargs['track_thresh'] = 0.3
#         tracker_kwargs['match_thresh'] = 0.8
#         tracker_kwargs['track_buffer'] = 30
#         tracker_kwargs['frame_rate'] = 30
#         tracker_kwargs['per_class'] = False
#     elif args.tracking_method.lower() == 'ocsort':
#         tracker_kwargs['det_thresh'] = 0.3
#         tracker_kwargs['max_age'] = 30
#         tracker_kwargs['min_hits'] = 3

#     try:
#         tracker = tracker_class(**tracker_kwargs)
#         print("跟踪器初始化成功。")
#     except Exception as e:
#         print(f"错误: 初始化跟踪器失败。")
#         print(e)
#         return

#     # 3. 处理输入源（视频文件 或 图片文件夹）
#     source_path = Path(args.source)
#     is_video_file = source_path.is_file() and source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    
#     image_files = []
#     cap = None
#     total_frames = 0
#     fps = 30  # 为图片文件夹设置一个默认帧率

#     if is_video_file:
#         cap = cv2.VideoCapture(args.source)
#         if not cap.isOpened():
#             print(f"错误: 无法打开视频文件 {args.source}")
#             return
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         video_fps = cap.get(cv2.CAP_PROP_FPS)
#         if video_fps > 0:
#             fps = video_fps
#         print(f"正在处理视频文件: {args.source}")
#     else:
#         image_files = sorted([os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
#         if not image_files:
#             print(f"错误: 在文件夹 {args.source} 中没有找到任何图片")
#             return
#         total_frames = len(image_files)
#         print(f"正在处理图片文件夹: {args.source}, 共 {total_frames} 帧")

#     # 4. 准备视频写入器和MOT结果列表
#     vid_writer = None
#     if args.save_video:
#         output_video_path = Path(args.output_dir) / f"{source_path.stem}_tracked.mp4"
#         output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
#         if is_video_file:
#             frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         else:
#             first_frame_img = cv2.imread(image_files[0])
#             frame_height, frame_width, _ = first_frame_img.shape
            
#         vid_writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#         print(f"结果视频将保存至: {output_video_path}")
        
#     mot_results = []

#     # 5. 逐帧处理
#     for frame_count in tqdm(range(1, total_frames + 1), desc="正在处理帧"):
#         if is_video_file:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#         else:
#             frame = cv2.imread(image_files[frame_count - 1])
#             if frame is None:
#                 print(f"警告: 无法读取图片文件 {image_files[frame_count - 1]}")
#                 continue

#         # 6. 运行YOLOv8检测
#         results = yolo.predict(frame, conf=args.conf, classes=args.classes, verbose=False)
        
#         # 7. 提取检测框给跟踪器
#         if results[0].boxes is not None and len(results[0].boxes) > 0:
#             detections = results[0].boxes.data.cpu().numpy()
#             print(f"帧 {frame_count}: 检测框数量 = {len(detections)}")  # 调试信息
#         else:
#             detections = np.empty((0, 6))
#             print(f"帧 {frame_count}: 检测框数量 = 0")  # 调试信息

#         # 8. 更新跟踪器
#         online_targets = tracker.update(detections, frame)
#         print(f"帧 {frame_count}: 跟踪目标数量 = {len(online_targets)}")  # 调试信息

#         annotator = Annotator(frame.copy(), line_width=2)
        
#         # 9. 处理并保存当前帧的跟踪结果（9 列格式）
#         if args.save_txt:
#             if len(online_targets) == 0:
#                 # 空帧记录，class_id 设置为 -1
#                 mot_results.append(f"{frame_count},-1,0,0,0,0,0,-1,-1\n")
#             else:
#                 for t in online_targets:
#                     # t 的格式: [x1, y1, x2, y2, id, conf, cls, ind]
#                     x1, y1, x2, y2, track_id, cls_conf, cls_id, _ = t
#                     track_id, cls_id = int(track_id), int(cls_id)
#                     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
#                     # 转换为MOT格式（9 列），使用 YOLO 的 cls_id
#                     bb_left, bb_top = x1, y1
#                     bb_width, bb_height = x2 - x1, y2 - y1
#                     mot_results.append(f"{frame_count},{track_id},{bb_left},{bb_top},{bb_width},{bb_height},{cls_conf},{cls_id},-1\n")
                
#                     # 绘制跟踪结果
#                     label = f"ID:{track_id} Class:{yolo.names[cls_id]}"
#                     annotator.box_label((x1, y1, x2, y2), label, color=colors(track_id, True))
        
#         # 10. 显示或保存处理后的帧
#         processed_frame = annotator.result()
#         if args.show:
#             cv2.imshow("BoxMOT - New Simple Tracker", processed_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         if args.save_video:
#             vid_writer.write(processed_frame)

#     # 11. 将所有MOT结果写入文件
#     if args.save_txt:
#         output_txt_path = Path(args.output_dir) / f"{source_path.stem}.txt"  # 输出 tamato-seq-01-30fps.txt
#         output_txt_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(output_txt_path, 'w') as f:
#             f.writelines(mot_results)
#         print(f"MOT轨迹已成功保存至: {output_txt_path}")
    
#     # 12. 释放资源
#     if cap:
#         cap.release()
#     if vid_writer:
#         vid_writer.release()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="一个基于新版BoxMOT的简洁多目标跟踪脚本")
#     parser.add_argument('--yolo-model', type=str, default=r'E:/Desktop/yolo/ultralytics-20240609/runs/Tomato_v7-5/90.9-deal-_dwhead-C2f_WTConv-LightweightCenterSurround/weights/yolov8-tomato.pt', help='YOLOv8模型文件的路径')
#     parser.add_argument('--source', type=str, required=True, help='输入视频文件或图片文件夹的路径')
#     parser.add_argument('--tracking-method', type=str, default='strongsort', choices=TRACKER_MAP.keys(), help='要使用的跟踪算法')
#     parser.add_argument('--reid-model', type=str, default="osnet_x0_25_msmt17.pt", help='ReID模型文件的路径或名称 (会自动下载)')
#     parser.add_argument('--conf', type=float, default=0.4, help='YOLOv8检测的置信度阈值')
#     parser.add_argument('--classes', type=int, nargs='+', help='(可选) 要跟踪的特定类别ID (e.g., --classes 0 2)')
#     parser.add_argument('--device', type=str, default='cuda:0', help='运行设备 (e.g., "cuda:0" or "cpu")')
#     parser.add_argument('--output-dir', type=str, default='runs/track_results', help='保存输出文件（视频和txt）的目录')
#     parser.add_argument('--save-video', action='store_true', help='是否将带跟踪框的视频保存下来')
#     parser.add_argument('--save-txt', action='store_true', help='是否将MOT轨迹保存为txt文件')
#     parser.add_argument('--show', action='store_true', help='是否实时显示跟踪结果窗口')
    
#     args = parser.parse_args()
#     main(args)


import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm
import os
import yaml  # 导入yaml库

# 导入所有可能用到的 boxmot 跟踪器类和 YOLO
# 请根据你的 boxmot 版本确保这些类都存在
from boxmot import (
    StrongSort, BotSort, OcSort, DeepOcSort, ByteTrack,
    BoostTrack, HybridSort
)
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# 将跟踪器名称（通常是yaml文件名的小写）映射到对应的类
# 这使得我们可以根据配置文件名动态选择要实例化的跟踪器类
TRACKER_MAP = {
    'strongsort': StrongSort,
    'botsort': BotSort,
    'ocsort': OcSort,
    'deepocsort': DeepOcSort,
    'bytetrack': ByteTrack,
    'boosttrack': BoostTrack,
    'hybridsort': HybridSort,

}

def get_available_trackers(config_dir: Path) -> list:
    """扫描配置目录，返回所有可用的跟踪器名称"""
    if not config_dir.is_dir():
        return []
    # 从yaml文件名中提取跟踪器名称 (例如: strongsort.yaml -> strongsort)
    return sorted([p.stem for p in config_dir.glob('*.yaml')])

def main(args):
    # ... (前面的代码不变) ...
    available_trackers = get_available_trackers(Path(args.tracker_config_dir))
    if not available_trackers:
        print(f"错误: 在目录 '{args.tracker_config_dir}' 中未找到任何跟踪器配置文件(.yaml)")
        return
    
    print("正在加载YOLOv8模型...")
    try:
        yolo = YOLO(args.yolo_model)
        print("模型的类别名称：", yolo.names)
    except Exception as e:
        print(f"错误: 无法加载YOLO模型，请检查路径: {args.yolo_model}")
        print(e)
        return

    # 2. 从YAML文件加载参数并初始化跟踪器
    tracker_name = args.tracking_method.lower()
    print(f"正在初始化跟踪器: {tracker_name}")
    if tracker_name not in TRACKER_MAP:
        print(f"错误: 代码中不支持的跟踪器 '{args.tracking_method}'。请检查 TRACKER_MAP 字典。")
        return

    config_file = Path(args.tracker_config_dir) / f"{tracker_name}.yaml"
    if not config_file.is_file():
        print(f"错误: 找不到配置文件: {config_file}")
        return

    # ==================== 代码修改的核心部分 START ====================
    try:
        # Step 1: 加载完整的“调优格式”YAML文件
        with open(config_file, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        # Step 2: 准备一个空的字典，用于存放最终的“运行参数”
        tracker_kwargs = {}
        
        # Step 3: 遍历原始配置，提取每一项的'default'值
        for param, config_dict in raw_config.items():
            if isinstance(config_dict, dict) and 'default' in config_dict:
                tracker_kwargs[param] = config_dict['default']
        
        print(f"成功从 {config_file} 提取了默认参数。")

        # 注意：此处已根据您的要求，移除了参数名翻译的逻辑
        
    except Exception as e:
        print(f"错误: 加载或解析YAML文件失败: {config_file}")
        print(e)
        return
    # ==================== 代码修改的核心部分 END ====================
        
    # 对于需要ReID模型的跟踪器，从命令行参数中获取并覆盖配置
    if tracker_name in ['strongsort', 'deepocsort', 'botsort', 'boosttrack', 'hybridsort']:
        tracker_kwargs['reid_weights'] = Path(args.reid_model)
        tracker_kwargs['device'] = args.device
        tracker_kwargs['half'] = False

    tracker_class = TRACKER_MAP[tracker_name]
    try:
        tracker = tracker_class(**tracker_kwargs)
        print("跟踪器初始化成功。")
    except Exception as e:
        print(f"错误: 初始化跟踪器 '{tracker_name}' 失败。请检查 {config_file} 中的参数是否与该跟踪器兼容。")
        print(e)
        return
        
    # 3. 处理输入源（视频文件 或 图片文件夹）
    source_path = Path(args.source)
    is_video_file = source_path.is_file() and source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    
    image_files = []
    cap = None
    total_frames = 0
    fps = 30  # 为图片文件夹设置一个默认帧率

    if is_video_file:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {args.source}")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps > 0:
            fps = video_fps
        print(f"正在处理视频文件: {args.source}, 共 {total_frames} 帧, FPS: {fps:.2f}")
    else:
        image_files = sorted([os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            print(f"错误: 在文件夹 {args.source} 中没有找到任何图片")
            return
        total_frames = len(image_files)
        print(f"正在处理图片文件夹: {args.source}, 共 {total_frames} 帧")

    # 4. 准备视频写入器和MOT结果列表
    vid_writer = None
    if args.save_video:
        output_video_path = Path(args.output_dir) / f"{source_path.stem}_{tracker_name}_tracked.mp4"
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        if is_video_file:
            frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            first_frame_img = cv2.imread(image_files[0])
            frame_height, frame_width, _ = first_frame_img.shape
            
        vid_writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        print(f"结果视频将保存至: {output_video_path}")
        
    mot_results = []

    # 5. 逐帧处理
    for frame_count in tqdm(range(1, total_frames + 1), desc="正在处理帧"):
        if is_video_file:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = cv2.imread(image_files[frame_count - 1])
            if frame is None:
                print(f"警告: 无法读取图片文件 {image_files[frame_count - 1]}")
                continue

        # 6. 运行YOLOv8检测
        results = yolo.predict(frame, conf=args.conf, classes=args.classes, verbose=False)
        
        # 7. 提取检测框给跟踪器
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            detections = results[0].boxes.data.cpu().numpy()
        else:
            detections = np.empty((0, 6))

        # 8. 更新跟踪器
        online_targets = tracker.update(detections, frame)

        annotator = Annotator(frame.copy(), line_width=2)
        
       # 9. 处理并保存当前帧的跟踪结果（MOT Challenge 格式）
        if len(online_targets) > 0:
            for t in online_targets:
                # t 的格式: [x1, y1, x2, y2, id, conf, cls, ind]
                x1, y1, x2, y2, track_id, cls_conf, cls_id, _ = t
                track_id, cls_id = int(track_id), int(cls_id)
                
                # 绘制跟踪结果
                label = f"ID:{track_id} {yolo.names[cls_id]}"
                annotator.box_label((x1, y1, x2, y2), label, color=colors(track_id, True))
                
                if args.save_txt:
                    # 转换为MOT Challenge格式的坐标
                    bb_left = x1
                    bb_top = y1
                    bb_width = x2 - x1
                    bb_height = y2 - y1
                    
                    # 写入标准的9列格式MOT结果
                    mot_results.append(
                        f"{frame_count},{track_id},{bb_left},{bb_top},{bb_width},{bb_height},{cls_conf:.6f},{cls_id},-1\n"
                    )
        
        # 10. 显示或保存处理后的帧
        processed_frame = annotator.result()
        if args.show:
            cv2.imshow("BoxMOT Tracker", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if args.save_video:
            vid_writer.write(processed_frame)

    # 11. 将所有MOT结果写入文件
    if args.save_txt:
        output_txt_dir = Path(args.output_dir) / tracker_name / "track_results"
        output_txt_dir.mkdir(parents=True, exist_ok=True)
        output_txt_path = output_txt_dir / f"{source_path.stem}.txt"
        with open(output_txt_path, 'w') as f:
            f.writelines(mot_results)
        print(f"MOT轨迹已成功保存至: {output_txt_path}")
    
    # 12. 释放资源
    if cap:
        cap.release()
    if vid_writer:
        vid_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # ... (命令行参数解析部分不变) ...
    temp_config_dir = Path('boxmot/configs/trackers')
    available_trackers = get_available_trackers(temp_config_dir)
    
    parser = argparse.ArgumentParser(description="一个基于新版BoxMOT和YAML配置的简洁多目标跟踪脚本")
    
    parser.add_argument('--yolo-model', type=str, default=r'E:/Desktop/yolo/ultralytics-20240609/runs/Tomato_v7-5/90.9-deal-_dwhead-C2f_WTConv-LightweightCenterSurround/weights/yolov8-tomato.pt', help='YOLOv8模型文件的路径')
    parser.add_argument('--reid-model', type=str, default="osnet_x0_25_msmt17.pt", help='ReID模型文件的路径或名称 (对于需要ReID的跟踪器)')
    parser.add_argument('--tracking-method', type=str,required= True , choices=available_trackers, help='要使用的跟踪算法')
    parser.add_argument('--tracker-config-dir', type=str, default='boxmot/configs/trackers', help='跟踪器YAML配置文件所在的目录')
    
    parser.add_argument('--source', type=str, required=True, help='输入视频文件或图片文件夹的路径')
    parser.add_argument('--conf', type=float, default=0.4, help='YOLOv8检测的置信度阈值')
    parser.add_argument('--classes', type=int, nargs='+', help='(可选) 要跟踪的特定类别ID (e.g., --classes 0 2)')
    parser.add_argument('--device', type=str, default='cuda:0', help='运行设备 (e.g., "cuda:0" or "cpu")')
    
    parser.add_argument('--output-dir', type=str, required=True, help='保存输出文件（视频和txt）的根目录')
    parser.add_argument('--save-video', action='store_true', help='是否将带跟踪框的视频保存下来')
    parser.add_argument('--save-txt', action='store_true', help='是否将MOT轨迹保存为txt文件')
    parser.add_argument('--show', action='store_true', help='是否实时显示跟踪结果窗口')
    
    args = parser.parse_args()
    
    if args.tracking_method not in available_trackers:
        print(f"错误: 跟踪器 '{args.tracking_method}' 的配置文件未在 '{args.tracker_config_dir}' 中找到。")
        print(f"可用选项: {available_trackers}")
    else:
        main(args)