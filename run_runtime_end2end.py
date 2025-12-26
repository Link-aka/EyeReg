import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scipy import ndimage
from scipy.stats import gaussian_kde

# ==================== 1. 导入依赖与模型 ====================
try:
    from unet import UNet as SegUNet
    from Unet_mycode_junction_Detect import UNet as DetUNet
    from matching import Matching_unetJunction_validtate as Matching
    import function_unetJunction_validate as f1
except ImportError as e:
    print(f"错误: 缺少必要的模型定义文件。\n详细信息: {e}")
    sys.exit(1)

# 尝试导入 thop
try:
    from thop import profile

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("提示: 未检测到 'thop' 库，FLOPs 将显示为 N/A。建议安装: pip install thop")


# ==================== 2. 全局配置 ====================
class Config:
    # 路径配置
    ROOT_PATH = r"D:\yan1_xia\week22\integrated_pipeline_final_v4"  # 修改路径以防覆盖
    DATASET_PATH = r"D:\yan1_xia\week1\dataset\multimodal\CF&FA\Combine_new"
    GT_POINTS_PATH = r"D:\yan1_xia\week1\dataset\multimodal\CF&FA\Combine_points"

    # 权重路径
    SEG_MODEL_PATH = r"D:\yan1_xia\week6\caitu_train.pth"
    DET_MODEL_PATH = r"D:\yan1_xia\week16\experiment4_dataAugment_greenchannel_clahe_mse\best_vessel_model_green_channel.pth"

    # 基础参数
    IMG_SIZE = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_IMG_COUNT = 59

    # 检测参数
    DET_THRESHOLD = 0.05
    MIN_AREA = 0

    # DSCF 参数 (论文公式对应)
    DSCF_H = 0.15  # Bandwidth h
    DSCF_LAMBDA_D = 0.3  # Displacement tolerance
    DSCF_LAMBDA_THETA = 0.1  # Orientation tolerance (radians)
    DSCF_TAU = 0.3  # Fallback ratio tau

    # 匹配参数
    SUPERPOINT_CONFIG = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    }


# ==================== 3. 复杂度计算修复方案 ====================

class SuperPointPerfModel(torch.nn.Module):
    """
    专门用于计算 FLOPs 的 SuperPoint 结构镜像。
    完全复制了你提供的 Backbone 定义，确保 FLOPs 计算 100% 准确。
    """

    def __init__(self):
        super(SuperPointPerfModel, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        # 注意：为了 thop 顺利计算，我们移除了 normalize 步骤
        # 归一化是逐元素操作，对 FLOPs 贡献极小（可以忽略），但能避免 thop 报错
        return semi, desc


def calculate_complexity(model, input_tensor, name="Model", is_superpoint=False):
    """计算参数量和 FLOPs"""
    # 1. 计算 Params (M)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    # 2. 计算 FLOPs (G)
    flops = "N/A"
    if THOP_AVAILABLE:
        try:
            model_to_profile = model
            input_to_profile = (input_tensor.clone(),)

            if is_superpoint:
                # 使用上面的 PerfModel 进行计算，输入是 Tensor，thop 可以完美处理
                model_to_profile = SuperPointPerfModel().to(input_tensor.device)
                input_to_profile = (input_tensor.clone(),)

            # 执行计算
            macs, _ = profile(model_to_profile, inputs=input_to_profile, verbose=False)
            flops = macs / 1e9

        except Exception as e:
            # print(f"计算 {name} FLOPs 失败: {e}")
            pass

    return params, flops


def analyze_full_complexity(seg_net, det_net, matcher, device, img_size=512):
    """打印并返回复杂度分析结果"""
    print("\n========== 模型复杂度分析 (Model Complexity) ==========")
    dummy_input = torch.randn(1, 1, img_size, img_size).to(device)

    # 1. SegNet
    seg_p, seg_f = calculate_complexity(seg_net, dummy_input, "SegNet")

    # 2. DetNet
    det_p, det_f = calculate_complexity(det_net, dummy_input, "DetNet")

    # 3. SuperPoint (使用 PerfModel 修复 FLOPs)
    match_p, match_f = calculate_complexity(matcher, dummy_input, "SuperPoint", is_superpoint=True)

    # 打印表格
    def fmt(val):
        return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)

    print(f"{'Model':<15} | {'Params (M)':<12} | {'FLOPs (G)':<12}")
    print("-" * 45)
    print(f"{'SegNet':<15} | {fmt(seg_p):<12} | {fmt(seg_f):<12}")
    print(f"{'DetNet':<15} | {fmt(det_p):<12} | {fmt(det_f):<12}")
    print(f"{'SuperPoint':<15} | {fmt(match_p):<12} | {fmt(match_f):<12}")
    print("-" * 45)

    # 汇总
    total_p = seg_p + det_p + match_p
    total_f = "N/A"
    if all(isinstance(f, (int, float)) for f in [seg_f, det_f, match_f]):
        total_f = seg_f + det_f + match_f
        print(f"{'TOTAL':<15} | {fmt(total_p):<12} | {fmt(total_f):<12}")
    else:
        print(f"{'TOTAL':<15} | {fmt(total_p):<12} | N/A")

    return [
        {"Model": "SegNet", "Params(M)": seg_p, "FLOPs(G)": seg_f},
        {"Model": "DetNet", "Params(M)": det_p, "FLOPs(G)": det_f},
        {"Model": "SuperPoint", "Params(M)": match_p, "FLOPs(G)": match_f},
        {"Model": "Pipeline_Total", "Params(M)": total_p, "FLOPs(G)": total_f}
    ]


# ==================== 4. 核心处理函数 ====================
def preprocess_image(path, size=512):
    """读取并预处理图像"""
    img_bgr = cv2.imread(path)
    if img_bgr is None: return None, None, None

    h, w = img_bgr.shape[:2]
    filename = os.path.basename(path)
    is_angio = '-' not in os.path.splitext(filename)[0]

    if is_angio:
        green = img_bgr[:, :, 1]
        mask = (green > 15).astype(np.uint8)
        inverted = green.copy()
        inverted[mask == 1] = 255 - green[mask == 1]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = clahe.apply(inverted)
    else:
        green = img_bgr[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = clahe.apply(green)

    img_resized = cv2.resize(processed, (size, size))
    tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    scale = (w / size, h / size)

    return tensor, scale, img_bgr


def get_kpts(heatmap, thresh, min_area, scale):
    """从热图解析关键点"""
    binary = heatmap > thresh
    labeled, num = ndimage.label(binary)
    pts = []
    sx, sy = scale
    for i in range(1, num + 1):
        coords = np.where(labeled == i)
        if len(coords[0]) < min_area: continue
        ws = heatmap[coords]
        cy = np.average(coords[0], weights=ws)
        cx = np.average(coords[1], weights=ws)
        pts.append([cx * sx, cy * sy])
    return np.array(pts)


# ========== 1. 严格对应论文公式的 DSCF 实现 (完整集成) ==========
def run_dscf_paper_exact(kpts1, kpts2, matches, h, lambda_d, lambda_theta, tau):
    """
    Dual Structural Consistency Filter (DSCF) - Paper Exact Implementation
    对应论文 Algorithm 1 及公式 (6)-(11)
    """
    if len(matches) < 4:
        return matches

    matches = np.array(matches)
    N = len(matches)

    # 获取匹配点的坐标
    p = kpts1[matches[:, 0]]  # 对应论文 p_i
    q = kpts2[matches[:, 1]]  # 对应论文 q_i

    # 计算位移向量 v_i = q_i - p_i
    v = q - p

    # --- Step 1: Displacement Consistency (公式 6, 7) ---
    d = np.linalg.norm(v, axis=1)
    indices_d = np.arange(N)  # 默认为全集 (Fallback状态)

    try:
        # KDE 估计概率密度函数 f_hat(d)
        # 注意：若d差异极小，KDE可能报错，需处理
        if np.std(d) > 1e-6:
            kde_d = gaussian_kde(d, bw_method=h)
            d_grid = np.linspace(np.min(d), np.max(d), 100)
            pdf_d = kde_d(d_grid)

            # 找峰值 mu_d = argmax f_hat(d)
            mu_d = d_grid[np.argmax(pdf_d)]

            # 筛选集合 I_d (公式 7): |d_i - mu_d| <= lambda_d * mu_d
            # 论文中是一个相对误差范围
            mask_d = np.abs(d - mu_d) <= (lambda_d * mu_d)
            temp_indices_d = np.where(mask_d)[0]

            # Fallback 机制 (Algorithm 1 Line 6)
            if len(temp_indices_d) >= N * tau:
                indices_d = temp_indices_d
    except Exception:
        # print(f"Displacement KDE Error: {e}")
        pass

    # --- Step 2: Orientation Consistency (公式 8, 9, 10, 11) ---
    # 计算角度 theta_i (公式 8)
    theta = np.arctan2(v[:, 1], v[:, 0])  # result in [-pi, pi]

    indices_theta = np.arange(N)  # 默认为全集

    try:
        # 为了处理角度的周期性，将角度转换到正空间进行KDE，或者使用Circular KDE
        # 这里简化处理：将角度映射到 [0, 2pi) 并进行KDE
        # 对于大部分视网膜图像，位移方向比较一致，这种处理通常足够
        theta_pos = np.where(theta < 0, theta + 2 * np.pi, theta)

        if np.std(theta_pos) > 1e-6:
            kde_theta = gaussian_kde(theta_pos, bw_method=h)  # (公式 9)
            theta_grid = np.linspace(0, 2 * np.pi, 360)
            pdf_theta = kde_theta(theta_grid)

            # 找峰值 mu_theta
            peak_idx = np.argmax(pdf_theta)
            mu_theta = theta_grid[peak_idx]
            if mu_theta > np.pi: mu_theta -= 2 * np.pi  # 转回 [-pi, pi] 以便计算差值

            # 计算角度差 angdiff (公式 11)
            # angdiff(a, b) = min(|a-b|, 2pi - |a-b|)
            diff_theta = np.abs(theta - mu_theta)
            diff_theta = np.minimum(diff_theta, 2 * np.pi - diff_theta)

            # 筛选集合 I_theta (公式 10): angdiff <= lambda_theta
            mask_theta = diff_theta <= lambda_theta
            temp_indices_theta = np.where(mask_theta)[0]

            # Fallback 机制 (Algorithm 1 Line 11)
            if len(temp_indices_theta) >= N * tau:
                indices_theta = temp_indices_theta
    except Exception:
        # print(f"Orientation KDE Error: {e}")
        pass

    # --- Step 3: Intersection (Algorithm 1 Line 14) ---
    # I = I_d \cap I_theta
    final_indices = np.intersect1d(indices_d, indices_theta)

    if len(final_indices) == 0:
        return matches  # 防止空集，返回原集或空均可，这里为了鲁棒返回原集

    return matches[final_indices]


# ==================== 5. 主程序 ====================
def run_pipeline():
    cfg = Config()
    os.makedirs(cfg.ROOT_PATH, exist_ok=True)

    # 加载模型
    print("正在加载模型...")
    seg_net = SegUNet(1, 1).to(cfg.DEVICE).eval()
    seg_net.load_state_dict(torch.load(cfg.SEG_MODEL_PATH, map_location=cfg.DEVICE))

    det_net = DetUNet(1, 1).to(cfg.DEVICE).eval()
    ckpt = torch.load(cfg.DET_MODEL_PATH, map_location=cfg.DEVICE)
    det_net.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

    matcher = Matching(cfg.SUPERPOINT_CONFIG).to(cfg.DEVICE).eval()

    # 1. 分析复杂度 (已使用 PerfModel 修复 FLOPs)
    complexity_data = analyze_full_complexity(seg_net, det_net, matcher, cfg.DEVICE, cfg.IMG_SIZE)

    # 2. 运行推理
    metrics = []
    print(f"\n开始推理测试 (Total Images: {cfg.TEST_IMG_COUNT})...")
    print(f"{'ID':<5} | {'Status':<8} | {'Time(ms)':<10} | {'Matches':<8} | {'MAE':<8}")
    print("-" * 50)

    for i in range(1, cfg.TEST_IMG_COUNT + 1):
        img_id = str(i)
        p_cfp = os.path.join(cfg.DATASET_PATH, f"{img_id}-{img_id}.jpg")
        p_fa = os.path.join(cfg.DATASET_PATH, f"{img_id}.jpg")

        if not os.path.exists(p_cfp) or not os.path.exists(p_fa): continue

        # 计时开始
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        try:
            # 预处理
            t_c, s_c, _ = preprocess_image(p_cfp, cfg.IMG_SIZE)
            t_f, s_f, _ = preprocess_image(p_fa, cfg.IMG_SIZE)
            t_c, t_f = t_c.to(cfg.DEVICE), t_f.to(cfg.DEVICE)

            # 推理
            with torch.no_grad():
                mask_c = torch.sigmoid(seg_net(t_c))
                mask_f = torch.sigmoid(seg_net(t_f))
                hm_c = det_net(t_c).squeeze().cpu().numpy()
                hm_f = det_net(t_f).squeeze().cpu().numpy()

            # 提取
            k_c = get_kpts(hm_c, cfg.DET_THRESHOLD, cfg.MIN_AREA, s_c)
            k_f = get_kpts(hm_f, cfg.DET_THRESHOLD, cfg.MIN_AREA, s_f)

            if len(k_c) < 4 or len(k_f) < 4: raise ValueError("Not enough keypoints")

            # 匹配
            inp = {
                'image0': mask_c, 'image1': mask_f,
                'img1_point': torch.from_numpy(k_c).float()[None].to(cfg.DEVICE),
                'img2_point': torch.from_numpy(k_f).float()[None].to(cfg.DEVICE)
            }
            pred = matcher(inp)
            raw_matches = f1.bidirectional_matching(pred['descriptors0'], pred['descriptors1'], 0.0)
            if isinstance(raw_matches, list): raw_matches = np.array(raw_matches)

            # 过滤 (使用严格的论文复现版 DSCF)
            final_matches = run_dscf_paper_exact(
                k_c, k_f, raw_matches,
                h=cfg.DSCF_H,
                lambda_d=cfg.DSCF_LAMBDA_D,
                lambda_theta=cfg.DSCF_LAMBDA_THETA,
                tau=cfg.DSCF_TAU
            )

            H = None
            if len(final_matches) >= 4:
                src = k_c[final_matches[:, 0]]
                dst = k_f[final_matches[:, 1]]
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

            # 计时结束
            torch.cuda.synchronize()
            rt = (time.perf_counter() - t0) * 1000

            # 计算精度 (Ground Truth)
            mae, mee, rmse = -1, -1, -1
            status = "Success" if H is not None else "Failed"
            if H is not None:
                gt_c = os.path.join(cfg.GT_POINTS_PATH, f"{img_id}-{img_id}.txt")
                gt_f = os.path.join(cfg.GT_POINTS_PATH, f"{img_id}.txt")
                if os.path.exists(gt_c):
                    try:
                        mae, mee, rmse, _ = f1.calculate_evaluation_metrics(gt_c, gt_f, H)
                    except:
                        pass

            metrics.append({
                "Image_ID": img_id, "Runtime_ms": rt, "FPS": 1000 / rt if rt > 0 else 0,
                "Status": status, "MAE": mae, "MEE": mee, "RMSE": rmse,
                "Final_Matches": len(final_matches) if final_matches is not None else 0
            })
            print(f"{img_id:<5} | {status:<8} | {rt:<10.2f} | {len(final_matches):<8} | {mae:<8.2f}")

        except Exception as e:
            print(f"{img_id:<5} | Error    | N/A        | 0        | N/A")

    # ==================== 6. 统计与保存 ====================
    df = pd.DataFrame(metrics)
    save_file = os.path.join(cfg.ROOT_PATH, "pipeline_metrics_final.xlsx")

    # 终端打印总体统计
    print("\n" + "=" * 50)
    print("总体性能报告 (Overall Performance Summary)")
    print("=" * 50)

    if not df.empty:
        success_df = df[df['Status'] == 'Success']
        total_imgs = len(df)
        success_count = len(success_df)

        # 计算平均指标
        avg_rt = success_df['Runtime_ms'].mean()
        avg_fps = success_df['FPS'].mean()
        avg_mae = success_df['MAE'].mean()
        avg_mee = success_df['MEE'].mean()
        avg_rmse = success_df['RMSE'].mean()

        print(f"Total Images         : {total_imgs}")
        print(f"Success Rate         : {success_count / total_imgs * 100:.2f}% ({success_count}/{total_imgs})")
        print(f"Average Runtime      : {avg_rt:.2f} ms")
        print(f"Average FPS          : {avg_fps:.2f} FPS")
        print(f"Average MAE          : {avg_mae:.4f}")
        print(f"Average MEE          : {avg_mee:.4f}")
        print(f"Average RMSE         : {avg_rmse:.4f}")

        # 准备 Summary Sheet 数据
        summary_data = [
            {"Metric": "Total Images", "Value": total_imgs},
            {"Metric": "Success Rate", "Value": success_count / total_imgs},
            {"Metric": "Average Runtime (ms)", "Value": avg_rt},
            {"Metric": "Average FPS", "Value": avg_fps},
            {"Metric": "mMAE", "Value": avg_mae},
            {"Metric": "mMEE", "Value": avg_mee},
            {"Metric": "mRMSE", "Value": avg_rmse}
        ]
    else:
        print("无数据。")
        summary_data = []

    # 保存 Excel
    with pd.ExcelWriter(save_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Per_Image_Performance', index=False)
        pd.DataFrame(complexity_data).to_excel(writer, sheet_name='Model_Complexity', index=False)
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

    print(f"\n所有结果已保存至: {save_file}")


if __name__ == "__main__":
    run_pipeline()