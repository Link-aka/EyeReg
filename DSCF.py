import numpy as np
from scipy.stats import gaussian_kde

def dscf(kpts1, kpts2, matches, h=0.15, lambda_d=0.3, lambda_theta=0.1, tau=0.3):
    """
    Dual Structural Consistency Filter (DSCF)

    params:
        h: KDE  (bandwidth)
        lambda_d: 位移容差比例
        lambda_theta: 角度容差 (弧度 Radian)
        tau: Fallback ratio
    """
    if len(matches) < 4:
        return matches

    matches = np.array(matches)
    N = len(matches)

    # 获取匹配点的坐标
    p = kpts1[matches[:, 0]]
    q = kpts2[matches[:, 1]]

    # 计算位移向量 v_i = q_i - p_i
    v = q - p

    # --- Step 1: Displacement Consistency  ---
    # 计算位移模长 d_i (magnitude)
    d = np.linalg.norm(v, axis=1)

    indices_d = np.arange(N)  # 默认为全集 (Fallback状态)

    try:
        # KDE 估计概率密度函数 f_hat(d)
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

            # Fallback 机制
            if len(temp_indices_d) >= N * tau:
                indices_d = temp_indices_d
    except Exception as e:
        # print(f"Displacement KDE Error: {e}")
        pass

    # --- Step 2: Orientation Consistency ---
    # 计算角度 theta_i
    theta = np.arctan2(v[:, 1], v[:, 0])  # result in [-pi, pi]

    indices_theta = np.arange(N)  # 默认为全集

    try:

        # 将角度映射到 [0, 2pi) 并进行KDE
        theta_pos = np.where(theta < 0, theta + 2 * np.pi, theta)

        if np.std(theta_pos) > 1e-6:
            kde_theta = gaussian_kde(theta_pos, bw_method=h)  # (公式 9)
            theta_grid = np.linspace(0, 2 * np.pi, 360)
            pdf_theta = kde_theta(theta_grid)

            # 找峰值 mu_theta
            peak_idx = np.argmax(pdf_theta)
            mu_theta = theta_grid[peak_idx]
            if mu_theta > np.pi: mu_theta -= 2 * np.pi  # 转回 [-pi, pi] 以便计算差值

            # 计算角度差 angdiff
            # angdiff(a, b) = min(|a-b|, 2pi - |a-b|)
            diff_theta = np.abs(theta - mu_theta)
            diff_theta = np.minimum(diff_theta, 2 * np.pi - diff_theta)

            # 筛选集合 I_theta : angdiff <= lambda_theta
            mask_theta = diff_theta <= lambda_theta
            temp_indices_theta = np.where(mask_theta)[0]

            # Fallback 机制
            if len(temp_indices_theta) >= N * tau:
                indices_theta = temp_indices_theta
    except Exception as e:
        # print(f"Orientation KDE Error: {e}")
        pass

    # --- Step 3: Intersection (Algorithm 1 Line 14) ---
    # I = I_d \cap I_theta
    final_indices = np.intersect1d(indices_d, indices_theta)

    if len(final_indices) == 0:
        return matches  # 防止空集，返回原集或空均可，这里为了鲁棒返回原集

    return matches[final_indices]