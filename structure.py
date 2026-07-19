import heapq
import time
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from utils.autosearch import structural_searching,structural_block_searching
from utils.mask import generate_structural_mask
from binary import rotation_weight
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.fft
import matplotlib.pyplot as plt
import random

'''
Used to generate masks for minor structural 2-bit salient data and split major 1-bit normal data according to different metric.
'''

LAYERBIT = 1 
GROUP_QPARAMS = 4
PARAMS_BIT = 3
CGN = 2

def random_mask_generate(origin_matrix, bit_1, bit_2,row_group_num, col_group_num):

    device = origin_matrix.device
    m, n = origin_matrix.shape
    row_group_size = m // row_group_num
    col_group_size = n // col_group_num
    group_num = (m * n) // (row_group_size * col_group_size)  
    mask = [torch.zeros((m, n), dtype=torch.bool, device=device) for _ in range(3)]
    mask_group = [torch.zeros((row_group_num, col_group_num), dtype=torch.bool, device=device) for _ in range(3)]
    mask_group[0][:,:]=True
    mask[0][:,:]= True
    iter = int(group_num * (bit_1+bit_2))
    update_count = 0
    while update_count < iter:
    # 随机选取一个位置
        i = random.randint(0, row_group_num - 1)
        j = random.randint(0, col_group_num - 1)
        
        # 检查该位置mask0或mask1是否为true
        if mask_group[0][i, j] or  mask_group[1][i, j]:
            # 更新操作
            r_start, r_end = i * row_group_size, (i + 1) * row_group_size
            c_start, c_end = j * col_group_size, (j + 1) * col_group_size
            if mask_group[0][i, j]:
                mask_group[0][i, j] = False
                mask[0][r_start:r_end, c_start:c_end] = False
                mask_group[1][i, j] = True
                mask[1][r_start:r_end, c_start:c_end] = True
            elif mask_group[1][i, j]:
                mask_group[1][i, j] = False
                mask[1][r_start:r_end, c_start:c_end] = False
                mask_group[2][i, j] = True
                mask[2][r_start:r_end, c_start:c_end] = True
            update_count += 1
    return mask[0], mask[1], mask[2]


def fit_quantizer(origin_matrix, ratio_list, high_order):
    H, W = origin_matrix.shape
    device = origin_matrix.device
    # 检查分块对齐，若不对齐则填充（示例中简化为报错）
    
    max_val_per_row = origin_matrix.max(dim=1, keepdim=True)[0]
    min_val_per_row = origin_matrix.min(dim=1, keepdim=True)[0]

    deltas = ratio_list[:,:W]* torch.where(
        (max_val_per_row - min_val_per_row) < 1e-8,
        torch.ones_like(max_val_per_row) * 1e-8,
        max_val_per_row - min_val_per_row
    )
    shifts = -ratio_list[:,W:2*W]*min_val_per_row / deltas  # [H, 1]

    
    # 扩展到原始矩阵大小
    range_matrix = deltas  # [H, W]
    zeros_matrix = shifts  # [H, W]
    
    # 计算多比特位量化参数
    bit_range = torch.arange(high_order, device=origin_matrix.device) * LAYERBIT + 1
    bit_scales = 2 ** (bit_range) 
    scale_matrix = range_matrix.unsqueeze(0) / bit_scales.view(-1, 1, 1).to(device)
    shift_matrix = zeros_matrix.unsqueeze(0) * bit_scales.view(-1, 1, 1).to(device)

    return scale_matrix, shift_matrix

def fit_quantizer_with_angle(origin_matrix, ratio_list, angle, high_order):
    H, W = origin_matrix.shape
    device = origin_matrix.device
    deltas_list = []
    shifts_list = []
    # 检查分块对齐，若不对齐则填充（示例中简化为报错）
    for i in range(high_order):
        origin_matrix_angle_i = rotation_weight(origin_matrix,angle[0].detach())
        max_val_per_row = origin_matrix_angle_i.max(dim=1, keepdim=True)[0]
        min_val_per_row = origin_matrix_angle_i.min(dim=1, keepdim=True)[0]

        deltas_i = ratio_list[:,:W]* torch.where(
            (max_val_per_row - min_val_per_row) < 1e-8,
            torch.ones_like(max_val_per_row) * 1e-8,
            max_val_per_row - min_val_per_row
        )
        shifts_i = -ratio_list[:,W:2*W]*min_val_per_row / deltas_i  # [H, 1]

    
        deltas_list.append(deltas_i)
        shifts_list.append(shifts_i)
    
    # 将列表转换为张量，形状为[high_order, H, W]
    range_matrix = torch.stack(deltas_list, dim=0)  # [high_order, H, W]
    zeros_matrix = torch.stack(shifts_list, dim=0)  # [high_order, H, 1]
    
    # 计算多比特位量化参数
    bit_range = torch.arange(high_order, device=device) * LAYERBIT + 1
    bit_scales = 2 ** bit_range  # [high_order]
    
    # 扩展维度以便广播
    bit_scales = bit_scales.view(-1, 1, 1)  # [high_order, 1, 1]
    
    # 计算尺度矩阵和偏移矩阵
    scale_matrix = range_matrix / bit_scales  # [high_order, H, W]
    shift_matrix = zeros_matrix * bit_scales  # [high_order, H, 1] -> 广播为[high_order, H, W]
    
    return scale_matrix, shift_matrix


def structural_group_searching_v0(origin_matrix, H, ratio_list, second_order_init_ratio, exbit_ratio, 
                             row_group_num, col_group_num, high_order, angle, file_name = None):
    
    device = origin_matrix.device
    m, n = origin_matrix.shape
    row_group_size = m // row_group_num
    col_group_size = n // col_group_num
    group_num = (m * n) // (row_group_size * col_group_size)
    # 预计算常用值
    
    # 初始化：当前设计为一阶 angle，先对整个权重矩阵旋转一次再做后续量化。
    if len(angle) == 1:
        rot_angle = angle[0].detach()
        scale_matrix, shift_matrix = fit_quantizer(rotation_weight(origin_matrix, rot_angle), ratio_list, high_order)
    else:
        # 如果外部传入了多阶 angle，仍然只取第一阶 angle 作为整体旋转矩阵
        rot_angle = angle[0].detach()
        scale_matrix, shift_matrix = fit_quantizer(rotation_weight(origin_matrix, rot_angle), ratio_list, high_order)

    # 避免存储完整的coarse_quant_result，改为即时计算误差
    error = torch.zeros((high_order, row_group_num, col_group_num), device=device)

    # 一次性计算所有阶的误差
    for j in range(high_order):
        w_rotated = rotation_weight(origin_matrix, rot_angle)
        w_clamp = torch.clamp(w_rotated / scale_matrix[j].detach() + shift_matrix[j].detach(), min=1e-8, max=2**(j+1) - 1e-8) - 0.5
        w_quant = (torch.round(w_clamp) - w_clamp).detach() + w_clamp
        w_dequant = scale_matrix[j].detach() * (w_quant - shift_matrix[j].detach() + 0.5)
        quantized = rotation_weight(w_dequant, rot_angle, mode="trans")

        delta_w_grouped = (origin_matrix - quantized).view(row_group_num, row_group_size, col_group_num, col_group_size)
        # 对第(r,c)个group，pad成[rs, n]，只有c列段非零
        # E(r,c) = sum_i (ΔW_(r,c)[i,:] @ H[c*cs:(c+1)*cs, :] @ ΔW_(r,c)[i,:].T) 只取对角
        H_reshaped = H.view(col_group_num, col_group_size, col_group_num, col_group_size)
        H_blocks = H_reshaped[torch.arange(col_group_num), :, torch.arange(col_group_num), :]
        # shape: [cg, cs, cs]

        # 只取 c==c' 的对角块
        error[j] = torch.einsum(
            'rscx,cxz,rscz->rc',
            delta_w_grouped,
            H_blocks,
            delta_w_grouped
        )

    
    # 使用稀疏矩阵存储mask
    mask = [torch.zeros((m, n), dtype=torch.bool, device=device) for _ in range(high_order)]
    mask_group = torch.zeros((row_group_num, col_group_num), dtype=torch.bool, device=device)
    
    gain12 = error[0] - error[1]   # [rg, cg]
    gain23 = error[1] - error[2]   # [rg, cg]

    # 可选：强制满足单调性假设 gain12 >= gain23
    # 防止个别block违反假设时，top-k选到 2->3 但没选 1->2
    gain23 = torch.minimum(gain23, gain12)

    # 总额外升级单位预算
    second_budget = int(group_num * second_order_init_ratio)
    exbit_budget = int(group_num * exbit_ratio)
    total_budget = second_budget + exbit_budget

    # 合法范围：[0, 2 * group_num]
    total_budget = max(0, min(total_budget, 2 * group_num))

    # group 状态：
    # 0 -> 1bit
    # 1 -> 2bit
    # 2 -> 3bit
    state_group = torch.zeros((row_group_num, col_group_num), dtype=torch.long, device=device)

    if total_budget > 0:
        flat_gain12 = gain12.flatten()
        flat_gain23 = gain23.flatten()

        # 拼成长度 2N 的“升级券”
        all_gains = torch.cat([flat_gain12, flat_gain23], dim=0)  # [2N]

        # 直接取前 total_budget 大
        _, top_idx = torch.topk(all_gains, k=total_budget, largest=True, sorted=False)

        N = group_num
        sel12 = top_idx[top_idx < N]
        sel23 = top_idx[top_idx >= N] - N

        # 先标记 1->2
        state_group.view(-1)[sel12] += 1
        # 再标记 2->3
        state_group.view(-1)[sel23] += 1

        # 理论上由于 gain23 <= gain12，不会出现非法状态
        # 保险起见再修正一下：
        state_group.clamp_(min=0, max=2)

    # 向量化生成 mask
    mask = [torch.zeros((m, n), dtype=torch.bool, device=device) for _ in range(high_order)]

    mask_group_1 = (state_group == 0)
    mask_group_2 = (state_group == 1)
    mask_group_3 = (state_group == 2)

    mask[0] = mask_group_1.repeat_interleave(row_group_size, dim=0).repeat_interleave(col_group_size, dim=1)
    mask[1] = mask_group_2.repeat_interleave(row_group_size, dim=0).repeat_interleave(col_group_size, dim=1)
    mask[2] = mask_group_3.repeat_interleave(row_group_size, dim=0).repeat_interleave(col_group_size, dim=1)

    # 如果 high_order > 3，后面的 mask 维持全 False
    for j in range(3, high_order):
        mask[j].zero_()

    msg = (
        f"Mask ratios: "
        f"{mask[0].float().mean().item():.4f}, "
        f"{mask[1].float().mean().item():.4f}, "
        f"{mask[2].float().mean().item():.4f}"
    )
    # 设置第一阶mask
    
    if not file_name:
        print(f"Mask ratios: {mask[0].float().mean().item():.4f}, {mask[1].float().mean().item():.4f}, {mask[2].float().mean().item():.4f}",flush = True)
    else:
        with open(file_name, 'a') as f:
            f.write(f"Mask ratios: {mask[0].float().mean().item():.4f}, {mask[1].float().mean().item():.4f}, {mask[2].float().mean().item():.4f}\n")
            f.flush()
    return mask[0], mask[1], mask[2], scale_matrix, shift_matrix
