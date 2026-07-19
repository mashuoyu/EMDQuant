"""SVD-based low-bit weight quantization (低层函数)。

将权重 W [H, C] 通过 SVD 分解为 W ≈ A[H,R] · B[R,C]（秩 R），对 A、B 分块量化：
    - A [H, R]：每行的 N 列作为一个 block（沿列分组，组大小 N）
    - B [R, C]：每列的 N 行作为一个 block（沿行分组，组大小 N）

每个 block 先经一个 [N,N] 正交旋转（"旋转→量化→反旋转"），旋转不改变分布（正交），
只把量化格点转到更适合量化的方向。旋转按 block-line 共享：A 按列组共享（R/N 个 R_A），
B 按行组共享（R/N 个 R_B）。

量化步长：delta_k = |scale_factor_k| · block内range / 2^b；零点：shift_k 由
−|shift_factor_k| · block内min / delta_k 给出（min/max 每 block 动态统计）。round 走 STE，可微。
scale_factor / shift_factor 为**逐坐标**倍率，与旋转同粒度按 line 共享：
每条 block-line 一组 [N]（block 内 N 个旋转坐标各一个），总形状 [R/N, N]——
A 按列组共享（全部行共用），B 按行组共享（全部列共用）。
每条 line 的量化格基 = 旋转 R（方向）× 逐坐标倍率 D（各轴步长），即 R·D。
这些低层函数被 quant.quant_layer 里的 SVD 分支调用，参数（free_line / scale_factor /
shift_factor）注册在 QuantLlamaDecoderLayer 上，通过原有的 layer_optim.optimize_layer
（整层输出误差）训练。比特分配用 greedy_bit_allocation（贪心 + 全局 top-k）。
"""

import math
import torch


# ---------------------------------------------------------------------------
# SVD 分解
# ---------------------------------------------------------------------------
def svd_decompose(weight: torch.Tensor, rank: int):
    """W ≈ A @ B，A=[H, r]，B=[r, C]，奇异值平方根均分给 A、B。"""
    W = weight.float()
    H, C = W.shape
    r = int(min(rank, H, C))
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U = U[:, :r]
    S = S[:r]
    Vh = Vh[:r, :]
    sqrt_s = torch.sqrt(S.clamp(min=0))
    A = U * sqrt_s.unsqueeze(0)      # [H, r]
    B = sqrt_s.unsqueeze(1) * Vh     # [r, C]
    return A, B, r


def svd_decompose_whitened(weight: torch.Tensor, rank: int, Hess: torch.Tensor,
                           percdamp: float = 0.01):
    """Hessian 加权（白化）SVD：min_{rank R} tr((W−AB)·H·(W−AB)ᵀ)。

    H = L·Lᵀ（Cholesky，GPTQ 同款阻尼 + 死通道处理），令 W̃ = W·L，对 W̃ 截断 SVD
    （Eckart–Young 在白化空间最优 = 原空间 H-加权最优），再把 L⁻¹ 折回 B：
        A = U_R·Σ_R^{1/2}，B = Σ_R^{1/2}·V_Rᵀ·L⁻¹，A·B ≈ W

    最后做**逐秩能量均衡**（对角阵 D 折入两侧，W = (A·D)·(D⁻¹·B) 不变）：
        d_k = sqrt(‖row_k(B)‖ / ‖col_k(A)‖)  →  ‖col_k(A·D)‖ = ‖row_k(D⁻¹·B)‖
    动机：白化使 B 的行范数带上不均匀的 ‖v_kᵀL⁻¹‖ 因子，block 内坐标幅值悬殊会撑大
    min-max 的 range、压扁小坐标；均衡把这种不均匀在 A/B 间对半分摊。一阶敏感度之积
    （A 侧 σ_k/d_k² × B 侧 σ_k·d_k² = σ_k²）与 D 无关，D 只做平衡不改变总预算。
    附带性质：B·H·Bᵀ ≈ D⁻¹ΣD⁻¹、AᵀA = DΣD 仍为对角 → 贪心块独立假设一阶精确保持。
    """
    W = weight.float().clone()
    Ho, C = W.shape
    r = int(min(rank, Ho, C))
    Hd = Hess.float().clone()
    dead = torch.diag(Hd) == 0
    Hd[dead, dead] = 1.0
    W[:, dead] = 0
    idx = torch.arange(C, device=W.device)
    Hd[idx, idx] += percdamp * torch.mean(torch.diag(Hd))
    L = torch.linalg.cholesky(Hd)                    # H = L·Lᵀ
    Wt = W @ L                                       # 白化
    U, S, Vh = torch.linalg.svd(Wt, full_matrices=False)
    U = U[:, :r]
    S = S[:r]
    Vh = Vh[:r, :]
    sqrt_s = torch.sqrt(S.clamp(min=0))
    A = U * sqrt_s.unsqueeze(0)                      # [Ho, r]
    Bt = sqrt_s.unsqueeze(1) * Vh                    # [r, C]（白化空间）
    B = torch.linalg.solve_triangular(L, Bt, upper=False, left=False)   # Bt·L⁻¹
    # 逐秩能量均衡：D 折入两侧，乘积不变
    a_norm = A.norm(dim=0).clamp(min=1e-8)           # ‖col_k(A)‖
    b_norm = B.norm(dim=1).clamp(min=1e-8)           # ‖row_k(B)‖
    d = torch.sqrt(b_norm / a_norm)                  # [r]
    A = A * d.unsqueeze(0)
    B = B / d.unsqueeze(1)
    return A, B, r


def _pad_rank(A: torch.Tensor, B: torch.Tensor, block: int):
    """把秩填充到 block 的整数倍（补零列 / 零行，不影响 A@B）。"""
    H, r = A.shape
    if r % block != 0:
        pad = block - (r % block)
        A = torch.cat([A, torch.zeros(H, pad, device=A.device, dtype=A.dtype)], dim=1)
        B = torch.cat([B, torch.zeros(pad, B.shape[1], device=B.device, dtype=B.dtype)], dim=0)
    return A, B


# ---------------------------------------------------------------------------
# 旋转矩阵：由 free_line 参数经 matrix_exp 生成一批 [N,N] 正交矩阵
# ---------------------------------------------------------------------------
def build_rotation(free_line: torch.Tensor, N: int) -> torch.Tensor:
    """free_line: [nb, N*(N-1)//2] -> R: [nb, N, N]（批量正交矩阵）。"""
    nb = free_line.shape[0]
    device = free_line.device
    dtype = free_line.dtype
    if N <= 1:
        return torch.ones(nb, 1, 1, device=device, dtype=dtype)
    tri = torch.triu_indices(N, N, 1, device=device)
    angle = math.pi * torch.tanh(free_line)                 # 约束到 (-pi, pi)
    skew = torch.zeros(nb, N, N, device=device, dtype=dtype)
    skew[:, tri[0], tri[1]] = angle
    skew = skew - skew.transpose(-1, -2)                    # 反对称
    return torch.matrix_exp(skew)                          # [nb, N, N]


# ---------------------------------------------------------------------------
# 可微分块仿射量化：对齐原框架 gather_ratio -> fit_quantizer -> quantize_with_params
# 的两步结构（旋转在外层 blockquant 里做，数学上与原框架"quantize 内旋转"等价）。
# ---------------------------------------------------------------------------
def svd_fit_quantizer(x, scale_factor, shift_factor, bits):
    """对齐 utils.structure.fit_quantizer + gather_ratio 的组合方式。

    x: [..., N]（旋转后的 block）；scale_factor/shift_factor: 可广播到 x 形状的
    **逐坐标**倍率（当前用法：每条 block-line 一组 [N]，block 内 N 个旋转坐标各一个）；
    bits: [...]（每 block 一个）。
        deltas_k = |scale_factor_k| · where(range<1e-8, 1e-8, max−min)   # 各坐标全范围 delta
        shifts_k = −min / deltas_k              （shift_factor=None，因子取 1）
                   −|shift_factor_k|·min/deltas_k （给定时，对齐原框架）
        scale_b = deltas / 2^b；shift_b = shifts · 2^b
    range(min/max) 是 block 内的标量统计；因子取 abs 对齐 gather_ratio。
    """
    xmin = x.min(dim=-1, keepdim=True)[0]
    xmax = x.max(dim=-1, keepdim=True)[0]
    rng = xmax - xmin
    rng = torch.where(rng < 1e-8, torch.ones_like(rng) * 1e-8, rng)
    deltas = torch.abs(scale_factor) * rng                           # [..., N] 逐坐标
    if shift_factor is None:
        shifts = -xmin / deltas
    else:
        shifts = -torch.abs(shift_factor) * xmin / deltas
    bit_scales = torch.pow(2.0, bits.float()).unsqueeze(-1)          # 2^b
    return deltas / bit_scales, shifts * bit_scales                  # scale_b, shift_b


def svd_quantize_with_params(x, scale, shift, bits):
    """对齐 quant_layer.quantize_with_params 的量化公式（不含旋转）。

        w_normal = x / scale_b + shift_b
        w_clamp  = clamp(w_normal, 1e-8, 2^b − 1e-8) − 0.5
        w_quant  = round STE
        dequant  = scale_b · (w_quant − shift_b + 0.5)
    """
    order_max = torch.pow(2.0, bits.float()).unsqueeze(-1)           # 2^b
    w_normal = x / scale + shift
    w_clamp = torch.minimum(
        torch.maximum(w_normal, torch.full_like(w_normal, 1e-8)),
        order_max - 1e-8,
    ) - 0.5
    w_quant = (torch.round(w_clamp) - w_clamp).detach() + w_clamp    # STE
    return scale * (w_quant - shift + 0.5)


def _rowblock_quant(X, block, bits, R, sf, tf):
    """行块量化核心：X [rows, r]，每行的 N 列一个 block（旋转→仿射量化→反旋转）。
    bits: [rows, nb]；sf/tf: [nb, N]（按 line 共享，广播到全部行）。"""
    rows, r = X.shape
    nb = r // block
    x = X.view(rows, nb, block)                       # [rows, nb, N]
    x = torch.einsum("hnj,njk->hnk", x, R)            # 旋转块内 N 个坐标
    sfe = sf.unsqueeze(0)                             # [1, nb, N] 广播到全部行
    tfe = tf.unsqueeze(0) if tf is not None else None
    scale, shift = svd_fit_quantizer(x, sfe, tfe, bits)
    xq = svd_quantize_with_params(x, scale, shift, bits)
    xq = torch.einsum("hnk,njk->hnj", xq, R)          # 反旋转 (xq @ R^T)
    return xq.reshape(rows, r)


def svd_block_quant(X, block, bits, R, scale_factor, shift_factor=None, block_axis=1):
    """A/B 统一的分块量化入口。

    block_axis=1（A 语义）：X [H, r]，每行的 N 列一个 block，bits [H, r/N]。
    block_axis=0（B 语义）：X [r, C]，每列的 N 行一个 block，bits [r/N, C]。
        内部按 Xᵀ 走同一行块核心——B 的列块即 Bᵀ 的行块，完全同构。
        旋转以 xᵀ·R 形式作用（相当于 Rᵀ·x）；SO(N) 对转置封闭、free_line
        初始为恒等，参数化家族与显式 R·x 等价。
    scale_factor/shift_factor: [r/N, N]（按 line 共享；None 时因子取 1）。"""
    if block_axis == 0:
        Xq = _rowblock_quant(X.t().contiguous(), block, bits.t(), R, scale_factor, shift_factor)
        return Xq.t()
    return _rowblock_quant(X, block, bits, R, scale_factor, shift_factor)


# ---------------------------------------------------------------------------
# 贪心比特分配：整体目标 tr(ΔW·H·ΔWᵀ)，A 侧等效 Hessian = B·H·Bᵀ（X 经 B 投影），
# B 侧 = H_cc × AᵀA（单列扰动精确）；各 block 独立评估 + 全局 top-k。
# ---------------------------------------------------------------------------
@torch.no_grad()
def _err_at_bit(X, Q_blocks, block, b, R, sf=None, tf=None, block_axis=1):
    """X 的各 block 单独量化到 b bit 时的加权二次型误差，按行块形式返回 [rows, nb]。

    统一公式 e(row, J) = δ · Q_JJ · δᵀ（δ 为该 block 的量化误差）：
      A（block_axis=1）：δ 污染 W 的整行，Q = G = B·H·Bᵀ（X 经 B 投影的输入协方差，
        列间相关不可丢），返回 [H, nb]；
      B（block_axis=0）：δ 只污染 W 的第 c 列，Q = M = AᵀA（输出侧度量），
        按 Bᵀ 行块形式返回 [C, nb]，调用方再乘 H_cc（单列扰动下精确）并转置。
    """
    Xr = X.t().contiguous() if block_axis == 0 else X
    rows, r = Xr.shape
    nb = r // block
    if sf is None:
        sf = torch.ones(nb, block, device=X.device)
    bits = torch.full((rows, nb), b, device=X.device)
    dX = (_rowblock_quant(Xr, block, bits, R, sf, tf) - Xr).view(rows, nb, block)
    return torch.einsum("hnj,njk,hnk->hn", dX, Q_blocks, dX)


@torch.no_grad()
def greedy_bit_allocation(A, B, Hess, block, budget_ratio, min_bit, max_bit, R_A, R_B,
                          sfA=None, sfB=None, tfA=None, tfB=None):
    """统一贪心：把 A/B 所有 block 的「1bit 升级券」按整体输出 MSE 下降收益全局 top-k 分配。

    整体目标 E = tr(ΔW·Hess·ΔWᵀ)，ΔW ≈ ΔA·B + A·ΔB（忽略二阶与块间交叉，
    与原框架 structural_group_searching_v0 的独立块假设一致）。X 的影响：
        A 侧经 B 投影 → G = B·Hess·Bᵀ 的对角块（列间相关不可丢）；
        B 侧单列扰动 → H_cc（精确）× 输出侧 M = AᵀA 的对角块。
    预算 budget = round(总 block 数 * budget_ratio)（平均每 block 额外比特数）。
    sfA/sfB/tfA/tfB：当前 scale/shift 逐坐标因子 [r/block, N]（None = 全 1）。"""
    H, r = A.shape
    C = B.shape[1]
    nb = r // block
    levels = list(range(min_bit, max_bit))            # b -> b+1 升级券等级
    L = len(levels)

    n_blocks_A = H * nb
    n_blocks_B = nb * C
    total_blocks = n_blocks_A + n_blocks_B

    if L == 0 or total_blocks == 0:
        bitsA = torch.full((H, nb), min_bit, dtype=torch.long, device=A.device)
        bitsB = torch.full((nb, C), min_bit, dtype=torch.long, device=B.device)
        return bitsA, bitsB

    # A/B 各自的等效 Hessian（只取 [nb,N,N] 对角块）
    G = B @ Hess @ B.t()                              # [r, r]  A 的输入协方差（X 经 B）
    M = A.t() @ A                                     # [r, r]  B 的输出侧度量
    idx = torch.arange(nb, device=A.device)
    G_blocks = G.view(nb, block, nb, block)[idx, :, idx, :]   # [nb, N, N]
    M_blocks = M.view(nb, block, nb, block)[idx, :, idx, :]   # [nb, N, N]
    d = torch.diag(Hess).clamp(min=0)                 # [C]  B 的输入侧对角（单列精确）

    errA = {b: _err_at_bit(A, G_blocks, block, b, R_A, sfA, tfA, block_axis=1)
            for b in range(min_bit, max_bit + 1)}                        # [H, nb]
    errB = {b: (_err_at_bit(B, M_blocks, block, b, R_B, sfB, tfB, block_axis=0)
                * d.unsqueeze(1)).t().contiguous()
            for b in range(min_bit, max_bit + 1)}                        # [nb, C]

    gainsA, gainsB = [], []
    prevA = prevB = None
    for b in levels:
        gA = errA[b] - errA[b + 1]
        gB = errB[b] - errB[b + 1]
        if prevA is not None:
            gA = torch.minimum(gA, prevA)             # 单调性
            gB = torch.minimum(gB, prevB)
        prevA, prevB = gA, gB
        gainsA.append(gA)
        gainsB.append(gB)

    flat = [g.reshape(-1) for g in gainsA] + [g.reshape(-1) for g in gainsB]
    all_gains = torch.cat(flat)

    budget = int(round(total_blocks * budget_ratio))
    budget = max(0, min(budget, all_gains.numel()))

    sel = torch.zeros_like(all_gains, dtype=torch.bool)
    if budget > 0:
        _, top_idx = torch.topk(all_gains, k=budget, largest=True, sorted=False)
        sel[top_idx] = True

    selA = sel[:L * n_blocks_A].view(L, H, nb)
    selB = sel[L * n_blocks_A:].view(L, nb, C)

    bitsA = (min_bit + selA.sum(dim=0)).clamp(min_bit, max_bit).long()
    bitsB = (min_bit + selB.sum(dim=0)).clamp(min_bit, max_bit).long()
    return bitsA, bitsB
