"""SVD 量化版 LLaMA Decoder 层（low_quant_method='svd' 专用）。

与 quant_layer.QuantLlamaDecoderLayer 相同的骨架（直接复用其 QuantLlamaMLP /
QuantLlamaAttention），但量化逻辑全部走 SVD 路径：
    W ≈ A·B（默认 Hessian 白化 SVD + 逐秩能量均衡）
    + A/B 统一分块量化（可学习旋转 free_line、逐坐标尺度/零点 scale/shift_factor）
    + 贪心比特分配（greedy_bit_allocation）
braq 相关代码不在此文件（见 quant_layer.py，两者互不依赖 SVD 逻辑）。

与 layer_optim.optimize_layer 的接口约定：
    - rotation/scale/shift_parameters() 按参数名收集（free_line / scale_factor / shift_factor）
    - binary_temporary()      每步量化出 temp_weight=Aq@Bq（保留计算图）
    - mask_generate_save()    分派为 svd_bit_alloc（disable_gptq 下每 epoch 重分配）
    - binary_inplace()        训练后把重构权重写回 module.weight
"""

import torch
from torch import nn
from typing import Optional, Tuple

from quant.int_linear import QuantLinear
from quant.int_matmul import QuantMatMul
from quant.ptq_norm import PTQLlamaRMSNorm
from quant.quant_layer import QuantLlamaMLP, QuantLlamaAttention
from transformers.models.llama.configuration_llama import LlamaConfig
from quant.svd_quant import (
    svd_decompose,
    svd_decompose_whitened,
    _pad_rank,
    build_rotation,
    svd_block_quant,
    greedy_bit_allocation,
)


class SVDQuantLlamaDecoderLayer(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 ori_layer,
                 args,
                 device_map):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantLlamaAttention(
            org_module=ori_layer.self_attn,
            config=config,
            args=args,
        )
        self.mlp = QuantLlamaMLP(
            org_module=ori_layer.mlp,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            args=args,
        )
        self.input_layernorm = PTQLlamaRMSNorm(ori_layer.input_layernorm, eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = PTQLlamaRMSNorm(ori_layer.post_attention_layernorm, eps=ori_layer.post_attention_layernorm.variance_epsilon)
        self.device_map = device_map or {}
        # ---- SVD 量化配置 ----
        self.svd_rank = getattr(args, "svd_rank", 256)
        # k/v 对秩最敏感（K 误差经 softmax 指数放大、V 线性混合；GQA 下 KV 冗余低），
        # 且 k/v 仅占层参数 ~6%、[1024,3072] 的分解收支平衡点是 768 —— 单独给高秩几乎免费
        self.svd_kv_rank = getattr(args, "svd_kv_rank", None)
        # 深层 MLP 单独秩（异构配置：浅层 MLP 低秩省预算，深层加秩）
        self.svd_mlp_deep_rank = getattr(args, "svd_mlp_deep_rank", None)
        self.svd_mlp_deep_from = getattr(args, "svd_mlp_deep_from", 22)
        self.svd_layer_idx = None   # 由 run 循环在 svd_prepare 前设置
        self.svd_block = getattr(args, "svd_block", 16)
        self.svd_budget_ratio = getattr(args, "svd_budget_ratio", 1.0)
        self.svd_min_bit = getattr(args, "svd_min_bit", 1)
        self.svd_max_bit = getattr(args, "svd_max_bit", 3)
        self.svd_whiten = getattr(args, "svd_whiten", True)
        self.svd_percdamp = getattr(args, "percdamp", 0.01)
        # 阶段1（FP 低秩微调）开关：True 时 binary_temporary 用 A@B（不量化）
        self.svd_fp_mode = False

    # ------------------------------------------------------------------
    # 前向 / 设备管理（与 quant_layer.QuantLlamaDecoderLayer 一致）
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if 'mlp' in self.device_map and hidden_states.device != self.device_map['mlp']:
            hidden_states = hidden_states.to(self.device_map['mlp'])

        hidden_states = self.mlp(hidden_states)

        if hidden_states.device != self.main_device:
            hidden_states = hidden_states.to(self.main_device)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    def move_model_part(self, main_device):
        self.main_device = main_device
        if 'self_attn' in self.device_map:
            self.self_attn.to(self.device_map['self_attn'])
        if 'mlp' in self.device_map:
            self.mlp.to(self.device_map['mlp'])
        if 'layernorms' in self.device_map:
            self.input_layernorm.to(self.device_map['layernorms'])
            self.post_attention_layernorm.to(self.device_map['layernorms'])

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for _, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)

    # ------------------------------------------------------------------
    # 参数收集（layer_optim.optimize_layer 按名字收集三组参数）
    # ------------------------------------------------------------------
    def rotation_parameters(self):
        return iter([m for n, m in self.named_parameters() if n.find("free_line") > -1])

    def scale_parameters(self):
        return iter([m for n, m in self.named_parameters() if n.find("scale_factor") > -1])

    def shift_parameters(self):
        return iter([m for n, m in self.named_parameters() if n.find("shift_factor") > -1])

    def factor_parameters(self):
        """SVD 因子 A/B（真权重容量，optimize_layer 里单独一组、用更小的 lr）。"""
        return iter([m for n, m in self.named_parameters()
                     if n.find("svd_A") > -1 or n.find("svd_B") > -1])

    def set_svd_fp_mode(self, enable: bool):
        """阶段1（FP 低秩微调）开关：True 时 binary_temporary 直接用 A@B，不量化，
        梯度只训练 A/B 本身；False 时走完整量化链路（阶段2 QAT）。"""
        self.svd_fp_mode = bool(enable)

    # ------------------------------------------------------------------
    # 与 optimize_layer 的调用约定对接（本类只有 SVD 路径，无需分派）
    # ------------------------------------------------------------------
    def mask_generate_save(self, highorder=None, layer_idx=None, disable_gptq=False, update=False, epoch=None):
        """SVD 路径：mask 即比特分配，用当前参数重新贪心分配。"""
        self.svd_bit_alloc()

    def binary_temporary(self, highorder=None, disable_gptq=False):
        self._svd_binary_temporary()

    def binary_inplace(self, highorder=None, disable_gptq=False):
        self._svd_binary_inplace()

    # ------------------------------------------------------------------
    # SVD 核心
    # ------------------------------------------------------------------
    def svd_prepare(self):
        """对每个 QuantLinear 做 SVD 分解并缓存 A/B（填充到 N 的整数倍）。

        svd_whiten=True（默认）时用 Hessian 白化 SVD（min tr(ΔW·H·ΔWᵀ)，含逐秩
        能量均衡 D，依赖已累积的 module.H）；False 时退化为普通 Frobenius SVD（消融）。
        """
        N = self.svd_block
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                rank = self.svd_rank
                if self.svd_kv_rank is not None and ("k_proj" in name or "v_proj" in name):
                    rank = self.svd_kv_rank
                elif (self.svd_mlp_deep_rank is not None
                      and self.svd_layer_idx is not None
                      and self.svd_layer_idx >= self.svd_mlp_deep_from
                      and any(t in name for t in ("gate_proj", "up_proj", "down_proj"))):
                    rank = self.svd_mlp_deep_rank
                with torch.no_grad():
                    Wf = module.weight.data.float()
                    if self.svd_whiten:
                        Hess = module.H.float().to(Wf.device)
                        A, B, _ = svd_decompose_whitened(Wf, rank, Hess, self.svd_percdamp)
                    else:
                        A, B, _ = svd_decompose(Wf, rank)
                    A, B = _pad_rank(A, B, N)
                    # 诊断：秩截断地板（FP A@B 与 W 的相对 Frobenius 误差），
                    # 与量化无关；偏大的模块说明该权重谱衰减慢、不低秩。
                    rel = ((Wf - A @ B).norm() / Wf.norm().clamp(min=1e-12)).item()
                    print(f"[SVD] {name}: rank-trunc rel err {rel:.4f}", flush=True)
                # A/B 注册为可学习参数（对分解因子做 QAT）：STE 量化链路对 A/B
                # 天然可微；阶段1(FP微调)/阶段2(QAT) 都训练它们。名字含 svd_A/svd_B，
                # 由 factor_parameters() 收集、optimize_layer 的关键词匹配保存/恢复。
                module.svd_A = nn.Parameter(A)
                module.svd_B = nn.Parameter(B)

    def svd_register_params(self):
        """为每个 QuantLinear 注册可学习旋转 free_line_A/B、尺度 scale_factor_A/B、
        零点 shift_factor_A/B。

        三者采用相同的按 line 共享粒度，且 scale/shift 为**逐坐标**倍率 [R/N, N]：
        每条 block-line 一组 N 个（block 内 N 个旋转坐标各一个）——
        A 按列组共享（全部行共用），B 按行组共享（全部列共用）。
        block 内的 min/max 每次前向动态统计，因子只做倍率调制。
        """
        N = self.svd_block
        tri = N * (N - 1) // 2
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                name_tmp = name.replace(".", "_")
                dev = module.weight.device
                nb = module.svd_A.shape[1] // N
                self.register_parameter(f"{name_tmp}_svd_free_line_A", nn.Parameter(torch.zeros(nb, tri, device=dev)))
                self.register_parameter(f"{name_tmp}_svd_free_line_B", nn.Parameter(torch.zeros(nb, tri, device=dev)))
                self.register_parameter(f"{name_tmp}_svd_scale_factor_A", nn.Parameter(torch.ones(nb, N, device=dev)))
                self.register_parameter(f"{name_tmp}_svd_scale_factor_B", nn.Parameter(torch.ones(nb, N, device=dev)))
                self.register_parameter(f"{name_tmp}_svd_shift_factor_A", nn.Parameter(torch.ones(nb, N, device=dev)))
                self.register_parameter(f"{name_tmp}_svd_shift_factor_B", nn.Parameter(torch.ones(nb, N, device=dev)))

    def svd_bit_alloc(self):
        """用当前旋转/尺度/零点参数对每个 QuantLinear 做贪心比特分配，
        结果存到 module.svd_bitsA / svd_bitsB。"""
        N = self.svd_block
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                name_tmp = name.replace(".", "_")
                dev = module.weight.device
                A, B = module.svd_A, module.svd_B
                free_A = getattr(self, f"{name_tmp}_svd_free_line_A").detach()
                free_B = getattr(self, f"{name_tmp}_svd_free_line_B").detach()
                sfA = getattr(self, f"{name_tmp}_svd_scale_factor_A").detach()
                sfB = getattr(self, f"{name_tmp}_svd_scale_factor_B").detach()
                tfA = getattr(self, f"{name_tmp}_svd_shift_factor_A").detach()
                tfB = getattr(self, f"{name_tmp}_svd_shift_factor_B").detach()
                R_A = build_rotation(free_A, N)
                R_B = build_rotation(free_B, N)
                Hess = module.H.float().to(dev)
                if Hess.shape[0] != B.shape[1]:
                    # Hessian 尺寸异常时退化为单位阵（等权）
                    Hess = torch.eye(B.shape[1], device=dev)
                bitsA, bitsB = greedy_bit_allocation(
                    A, B, Hess, N, self.svd_budget_ratio, self.svd_min_bit, self.svd_max_bit, R_A, R_B,
                    sfA=sfA, sfB=sfB, tfA=tfA, tfB=tfB,   # [nb, N] 逐坐标因子
                )
                module.svd_bitsA = bitsA
                module.svd_bitsB = bitsB

    def _svd_quantize_module(self, module, name_tmp):
        """用当前旋转/尺度/零点参数量化 A、B 并返回重构权重 Aq @ Bq（可微）。

        A/B 走统一的 svd_block_quant：block_axis=1 → A（每行 N 列一块）；
        block_axis=0 → B（每列 N 行一块，内部按 Bᵀ 同构处理）。
        """
        N = self.svd_block
        free_A = getattr(self, f"{name_tmp}_svd_free_line_A")
        free_B = getattr(self, f"{name_tmp}_svd_free_line_B")
        sfA = getattr(self, f"{name_tmp}_svd_scale_factor_A")     # [nb, N]
        sfB = getattr(self, f"{name_tmp}_svd_scale_factor_B")     # [nb, N]
        tfA = getattr(self, f"{name_tmp}_svd_shift_factor_A")     # [nb, N]
        tfB = getattr(self, f"{name_tmp}_svd_shift_factor_B")     # [nb, N]
        R_A = build_rotation(free_A, N)
        R_B = build_rotation(free_B, N)
        Aq = svd_block_quant(module.svd_A, N, module.svd_bitsA, R_A, sfA, tfA, block_axis=1)
        Bq = svd_block_quant(module.svd_B, N, module.svd_bitsB, R_B, sfB, tfB, block_axis=0)
        return Aq @ Bq

    def _svd_binary_temporary(self):
        """把重构权重写到 temp_weight（保留计算图，供 optimize_layer 反传）。

        svd_fp_mode=True（阶段1）：temp_weight = A@B（FP，不量化，只训练因子）；
        svd_fp_mode=False（阶段2）：完整「旋转→量化→反旋转」链路（QAT）。
        """
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                name_tmp = name.replace(".", "_")
                if self.svd_fp_mode:
                    module.temp_weight = module.svd_A @ module.svd_B
                else:
                    module.temp_weight = self._svd_quantize_module(module, name_tmp)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter = True

    def _svd_binary_inplace(self):
        """用最终参数写回 module.weight，并清理缓存。

        先统一释放训练残留（temp_weight 半精度副本）与不再需要的 Hessian
        （比特分配已完成，inplace 不需要 H），再做 fp32 重构——否则重构第一个
        模块时其余模块的 temp_weight + 全部 H 仍压着显存，容易在此处 OOM。
        """
        for _, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.use_temporary_parameter = False
                module.use_temporary_lora_parameter = False
                for attr in ["temp_weight", "temp_bias", "lora_weight", "H"]:
                    if hasattr(module, attr):
                        try:
                            delattr(module, attr)
                        except Exception:
                            pass
        torch.cuda.empty_cache()

        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                name_tmp = name.replace(".", "_")
                with torch.no_grad():
                    q = self._svd_quantize_module(module, name_tmp)
                    module.weight = q.to(module.weight.dtype)
                    del q
                for attr in ["svd_A", "svd_B", "svd_bitsA", "svd_bitsB"]:
                    if hasattr(module, attr):
                        try:
                            delattr(module, attr)
                        except Exception:
                            pass
                torch.cuda.empty_cache()

    def svd_fp_inplace(self):
        """不量化：直接把 FP 低秩重构 A@B 写回 module.weight。

        Stage 1 地板评测专用（--svd_fp_only）：端到端度量微调后低秩表示
        A1·B1 的质量，无量化混淆、无贪心/量化计算。"""
        # 与 _svd_binary_inplace 一样，先统一释放所有模块的 Hessian 与训练期
        # temp_weight；否则在重构第一个大 MLP 矩阵时，其余模块的缓存仍占显存，
        # 很容易在纯 Stage-1 地板评测的写回阶段 OOM。
        for _, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.use_temporary_parameter = False
                module.use_temporary_lora_parameter = False
                for attr in ["temp_weight", "temp_bias", "lora_weight", "H"]:
                    if hasattr(module, attr):
                        try:
                            delattr(module, attr)
                        except Exception:
                            pass
        torch.cuda.empty_cache()

        for _, module in self.named_modules():
            if isinstance(module, QuantLinear):
                with torch.no_grad():
                    q = module.svd_A @ module.svd_B
                    module.weight = q.to(module.weight.dtype)
                    del q
                for attr in ["svd_A", "svd_B", "svd_bitsA", "svd_bitsB"]:
                    if hasattr(module, attr):
                        try:
                            delattr(module, attr)
                        except Exception:
                            pass
                torch.cuda.empty_cache()
