import time
import copy
import math
from eval_ppl_utils import zero_short_eval
from quant.int_linear import QuantLinear
from quant.svd_quant_layer import SVDQuantLlamaDecoderLayer as QuantLlamaDecoderLayer
import layer_optim
import utils
import os
import pdb
import gc

import torch
import torch.nn as nn

from modelutils import find_layers
from contextlib import nullcontext

from utils.structure import fit_quantizer
from utils.util import NativeScalerWithGradNormCount

import gc


RGN_IN = 16
BLOCK_IN = 2
IDX="2026_05_04"
MODEL_NAME="llama2_7b"

# 注意：不能连续赋值两次（第二次会覆盖第一次）。expandable_segments 是
# 逐层大块分配/释放（Hessian/SVD workspace）场景下抗碎片化的关键开关。
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                      
def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if "opt" in model:
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = 2048
    return model


def svd_rank_floor(qlayer, quant_inps, fp_inps, quant_inps_fp, attention_mask, position_ids, nsamples, layer_idx, tag=""):
    """诊断：用 FP 的 A@B（不量化）前向，给出该层输出损失的不可约地板。

    floor loss 与训练日志的 loss 同尺度；mse 与末尾 "svd quant mse loss" 同尺度。
    训练后 loss 若已贴近地板，说明可约（量化）部分已被吃完，瓶颈在秩截断。
    在阶段1（FP 微调）前后各调一次即可量化微调带来的地板下降。
    """
    loss_func = torch.nn.MSELoss()
    with torch.no_grad():
        for _, module in qlayer.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.svd_A @ module.svd_B
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter = True
        m1_sum, m2_sum = 0.0, 0.0
        with torch.amp.autocast("cuda"):
            for j in range(nsamples):
                out_j = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                m1_sum += loss_func(fp_inps[j].unsqueeze(0), out_j).item()
                m2_sum += loss_func(quant_inps_fp[j].unsqueeze(0), out_j).item()
        m1, m2 = m1_sum / nsamples, m2_sum / nsamples
        floor_loss = 10 * (0.8 * m1 + 0.2 * m2)
        print(f"layer {layer_idx} rank-floor{tag}: loss={floor_loss:.6f}, mse={m1:.6e}", flush=True)
        for _, module in qlayer.named_modules():
            if isinstance(module, QuantLinear):
                module.use_temporary_parameter = False
                if hasattr(module, "temp_weight"):
                    del module.temp_weight
        torch.cuda.empty_cache()


'''
The function is employed to calibrate and quantize models layer by layer.
'''
def quant_sequential(model, dataloader, dev):
    print("Starting ...")
    
    # move embedding layer and first layer to target device

    use_cache = model.config.use_cache
    model.config.use_cache = False
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if len(devices) > 1:
        secondary_device = devices[1] if devices[0] == dev else devices[0]
    else:
        secondary_device = dev
    is_llama = False
    if "llama" in args.model:
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.model:
        layers = model.model.decoder.layers
        print(layers)
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        #DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.model:
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        #DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")
    
    
    layers[0] = layers[0].to(dev)
    print("------------")
    if args.deactive_amp and args.epochs>0:
        dtype = next(iter(model.parameters())).dtype
        traincast = nullcontext
    else:
        dtype = next(iter(model.parameters())).dtype
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {"i": 0}
    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    quant_inps_fp = copy.deepcopy(inps)
    

    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch_inter = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        print(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch_inter = None
    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None
    cossim = nn.CosineSimilarity(dim=2)
    
    for i in range(len(layers)):
        print(f"=== Start quantize layer {i} ===")
        optim_save_dir = os.path.join(args.optim_save_dir, f"layer_{i}")
        optim_save_path = os.path.join(optim_save_dir, f"optim_params.pth")
        if not os.path.exists(optim_save_dir):
            os.mkdir(optim_save_dir)
        # ------------------------------------------------------------------
        # SVD 量化路径：W≈A@B（秩 R），对 A/B 分块量化（可学习旋转 + 仿射）+ 贪心比特分配。
        # affine/旋转参数复用 layer_optim.optimize_layer（整层输出误差）训练。
        # 不使用 GPTQ 误差传播 / LoRA 补偿。
        # ------------------------------------------------------------------
        if args.low_quant_method == "svd":
            # SVD 阶段默认整层放在主卡。跨 GPU 的 MLP 会在量化临时权重、AMP 与
            # 残差相加之间引入额外的异步 copy；仅在显式请求时启用，便于隔离并避免
            # 非法 CUDA 访问。
            device_map = {"mlp": secondary_device} if args.svd_split_mlp else {}
            layer = layers[i].to(dev)
            qlayer = DecoderLayer(model.config, layer, args, device_map = device_map)
            qlayer = qlayer.to(dev)
            qlayer.move_model_part(dev)
            qlayer.svd_layer_idx = i          # 供逐层异构秩（svd_mlp_deep_rank）使用
            qlayer.set_quant_state(weight_quant=False, act_quant=False)

            # 1) 全精度参考输出（向后传递 FP 隐状态，作为训练目标）
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            # 2) 在量化模型激活上累积 Hessian，同时记录 FP 权重输出到 quant_inps_fp
            handles = []
            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    def hook_func(_, inp, out, mod=module):
                        mod.add_batch(inp[0].data, out.data)
                    handles.append(module.register_forward_hook(hook_func))
            with torch.no_grad():
                for j in range(args.nsamples):
                    quant_inps_fp[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for handle in handles:
                handle.remove()

            # 3) SVD 分解 + 注册可学习参数
            qlayer.svd_prepare()
            qlayer.svd_register_params()

            # 有本层存档则加载并跳过训练（断点续跑，语义同 braq 路径）
            loaded = False
            if os.path.exists(optim_save_path):
                saved_params = torch.load(optim_save_path, map_location="cpu")
                state_dict_to_load = {}
                for name, param in qlayer.named_parameters():
                    if name in saved_params:
                        state_dict_to_load[name] = saved_params[name].to(param.device)
                if state_dict_to_load:
                    qlayer.load_state_dict(state_dict_to_load, strict=False)
                    loaded = True
                    print(f"Loading optimized parameters for layer {i} (saved at {optim_save_path})", flush=True)
                del state_dict_to_load, saved_params

            # 诊断：SVD 初始点的秩截断地板
            svd_rank_floor(qlayer, quant_inps, fp_inps, quant_inps_fp,
                           attention_mask, position_ids, args.nsamples, i, tag="(svd-init)")

            # 4) 阶段1：FP 低秩微调 —— A/B 作为可学习参数直接训练（不量化）。
            #    上游误差经 quant_inps 流天然由本层补偿；目标是把秩-R 点从
            #    "SVD 权重最优" 挪到 "输出最优"，压低地板。
            best_optim_params = None
            if args.svd_fp_epochs > 0 and not loaded:
                qlayer.set_svd_fp_mode(True)
                best_optim_params = layer_optim.optimize_layer(
                    qlayer=qlayer,
                    quant_inps=quant_inps,
                    fp_inps=fp_inps,
                    quant_inps_fp=quant_inps_fp,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    args=args,
                    layer_idx=i,
                    num_epochs=args.svd_fp_epochs,
                    factors_only=True,
                )
                with torch.no_grad():
                    if best_optim_params is not None:
                        for name, param in qlayer.named_parameters():
                            if name in best_optim_params:
                                param.data.copy_(best_optim_params[name].to(param.device))
                qlayer.set_svd_fp_mode(False)
                # 微调后的地板（与 svd-init 对比 = 阶段1 的收益）
                svd_rank_floor(qlayer, quant_inps, fp_inps, quant_inps_fp,
                               attention_mask, position_ids, args.nsamples, i, tag="(after-fp-ft)")

            # 5) 贪心比特分配（用微调后的 A/B 与当前格子参数）
            #    --svd_fp_only 时跳过分配与量化训练（Stage 1 地板评测）
            if not args.svd_fp_only:
                qlayer.svd_bit_alloc()

            # 6) 阶段2：量化感知校准 —— 格子参数 + A/B 继续联合训练
            if args.epochs > 0 and not loaded and not args.svd_fp_only:
                best_optim_params = layer_optim.optimize_layer(
                    qlayer=qlayer,
                    quant_inps=quant_inps,
                    fp_inps=fp_inps,
                    quant_inps_fp=quant_inps_fp,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    args=args,
                    layer_idx=i,
                    num_epochs=args.epochs,
                )
                with torch.no_grad():
                    if best_optim_params is not None:
                        for name, param in qlayer.named_parameters():
                            if name in best_optim_params:
                                param.data.copy_(best_optim_params[name].to(param.device))
                    # QAT 后参数变了，重新同步比特分配再写回
                    qlayer.svd_bit_alloc()

            if best_optim_params is not None and not loaded:
                torch.save(best_optim_params, optim_save_path)
                print(f"Saved optimized parameters to {optim_save_path}")

            # 7) 写回：--svd_fp_only 时直接写 FP 低秩重构（地板评测），否则量化写回
            with torch.no_grad():
                if args.svd_fp_only:
                    qlayer.svd_fp_inplace()
                else:
                    qlayer.binary_inplace(args.highorder)

            # 8) 量化后前向，更新量化隐状态并打印 MSE
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                        if torch.isnan(quant_inps[j]).any():
                            raise RuntimeError(f"quant_inps is NAN at layer {i}")
                    loss = loss_func(fp_inps, quant_inps)
            print(f"layer {i} svd quant mse loss: {loss.item():.6e}", flush=True)

            qlayer.half()
            layers[i] = qlayer.to("cpu")
            del layer, qlayer, best_optim_params
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"layer {i} done. cuda allocated: {torch.cuda.memory_allocated()/2**30:.2f} GiB, "
                f"reserved: {torch.cuda.memory_reserved()/2**30:.2f} GiB",
                flush=True,
            )

              
    del inps
    del quant_inps
    del fp_inps
    torch.cuda.empty_cache()
    
    model.config.use_cache = use_cache
    return model


if __name__ == "__main__":
    import argparse
    from datautils import *

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=["xnor", "sign", "no", "2bit", "4bit", "prune", "braq", "svd"],
        help="quantization method; `xnor` is the method using XNOR to adapt hardware calculation; `prune` is the method used in sparseGPTQ; braq is the method used in BiLLM; `svd` decomposes W=A@B (rank R) and block-quantizes A/B with greedy bit allocation",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--CGN",
        type=int,
        default=16,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--RGN",
        type=int,
        default=1,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        default="magnitude",
        choices=["magnitude", "hessian"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )

    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--highorder", type=int, default=1)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--mask_save_dir", type=str, default=f"/sharedata/home/msy/code_llm_low_bit/LORA_LLM/embedding_lattice_lora/params_path/{MODEL_NAME}/{IDX}/mask_path")
    parser.add_argument("--lora_save_dir", type=str, default=f"/sharedata/home/msy/code_llm_low_bit/LORA_LLM/embedding_lattice_lora/params_path/{MODEL_NAME}/{IDX}/lora_path")
    parser.add_argument("--optim_save_dir", type=str, default=f"/sharedata/home/msy/code_llm_low_bit/LORA_LLM/embedding_lattice_lora/params_path/{MODEL_NAME}/{IDX}/optim_path")
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--lora_rank", type=int, default=None, help="Rank for LoRA error compensation Kronecker factorization.")
    parser.add_argument("--lora_a_shape", type=str, default="", help="Shape of matrix A for LoRA compensation, format 'r,c'.")
    parser.add_argument("--lora_b_shape", type=str, default="", help="Shape of matrix B for LoRA compensation, format 'r,c'.")
    parser.add_argument("--lora_init_scale", type=float, default=1e-3, help="Initialization scale for LoRA compensation matrices.")
    parser.add_argument("--lora_fit_steps", type=int, default=0, help="Number of LoRA error compensation fitting steps after quantization.")
    parser.add_argument("--lora_lr", type=float, default=1e-2, help="Learning rate for LoRA compensation fitting.")
    parser.add_argument("--epochs", type=int, default=1)
    # ---- SVD 量化（low_quant_method='svd'）相关参数 ----
    parser.add_argument("--svd_rank", type=int, default=256, help="SVD rank R for W≈A@B decomposition.")
    parser.add_argument("--svd_block", type=int, default=16, help="Block size N: A groups N cols per row, B groups N rows per col.")
    parser.add_argument("--svd_budget_ratio", type=float, default=1.0, help="Greedy bit-upgrade budget as average extra bits per block (0..max_bit-min_bit).")
    parser.add_argument("--svd_min_bit", type=int, default=1, help="Minimum bits per A/B block.")
    parser.add_argument("--svd_max_bit", type=int, default=3, help="Maximum bits per A/B block.")
    parser.add_argument("--svd_no_whiten", action="store_true", help="Disable Hessian-whitened SVD (fall back to plain Frobenius SVD, for ablation).")
    parser.add_argument("--svd_fp_epochs", type=int, default=0, help="Phase-1 FP low-rank fine-tuning epochs (train A/B without quantization) before QAT; 0 disables.")
    parser.add_argument("--svd_fp_only", action="store_true", help="Skip quantization entirely: write back FP low-rank A@B (Stage-1 floor evaluation).")
    parser.add_argument("--svd_kv_rank", type=int, default=None, help="Separate (higher) rank for k_proj/v_proj, which are most rank-sensitive (K>V, softmax amplification + GQA density). Break-even rank for [1024,3072] is 768. Default: same as --svd_rank.")
    parser.add_argument("--svd_mlp_deep_rank", type=int, default=None, help="Separate rank for MLP (gate/up/down) in deep layers [svd_mlp_deep_from, end]. Default: same as --svd_rank.")
    parser.add_argument("--svd_mlp_deep_from", type=int, default=22, help="First layer index (0-based) that uses --svd_mlp_deep_rank.")
    parser.add_argument("--svd_fp_lr", type=float, default=2e-4, help="Stage-1 learning rate for SVD factors A/B.")
    parser.add_argument("--svd_split_mlp", action="store_true", help="Opt in to placing the SVD MLP on a second GPU; disabled by default for stability.")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    args.svd_whiten = not args.svd_no_whiten

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc
    }
    args.lora_a_shape = list_of_ints(args.lora_a_shape) if args.lora_a_shape else None
    args.lora_b_shape = list_of_ints(args.lora_b_shape) if args.lora_b_shape else None
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    groupsize = args.blocksize
    base_params_path= f"/sharedata/home/msy/code_llm_low_bit/LORA_LLM/embedding_lattice_lora/params_path/{MODEL_NAME}/{IDX}"

    if base_params_path is not None:
        os.makedirs(args.mask_save_dir, exist_ok=True)
        os.makedirs(args.lora_save_dir, exist_ok=True)          
        os.makedirs(args.optim_save_dir, exist_ok=True) 
    else:
        os.makedirs(base_params_path,exist_ok=True)
        os.makedirs(args.mask_save_dir, exist_ok=True)
        os.makedirs(args.lora_save_dir, exist_ok=True)          
        os.makedirs(args.optim_save_dir, exist_ok=True)

    device = args.device
    save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}"
    save_file = "/sharedata/home/msy/code_llm_low_bit/LORA_LLM/embedding_lattice_lora/output/" + args.optim_save_dir + ".pt"
    if args.load_quantized:
        model = get_model(save_file)
        model.eval()
        model.to(device)
    else: # braq
        model = get_model(args.model)
        model.eval()
        tick = time.time()
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        quant_sequential(model, dataloader, device)
        print("quantization time:", time.time() - tick, "s")

    if args.save:
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_file)
    if args.tasks == "":
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
            )
            print(dataset)
            if "opt" in args.model:
                from eval_ppl_utils import opt_eval

                opt_eval(model, testloader, device, dataset, args.log_wandb)
            elif "llama" in args.model:
                from eval_ppl_utils import llama_eval

                llama_eval(model, testloader, device, dataset, args.log_wandb)
    else:
        zero_short_eval(model, device, args)
        
        
