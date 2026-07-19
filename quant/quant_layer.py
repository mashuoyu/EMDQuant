import os
import torch
from torch import nn
from typing import Optional, Tuple, List
from quant.int_linear import QuantLinear
from quant.int_matmul import QuantMatMul
import torch.nn.functional as F
from quant.ptq_norm import PTQLlamaRMSNorm
from collections import OrderedDict
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb,LlamaRMSNorm,repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
import pdb
import copy
import logging
from quant.transformation import *
from utils.structure import fit_quantizer, structural_group_searching_v0, fit_quantizer_with_angle, random_mask_generate
from contextlib import nullcontext
RANDOM = False
RGN_IN = 16
BLOCK_IN = 2
BLOCK_PARAMS = 258
def gather_ratio(CGN, scale_lines, shift_lines, blocki):

    scale_ratio = torch.abs(repeat_scale_shift(CGN,scale_lines[blocki]))
    shift_ratio = torch.abs(repeat_scale_shift(CGN, shift_lines[blocki]))

    ratio_matrix = torch.cat([
        scale_ratio, 
        shift_ratio, 
        ], dim=1)
    return ratio_matrix

def gather_ratio_omni(scale_line,shfit_line, CGN, columns, blocksize):
    scale_parts = []
    shift_parts = []
    torch.autograd.set_detect_anomaly(True)
    for blocki, col_st in enumerate(range(0, columns, blocksize)):
        col_ed = min(col_st + blocksize, columns)
        ratio = gather_ratio(CGN,scale_line,shfit_line,blocki)
        scale_parts.append(ratio[:, :blocksize])
        shift_parts.append(ratio[:, blocksize:])

    scale_combined = torch.cat(scale_parts, dim=1)  # 形状 [H, N*W]
    shift_combined = torch.cat(shift_parts, dim=1)  # 形状 [H, N*W]
    ratio_list = torch.cat([scale_combined, shift_combined], dim=1)

    return ratio_list

def repeat_scale_shift(CGN,s):
    W = s.repeat_interleave(RGN_IN, dim=0)  # 形状变为 [H, blocksize//CGN//2]
    W = W.repeat_interleave(BLOCK_IN, dim=1)  # 最终形状 [H, blocksize//CGN]
    W1 =  W.repeat_interleave(CGN, dim=1)
    return W1

def quantize_with_params( w, mask, order=2, deltas=None, shifts=None, angle=None, groupi=0):
    w_rotated = rotation_weight(w,angle)
    w_normal = w_rotated / deltas + shifts
    w_clamp = torch.clamp(w_normal, min = 1e-8, max = 2**(order) - 1e-8) - 0.5
    w_quant = (torch.round(w_clamp) - w_clamp).detach() + w_clamp
    w_dequant = deltas * (w_quant - shifts + 0.5)
    w_dequant_trans_rotated = rotation_weight(w_dequant,angle,mode="trans")
    w = w_dequant_trans_rotated*mask
    return w

def rotation_weight(weight_matrix, angle, mode = "default"):
    group_size = angle.shape[0]
    weight_matrix_reshaped = weight_matrix.view(weight_matrix.shape[0], -1, group_size)  # [4096, 512, 8]
    if mode == "default":
        weight_rotated = torch.einsum("bni,ij->bnj", weight_matrix_reshaped, angle).view(weight_matrix.shape[0], weight_matrix.shape[1])  # 或使用 matmul
    else:
        weight_rotated = torch.einsum("bni,ij->bnj", weight_matrix_reshaped, angle.T).view(weight_matrix.shape[0], weight_matrix.shape[1])  # 或使用 matmul
    return weight_rotated  
          
def mask_generate(
                    weight,
                    H,
                    dev,
                    CGN,
                    scale_line,
                    shfit_line,
                    blocksize=128, 
                    percdamp=0.01, 
                    partition=3, 
                    orders=(1,2,3),
                    bit_init = 0.15,
                    bit_ad = 0.1,
                    angle = None,
                    disable_gptq = False,
                    file_name = None
                    ):
        
        W = weight.float().clone()
        H_mask = H.clone()

        columns = W.shape[1]
        
        dead = torch.diag(H_mask ) == 0
        H_mask[dead, dead] = 1
        W[:, dead] = 0
        

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(W.shape[1], device =  dev)
        
        
        H_mask[diag, diag] += damp
        H_mask = torch.linalg.cholesky(H_mask)
        H_mask = torch.cholesky_inverse(H_mask)
        H_mask = torch.linalg.cholesky(H_mask, upper=True)
        Hinv = H_mask
        mask_QuantLinear = torch.zeros_like(W, dtype=torch.bool,requires_grad=False).unsqueeze(0).repeat_interleave(partition, dim=0,)
        mask = torch.zeros_like(W[:, 0:blocksize], dtype=torch.bool,requires_grad=False).unsqueeze(0).repeat_interleave(partition, dim=0,)
        scale = torch.zeros_like(mask)
        shift = torch.zeros_like(mask)
        Q1 = torch.zeros_like(W[:, 0:blocksize])
        Err1 = torch.zeros_like(W[:, 0:blocksize])

        torch.autograd.set_detect_anomaly(True)
        for blocki, col_st in enumerate(range(0, columns, blocksize)):
            col_ed = min(col_st + blocksize, columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed
            ratio_list = gather_ratio(CGN,scale_line,shfit_line, blocki)
            RGN = W[:, st:ed].shape[0] // 1          
            with torch.no_grad():               
                mask[0], mask[1], mask[2] ,scale, shift = structural_group_searching_v0(W[:, st:ed].detach(), H_mask[st:ed, st:ed].detach(), ratio_list, bit_init, bit_ad, RGN, CGN, partition, angle, file_name)
                torch.cuda.empty_cache()
                mask_QuantLinear[:,:,st:ed] = mask
            with torch.enable_grad():
                if disable_gptq:
                    # RTN
                    # print("RTN")
                    w = W[:, col_st:col_ed]                
                    # from low to high group
                    q_part_groups = []
                    for i in range(mask.shape[0]):
                        q_part_groups.append(quantize_with_params( w, mask[i], order=orders[i], deltas=scale[i], shifts=shift[i], angle=angle[i]))

                    q = torch.zeros_like(w)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:] * mask[j, :]
                    W[:, col_st:col_ed] = q
                else:
                    # shape of W1: [oc, n_cols]
                    W1 = W[:, col_st:col_ed].clone()
                    Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]
                    q_part_groups = []              
              
                    for i in range(mask.shape[0]):
                        q_part_groups.append(quantize_with_params( W1, mask[i], order=orders[i], deltas=scale[i], shifts=shift[i], angle=angle[0]))
                         
                    for i in range(n_cols):
                        # shape of w: [oc, 1]
                        w = W1[:, i]
                        d = Hinv1[i, i]
                        q = torch.zeros_like(w)
                        for j in range(mask.shape[0]):
                            q_mask = q_part_groups[j][:, i] * mask[j, :, i]
                            if  W.shape[0]% RGN !=0 or blocksize % CGN !=0:
                                q_mask = q_mask[:q.shape[0]]
                            q += q_mask

                        Q1[:, i] = q
                        # breakpoint()
                        err1 = (w - q) / d
                        Err1[:, i] = err1
                    W[:, col_st:col_ed] = Q1
                    W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])
            
        if not disable_gptq:
            del W1, Q1, Err1, Hinv1, scale, shift
            del Hinv, W
        else:
            del w, q

        torch.cuda.empty_cache()
        return mask_QuantLinear


def fastquant_with_params(
                    W,
                    H,
                    dev,
                    CGN,
                    mask,
                    scale_line,
                    shfit_line,
                    blocksize=128, 
                    percdamp=0.01, 
                    high_order=3, 
                    orders=(1,2,3),
                    angle = None,
                    disable_gptq = False,
                    ):
        
        W = W.float()
        columns = W.shape[1]
        if not disable_gptq:
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0
        

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(W.shape[1], device =  dev)
            H[diag, diag] += damp
            H_c = torch.linalg.cholesky(H)
            H_c_i = torch.cholesky_inverse(H_c)
            Hinv = torch.linalg.cholesky(H_c_i, upper=True)
    
            Q1 = torch.zeros_like(W[:, 0:blocksize])
            Err1 = torch.zeros_like(W[:, 0:blocksize])


        torch.autograd.set_detect_anomaly(True)
        for blocki, col_st in enumerate(range(0, columns, blocksize)):
            col_ed = min(col_st + blocksize, columns)
            ratio_list = gather_ratio(CGN,scale_line,shfit_line,blocki)
            st = col_st
            ed = col_ed
            W1 = W[:, col_st:col_ed]      
            scale, shift = fit_quantizer(rotation_weight(W1.detach(),angle[0]), ratio_list, high_order)
                
            if disable_gptq:
                # RTN
                # print("RTN")
                w = W[:, col_st:col_ed]                
                # from low to high group
                q_part_groups = []
                for i in range(mask.shape[0]):
                    q_part_groups.append(quantize_with_params(W1, mask[i][:,st:ed], order=orders[i], deltas=scale[i], shifts=shift[i], angle=angle[0]))

                q = torch.zeros_like(w)
                for j in range(mask.shape[0]):
                    q += q_part_groups[j][:] * mask[j, :, st:ed]
                W[:, col_st:col_ed] = q
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed]
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]
                q_part_groups = []              
              
                for i in range(mask.shape[0]):
                    q_part_groups.append(quantize_with_params(W1, mask[i][:,st:ed], order=orders[i], deltas=scale[i], shifts=shift[i], angle=angle[0]))
                         
                q = torch.zeros_like(Q1)
                for j in range(mask.shape[0]):
                    q_mask = q_part_groups[j] * mask[j, :, st:ed]
                    q += q_mask
                Q1 = q
                # breakpoint()
                Err1 = (W1 - Q1) / torch.diag(Hinv1).unsqueeze(0)
                W[:, col_st:col_ed] = Q1
                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])
            
        if not disable_gptq:
            del W1, Q1, Err1, Hinv1, scale, shift, ratio_list
            del H, Hinv,q,q_mask,q_part_groups,H_c, H_c_i, columns, damp, diag
            torch.cuda.empty_cache()
        return W

def fast_quant_wo_gptq(
                    W,
                    CGN,
                    mask,
                    scale_line,
                    shfit_line,
                    blocksize=128, 
                    high_order=3, 
                    orders=(1,2,3),
                    angle = None,
                    ):
        
        W = W.float()
        columns = W.shape[1]

        ratio_list = gather_ratio_omni(scale_line,shfit_line, CGN, columns, blocksize)
      
        scale, shift = fit_quantizer_with_angle(W.detach(), ratio_list, angle, high_order) #(rotation_weight(W.detach(),angle[0]), ratio_list, high_order)
             
        # from low to high group
        q_part_groups = []
        for i in range(mask.shape[0]):
            q_part_groups.append(quantize_with_params(W, mask[i], order=orders[i], deltas=scale[i], shifts=shift[i], angle=angle[0]))
        q = torch.zeros_like(W)
        for j in range(mask.shape[0]):
            q += q_part_groups[j][:] * mask[j]
        W = q
            
        del scale, shift, ratio_list
        del q_part_groups,columns
        torch.cuda.empty_cache()
        return W
 
class QuantLlamaMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        args=None,
    ):
        super().__init__()
        # self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = QuantLinear(
            org_module.gate_proj,
            args.weight_quant_params,
            args.act_quant_params,
            lora_rank=getattr(args, 'lora_rank', None),
            lora_a_shape=getattr(args, 'lora_a_shape', None),
            lora_b_shape=getattr(args, 'lora_b_shape', None),
            lora_init_scale=getattr(args, 'lora_init_scale', 1e-3),
        )
        self.down_proj = QuantLinear(
            org_module.down_proj,
            args.weight_quant_params,
            args.act_quant_params,
            lora_rank=getattr(args, 'lora_rank', None),
            lora_a_shape=getattr(args, 'lora_a_shape', None),
            lora_b_shape=getattr(args, 'lora_b_shape', None),
            lora_init_scale=getattr(args, 'lora_init_scale', 1e-3),
        )
        self.up_proj = QuantLinear(
            org_module.up_proj,
            args.weight_quant_params,
            args.act_quant_params,
            lora_rank=getattr(args, 'lora_rank', None),
            lora_a_shape=getattr(args, 'lora_a_shape', None),
            lora_b_shape=getattr(args, 'lora_b_shape', None),
            lora_init_scale=getattr(args, 'lora_init_scale', 1e-3),
        )
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, 
                 org_module: nn.Module,
                 config: LlamaConfig,
                 args=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = copy.deepcopy(org_module.rotary_emb)

        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.weight_quant_params,
            args.act_quant_params,
            lora_rank=getattr(args, 'lora_rank', None),
            lora_a_shape=getattr(args, 'lora_a_shape', None),
            lora_b_shape=getattr(args, 'lora_b_shape', None),
            lora_init_scale=getattr(args, 'lora_init_scale', 1e-3),
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.weight_quant_params,
            args.act_quant_params,
            lora_rank=getattr(args, 'lora_rank', None),
            lora_a_shape=getattr(args, 'lora_a_shape', None),
            lora_b_shape=getattr(args, 'lora_b_shape', None),
            lora_init_scale=getattr(args, 'lora_init_scale', 1e-3),
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.weight_quant_params,
            args.act_quant_params,
            lora_rank=getattr(args, 'lora_rank', None),
            lora_a_shape=getattr(args, 'lora_a_shape', None),
            lora_b_shape=getattr(args, 'lora_b_shape', None),
            lora_init_scale=getattr(args, 'lora_init_scale', 1e-3),
        )
        self.o_proj = QuantLinear(
            org_module.o_proj,
            args.weight_quant_params,
            args.act_quant_params,
            lora_rank=getattr(args, 'lora_rank', None),
            lora_a_shape=getattr(args, 'lora_a_shape', None),
            lora_b_shape=getattr(args, 'lora_b_shape', None),
            lora_init_scale=getattr(args, 'lora_init_scale', 1e-3),
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul
        )

        self.use_weight_quant = False
        self.use_act_quant = False

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states =self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)
                

class QuantLlamaDecoderLayer(nn.Module):
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
        self.input_layernorm = PTQLlamaRMSNorm(ori_layer.input_layernorm,eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = PTQLlamaRMSNorm(ori_layer.post_attention_layernorm,eps=ori_layer.post_attention_layernorm.variance_epsilon)
        self.group_size = args.blocksize // args.CGN
        self.CGN = args.CGN
        self.blocksize = args.blocksize
        self.mask_save_dir = args.mask_save_dir
        self.lora_save_dir = args.lora_save_dir
        self.sigmoid = nn.Sigmoid()
        self.device_map = device_map or {}


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
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
            #residual = residual.to(self.main_device)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs        

    def move_model_part(self,main_device):
        self.main_device = main_device
        if 'self_attn' in self.device_map:
            self.self_attn.to(self.device_map['self_attn'])
        if 'mlp' in self.device_map:
            self.mlp.to(self.device_map['mlp'])
        if 'layernorms' in self.device_map:
            self.input_layernorm.to(self.device_map['layernorms'])
            self.post_attention_layernorm.to(self.device_map['layernorms'])

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)
    
    def mask_generate_save(self, highorder, layer_idx, disable_gptq = False, update = False, epoch = None):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.float()
                module.temp_weight = module.weight
        # quant


        save_dir  = os.path.join(self.mask_save_dir, f"layer_{layer_idx}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if epoch is not None:
            train_save_dir = os.path.join(save_dir, "train")
            if not os.path.exists(train_save_dir):
                os.mkdir(train_save_dir)    
            file_name = os.path.join(train_save_dir, f"mask_ratio_{epoch}.txt")
        else:   
            file_name = os.path.join(save_dir, f"mask_ratio.txt")
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                dev = module.weight.device
                H = module.H.float()
                with open(file_name, 'a') as f:
                    f.write(f"dealing_mask_for_module_{name}\n")
                name_tmp = name.replace(".","_")
                if module.temp_weight is not None:
                    save_path = os.path.join(save_dir, f"layer_{layer_idx}_{name}_masks.pt")

                    if os.path.exists(save_path) and (not update):
                        mask_restored = torch.load(save_path)
                        mask = mask_restored["masks"]
                        module.weight_mask = mask.to(dev)
                        continue

                    else:
                        free_lines = []
                        scale_factors = []
                        shift_factors = []


                        for order in range(highorder):
                            free_line = getattr(self, f"{name_tmp}_free_line_{order}").to(dev)
                            free_lines.append(free_line)

                        _, W = module.weight.shape[0], module.weight.shape[1]

                        for blocki, col_st in enumerate(range(0, W, self.blocksize)):
                            scale_factor = getattr(self, f"{name_tmp}_scale_factor_{blocki}").to(dev)
                            shift_factor = getattr(self, f"{name_tmp}_shift_factor_{blocki}").to(dev)
                            scale_factors.append(scale_factor)
                            shift_factors.append(shift_factor)


                        angles = self.cal_rotation(free_lines,highorder,dev)
                        if not RANDOM:
                            mask = mask_generate(weight = module.temp_weight.clone(), H = H, blocksize=self.blocksize,
                                         scale_line = scale_factors, shfit_line = shift_factors, 
                                         dev = dev, CGN = self.CGN,angle=angles,file_name=file_name,disable_gptq=disable_gptq)

                        module.weight_mask = mask
                    
                        mask_data = {
                                'layer_name': name,
                                'layer_idx': layer_idx,
                                'masks': mask,
                                'groupsize': self.group_size,
                                'CGN': self.CGN,
                                'blocksize': self.blocksize
                                }
                        torch.save(mask_data, save_path)

        print(f"Saved masks to {save_dir}")


    def init_lora_weight(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.lora_weight = module.temp_weight.clone()

    def lora_save(self, layer_idx, update=False):
        if self.lora_save_dir is None:
            return
        save_dir = os.path.join(self.lora_save_dir, f"layer_{layer_idx}")
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                lora_state = module.save_lora_state()
                if lora_state is None:
                    continue
                save_path = os.path.join(save_dir, f"layer_{layer_idx}_{name}_lora.pt")
                if os.path.exists(save_path) and not update:
                    continue
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                torch.save({"lora_state": lora_state}, save_path)

        print(f"Saved LoRA state to {save_dir}")

    def lora_load(self, layer_idx):

        if self.lora_save_dir is None:
            lora_load = False
            return lora_load
        
        lora_load_num = 0
        lora_load = True

        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                save_path = os.path.join(self.lora_save_dir, f"layer_{layer_idx}", f"layer_{layer_idx}_{name}_lora.pt")
                if not os.path.exists(save_path):
                    lora_load_num += 1
                    if lora_load_num ==7:
                        lora_load = False
                    continue
                data = torch.load(save_path, map_location=module.weight.device)
                module.load_lora_state(data.get("lora_state"), device=module.weight.device)
                del data
        return lora_load

    def binary_temporary(self, highorder,disable_gptq=False):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight.clone()
 

        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                dev =module.weight.device
                H_cal = module.H.clone()
    
                name_tmp = name.replace(".","_")
                if hasattr(module, "temp_weight"):
                    free_lines = []
                    scale_factors = []
                    shift_factors = []


                    for order in range(highorder):
                        free_line = getattr(self, f"{name_tmp}_free_line_{order}").to(dev)
                        free_lines.append(free_line)

                    _, W = module.weight.shape[0], module.weight.shape[1]

                    for blocki, _ in enumerate(range(0, W, self.blocksize)):
                        scale_factor = getattr(self, f"{name_tmp}_scale_factor_{blocki}").to(dev)
                        shift_factor = getattr(self, f"{name_tmp}_shift_factor_{blocki}").to(dev)
                        scale_factors.append(scale_factor)
                        shift_factors.append(shift_factor)
                       

                    angles = self.cal_rotation(free_lines,highorder,dev)
                    mask = module.weight_mask
                    if not disable_gptq:
                        q_mix= fastquant_with_params(W = module.temp_weight, H = H_cal, mask = mask,
                                                 scale_line = scale_factors, shfit_line = shift_factors, 
                                                 dev = dev, CGN = self.CGN,  angle = angles)
                    else:
                        q_mix = fast_quant_wo_gptq(W = module.temp_weight, mask = mask, blocksize=self.blocksize,
                                                scale_line = scale_factors, shfit_line = shift_factors, 
                                                CGN = self.CGN,  angle = angles)
                    
                    module.temp_weight = q_mix

                else:
                    free_lines = []
                    scale_factors = []
                    shift_factors = []


                    for order in range(highorder):
                        free_line = getattr(self, f"{name_tmp}_free_line_{order}").to(dev)
                        free_lines.append(free_line)

                    _, W = module.weight.shape[0], module.weight.shape[1]

                    for blocki, _ in enumerate(range(0, W, self.blocksize)):
                        scale_factor = getattr(self, f"{name_tmp}_scale_factor_{blocki}").to(dev)
                        shift_factor = getattr(self, f"{name_tmp}_shift_factor_{blocki}").to(dev)
                        scale_factors.append(scale_factor)
                        shift_factors.append(shift_factor)
                       
                    mask = module.weight_mask
                    angles = self.cal_rotation(free_lines,highorder,dev)

                    q_mix = fast_quant_wo_gptq(W = module.weight, mask = mask,
                                                scale_line = scale_factors, shfit_line = shift_factors, 
                                                CGN = self.CGN,  angle = angles)
                    
                    module.temp_weight = q_mix



                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True 

    def binary_inplace(self, highorder, disable_gptq=False):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                dev = module.weight.device
                H_cal = module.H.clone()
                
                name_tmp = name.replace(".","_")
                free_lines = []
                scale_factors = []
                shift_factors = []

                for order in range(highorder):
                    free_line = getattr(self, f"{name_tmp}_free_line_{order}").to(dev)
                    free_lines.append(free_line)

                _, W = module.weight.shape[0], module.weight.shape[1]

                for blocki, _ in enumerate(range(0, W, self.blocksize)):
                    scale_factor = getattr(self, f"{name_tmp}_scale_factor_{blocki}").to(dev)
                    shift_factor = getattr(self, f"{name_tmp}_shift_factor_{blocki}").to(dev)
                    scale_factors.append(scale_factor)
                    shift_factors.append(shift_factor)

                angles = self.cal_rotation(free_lines, highorder, dev)
                mask = module.weight_mask
                temp_weight = module.weight.clone()
                if not disable_gptq:
                    q_mix= fastquant_with_params(W = temp_weight, H = H_cal, mask = mask,
                                                scale_line = scale_factors, shfit_line = shift_factors, 
                                                dev = dev, CGN = self.CGN,  angle = angles)
                else:
                    q_mix = fast_quant_wo_gptq(
                        W=temp_weight,
                        mask=mask,
                        blocksize=self.blocksize,
                        scale_line=scale_factors,
                        shfit_line=shift_factors,
                        CGN=self.CGN,
                        angle=angles,
                    )

                if module.lora_compensation is not None:
                        q_mix = module.lora_compensation(q_mix)

                with torch.no_grad():
                    module.weight = q_mix
                    module.use_temporary_parameter = False
                    module.use_temporary_lora_parameter = False
                    #module.weight_error()
                    
                del module.H, mask, q_mix, shift_factors, scale_factors, free_lines, H_cal, temp_weight, module.weight_mask
                torch.cuda.empty_cache()


    def lora_compensation(self):
        for _, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.lora_weight = module.lora_weight.detach()
                module.use_temporary_lora_parameter = True


    def load_best_lora_params(self, best_lora_params: dict):
        """加载最优 LoRA 参数到各个 QuantLinear 模块"""
        if best_lora_params is None:
            print("Warning: No best_lora_params to load")
            return
        
        loaded_count = 0
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if name in best_lora_params:
                    lora_state = best_lora_params[name]
                    
                    # 如果 lora_compensation 不存在，先初始化
                    if module.lora_compensation is None:
                        module.init_lora_compensation(
                            a_shape=lora_state['a_shape'],
                            b_shape=lora_state['b_shape'],
                            device=module.weight.device,
                            init_scale=module.lora_init_scale,
                        )
                    
                    # 加载参数
                    module.lora_compensation.A.data.copy_(
                        lora_state['A'].to(module.weight.device)
                    )
                    module.lora_compensation.B.data.copy_(
                        lora_state['B'].to(module.weight.device)
                    )
                    loaded_count += 1
        
        print(f"Successfully loaded LoRA params")

    def clear_temp_variable(self):
       for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                if hasattr(module, "lora_weight"):
                    del module.lora_weight 

                if hasattr(module, "temp_bias"):
                    del module.temp_bias
                torch.cuda.empty_cache() 

    def rotation_parameters(self):
        params = []
        template = "free_line" 
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params) 

    def scale_parameters(self):
        params = []
        template = "scale_factor"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def shift_parameters(self):
        params = []
        template = "shift_factor"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params) 
    
    def upbound_parameters(self):
        params = []
        template = "upbound_factor"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params) 
    
    def lowbound_parameters(self):
        params = []
        template = "lowbound_factor"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params) 
    
    def PTQ_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
    
    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()
 
    def cal_rotation(self, free_line, highorder, dev):
        angle_group = []
        pi = np.pi
        triu_idx = torch.triu_indices(self.group_size, self.group_size, 1).to(dev)
        with torch.enable_grad():
            for i in range(highorder):
                tanh = torch.nn.Tanh()
                free_line_norm = pi * tanh(free_line[i])
                free_matrix = torch.zeros(self.group_size, self.group_size, device=dev)
                free_matrix[triu_idx[0], triu_idx[1]] = free_line_norm  # 关键：依赖 free_line
                free_matrix = free_matrix - free_matrix.T
                angle_matrix = torch.matrix_exp(free_matrix)
                angle_group.append(angle_matrix)

        return angle_group
    
