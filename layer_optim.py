import math
import time
import torch
from torch import nn
from contextlib import nullcontext

from quant.int_linear import QuantLinear


class SchedulerWithRestore(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def __init__(self, optimizer, model, **kwargs):
            super().__init__(optimizer, **kwargs)
            self.model = model
            self.best_params = None
        
        def update_best_params(self, params):
            self.best_params = params

        def step(self, metrics):
            old_lr = self.optimizer.param_groups[0]['lr']
            super().step(metrics)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr < old_lr and self.best_params is not None:
                self._restore_params()

        def _restore_params(self):
            # 检查是否为LoRA参数（字典的值是字典）
            first_key = next(iter(self.best_params))
            if isinstance(self.best_params[first_key], dict) and 'A' in self.best_params[first_key]:
                # LoRA参数恢复
                for name, module in self.model.named_modules():
                    if name in self.best_params and hasattr(module, 'lora_compensation'):
                        lora_params = self.best_params[name]
                        module.lora_compensation.A.data.copy_(lora_params['A'].to(module.lora_compensation.A.device))
                        module.lora_compensation.B.data.copy_(lora_params['B'].to(module.lora_compensation.B.device))
            else:
                # 普通参数恢复
                for name, param in self.model.named_parameters():
                    if name in self.best_params:
                        param.data.copy_(self.best_params[name].to(param.device))

def optimize_layer(
    qlayer,
    quant_inps: torch.Tensor,
    fp_inps: torch.Tensor,
    quant_inps_fp: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    args,
    layer_idx: int,
    num_epochs=None,
    factors_only: bool = False,
):
    """Optimize rotation/scale/shift (and SVD factors A/B) for one layer.

    num_epochs=None 时用 args.epochs；SVD 两阶段训练用它分别传
    svd_fp_epochs（阶段1 FP 微调）和 epochs（阶段2 QAT）。

    factors_only=True 用于 Stage 1：仅更新 SVD 因子 A/B，不把 rotation、
    scale 或 shift 加入优化器。此时 qlayer 应处于 svd_fp_mode，临时权重为 A@B。
    """
    if attention_mask is not None:
        attention_mask_batch_inter = attention_mask.repeat(args.batch_size, 1, 1, 1).float()
    else:
        attention_mask_batch_inter = None

    traincast = nullcontext if args.deactive_amp else torch.amp.autocast
    loss_func = nn.MSELoss()
    cossim = nn.CosineSimilarity(dim=2)

    # SVD 因子 A/B（真权重容量，步长比格子参数小）
    factor_params = list(qlayer.factor_parameters()) if hasattr(qlayer, "factor_parameters") else []
    if factors_only:
        if not factor_params:
            raise ValueError("factors_only=True requires qlayer.factor_parameters().")
        param_groups = [{"params": factor_params, "lr": args.svd_fp_lr}]
    else:
        param_groups = [
            {"params": qlayer.rotation_parameters(), "lr": 1e-3},
            {"params": qlayer.scale_parameters(), "lr": 5e-2},
            {"params": qlayer.shift_parameters(), "lr": 5e-2},
            {"params": factor_params, "lr": 2e-4},
        ]
    optimizer = torch.optim.AdamW(param_groups)

    best_params_ref = [None]

    scheduler = SchedulerWithRestore(
        optimizer, qlayer,
        mode="min", factor=0.1, patience=2, verbose=True
    )

    best_loss = float("inf")
    best_optim_params = None
    batch_num = math.ceil(args.nsamples / args.batch_size)
    epochs = num_epochs if num_epochs is not None else args.epochs

    for epoch in range(epochs):
        # Shuffle indices at the start of each epoch
        indices = torch.randperm(args.nsamples)
        
        time_start = time.time() if hasattr(torch, 'cuda') else None
        loss_list = []
        attention_mask_batch = attention_mask_batch_inter

        for j in range(batch_num):
            index = j * args.batch_size
            if (j == batch_num - 1) and ((j + 1) * args.batch_size > args.nsamples):
                index_end = args.nsamples
                attention_mask_batch = (
                    attention_mask.repeat(args.nsamples - j * args.batch_size, 1, 1, 1)
                    if attention_mask is not None
                    else None
                )
            else:
                index_end = index + args.batch_size

            batch_indices = indices[index:index_end]

            with traincast("cuda"):
                optimizer.zero_grad()
                qlayer.binary_temporary(args.highorder) 
                quant_out = qlayer(quant_inps[batch_indices], attention_mask=attention_mask_batch, position_ids=position_ids)[0]

                if torch.isnan(quant_out).any():
                    raise RuntimeError(f"quant_out is NAN on epoch {epoch}, batch {j}")
            
                loss = loss_form(quant_out, fp_inps[batch_indices], quant_inps_fp[batch_indices])

            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Loss is NAN on epoch {epoch}, batch {j}")

            loss_list.append(loss.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                qlayer.parameters(),
                max_norm=2.0,
                norm_type=2.0
            )
            optimizer.step()

        torch.cuda.empty_cache()
        time_end = time.time() if hasattr(torch, 'cuda') else None
        loss_mean = torch.stack(loss_list).mean()

        if loss_mean < best_loss:
            best_loss = loss_mean
            best_optim_params = {
                name: param.data.clone().cpu()
                for name, param in qlayer.named_parameters()
                if any(keyword in name for keyword in ["free_line", "scale_factor", "shift_factor", "svd_A", "svd_B"])
            }
            scheduler.update_best_params(best_optim_params)

        scheduler.step(torch.tensor(loss_mean).to(next(qlayer.parameters()).device))

        if factors_only:
            print(
                f"layer {layer_idx} Stage-1 iter {epoch} loss:{loss_mean}, "
                f"lr_factor: {optimizer.param_groups[0]['lr']}, "
                f"time {time_end - time_start if time_start is not None else 'N/A'}",
                flush=True,
            )
        else:
            lr_rotation = optimizer.param_groups[0]["lr"]
            lr_scale = optimizer.param_groups[1]["lr"]
            lr_shift = optimizer.param_groups[2]["lr"]
            print(
                f"layer {layer_idx} iter {epoch} loss:{loss_mean}, lr_rotation: {lr_rotation}, lr_scale: {lr_scale}, lr_shift: {lr_shift}, time {time_end - time_start if time_start is not None else 'N/A'}",
                flush=True,
            )

        with torch.no_grad():
            if args.disable_gptq:
                qlayer.mask_generate_save(
                    highorder=args.highorder,
                    layer_idx=layer_idx,
                    disable_gptq=args.disable_gptq,
                    update=True,
                    epoch=epoch+1
                )

    del optimizer, scheduler
    return best_optim_params

def optimize_lora(
    qlayer,
    quant_inps: torch.Tensor,
    fp_inps: torch.Tensor,
    quant_inps_fp: torch.Tensor = None,
    steps: int = 8,
    lr: float = 1e-2,
    attention_mask: torch.Tensor = None,
    position_ids: torch.LongTensor = None,
    args = None,
    layer_idx: int = None,
    lora_parameters = None
    ):
    if steps <= 0:
        return
        
    if attention_mask is not None:
        attention_mask_batch_inter = attention_mask.repeat(args.batch_size, 1, 1, 1).float()
    else:
        attention_mask_batch_inter = None
    
    traincast = nullcontext if args.deactive_amp else torch.amp.autocast
    best_loss = float("inf")
    optimizer = torch.optim.Adam(lora_parameters, lr=lr)
    scheduler = SchedulerWithRestore(
        optimizer, qlayer,
        mode="min", factor=0.1, patience=5, verbose=True
    )


    batch_num = math.ceil(args.nsamples / args.batch_size)
    time_begin = 0
    for step in range(steps):
        # Shuffle indices at the start of each epoch
        indices = torch.randperm(args.nsamples)
        
   
        attention_mask_batch = attention_mask_batch_inter
        loss_list = []
        if time_begin == 0:
            time_start = time.time() if hasattr(torch, 'cuda') else None
            time_begin = 1
        for j in range(batch_num):
            index = j * args.batch_size
            if (j == batch_num - 1) and ((j + 1) * args.batch_size > args.nsamples):
                index_end = args.nsamples
                attention_mask_batch = (
                    attention_mask.repeat(args.nsamples - j * args.batch_size, 1, 1, 1)
                    if attention_mask is not None
                    else None
                )
            else:
                index_end = index + args.batch_size

            batch_indices = indices[index:index_end]

            with torch.enable_grad():
                with traincast("cuda"):
                    optimizer.zero_grad()
                    qlayer.lora_compensation()
                    quant_out = qlayer(quant_inps[batch_indices], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                    loss = loss_form(quant_out, fp_inps[batch_indices], quant_inps_fp[batch_indices])

                if not math.isfinite(loss.item()):
                    raise RuntimeError(f"Loss is NAN on epoch {step}, batch {j}")
                
                loss_list.append(loss.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_( # 5. 梯度裁剪
                    lora_parameters,
                    max_norm=10.0,
                    norm_type=2.0
                    )
                optimizer.step()

                torch.cuda.empty_cache()

        loss_mean = torch.stack(loss_list).mean()

        if step % 5 == 0 or step == steps - 1:
            time_end = time.time() if hasattr(torch, 'cuda') else None
            time_begin = 0
            print(
                f"LoRA compensation output fitting: Layer {layer_idx}, iteration:{step}, loss:{loss_mean.item()}, time:{time_end - time_start if time_start is not None else 'N/A'} ",
                flush=True,
            )
       

        if loss_mean < best_loss:
            best_loss = loss_mean
            best_lora_params = {}
            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    if module.lora_compensation is not None:
                        best_lora_params[name] = {
                            'A': module.lora_compensation.A.data.clone().cpu(),
                            'B': module.lora_compensation.B.data.clone().cpu(),
                            'a_shape': module.lora_compensation.a_shape,
                            'b_shape': module.lora_compensation.b_shape,
                        }
            scheduler.update_best_params(best_lora_params)

        scheduler.step(torch.tensor(loss_mean).to(next(qlayer.parameters()).device))

    del optimizer, scheduler
    return best_lora_params 

def loss_form(quant_out, fp_inps, quant_inps_fp):
    loss_func = torch.nn.MSELoss()
    cossim = nn.CosineSimilarity(dim=2)
    loss1 =  0.1*loss_func(fp_inps, quant_out) + 0.9*loss_func(quant_inps_fp, quant_out)
    cos1 = cossim(quant_out,fp_inps).mean().abs()
    cos2 = cossim(quant_inps_fp,quant_out).mean().abs()
    loss2 = -torch.log(cos1) -torch.log(cos2) 

    return 9*loss1 + 12*loss2
