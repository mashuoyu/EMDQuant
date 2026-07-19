import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from quant.quantizer import UniformAffineQuantizer
from utils.lora import LoRAErrorCompensation



class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
        lora_rank: int = None,
        lora_a_shape=None,
        lora_b_shape=None,
        lora_init_scale: float = 1e-3,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
    
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.use_temporary_lora_parameter = False
        self.register_buffer('H', torch.zeros((self.weight.shape[1], self.weight.shape[1])))
        self.nsamples = 0
        self.weight_mask = None
        self.lora_compensation = None
        self.lora_rank = lora_rank
        self.lora_a_shape = lora_a_shape
        self.lora_b_shape = lora_b_shape
        self.lora_init_scale = lora_init_scale
        self.use_lora = lora_rank is not None or lora_a_shape is not None or lora_b_shape is not None


    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_temporary_lora_parameter:
            weight = self.lora_compensation(self.lora_weight)


        #if self.use_act_quant and not self.disable_input_quant:
            #input = self.act_quantizer(input)
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def init_lora_compensation(
        self,
        rank: int = None,
        a_shape=None,
        b_shape=None,
        device=None,
        init_scale: float = 1e-3
    ) -> LoRAErrorCompensation:
        """Create a low-rank error compensation module for this weight."""
        init_device = torch.device(device) if device is not None else self.weight.device
        self.lora_compensation = LoRAErrorCompensation(
            self.weight.shape,
            a_shape=a_shape,
            b_shape=b_shape,
            rank=rank,
            device=init_device,
            init_scale=init_scale,
        )
        return self.lora_compensation

    def save_lora_state(self) -> dict:
        if self.lora_compensation is None:
            return None
        return self.lora_compensation.save_state_dict()

    def load_lora_state(self, state: dict, device=None):
        if state is None:
            return
        load_device = torch.device(device) if device is not None else self.weight.device
        if self.lora_compensation is None:
            self.init_lora_compensation(
                rank=self.lora_rank,
                a_shape=state.get('a_shape', self.lora_a_shape),
                b_shape=state.get('b_shape', self.lora_b_shape),
                device=load_device,
                init_scale=self.lora_init_scale,
            )
        self.lora_compensation.A.data.copy_(state['A'].to(load_device))
        self.lora_compensation.B.data.copy_(state['B'].to(load_device))

    def add_batch(self, inp, dev):
        inp = inp
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        mult_inp = inp.matmul(inp.t())
        self.H += mult_inp
    
    def optim_quant(self, x):
        return x
    