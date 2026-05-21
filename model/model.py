from transformers import PretrainedConfig

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

import torch
import torch.nn as nn
## RMNorm:继承Moudle类
class RMSNorm(nn.Moudle):
## init初始化
    def __init__(self, dim:int,eps:float=1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.wight = nn.parameter(torch.ones(dim))
#norm(token输入：(batch_size, seq_len, hidden_dim)，故mean维度为-1，即hidden_dim维度，keepdim=True保持维度不变，输出(batch_size, seq_len, 1)，再乘以权重weight，输出(batch_size, seq_len, hidden_dim)
    def _norm(self,x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
# forward：最后保证为原始的类型
    def forward(self,x):
        return self.wight * x * self._norm(x.float()).type_as(x)
        

# RoPE
# 先写Yarn
from typing import Optional
def precompute_freqs_cis(dim:int, end:int=int(32 * 1024), rope_base: float=1e-6, rope_scaling:Optional[dict] = None):
    #初始化PoPE频率
    freqs,attn_factor = (1/(rope_base**(torch.arange(0, dim, 2)[:dim//2].float()/dim)), 1.0)
    #配置（上述复制的类把超参数取出来）
    if rope_scaling is not None :
        orig_max, factor, beta_fast, betas_slow = (rope_scaling["original_max_position_embeddings"], 
        rope_scaling["factor"], 
        rope_scaling["beta_fast"], 
        rope_scaling["beta_slow"]
    )

    # 推断的长度大于训练长度，使用缩放
    if end > orig_max:
        # 求出波长b到i的映射
        inv_dim = lambda b:(dim*match.log(orig_max / (b*math.pi)))/(
            2*math.log(rope_base)
        )
        # 划分高低维度（频）,low不需要缩放的部分，high需要
        low, high =(max(math.floor(inv_dim(beta_fast)), 0),
                     min(math.ceil(inv_dim(beta_slow)), dim//2-1))
        # 计算缩放因子，low之前因子ramp为0，之后为1，在之间线性过渡
        ramp = torch.clamp(
            (torch.arange(dim//2, device=freqs.device).float() - low) 
            / max(high - low, 1),
            0, 
            1)
        
        # ramp=0（高频），系数为1，保持不变
        # ramp=1（低频），系数为1/factor，即对频率进行线性插值缩放
        # ramp在0和1之间，平滑过渡
        freaqs = freqs * (1 - ramp + ramp * factor)
    
    # 根据end,生成位置索引
    t = torch.arange(end, device=freqs.device).float()

    # 计算外积，将t与频率相乘，得到每个位置的旋转角度
    freqs = torch.outer(t, freqs).flaot()
    freq_cos =(
        torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1)*attn_factor
    )
    freq_sin =(
        torch.cat((torch.sin(freqs), -torch.sin(freqs)), dim=-1)*attn_factor
    )
    return freq_cos, freq_sin

# 编写RoPE
def apply_rotary_pos_emb(q, k, cos, sin, position_ids = None, unsqueze_dim =1):
    # [a,b]->[-b,a]
    def rotate_half(x):
        # shape[-1]取最后一个维度的重点
        # -x[..., x.shape[-1]//2:取出X的后半部分，x[..., :x.shape[-1]//2]取出前半部分
        return torch.cat(
            (-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), 
            dim=-1)
    # 计算旋转位置编码，x_rotated = (x * cos) + (rotate_half(x) * sin)，其中x是q或k
    q_emded = (q * cos.unsqueeze(unsqueze_dim)) +(
        rotate_half(q) * sin.unsqueeze(unsqueze_dim))
    k_emded = (k * cos.unsqueeze(unsqueze_dim)) +(
        rotate_half(k) * sin.unsqueeze(unsqueze_dim))
    return q_emded, k_emded