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
        self.use_moe = kwargs.get("use_moe", False)
        self.n_routed_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 2)
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)
        self.scoring_func = kwargs.get("scoring_func","softmax")
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.seq_aux = kwargs.get("seq_aux", True)
        self.aux_loss_alpha = kwargs.get("aux_loss_alpha", 0.01)

import torch
import torch.nn as nn
## RMNorm:继承Moudle类
class RMSNorm(nn.Module):
## init初始化
    def __init__(self, dim:int,eps:float=1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.wight = nn.Parameter(torch.ones(dim))
#norm(token输入：(batch_size, seq_len, hidden_dim)，故mean维度为-1，即hidden_dim维度，keepdim=True保持维度不变，输出(batch_size, seq_len, 1)，再乘以权重weight，输出(batch_size, seq_len, hidden_dim)
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
# forward：最后保证为原始的类型
    def forward(self,x):
        return self.wight * x * self._norm(x.float()).type_as(x)
        

# RoPE
# 先写Yarn
from typing import Optional, Tuple, Union
import math
def precompute_freqs_cis(dim:int, end:int=int(32 * 1024), rope_base: float=1e-6, rope_scaling:Optional[dict] = None):
    #初始化PoPE频率
    freqs,attn_factor = (1/(rope_base**(torch.arange(0, dim, 2)[:dim//2].float()/dim)), 1.0)
    #配置（上述复制的类把超参数取出来）
    if rope_scaling is not None :
        orig_max, factor, beta_fast, beta_slow = (rope_scaling["original_max_position_embeddings"], 
        rope_scaling["factor"], 
        rope_scaling["beta_fast"], 
        rope_scaling["beta_slow"]
    )

        # 推断的长度大于训练长度，使用缩放
        if end > orig_max:
            # 求出波长b到i的映射
            inv_dim = lambda b:(dim*math.log(orig_max / (b*math.pi)))/(
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
    freqs = torch.outer(t, freqs).float()
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

# K和V的复制
def repeat_kv(x:torch.Tensor, n_rep:int)->torch.Tensor:
    # x的形状为(batch_size, seq_len, num_key_value_heads, head_dim)
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:,:,:,None,:].expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

from torch.nn import functional as F
from transformers.activations import ACT2FN
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0,"num_attention_heads must be divisible by num_key_value_heads"
        self.n_local_heads = args.num_attention_heads
        # self.num_key_value_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.head_dim

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads *self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads *self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads *self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads *self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.n_resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attn

    
    def forward(self, 
                x:torch.Tensor, 
                position_embedding: Tuple[torch.Tensor, torch.Tensor], 
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, #一个元组，有序，包含两个元素，分别是过去的键和值的张量,每个都是形状为[B, H, T_past, D]的张量
                use_cache: bool = False, 
                attention_mask: Optional[torch.Tensor] = None
                )->torch.Tensor:
        # x的形状为(batch_size, seq_len, hidden_dim)
    # 投影，计算Q,K,V
        bsz, seq_len, _ = x.shape
        xq,xk,xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    # 把输入拆分为多个头，使用view
        xq =xq.view(bsz,seq_len,self.n_local_heads, self.head_dim)
        xk =xk.view(bsz,seq_len,self.num_key_value_heads, self.head_dim)
        xv =xv.view(bsz,seq_len,self.num_key_value_heads, self.head_dim)
    # Q和K进行旋转位置编码RoPE
        cos,sin = position_embedding
        xq,xk = apply_rotary_pos_emb(xq,xk,cos[:seq_len],sin[:seq_len])
    # K和V进行复制，使得每个头都有对应的K和V
        if past_key_value is not None:
            xk = torch.cat((past_key_value[0], xk), dim=1)
            xv = torch.cat((past_key_value[1], xv), dim=1)
        past_key_value = (xk, xv) if use_cache else None #不推理一开始就不进，前面这个就为空，这里的use只是判断进不进

        xq,xk,xv =(
            xq.transpose(1,2), # (bsz, n_local_heads, seq_len, head_dim),记得交换一下注意力头数和序列长的维度，每个头都看的见整个序列的计算，下同
            repeat_kv(xk, self.n_rep).transpose(1,2), # (bsz, num_key_value_heads*n_rep, seq_len_kv, head_dim) ,
            repeat_kv(xv, self.n_rep).transpose(1,2)  # (bsz, num_key_value_heads*n_rep, seq_len_kv, head_dim)
        ) 
    # 计算注意力得分
        if self.flash and seq_len>1 and (attention_mask is None or torch.all(
            attention_mask == 1)):
            # 官方的高速实现
            attn_mask = (
                None 
                if attention_mask is None 
                else attention_mask.view(bsz, 1, 1, -1).expand(-1, self.n_local_heads, 
                                                               seq_len, -1).bool()
            )
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask = attn_mask, 
                                                    dropout_p=self.dropout if self.training else 0.0,is_causal=True)
        # if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
        #     output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:# 上述是官方的，直接复制，我们自己的走这里
            scores = (xq @ xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float('-inf'),device = scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
    
            # 扩展的mask：消除之前tokenr为了序列同长补充的padding影响
            if attention_mask is not None: 
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
    # 多个头拼接，输出投影，返回
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
        
        # [bsz,n_local_heads, seq_len, head_dim] -> [bsz, seq_len, n_local_heads*head_dim]
        output = output.transpose(1,2).reshape(bsz,seq_len,-1)
        output = self.n_resid_dropout(self.o_proj(output)) #最后的线性层不要忘记了
        return output, past_key_value
    
## FFN
class FeedForward(nn.Module):
    # 初始化
    # 升维
    # 降维
    # 门控
    # dropout
    # 激活函数
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8/3) #同等规模下SWiLU要求升维的维度系数
            args.intermediate_size = 64*((intermediate_size+64-1)//64) #为了效率，升维的维度最好是64的倍数
        
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        # 门控机制：与升维的相同，
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False) 
        self.dropout = nn.Dropout(args.dropout)
        # SWiGLU激活函数，输入是升维后的张量，输出也是升维后的张量
        self.act_fn = ACT2FN[args.hidden_act]

    # 前向传播：一个升维经过激活一个不经过
    def forward(self,x):
        return self.dropout(
            self.down_proj(self.act_fn(self.up_proj(x)) * self.gate_proj(x))
        )
    

from torch.nn import init    
# MOE Gates
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok #每个token选择几个专家
        self.n_routed_experts = config.n_routed_experts # 可路由专家总数，有几个候选专家

        self.scoring_func = config.scoring_func # 损失函数，这里后面只有softmax，把logits转化为概率
        self.alpha = config.aux_loss_alpha # loss = loss + α * aux_loss的α
        self.seq_aux = config.seq_aux # 是否按sequence做辅助loss

        self.norm_topk_prob = config.norm_topk_prob # Top-K后是否重新归一化概率,不然原先做的softmax取出topk后总和不为1
                                                    # 了，能量变小
        self.gating_dim = config.hidden_size    # Gate输入维度
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:  #何凯明（残差提出者）提出的初始化方法有利于模型训练
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h= hidden_states.shape
        hidden_states = hidden_states.view(-1,h) # [bsz, slen, hiddendim] ->[bsz*slen=num_tokens, hiddendim] 
        logits = F.linear(hidden_states, self.weight, None) # @W(h,n_experts),计算每个专家分数[bsz*slen=num_tokens, hiddendim] ->[num_tokens, n_experts]

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1) # 对各专家的分数变成概率，softmax不会改变形状，只会把“数值”变成概率分布
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )
        
        top_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False) # 取出最大的topk值和其索引，
                                                                                        # topk_idx:[1, 5]topk_weight:[0.7, 0.2]，
                                                                                        # 其大小都为[bsz * seq_len, top_k]

        if self.top_k > 1 and self.norm_topk_prob: # 取出topk需要重新归一归一化
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20 # 防止除以0下面
            topk_weight = topk_weight / denominator
        
        if self.training and self.alpha > 0.0: # 辅助损失aux_loss计算，想让所有专家都被使用，即token分布尽量均匀
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1) # 把所有token重新按batch组织回来
            if self.seq_aux: # sequence级均衡，先看下面的batch级，这里可能是为了句子A走专家A，句子B走专家B，整体均匀但是单个样本不均匀
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)# 恢复[batch, seq, experts]
                ce = torch.zeros( #ce同样是为了各个专家的概率
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_( # 统计每个sequence用了多少次各专家
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device), #seq_len*topk token数*topk设定数=每个seq总专家调用次数
                ).div_(seq_len * aux_topk / self.n_routed_experts) # div作用,归一化到均匀期望,每专家期望 总调用次数/候选专家数 次
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha # 每个sequence内部都做负载均衡，比batch更强
            else: # batch级均衡
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )   # 上面的展平后[[0,1],[1,2]] -> [0,1,1,2],我们又进行onehot变成:[
                    # [1,0,0,0],
                    # [0,1,0,0],
                    # [0,1,0,0],
                    # [0,0,1,0]
                    # ]
                ce = mask_ce.float().mean(0)    # 每个专家的真实使用频率，沿row维度平均，即按列
                Pi = scores_for_aux.mean(0)     # softmax后的完整概率，token1:[0.7,0.2,0.1,0]，token2:[0.1,0.6,0.2,0.1]
                                                # 平均[0.4,0.4,0.15,0.05]，Gate理论上想给各专家多少概率
                fi = ce * self.n_routed_experts #为了均匀分布时 fi≈1时更稳定
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss
    

class MoEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 专家层
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 门控层
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 展开x以便处理
        x = x.view(-1, x.shape[-1]) # 变成[num_token,h]

        flat_topk_idx = topk_idx.view(-1) # 展平
        if self.training:
            # 按照定义的num_experts_per_tok重复输入token
            # 每个token安排num_experts_per_tok个专家处理
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # y是空张量，和x形状相同
            y = torch.empty_like(x, dtype=x.dtype)
            # 遍历所有专家
            for i, expert in enumerate(self.experts):
                # 找到所有指向专家i的token
                # 然后将这些token输入专家i进行处理
                # 最后将结果放回y对应位置
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                        )
            # 加权求和
            # 最后的y意义是每个token经过专家处理后的加权结果
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)

        # 如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    # MoE推理方法
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache




# 好了，现在已经完成了想要的模块，直接把其拼成一个block就行
class MokioMindBlock(nn.Module):
    def __init__(self, layer_id: int, args: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.head_dim = args.head_dim
        self.self_attn = Attention(args) #注意力模块

        self.layer_id = layer_id # MOE会用到
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = FeedForward(args)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, 
                use_cache=False, attention_mask=None):
        # 注意力模块
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = residual + hidden_states # 残差连接
        # FFN模块
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
            )
        return hidden_states, present_key_value
    
# 最后把block堆起来，形成模型主体
class MokioMindModel(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.config = args
        self.vocab_size, self.num_hidden_layers = (#字符表大小，Transformer block层数
            args.vocab_size,
            args.num_hidden_layers
        ) 
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size) #前面的linear映射到向量

        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList( # 对应图中的Transformer重复k层
            [MokioMindBlock(i, args) for i in range(args.num_hidden_layers)]
        )

        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps) # k transformer layer后后的RMSNorm

        # RoPE预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim = args.head_dim, 
            end = args.max_position_embeddings,
            rope_base = args.rope_theta,
            rope_scaling = args.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos,persistent=False) #persistent=False表示这个buffer不会被保存到模型的state_dict中，也就是说在保存和加载模型时，这个buffer不会被包含在内。这通常用于那些可以在运行时动态计算或重建的值，比如位置编码等。
        self.register_buffer("freqs_sin", freqs_sin,persistent=False)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            **kwargs, # 其他补充的参数
    ):
        batch_size, seq_len = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None # 兼容之前的版本，之前的版本是个对象，现在改成了一个元组

        past_key_values = past_key_values or [None] * len(self.layers)
        
        start_pos= (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(
            self.embed_tokens(input_ids) #输入的token id映射到向量
        )
        
        position_embeddings = (self.freqs_cos[start_pos: start_pos + seq_len],
                               self.freqs_sin[start_pos: start_pos + seq_len]) #根据输入序列的长度，截取对应位置的RoPE编码
        
        presents =[]
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
            ):
            hidden_states, present = layer( #就是一层transformer block的前向传播，输入是上一层的输出，位置编码，过去的K和V，是否使用缓存，注意力掩码，输出是当前层的输出和当前层的K和V
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask
            )
            presents.append(present) #把每层的K和V都保存下来，最后一起返回
    
        hidden_states =self.norm(hidden_states) #最后的RMSNorm
        ##剩下的linear，softmax和toknizer解码器先不做,放到下面做

        # MOE部分补充
        aux_loss = sum(
            [
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(
                    layer.mlp, MoEFeedForward
                )  # ！修正：原MoEFeedForaward拼写错误
            ],
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss
    

#封装成一个更高层的接口，方便后续添加语言模型头或者其他任务的头,即inear和foftmax层
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast  
class mokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    def __init__(self, config: MiniMindConfig):
        self.config = config

        super().__init__(config) #必须在上一句之后，因为这个父类定义一个confi需要我们自己定义一个config信息
        self.model = MokioMindModel(config) #实例化，config传入

        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False) #语言模型头，输入是transformer的输出，输出是每个token的概率分布
        #于是我们512维的隐藏层能够映射安东6400多个词的词表上，表示出每个词的概率

        self.model.embed_tokens.weight = self.lm_head.weight #权重共享，输入的embedding和输出的lm_head共享权重，计算更加简单

        # self.OUT = CausalLMOutputWithPast() #这个是transformers库里定义的一个输出类，包含了语言模型输出需要的几个字段，比如logits和past_key_values等

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            logits_to_keep: Union [int, torch.Tensor] = 0, # 这个参数是为了支持只返回前k个token的logits，减少计算量和内存占用，-1表示返回全部
            labels=None,
            **args, # 其他补充的参数
    ):
        hidden_states, past_key_values, aux_loss= self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            use_cache = use_cache,
            **args,
        )
        #如果我们的logits_to_keep是一个整数，表示只保留前k个token的logits，那么我们就把hidden_states的最后一个维度切片，只保留前k个token的logits
        # 作用：生成时只需要最后的logits来预测下一个token
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep,int)
            else logits_to_keep #如果不是int类型而是一个tensor，那么等于0，表示不切片，保留所有位置返回全部logits
            )
        logits = self.lm_head(hidden_states[:,slice_indices,:])

        #得自己计算loss在模型中
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output

