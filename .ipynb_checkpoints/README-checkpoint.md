# Note

## RmsNorm
用于归一化神经网络中某一层的输出，使其数值保持稳定.同时计算复杂度小
token(x)输入：(batch_size, seq_len, hidden_dim)，故mean维度为-1，即hidden_dim维度，keepdim=True保持维度不变，输出(batch_size, seq_len, 1),而不是变成unsqueeze()了，再乘以权重weight，输出(batch_size, seq_len, hidden_dim)

## RoPE旋转位置编码
绝对位置编码，如果“A...是x”如果前面加上一段“...A...是x”绝对位置就变了，太呆板
相对位置编码：RoPE最常见，cos编码
感兴趣可以从3角不等式看看使用RoPE的数学原因：把Q[q1,q2]与K[k1,k2]旋转m*θ和n*θ角度，再点积，由3角恒等式得到最终结果因为Q和K是原始值，位置信息只由cos[(m-n)θ]和sin[(m-n)θ]决定，即只和m-n有关
使用时，假设初始“cat”"sat"的位置分别为2和3每个Token有自己的向量列，我们会把列中的多个向量两两分为1组进行旋转，每个向量都有自己的Q和K，第一组旋转θ1角度，第二组旋转θ2角度...点积了之后就变成(2-3)θ1,(2-3)θ2...相对位置信息就隐藏再这里

## YaRNd对RoPE的外推，优化
“外推”是处理超出计算长度的意思
原始的RoPE原始频率在i越大越小，那这样就会：比如说我们的模型是在2048的序列长度进行计算的，那再80亿的一个模型中，都是按照这样的一个长度来理解，如果哪天，给模型塞入一个4000长度，模型会疯掉，因为它处理2048长度的，80亿参数与4000长的匹配不上。
原始方法将4096长压缩到2048（比如乘以一个0.5的系数），但这样会损失一定信息
YaRN对高低频使用不同的处理（将注意力视为一个钟表，高频转的块，能覆盖360度的信息，比如处理2048这样0-6这样一个对数，都落在某个扇形，全在圆内。低频转的慢可能不能覆盖要求的扇形，比其小的扇形）：高频不缩放，保持原样。比如0-6的部分，对低频使用一个线性缩放，对于中频，使用线性插值平滑过渡。**高频对信息精细度的把握,低频是对全局信息的把握**
还引入了一个温度系数，softmax分配注意力机制（20个学生一开始比较关注5个学生，后面塞入3000个学生，还想比较关注这5个学生我的注意力会被分散，因为还要考虑其他3000个）会导致这5个学生的注意力会被稀释，我们加入这个系数再RoPE计算之后传入这个系数让这5个学生依然保持高注意力，其他部分缩放。
使用的话，要找到高低频的界限：2pai*base^(2i/d)
- torch.where(条件,x,y): y为原始输入张量，x张量中符合条件的才通过，并保持原位置，替换掉y中的相应位置元素
- torch.arange(起点，终点，步差):生成等差序列
- torch.outer(x1,x2):x1和x2的外积
- torch.cat(x1,x2,dim=0):在第0个维度进行拼接：size(223->423),如果dim=1则是百年未423的形状
- torch.unsqueeze:自适应增加一个维度
- torch.clamp(x,min,max):计算x中在min和max之间的

## GQA
具体可看3B1B的视频
- torch.tensor.dropout(p=0.x):以多少概率丢弃，防止过拟合
- torch.tensor.liner(x):线性变换，就是乘以ige矩阵
- torch.tensor.view(x):改变形状 [3,4]->[4,3]，但是共享一个物理层
- torch.tensor.transpose(a,b):交换a维和b维(size:233--0,1-->323)
- torch.tensor.triu(x,diagonal=S):只有x时，从对角线下半为0，掩码，diagonal为正数上移S，为负数时下移
- torch.tensor.reshape(x，(...)):类似view(),但是底层不一样,括号里面的维度可以使用-1自动推断减去一个维度
这里的hasattr时查看类是否有什么属性

## FFN
简单的几个线性层而已

## 拼接为Block
- 注意进入前都要使用一个RMSNorm做归一化，自注意力的已经画在图里，但FFN的输入前也要，这里的post是attention“之后”的意思
- 记得残差连接
- FFN在跨层功能不应该一样，用layerid,同时也可以额外MOE使用

## 模型主体model组装
- RoPE预计算：
- forward时，input_ids是经过toker Encorder之后，进入input embedding之前的，给每个单词都分配了一个id,"I love NLP"→ [101, 234, 987],而 input embedding模型有一个矩阵：table W ∈ R^{V × H}，把token_id → W[token_id]，比如234 → embedding vector (size H)。为什么不能直接用 id 做输入？因为：id 是离散的，Transformer 需要连续空间（vector space）
- 之所以要在past_key_value里查找“layer”属性，是因为兼容之前的版本，之前的版本是个对象，现在改成了一个元组
- 记得input embedding转化为向量时也要转化位置信息
- 位置编码position_embeddings之所以是 “整个 seq_len 的一段”，是因为 attention 需要知道每个 token 的绝对位置（或相对旋转角度）,第一次：start_pos = 0,seq_len = → positions: [0,1,2]第二次生成:start_pos = 3,seq_len = 1→ positions: [3],KV cache 让序列变长，但 position 不能重置,position embeddings 不是处理单个 token，而是处理 seq_len 个 token，整个序列中所有 token 的“坐标系”每个 token 都必须有对应的位置编码（RoPE），并且 KV cache 机制要求位置是连续累加的。
- 至于start_pos = past_key_values[0][0].shape[1]?回顾K_cache的维度[B, T_past，H(头数),  D],于是变成T_past，即之前的token数

## 封装，与官方类对应，标准化
- PretrainedModel, GenerationMixin是huggingface的两个标准类，网上的都要继承它两，前者定义模型管理配置，第二个文本生成方法
- 为什么不在 forward 里做 softmax？mokioMindForCausalLM其实只做了一件事lm_head 是：nn.Linear(hidden_size → vocab_size)。输出：logits（未归一化分数）: [B, T, V]所以 softmax 去哪了？在 HuggingFace 体系里：
**情况 A**：训练阶段（最常见）loss 是这样算的：loss = CrossEntropyLoss(logits, labels)而 CrossEntropyLoss 内部做了：log_softmax + NLLLoss 👉 等价于：softmax(logits)但被融合进 loss 里了✔ 所以 softmax 在这里：❗CrossEntropyLoss 内部
**情况 B**：生成（generate）你继承了：GenerationMixin生成时流程是：logits → softmax → sampling/argmax → next token但注意：softmax 在 sampling 函数内部，而不是 model.forward
- 实际上没有完成tokenizer decoder


## Dataset
- 给PAD进行labels防止参与计算时：我们一般因为后面crossloss会自动忽略-100，所以给label=-100。即转化为ids后[1,2,3,PAD] -->[1,2,3,-100]
- 虽然我们时自回归，也就是上述的传入模型的顺序时1，2，3...，为什么时clone下来而不是平移1位呢？因为前面model的部分已经给我们内置了自回归部分
- 虽然input_ids、attention_mask、labels 3者似乎都来自input_ids只是PAD的位置修改了一下，为什么要传3种？但是其实用法不一样：
| 名字               | 给谁用       | 作用                 | 示例   |
| ---------------- | --------- | ------------------ | ---------------|
| `input_ids`      | 模型输入      | 告诉模型“看到什么”         |[BOS, 我, 爱, NLP, EOS, PAD, PAD]|
| `attention_mask` | Attention | 告诉模型“哪些 token 可以看” |[1 1 1 0 0] |
| `labels`         | Loss      | 告诉模型“哪些位置需要算loss”  |为了去预测的监督信号，不是输入|
对于最后一个labels,用法：
| 输入  | 目标labels |
| --- | -------- |
| BOS | 我        |
| 我   | 爱        |
| 爱   | NLP      |
| NLP | EOS      |
PAD 不应该参与 loss故取-100。

## 预训练
- 动态学习率：lr*(0.1 + 0.45*(1+cos(PI*CS/TS)))  #最大为1，最小为0.1，逐渐下降
- 从dataset处获取input_ids、attention_mask、labels
- 向前传播：计算LOSS，反向传播，梯度下降
- 梯度累计：显存时有限的，无法大batch加载，把小的batch加载进来，如把1280的分为160的一次一次处理，每次的梯度累加，最后/8
- 混合精度:32位和16位，使用混合精度的下降,加个scaler放大器:前向传播没事，但 fp16 梯度可能太小（1e-10），变量表示数值范围不够，会溢出，于是反向传播时缩放（*一个炒鸡大的数）fp16 能表示
- 记得datasets是huggingface一个库要安装，跟我们的dataset模块是区分开