# 导入必要的库
import copy  # 深拷贝工具
import math  # 数学运算工具
import torch  # PyTorch库
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式工具

# 定义LayerNorm层
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        初始化LayerNorm层。
        :param hidden_size: 输入隐藏层的大小
        :param eps: 防止除零的小值，默认1e-12
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 初始化可学习的权重为1
        self.bias = nn.Parameter(torch.zeros(hidden_size))  # 初始化可学习的偏置为0
        self.variance_epsilon = eps  # 防止除零的小值

    def forward(self, x):
        """
        前向传播函数。
        :param x: 输入张量
        :return: 归一化后的张量
        """
        u = x.mean(-1, keepdim=True)  # 按最后一个维度计算均值
        s = (x - u).pow(2).mean(-1, keepdim=True)  # 按最后一个维度计算方差
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)  # 归一化
        return self.weight * x + self.bias  # 加权并加上偏置

# 定义FeedForward前馈网络
class FeedForward(nn.Module):
    def __init__(self, args):
        """
        初始化前馈网络。
        :param args: 参数配置对象
        """
        super(FeedForward, self).__init__()
        hidden_size = args.hidden_size  # 隐藏层大小
        inner_size = 4 * args.hidden_size  # 内部隐藏层大小，通常是输入的4倍

        self.dense_1 = nn.Linear(hidden_size, inner_size)  # 第一层全连接
        self.intermediate_act_fn = self.get_hidden_act(args.hidden_act)  # 激活函数
        self.dense_2 = nn.Linear(inner_size, hidden_size)  # 第二层全连接
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)  # LayerNorm层
        self.dropout = nn.Dropout(args.hidden_dropout_prob)  # Dropout层

    def get_hidden_act(self, act):
        """
        获取指定的激活函数。
        :param act: 激活函数名称
        :return: 激活函数
        """
        ACT2FN = {
            "gelu": self.gelu,  # GELU激活
            "relu": F.relu,  # ReLU激活
            "swish": self.swish,  # Swish激活
            "tanh": torch.tanh,  # Tanh激活
            "sigmoid": torch.sigmoid,  # Sigmoid激活
        }
        return ACT2FN[act]

    def gelu(self, x):
        """
        GELU激活函数。
        :param x: 输入张量
        :return: 激活后的张量
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        """
        Swish激活函数。
        :param x: 输入张量
        :return: 激活后的张量
        """
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        """
        前向传播函数。
        :param input_tensor: 输入张量
        :return: 输出张量
        """
        hidden_states = self.dense_1(input_tensor)  # 第一层全连接
        hidden_states = self.intermediate_act_fn(hidden_states)  # 激活函数
        hidden_states = self.dense_2(hidden_states)  # 第二层全连接
        hidden_states = self.dropout(hidden_states)  # 应用Dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 残差连接 + 归一化
        return hidden_states

#######################
## Basic Transformer ##
#######################

# 定义多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        """
        初始化多头注意力机制。
        :param args: 参数配置对象
        """
        super(MultiHeadAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError("隐藏层大小必须是注意力头数量的整数倍。")
        self.args = args  # 保存参数
        self.num_attention_heads = args.num_attention_heads  # 注意力头数量
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)  # 每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 所有头的总大小
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)  # 归一化因子

        self.query = nn.Linear(args.hidden_size, self.all_head_size)  # Query层
        self.key = nn.Linear(args.hidden_size, self.all_head_size)  # Key层
        self.value = nn.Linear(args.hidden_size, self.all_head_size)  # Value层

        self.softmax = nn.Softmax(dim=-1)  # Softmax函数
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)  # 注意力Dropout
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)  # 输出全连接层
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)  # LayerNorm层
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)  # 输出Dropout

    def transpose_for_scores(self, x):
        """
        转置张量以适配多头注意力机制。
        :param x: 输入张量
        :return: 转置后的张量
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # 调整形状
        x = x.view(*new_x_shape)  # 改变张量形状
        return x

    def forward(self, input_tensor, attention_mask):
        """
        多头注意力的前向传播。
        :param input_tensor: 输入张量
        :param attention_mask: 注意力掩码
        :return: 输出张量
        """
        mixed_query_layer = self.query(input_tensor)  # 计算Query
        mixed_key_layer = self.key(input_tensor)  # 计算Key
        mixed_value_layer = self.value(input_tensor)  # 计算Value

        # 调整形状以适配多头注意力机制
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size  # 缩放
        attention_scores = attention_scores + attention_mask  # 应用掩码

        # 将注意力分数转为概率
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)  # 应用Dropout

        # 计算上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 应用输出层和残差连接
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 残差连接 + LayerNorm
        return hidden_states

# 定义Transformer块，包含多头注意力和前馈网络
class TransformerBlock(nn.Module):
    def __init__(self, args):
        """
        初始化Transformer块。
        :param args: 参数配置对象，包含隐藏层大小、注意力头数量等参数。
        """
        super(TransformerBlock, self).__init__()
        self.layer = MultiHeadAttention(args)  # 定义多头注意力层
        self.feed_forward = FeedForward(args)  # 定义前馈网络层

    def forward(self, hidden_states, attention_mask):
        """
        Transformer块的前向传播。
        :param hidden_states: 输入的隐藏状态张量 [batch_size, seq_length, hidden_size]
        :param attention_mask: 注意力掩码张量
        :return: 前馈网络的输出 [batch_size, seq_length, hidden_size]
        """
        layer_output = self.layer(hidden_states, attention_mask)  # 通过多头注意力层
        feedforward_output = self.feed_forward(layer_output)  # 通过前馈网络
        return feedforward_output  # 返回输出

# 定义Transformer编码器，包含多个Transformer块
class TransformerEncoder(nn.Module):
    def __init__(self, args):
        """
        初始化Transformer编码器。
        :param args: 参数配置对象，包含隐藏层大小、注意力头数量、层数等参数。
        """
        super(TransformerEncoder, self).__init__()
        self.args = args  # 保存参数配置
        block = TransformerBlock(args)  # 定义一个Transformer块
        # 使用深拷贝创建多个Transformer块
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        """
        Transformer编码器的前向传播。
        :param hidden_states: 输入的隐藏状态张量 [batch_size, seq_length, hidden_size]
        :param attention_mask: 注意力掩码张量 [batch_size, seq_length, seq_length]
        :param output_all_encoded_layers: 是否输出每一层的编码结果，默认为False
        :return: 编码器的所有层输出（如果output_all_encoded_layers=True）或最后一层输出
        """
        all_encoder_layers = [hidden_states]  # 保存所有层的编码输出，初始值为输入状态

        for layer_module in self.blocks:  # 遍历每个Transformer块
            hidden_states = layer_module(hidden_states, attention_mask)  # 通过Transformer块
            if output_all_encoded_layers:  # 如果需要输出所有层
                all_encoder_layers.append(hidden_states)  # 保存当前层的输出
        if not output_all_encoded_layers:  # 如果只需要最后一层
            all_encoder_layers.append(hidden_states)  # 保存最后一层的输出

        return all_encoder_layers  # 返回所有编码层的输出



# 定义一个获取1D卷积层的函数
def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    # 返回一个Conv1d层，使用指定的参数初始化
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                     kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, 
                     groups=groups, bias=bias)

# 定义一个获取Batch Normalization层的函数
def get_bn(channels):
    # 返回一个BatchNorm1d层，通道数为channels
    return nn.BatchNorm1d(channels)

# 定义一个组合卷积和BatchNorm的函数
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    # 如果没有提供padding，默认使用kernel_size的一半作为padding
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()  # 初始化一个顺序容器
    # 添加卷积层到result
    result.add_module('conv', get_conv1d(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       kernel_size=kernel_size,
                                       stride=stride, 
                                       padding=padding, 
                                       dilation=dilation, 
                                       groups=groups, 
                                       bias=bias))
    # 添加Batch Normalization层到result
    result.add_module('bn', get_bn(out_channels))
    return result  # 返回组合的卷积和BatchNorm模块

# 定义一个ReparamLargeKernelConv类，用于实现重参数化的大核卷积层
class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()  # 调用父类构造函数
        self.kernel_size = kernel_size  # 设置卷积核大小
        self.small_kernel = small_kernel  # 设置小卷积核的大小
        padding = kernel_size // 2  # 计算padding，确保卷积结果的尺寸不变
        
        if small_kernel_merged:
            # 如果小卷积核和大卷积核合并，则使用一个卷积层
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=kernel_size,
                                        stride=stride, 
                                        padding=padding, 
                                        dilation=1, 
                                        groups=groups, 
                                        bias=True)
        else:
            # 否则，先用大卷积核进行卷积并加BatchNorm
            self.lkb_origin = conv_bn(in_channels=in_channels, 
                                    out_channels=out_channels, 
                                    kernel_size=kernel_size,
                                    stride=stride, 
                                    padding=padding, 
                                    dilation=1, 
                                    groups=groups,
                                    bias=False)
            if small_kernel is not None:
                # 确保小卷积核大小不大于大卷积核
                assert small_kernel <= kernel_size, 'Small kernel size must be <= large kernel size!'
                # 如果有小卷积核，则添加小卷积核的卷积和BatchNorm层
                self.small_conv = conv_bn(in_channels=in_channels, 
                                        out_channels=out_channels,
                                        kernel_size=small_kernel,
                                        stride=stride, 
                                        padding=small_kernel // 2, 
                                        groups=groups, 
                                        dilation=1,
                                        bias=False)

    def forward(self, x):
        if hasattr(self, 'lkb_reparam'):
            # 如果使用重参数化的卷积核，直接计算输出
            out = self.lkb_reparam(x)
        else:
            # 否则，先计算大卷积核的输出
            out = self.lkb_origin(x)
            if hasattr(self, 'small_conv'):
                # 如果有小卷积核，则将小卷积核的输出加到结果中
                out += self.small_conv(x)
        return out  # 返回输出

class ModernTCNBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, drop=0.1):
        super(ModernTCNBlock, self).__init__()
        
        # 使用ReparamLargeKernelConv替换原来的普通深度卷积
        self.dw_conv = ReparamLargeKernelConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            groups=hidden_size,  # 保持深度卷积特性
            small_kernel=kernel_size // 3,  # 设置小卷积核大小
            small_kernel_merged=False  # 训练时不合并
        )
        
        # 第一层FFN保持不变：时间序列混合
        self.ffn1 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 4, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(hidden_size * 4, hidden_size, 1),
            nn.Dropout(drop)
        )
        
        # 第二层FFN保持不变：通道混合
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(drop)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        residual = x

        # 时间卷积部分
        x = x.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        x = self.dw_conv(x)    # 使用ReparamLargeKernelConv
        x = x.transpose(1, 2)
        x = self.norm1(x + residual)

        # FFN1：时间序列混合
        residual = x
        x = x.transpose(1, 2)
        x = self.ffn1(x)
        x = x.transpose(1, 2)
        x = self.norm2(x + residual)

        # FFN2：通道混合
        residual = x
        x = self.ffn2(x)
        x = self.norm3(x + residual)

        return x


class ModernTCN(nn.Module):
    def __init__(self, hidden_size, num_layers, kernel_size, dropout=0.1):
        super(ModernTCN, self).__init__()
        
        self.blocks = nn.ModuleList([
            ModernTCNBlock(
                hidden_size=hidden_size,
                kernel_size=kernel_size,
                drop=dropout
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
    
    def merge_kernels(self):
        """在训练后调用此函数合并卷积核"""
        for block in self.blocks:
            block.dw_conv.merge_kernel()
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class STMIXBlock(nn.Module):
    def __init__(self, args):
        super(STMIXBlock, self).__init__()
        
        kernel_size = min(15, args.max_seq_length // 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.modern_tcn = ModernTCN(
            hidden_size=args.hidden_size,
            num_layers=args.num_hidden_layers,
            kernel_size=kernel_size,
            dropout=args.hidden_dropout_prob
        )
    
    def merge_modernTCN(self):
        """训练后调用此函数进行参数合并"""
        self.modern_tcn.merge_kernels()
        
    def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=False):
        tcn_output = self.modern_tcn(hidden_states)
        
        if output_all_encoded_layers:
            return [hidden_states, tcn_output]
        return tcn_output