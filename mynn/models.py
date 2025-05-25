# mynn/models.py

import torch
import torch.nn as nn
# 导入新的配置系统
from . import config as mynn_config  # mynn_config 在此文件中主要被 DynamicCNN 使用


def get_activation_function(name="ReLU", **kwargs):
    """ 根据名称获取激活函数实例 """
    name_lower = name.lower()
    if name_lower == "relu":
        return nn.ReLU(**kwargs)
    elif name_lower == "leakyrelu":
        return nn.LeakyReLU(**kwargs)
    elif name_lower == "gelu":
        return nn.GELU()
    elif name_lower == "sigmoid":
        return nn.Sigmoid()
    elif name_lower == "tanh":
        return nn.Tanh()
    else:
        try:
            activation_module = getattr(nn, name)
            return activation_module(**kwargs)
        except AttributeError:
            raise ValueError(f"不支持的激活函数: {name}")


class DynamicCNN(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_classes=10, architecture_config=None, exp_config=None):
        super(DynamicCNN, self).__init__()
        if architecture_config is None:
            raise ValueError("DynamicCNN 必须提供 architecture_config")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture_config = architecture_config
        self.exp_config = exp_config if exp_config else {}

        self.features_sequence = self._create_layers()

    @staticmethod
    def _get_spatial_output_size(in_size_hw, kernel_size, stride=1, padding=0, dilation=1):
        h_in, w_in = in_size_hw
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride = (stride, stride) if isinstance(stride, int) else stride
        padding = (padding, padding) if isinstance(padding, int) else padding
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        w_out = (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
        return h_out, w_out

    def _create_layers(self):
        layers = []
        current_channels = self.input_shape[0]
        current_h, current_w = self.input_shape[1], self.input_shape[2]
        is_flattened = False

        last_fc_idx = -1
        for idx, layer_def in reversed(list(enumerate(self.architecture_config))):
            if layer_def['type'].lower() == 'fc':
                last_fc_idx = idx
                break

        for i, layer_def in enumerate(self.architecture_config):
            layer_type = layer_def['type'].lower()
            params = layer_def.get('params', {}).copy()

            if layer_type == 'conv':
                params['in_channels'] = current_channels
                if 'out_channels' not in params:
                    raise ValueError(f"卷积层索引 {i} 缺少 'out_channels'。参数: {params}")
                layers.append(nn.Conv2d(**params))
                current_channels = params['out_channels']
                current_h, current_w = self._get_spatial_output_size(
                    (current_h, current_w), params['kernel_size'],
                    params.get('stride', 1), params.get('padding', 0), params.get('dilation', 1)
                )
            elif layer_type == 'bn':
                if 'num_features' not in params:  # For BatchNorm2d
                    if not is_flattened:  # typically BatchNorm2d before flatten
                        params['num_features'] = current_channels
                    else:  # BatchNorm1d after flatten
                        params['num_features'] = current_channels  # in_features of FC

                # Decide between BatchNorm1d and BatchNorm2d based on context
                if not is_flattened:
                    layers.append(nn.BatchNorm2d(**params))
                else:  # after flatten, use BatchNorm1d
                    # BatchNorm1d expects num_features as the size of the input
                    # (C from an NCHW tensor or L from an NL tensor)
                    # params['num_features'] should be current_channels (which is the flattened dim)
                    layers.append(nn.BatchNorm1d(**params))

            elif layer_type in ['relu', 'leakyrelu', 'gelu', 'sigmoid', 'tanh'] or hasattr(nn, layer_type.upper()):
                act_name = layer_type
                if 'name' in params:
                    act_name = params.pop('name')
                layers.append(get_activation_function(act_name, **params))
            elif layer_type == 'pool':
                layers.append(nn.MaxPool2d(**params))
                current_h, current_w = self._get_spatial_output_size(
                    (current_h, current_w), params['kernel_size'],
                    params.get('stride', params['kernel_size']), params.get('padding', 0)
                )
            elif layer_type == 'avgpool':
                layers.append(nn.AvgPool2d(**params))
                current_h, current_w = self._get_spatial_output_size(
                    (current_h, current_w), params['kernel_size'],
                    params.get('stride', params['kernel_size']), params.get('padding', 0)
                )
            elif layer_type == 'adaptiveavgpool':
                layers.append(nn.AdaptiveAvgPool2d(**params))
                output_size = params['output_size']
                current_h = output_size[0] if isinstance(output_size, tuple) else output_size
                current_w = output_size[1] if isinstance(output_size, tuple) else output_size
            elif layer_type == 'resblock':  # 新增：处理 ResBlock
                if 'out_channels' not in params:
                    raise ValueError(f"ResBlock 层索引 {i} 缺少 'out_channels'。参数: {params}")

                block_stride = params.get('stride', 1)
                block_out_channels = params['out_channels']

                resblock_module = ResBlock(in_channels=current_channels,
                                           out_channels=block_out_channels,
                                           stride=block_stride)
                layers.append(resblock_module)

                current_channels = block_out_channels

                # 更新空间维度，假设 ResBlock 的第一个卷积 (通常 k=3, p=1) 决定尺寸变化
                if block_stride > 1:  # 只有当 stride > 1 时空间维度才会改变
                    current_h, current_w = self._get_spatial_output_size(
                        (current_h, current_w), kernel_size=3,  # ResBlock 的第一个卷积通常 kernel_size=3
                        stride=block_stride,
                        padding=1  # ResBlock 的第一个卷积通常 padding=1
                    )
                # 如果 stride=1 (且 k=3, p=1), 空间维度不变
            elif layer_type == 'flatten':
                layers.append(nn.Flatten())
                # current_channels should be C*H*W after flatten
                if current_h is not None and current_w is not None:  # If spatial dims are tracked
                    current_channels = current_channels * current_h * current_w
                is_flattened = True
            elif layer_type == 'fc':
                if not is_flattened:  # 如果前面没有显式的 flatten 层，则添加一个
                    layers.append(nn.Flatten())
                    if current_h is not None and current_w is not None:
                        current_channels = current_channels * current_h * current_w  # 更新展平后的通道数
                    # else: pass, assume in_features for Linear will be calculated dynamically if needed by a wrapper
                    is_flattened = True
                params['in_features'] = current_channels
                if i == last_fc_idx and 'out_features' not in params:  # 如果是最后一个FC层且未指定输出特征数
                    params['out_features'] = self.num_classes  # 自动设置为类别数
                elif 'out_features' not in params:
                    raise ValueError(f"全连接层索引 {i} 需要 'out_features'。参数: {params}")
                layers.append(nn.Linear(**params))
                current_channels = params[
                    'out_features']  # This is now the new "channel" count for subsequent 1D layers
            elif layer_type == 'dropout':
                p_val = params.get('p', self.exp_config.get('dropout_rate', 0.0))
                if p_val > 0:  # 只有当 p > 0 时才添加 Dropout 层
                    layers.append(nn.Dropout(p=p_val))
            else:
                raise ValueError(f"不支持的层类型: {layer_type} 在索引 {i}")

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.features_sequence(x)


class ResBlock(nn.Module):
    """
    ResNet 基本块 (与问题描述中提供的版本一致)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 快捷连接 (Shortcut connection)
        self.shortcut = nn.Sequential()  # 默认为空，即恒等映射 (如果维度匹配)
        if stride != 1 or in_channels != out_channels:
            # 如果步长不为1或输入输出通道数不同，则需要通过1x1卷积调整维度
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 添加快捷连接 (残差连接)
        out += self.shortcut(x)

        # 最终激活
        out = self.relu(out)
        return out
