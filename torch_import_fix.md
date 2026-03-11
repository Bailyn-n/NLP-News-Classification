## PyTorch 导入错误修复指南

如果你遇到 `AttributeError: partially initialized module 'torch' has no attribute 'nn' (most likely due to a circular import)` 错误，请按照以下步骤解决问题：

### 问题原因分析
这个错误通常由以下几个原因造成：
1. 在代码中将变量命名为 `torch`，覆盖了导入的torch库
2. 工作目录中存在名为 `torch.py` 的文件
3. 循环导入问题
4. PyTorch安装不完整或损坏

### 解决方案

#### 方案1: 检查变量名冲突
确保你没有使用 `torch` 作为变量名，例如：
```python
# 错误做法 - 不要这样做
import torch
torch = some_other_variable  # 这会覆盖torch模块

# 正确做法
import torch
my_torch_data = some_other_variable  # 使用其他名称
```

#### 方案2: 检查同名文件
确认项目目录中没有名为 `torch.py` 或 `torch/` 目录的文件

#### 方案3: 重启Python环境
如果在Jupyter Notebook中遇到此问题：
1. 重启内核 (Kernel -> Restart)
2. 清除所有输出 (Cell -> All Output -> Clear)
3. 重新运行代码

#### 方案4: 重新安装PyTorch
如果上述方法都不起作用，尝试重新安装PyTorch：
```bash
pip uninstall torch
pip install torch
```

#### 方案5: 检查导入顺序
确保在代码中正确地首先导入torch，然后再使用它：
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 其他代码...
```

### 预防措施
- 避免使用库名作为变量名
- 定期清理Python环境
- 使用虚拟环境管理依赖