import torch
import torch.nn.functional as F


def compute_kernel(x, y, kernel_type='gaussian', kernel_param=1.0):
    """
    计算核矩阵
    Args:
        x: 样本集合X，大小为 (batch_size, feature_dim)
        y: 样本集合Y，大小为 (batch_size, feature_dim)
        kernel_type: 核函数类型，可以是 'gaussian' 或 'linear'
        kernel_param: 核函数参数，对于高斯核，表示高斯核的方差

    Returns:
        kernel_matrix: 核矩阵，大小为 (batch_size, batch_size)
    """
    if kernel_type == 'gaussian':
        x_norm = (x**2).sum(dim=-1, keepdim=True)  # 计算 x 的范数的平方
        y_norm = (y**2).sum(dim=-1, keepdim=True)  # 计算 y 的范数的平方
        xy = torch.matmul(x, y.t())  # 计算 x 和 y 的内积
        pairwise_distance = x_norm + y_norm.t() - 2 * xy  # 计算欧氏距离的平方
        kernel_matrix = torch.exp(-pairwise_distance / (2 * kernel_param**2))  # 高斯核函数
    elif kernel_type == 'linear':
        kernel_matrix = torch.matmul(x, y.t())  # 线性核函数
    else:
        raise ValueError("Unsupported kernel type.")

    return kernel_matrix

def mk_mmd_loss(x, y, kernel_types=['gaussian','gaussian','gaussian','gaussian','gaussian'], kernel_params=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """
    计算MK-MMD损失函数
    Args:
        x: 样本集合X，大小为 (batch_size, feature_dim)
        y: 样本集合Y，大小为 (batch_size, feature_dim)
        kernel_types: 核函数类型列表，例如 ['gaussian', 'linear']
        kernel_params: 核函数参数列表，例如 [1.0, 0.5]

    Returns:
        mk_mmd: MK-MMD损失值
    """
    batch_size = x.size(0)
    n_kernels = len(kernel_types)

    # 计算各个核矩阵
    xx_kernels = [compute_kernel(x, x, kernel_type, kernel_param) for kernel_type, kernel_param in zip(kernel_types, kernel_params)]
    yy_kernels = [compute_kernel(y, y, kernel_type, kernel_param) for kernel_type, kernel_param in zip(kernel_types, kernel_params)]
    xy_kernels = [compute_kernel(x, y, kernel_type, kernel_param) for kernel_type, kernel_param in zip(kernel_types, kernel_params)]

    # 计算MK-MMD值
    mk_mmd = 0.0
    for i in range(n_kernels):
        xx = xx_kernels[i]
        yy = yy_kernels[i]
        xy = xy_kernels[i]
        mmd = torch.mean(xx) - 2 * torch.mean(xy) + torch.mean(yy)
        mk_mmd += torch.sqrt(torch.max(torch.tensor(0.0), mmd))

    mk_mmd /= n_kernels

    return mk_mmd


if __name__ == '__main__':
    a=torch.rand([2,5])
    b=torch.rand([3,5])
    print(mk_mmd_loss(a,b))