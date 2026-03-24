import warnings

warnings.filterwarnings("ignore")
import torch
import pandas as pd
import os
import scipy.io as sio
from botorch.models import SingleTaskMultiFidelityGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from torch import Tensor


def load_source_data(meta_data_dir, mean, std):
    """
    加载所有元数据并添加低保真度标记
    返回标准化后的特征张量和目标张量
    """
    all_X, all_Y = [], []
    for i in range(1):  # 假设有1个元任务
        data = sio.loadmat(os.path.join(meta_data_dir, f'task_{i + 1}_date.mat'))
        X = torch.tensor(data['X_meta'], dtype=torch.float64)
        X_norm = (X - mean) / std

        # 添加低保真度标记 (fidelity=0)
        fidelity_dim = torch.zeros(X_norm.shape[0], 1, dtype=torch.float64)
        X_with_fidelity = torch.cat([X_norm, fidelity_dim], dim=-1)

        Y = torch.tensor(data['Y_meta'], dtype=torch.float64)

        all_X.append(X_with_fidelity)
        all_Y.append(Y)

    # 合并所有元数据
    return torch.cat(all_X, dim=0), torch.cat(all_Y, dim=0)


def main():
    # 路径定义
    meta_data_dir = 'meta_tasks/rths_m1/'
    train_data_path = 'main_tasks/rths_m1/'
    test_data_path = 'main_tasks/rths_m1/'
    output_path = 'main_tasks/rths_m1/test_ver_prediction.mat'

    # 加载训练数据 (高保真数据)
    train_data = sio.loadmat(os.path.join(train_data_path, f'rmse_task_main_date.mat'))
    X_train = torch.tensor(train_data['Xs_rese'], dtype=torch.float64)
    Y_train = torch.tensor(train_data['Ys_rese'], dtype=torch.float64)

    # 计算标准化参数
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    X_train_norm = (X_train - mean) / std

    # 添加高保真度标记 (fidelity=1)
    fidelity_dim = torch.ones(X_train_norm.shape[0], 1, dtype=torch.float64)
    X_train_with_fidelity = torch.cat([X_train_norm, fidelity_dim], dim=-1)

    # 加载元数据 (低保真数据)
    X_meta, Y_meta = load_source_data(meta_data_dir, mean, std)

    # 合并低保真和高保真数据
    X_full = torch.cat([X_meta, X_train_with_fidelity], dim=0)
    Y_full = torch.cat([Y_meta, Y_train], dim=0)

    # 定义数据维度

    # 初始化多保真度GP模型
    model = SingleTaskMultiFidelityGP(
        train_X=X_full,
        train_Y=Y_full,data_fidelity=-1,
        outcome_transform=Standardize(m=1)
    )

    # 训练模型
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # 加载测试数据
    test_data = sio.loadmat(os.path.join(test_data_path, f'rmse_task_test_date.mat'))
    X_test = torch.tensor(test_data['X_ver'], dtype=torch.float64)
    X_test_norm = (X_test - mean) / std

    # 为测试数据添加高保真度标记
    fidelity_dim_test = torch.ones(X_test_norm.shape[0], 1, dtype=torch.float64)
    X_test_with_fidelity = torch.cat([X_test_norm, fidelity_dim_test], dim=-1)

    # 预测
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X_test_with_fidelity)
        mean_pred = posterior.mean

    # 保存结果
    sio.savemat(output_path, {'mean': mean_pred.numpy()})


if __name__ == '__main__':
    main()