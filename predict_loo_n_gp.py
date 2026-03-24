import torch
# import pandas as pd
import os
import scipy.io as sio
# from model import ScaMLGP, meta_fit_scamlgp, optimize_marginal_likelihood
# from botorch.utils.datasets import SupervisedDataset
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


def main():
    # 路径定义

    train_data_path = 'main_tasks/'
    test_data_path = 'main_tasks/'
    output_path = 'main_tasks/CV_test_prediction.mat'

    # 加载训练数据
    train_data = sio.loadmat(os.path.join(train_data_path, f'CV_task_main_date.mat'))
    X_train = torch.tensor(train_data['Xs_Selec'], dtype=torch.float64)
    Y_train = torch.tensor(train_data['Ys_Selec'], dtype=torch.float64)
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    X_train_norm = (X_train - mean) / (std)
    # 加载元模型

    # 加载测试点
    test_data = sio.loadmat(os.path.join(test_data_path, f'CV_task_test_date.mat'))
    X_test = torch.tensor(test_data['Xs_test'], dtype=torch.float64)
    X_test_norm = (X_test - mean) / (std)

    # 初始化并优化主任务模型

    model = SingleTaskGP(train_X=X_train_norm, train_Y=Y_train, outcome_transform=Standardize(1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.train()
    model.eval()

    # 预测
    with torch.no_grad():
        posterior = model.posterior(X_test_norm)
        mean = posterior.mean
        variance = posterior.variance
    # print("训练后的权重:", model.weights.detach().numpy())
    # 保存结果
    # sio.savemat(output_path, {'mean': mean})
    sio.savemat(output_path, {'mean': mean.cpu().numpy(),'variance': variance.cpu().numpy()})

if __name__ == '__main__':
    main()