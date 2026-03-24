import warnings
warnings.filterwarnings("ignore")
import torch
import pandas as pd
import os
import scipy.io as sio
from model import ScaMLGP, meta_fit_scamlgp, optimize_marginal_likelihood
from botorch.utils.datasets import SupervisedDataset
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood



def load_source_models(meta_data_dir,mean,std):
    meta_data = {}
    for i in range(1):
        data = sio.loadmat(os.path.join(meta_data_dir, f'task_{i + 1}_date.mat'))
        X = torch.tensor(data['X_meta'], dtype=torch.float64)
        X_norm = (X - mean) / (std)
        Y = torch.tensor(data['Y_meta'], dtype=torch.float64)
        meta_data[i] = SupervisedDataset(X_norm, Y)
        source_models = meta_fit_scamlgp(meta_data, num_restarts_log_likelihood=5)
    return source_models

def main():
    # 路径定义
    meta_data_dir = 'meta_tasks/rths_m1/'
    train_data_path = 'main_tasks/rths_m1/'
    test_data_path = 'main_tasks/rths_m1/'
    output_path = 'main_tasks/rths_m1/test_ver_prediction.mat'

    # 加载训练数据
    train_data = sio.loadmat(os.path.join(train_data_path, f'rmse_task_main_date.mat'))
    X_train = torch.tensor(train_data['Xs_rese'], dtype=torch.float64)
    Y_train = torch.tensor(train_data['Ys_rese'], dtype=torch.float64)
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    X_train_norm = (X_train - mean) / (std)
    # 加载元模型
    source_models = load_source_models(meta_data_dir,mean,std)
    # 加载测试点
    test_data = sio.loadmat(os.path.join(test_data_path, f'rmse_task_test_date.mat'))
    X_test = torch.tensor(test_data['X_ver'], dtype=torch.float64)
    X_test_norm = (X_test - mean) / (std)

    # 初始化并优化主任务模型
    model = ScaMLGP(train_X=X_train_norm, train_Y=Y_train, source_gps=source_models)
    optimize_marginal_likelihood(model, num_restarts=10)
    # for gp in model.source_gps.values():
    #     mll_source = ExactMarginalLogLikelihood(gp.likelihood, gp)
    #     fit_gpytorch_mll(mll_source)
    #
    # mll_main = ExactMarginalLogLikelihood(model.likelihood, model)
    # fit_gpytorch_mll(mll_main)
    model.train()
    # print("训练前权重:", model.weights.detach().numpy())
    # optimize_marginal_likelihood(model, num_restarts=10)
    # print("训练后权重:", model.weights.detach().numpy())
    model.eval()
    weights = model.weights.detach().numpy()
    # 预测
    with torch.no_grad():
        posterior = model.posterior(X_test_norm)
        mean = posterior.mean
    # 保存结果
    sio.savemat(output_path, {'mean': mean,'weights': weights})

if __name__ == '__main__':
    main()