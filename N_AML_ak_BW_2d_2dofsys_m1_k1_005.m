clear
clc
uqlab;
close all
task_p=[4e5,3e5,2e5];
for ttn=1:3
for hh=1:10
tic;
%%
rng(hh,'twister');
load('Vv2GM.mat');
 GG=10;
% GG=8;% Select the earthquake
Earthquake_record_X = Stochastic_earthquake{GG}; %Stochastic Ground Motion
Earthquake_record_X(:,2) = Earthquake_record_X(:,2)*9.8;
time=0:1/1024:30;
GM=interp1(Earthquake_record_X(:,1),Earthquake_record_X(:,2),time);
SS=1e5; % number of samples
ini_Xn=5; % The initial sample
%% 辅助任务的不确定性
X1_l1=4*1e5;
X1_u1=6*1e5;
InputOpts1.Marginals(1).Name = 'k1';      %define the variable distribution
InputOpts1.Marginals(1).Type = 'Uniform';
InputOpts1.Marginals(1).Parameters = [X1_l1,X1_u1];
X2_l1=5*1e5;
X2_u1=1.5*1e6;
InputOpts1.Marginals(2).Name = 'k2';      %define the variable distribution
InputOpts1.Marginals(2).Type = 'Uniform';
InputOpts1.Marginals(2).Parameters = [X2_l1,X2_u1];
X_m=[X1_l1 X2_l1];
X_v=[X1_u1-X1_l1 X2_u1-X2_l1];
myInput1 = uq_createInput(InputOpts1);
X_inim = uq_getSample(myInput1,ini_Xn,'LHS');
X_cand1= uq_getSample(myInput1,SS,'MC'); 
%% 主要任务的不确定性

%% 生成辅助任务数据ak1
cn=0.05;
X_cand=X_cand1;
x_mate=task_p(ttn);
meta_data_dir = '.\meta_tasks\rths_m1\';
num_meta_ini = 5;
i_mate = 1;
X_meta_ini=X_inim;
[Y_meta_ini,~,~] = DDOF_Boucwen_2dm(X_meta_ini,GM,x_mate);
nn=0;
Xs_mete1=X_meta_ini;
Ys_mete1=Y_meta_ini;
N_max=50;
while  (nn<N_max)
[Xs_new,error]=CVV_GP(Xs_mete1,Ys_mete1,X_cand,X_m,X_v);

[Ys_new,~,~] = DDOF_Boucwen_2dm(Xs_new,GM,x_mate); 
nn=nn+1
Xs_mete1(num_meta_ini+nn,:)=Xs_new;
Ys_mete1(num_meta_ini+nn,:)=Ys_new;
    RMSE_gp1(nn,1) = sqrt(sum(error.^2)./size(Xs_mete1,1))./(max(Ys_mete1)-min(Ys_mete1));
RMSE_gp1(nn,1)

if nn>2
    if RMSE_gp1(nn,1)<cn && RMSE_gp1(nn-1,1)<cn
        break
    end
end


index5=find(X_cand==Xs_new,1);
X_cand(index5,:)=[];
end
X_meta=Xs_mete1;
Y_meta=Ys_mete1;
filename = [meta_data_dir,'task_', num2str(i_mate),'_date', '.mat'];
save(filename,"X_meta","Y_meta")
%%  主要任务的迭代
X_cand=X_cand1;
X_ini=X_inim;
main_data_dir='.\main_tasks\rths_m1\';
load 2dof_ver_k1_1000.mat
x_mate=5e5;
[Y_ini,~,~] = DDOF_Boucwen_2dm(X_ini,GM,x_mate);
X_main=X_ini;
Y_main=Y_ini;
filename = [main_data_dir,'task_', 'main_','date', '.mat'];
save(filename,"X_main","Y_main")

nn=0;
Xs=X_ini;
Ys=Y_ini;
N_max=50;
while  (nn<N_max)

[Xs_new,error]=CVV_GP(Xs,Ys,X_cand,X_m,X_v);
[~,errorm]=CVV_GPML(Xs,Ys,X_cand,X_m,X_v);
[Ys_new,~,~] = DDOF_Boucwen_2dm(Xs_new,GM,x_mate); 
hh
nn=nn+1
Xs(ini_Xn+nn,:)=Xs_new;
Ys(ini_Xn+nn,:)=Ys_new;
    RMSE_gp3(nn,1) = sqrt(sum(error.^2)./size(Xs,1))./(max(Ys)-min(Ys));
    RMSE_gp3(nn,1)
    RMSE_mlgp3m(nn,1) = sqrt(sum(errorm.^2)./size(Xs,1))./(max(Ys)-min(Ys));
    RMSE_mlgp3m(nn,1)
    Xs_rese=Xs;
    Ys_rese=Ys;
    filename = [main_data_dir,'rmse_task_', 'main_','date', '.mat'];
    save(filename,"Xs_rese","Ys_rese")

    filename = [main_data_dir,'rmse_task_', 'test_','date', '.mat'];
    save(filename,"X_ver")
    system('D:\ProgramData\anaconda3\envs\py391\python.exe predict_ver_n_1.py');
    filename = [main_data_dir,'test_ver_prediction','.mat'];
    Y11=load (filename);
    Y_mlgp=Y11.mean;
    mlgp_weig(nn,1)=Y11.weights;
    RMSEmlgp(nn,1) = sqrt(sum((Y_ver(:,1)-Y_mlgp(:,1)).^2)./size(X_ver,1))./(max(Y_ver)-min(Y_ver));
    RMSEmlgp(nn,1)

    RRmlgp(nn,1) = (sum((Y_ver(:,1)-mean(Y_ver(:,1))).^2)-sum((Y_ver(:,1)-Y_mlgp(:,1)).^2))/(sum((Y_ver(:,1)-mean(Y_ver(:,1))).^2))*100;
    RRmlgp(nn,1)

 index5=find(X_cand==Xs_new,1);
 X_cand(index5,:)=[];

% if RMSEmlgp(nn,1)<0.02
%     break
% end

if nn>=2
    if RMSE_mlgp3m(nn,1)<cn && RMSE_mlgp3m(nn-1,1)<cn
        break
    end
end

end
elapsedTime = toc;
filename= strcat('result1\AML_m1_k1_005\Amlgp_ak_2dof_2d_',num2str(cn),'_', num2str(nn),'_',num2str(hh),'_',num2str(ttn),'.mat');
save(filename);
clearvars -except hh ttn task_p
end
clearvars -except ttn task_p hh
end
