clear
clc
addpath('dace')
uqlab;
SS=1e6; % number of samples
ver_n=1e4; % The initial sample
rng(1,'twister');
close all
%%
load('Vv2GM.mat');
 GG=10;
% GG=8;% Select the earthquake
Earthquake_record_X = Stochastic_earthquake{GG}; %Stochastic Ground Motion
Earthquake_record_X(:,2) = Earthquake_record_X(:,2)*9.8;
time=0:1/1024:30;
GM=interp1(Earthquake_record_X(:,1),Earthquake_record_X(:,2),time);
%%
load sensiti\2dof_sensiti_all_10000.mat X_ver1 X_ver2 s_totall ver_n
sensiall=zeros(9,2);
x_mate=5e5;
% [Y_ver1,~,~] = DDOF_Boucwen_2dm(X_ver1,GM,x_mate);
%%
meta_data_dir = '.\meta_tasks\rths_m1\';
main_data_dir='.\main_tasks\rths_m1\';
i_mate = 1;
N_max=2;
for hh=1:1
matname=strcat('result1\AML_m1_k1_005\','Amlgp_ak_2dof_2d_',num2str(cn),'_',num2str(N_max),'_',num2str(hh),'_',num2str(ttn),'.mat');
load (matname,'Xs_rese','Ys_rese','Xs_mete1','Ys_mete1')
end
X_meta=Xs_mete1;
Y_meta=Ys_mete1;
filename = [meta_data_dir,'task_', num2str(i_mate),'_date', '.mat'];
save(filename,"X_meta","Y_meta")
sensiall(1,:)=s_totall;
X_ver=X_ver1;
    filename = [main_data_dir,'rmse_task_', 'main_','date', '.mat'];
    save(filename,"Xs_rese","Ys_rese")

    filename = [main_data_dir,'rmse_task_', 'test_','date', '.mat'];
    save(filename,"X_ver")
    system('D:\ProgramData\anaconda3\envs\py391\python.exe predict_ver_n_1.py');
    filename = [main_data_dir,'test_ver_prediction','.mat'];
    Y11=load (filename);
    Y_ver1=Y11.mean;

s_tot=zeros(1,2);
y_stota=zeros(ver_n,2);

for i = 1:2
x_stot = X_ver1;
x_stot(:,i) = X_ver2(:,i);

% [y_stot,~,~] = DDOF_Boucwen_2dm(x_stot,GM,x_mate);

    X_ver=x_stot;
    filename = [main_data_dir,'rmse_task_', 'main_','date', '.mat'];
    save(filename,"Xs_rese","Ys_rese")

    filename = [main_data_dir,'rmse_task_', 'test_','date', '.mat'];
    save(filename,"X_ver")
    system('D:\ProgramData\anaconda3\envs\py391\python.exe predict_ver_n_1.py');
    filename = [main_data_dir,'test_ver_prediction','.mat'];
    Y11=load (filename);
    y_stot=Y11.mean;


y_stota(:,i)=y_stot;
s_tot(1,i) = mean( (Y_ver1 - y_stot) .^ 2 );

end
y_var=var(Y_ver1);
s_totallk = s_tot / ( 2*  y_var );
sensiall(9,:)=s_totallk;
