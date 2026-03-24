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

X1_l=4*1e5;
X1_u=6*1e5;
InputOpts1.Marginals(1).Name = 'k1';      %define the variable distribution
InputOpts1.Marginals(1).Type = 'Uniform';
InputOpts1.Marginals(1).Parameters = [X1_l,X1_u];
X2_l=5*1e5;
X2_u=1.5*1e6;
InputOpts1.Marginals(2).Name = 'k2';      %define the variable distribution
InputOpts1.Marginals(2).Type = 'Uniform';
InputOpts1.Marginals(2).Parameters = [X2_l,X2_u];
myInput1 = uq_createInput(InputOpts1);

X_ver1 = uq_getSample(myInput1,ver_n,'LHS');
X_ver2 = uq_getSample(myInput1,ver_n,'LHS');
x_mate=5e5;

[Y_ver1,~,~] = DDOF_Boucwen_2dm(X_ver1,GM,x_mate);
s_tot=zeros(1,2);
y_stota=zeros(ver_n,2);

for i = 1:2
x_stot = X_ver1;
x_stot(:,i) = X_ver2(:,i);

[y_stot,~,~] = DDOF_Boucwen_2dm(x_stot,GM,x_mate);

y_stota(:,i)=y_stot;
s_tot(1,i) = mean( (Y_ver1 - y_stot) .^ 2 );

end
y_var=var(Y_ver1);
s_totall = s_tot / ( 2*  y_var )

filename= strcat('sensiti/2dof_sensiti_all','_', num2str(ver_n),'.mat');
save(filename);