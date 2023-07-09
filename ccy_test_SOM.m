clc

clear

%% 录入输入数据

% 载入数据
%转置后符合神经网络的输入格式

load = xlsread('2021load','重庆负荷');
load = reshape(load.',96,365);


%% 网络建立和训练

% 建立SOM网络,竞争层为6*6=36个神经元

net=selforgmap([20 20]);

plotsom(net.layers{1}.positions)

% 5次训练的步数

a=[10 30 50 100 200 500 100];

% 随机初始化一个1*10向量。

yc=rands(7,365);

%% 进行训练

% 训练次数为10次

net.trainparam.epochs=a(7);

% 训练网络和查看分类

net=train(net,load);

y=sim(net,load);

yc(7,:)=vec2ind(y);

plotsom(net.IW{1,1},net.layers{1}.distances)
% %% 网络作分类的预测
% 
% % 测试样本输入
% 
% % load_in = xlsread('2021load','重庆负荷');
% % load_in = load_in(1:96,1);
% 
% % sim()来做网络仿真
% 
% r=sim(net,load);
% 
% % 变换函数 将单值向量转变成下标向量。
% 
% rr=vec2ind(r)

%% 网络神经元分布情况

% 查看网络拓扑学结构

plotsomtop(net)
% % 查看临近神经元直接的距离情况
% 
% plotsomnd(net)
% % 查看每个神经元的分类情况
% 
%  plotsomhits(net,load)

 % plotsompos(net,load)