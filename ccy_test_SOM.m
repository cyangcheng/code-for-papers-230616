clc

clear

%% ¼����������

% ��������
%ת�ú����������������ʽ

load = xlsread('2021load','���츺��');
load = reshape(load.',96,365);


%% ���罨����ѵ��

% ����SOM����,������Ϊ6*6=36����Ԫ

net=selforgmap([20 20]);

plotsom(net.layers{1}.positions)

% 5��ѵ���Ĳ���

a=[10 30 50 100 200 500 100];

% �����ʼ��һ��1*10������

yc=rands(7,365);

%% ����ѵ��

% ѵ������Ϊ10��

net.trainparam.epochs=a(7);

% ѵ������Ͳ鿴����

net=train(net,load);

y=sim(net,load);

yc(7,:)=vec2ind(y);

plotsom(net.IW{1,1},net.layers{1}.distances)
% %% �����������Ԥ��
% 
% % ������������
% 
% % load_in = xlsread('2021load','���츺��');
% % load_in = load_in(1:96,1);
% 
% % sim()�����������
% 
% r=sim(net,load);
% 
% % �任���� ����ֵ����ת����±�������
% 
% rr=vec2ind(r)

%% ������Ԫ�ֲ����

% �鿴��������ѧ�ṹ

plotsomtop(net)
% % �鿴�ٽ���Ԫֱ�ӵľ������
% 
% plotsomnd(net)
% % �鿴ÿ����Ԫ�ķ������
% 
%  plotsomhits(net,load)

 % plotsompos(net,load)