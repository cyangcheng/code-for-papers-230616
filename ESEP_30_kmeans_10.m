

clear
clc
tic

%%系统参数
%所有参数均用有名值表示
paragen=xlsread('excel2017','机组参数');
loadcurve=xlsread('excel2017','负荷曲线');
netpara=xlsread('excel2017','网络参数');
%% 规模变量
%机组数
gennum=size(paragen);
gennum=gennum(1,1);
%节点数
numnodes=size(loadcurve);
numnodes=numnodes(1,1)-1;
%时间范围
% T=96*12; %典型日数量
T = 24*30;
%% 储能参数
% bus   eff_c  eff_dc   E_init E_lower
data_s = [
	6	0.95  0.95  0.5   0.2
];
%% 新能源参数
% bus   capacity
data_r = [
 	6	1.5;
];
branch_num=size(netpara);%网络中的支路
branch_num=branch_num(1,1);
PL_max=netpara(:,6);%线路最大负荷
PL_min=netpara(:,7);%线路最小负荷
limit=paragen(:,3:4);%机组出力上下限//limit(:,1)表示上限，limit(:,2)表示下限
Pmin=limit(:,2);             
Pmin = 2*Pmin;
Pmax=limit(:,1);
Ru = [0.3;0.3;0.2;0.2;0.15;0.15];
Rd = [0.3;0.3;0.2;0.2;0.15;0.15];
% Su=(limit(:,1)+limit(:,2))/2;
% Sd=(limit(:,1)+limit(:,2))/2;
c = 3;
eff_c  = data_s(:,2);
eff_dc = data_s(:,3);
E_lower = data_s(:,5);
N_r = length(data_r(:,1));
N_s = length(data_s(:,1));
 beta = 0.5;% 新能源渗透率
% beta = sdpvar(1);
H = 2; %持续充放电时间
P_c = sdpvar(N_s,T,'full'); % 储能充电
P_dc = sdpvar(N_s,T,'full'); % 储能放电
E_init = sdpvar(N_s,1,'full'); %储能初始状态
P_s = sdpvar(N_s,1,'full'); % 
E_s = sdpvar(N_s,T,'full');% 
P_r = sdpvar(N_r,T,'full');% 新能源变量

para=paragen(:,5:7);%成本系数//para(:,1)表示系数a,para(:,2)表示系数b,para(:,3)表示系数c。
price=100;
para=price*para;%价格换算
lasttime=paragen(:,9);%持续时间
lasttime=3*lasttime;
Rud=paragen(:,8);%上下爬坡速率//因题中简化上坡下坡速度相同
H_start=paragen(:,10);%启动成本
J_stop =paragen(:,11);%关停成本
power_gen=paragen(:,2);%发电机对应节点
BaseMVA=100;%参考电压
slack_bus=26;%参考节点



%% 直流潮流下的导纳矩阵节点参数初始化
netpara(:,4)=1./netpara(:,4);  %电抗求倒数成电纳，这个地方只能计算一次，之前这个值多次计算出了问题
Y=zeros(numnodes,numnodes);
YY=zeros(numnodes,numnodes);
% 直流潮流的导纳矩阵计算
for k=1:branch_num
    i=netpara(k,2);%首节点
    j=netpara(k,3);%尾节点
    Y(i,j)=-netpara(k,4);%导纳矩阵中非对角元素
    Y(j,i)= Y(i,j);
end
for k=1:numnodes
    Y(k,k)=-sum(Y(k,:)); %导纳矩阵中的对角元素
end
%再删除掉平衡节点所在的行与列
YY=Y;
Y(slack_bus,:)=[];
Y(:,slack_bus)=[];

%% 各时刻节点负荷
pd = [2.9875
4.2727
2.1952
3.3239
2.4187
2.0997
2.1838
3.0257
2.156
2.5129
2.6846
3.2174
3.1818
2.8222
2.8064
3.2757
2.1116
2.1226
2.4509
4.1734
2.7695
2.1889
2.7616
3.3533
2.1823
2.461
2.0537
2.2167
2.4696
2.1453
2.6064
2.354
2.0512
2.0173
3.9579
3.2395
2.1755
2.654
2.3376
4.0893
2.8529
2.6582
2.6268
2.9304
3.1189
2.7102
2.0806
3.3142
2.1055
2.3257
3.076
2.3795
2.9082
2.0254
2.3813
4.0136
3.1871
2.0757
2.1182
2.1328
2.7739
3.8664
2.0339
3.1228
2.2389
1.9763
2.0363
2.8151
2.0318
2.2992
2.5309
2.9791
2.9612
2.572
2.5817
3.0505
1.9833
1.9949
2.2974
3.8352
2.5476
2.0571
2.5414
3.1619
2.0622
2.2603
1.9772
2.0803
2.3049
2.0174
2.4546
2.2065
1.9728
1.944
3.6562
3.0594
2.0512
2.4769
2.2144
3.7736
2.6809
2.4876
2.4637
2.7381
2.9146
2.5183
1.997
3.0913
2.0133
2.2054
2.8904
2.2685
2.6927
1.947
2.2491
3.7327
3.0203
1.9503
2.0179
2.0014
2.6056
3.6023
1.9743
2.9477
2.1443
1.9175
1.9547
2.639
1.9685
2.1679
2.4153
2.8357
2.7976
2.435
2.4517
2.8697
1.932
1.9282
2.1913
3.575
2.434
1.992
2.4238
2.9806
1.9676
2.1617
1.8948
1.9909
2.2104
1.9387
2.4186
2.132
1.9135
1.8797
3.4581
2.8971
1.956
2.3983
2.184
3.5335
2.567
2.4274
2.4047
2.6075
2.7965
2.4381
1.943
2.8994
1.9622
2.1309
2.7426
2.1984
2.5726
1.8966
2.202
3.4958
2.848
1.9307
1.9554
1.9428
2.5359
3.4384
1.9297
2.8048
2.1139
1.8878
1.9209
2.5432
1.941
2.1107
2.3915
2.724
2.6915
2.4109
2.4074
2.7476
1.9017
1.8851
2.1825
3.3948
2.3996
1.9731
2.3818
2.8255
1.9478
2.1233
1.8729
1.9509
2.1854
1.8978
2.3947
2.1125
1.9092
1.9047
3.3241
2.7561
1.9349
2.365
2.1839
3.3649
2.4946
2.3985
2.3886
2.5047
2.7022
2.3922
1.9488
2.792
1.9782
2.1124
2.6861
2.1658
2.502
1.8963
2.1611
3.3348
2.7477
1.8983
1.961
1.9077
2.495
3.2904
1.9904
2.753
2.1723
1.9465
1.9137
2.4929
2.0504
2.1185
2.418
2.7047
2.7414
2.3956
2.4231
2.6845
1.9572
1.9104
2.2069
3.2937
2.4373
2.0016
2.4162
2.7559
2.0122
2.1273
1.9413
1.9373
2.192
1.9157
2.5437
2.1776
2.0275
2.0803
3.2808
2.7564
2.013
2.4758
2.3439
3.2916
2.572
2.5127
2.5475
2.5246
2.7422
2.452
2.0786
2.7563
2.1961
2.2284
2.7743
2.2701
2.5451
1.9738
2.3043
3.2652
2.7663
1.9767
2.1483
2.049
2.607
3.2572
2.2579
2.8279
2.4083
2.2818
2.1205
2.5925
2.3596
2.2831
2.882
2.8437
2.8885
2.6536
2.6763
2.8186
2.291
2.1736
2.5398
3.2836
2.7557
2.2849
2.7288
2.7807
2.3303
2.3255
2.3893
2.1913
2.4612
2.1681
3.0743
2.4601
2.4421
2.4558
3.3696
2.8466
2.3355
2.9574
2.8217
3.2843
2.7754
2.9432
3.0487
2.6769
2.9097
2.8839
2.4281
2.8287
2.4307
2.5769
2.955
2.6436
2.7361
2.4034
2.7101
3.3012
2.8863
2.3618
2.3871
2.3727
2.9512
3.4549
2.5832
3.008
2.9547
2.553
2.5248
2.8493
2.5958
2.6699
3.407
3.1252
3.2452
3.1685
3.1427
3.0268
2.6911
2.5094
2.9314
3.5844
3.2258
2.4529
3.2797
2.9145
2.5143
2.8288
2.6124
2.4877
2.88
2.5905
3.6968
3.0772
2.8585
2.7682
4.0377
3.1486
2.7073
3.5606
3.2691
3.6947
3.4022
3.4727
3.6493
3.1111
3.4426
3.3451
2.9667
3.0904
2.7406
3.204
3.573
3.0454
3.2418
2.6847
3.1431
3.8902
3.2326
2.7651
2.6622
2.8442
3.654
4.1974
3.1354
3.4379
3.4951
2.8791
2.9511
3.5314
2.978
3.3242
4.0693
3.8103
3.9316
3.8261
3.7945
3.6661
3.0904
2.8146
3.4267
4.3849
3.9142
2.8361
3.965
3.3394
2.8777
3.4185
2.9415
3.0516
3.3326
3.0332
4.1871
3.5557
3.1671
3.0058
4.7466
3.6712
3.2968
4.1371
3.5888
4.4965
3.9552
4.095
4.1496
3.8055
3.9916
4.0485
3.4311
3.5653
3.0447
3.5777
4.0926
3.5223
3.8763
2.9653
3.5271
4.6323
3.7613
3.128
2.9872
3.3688
4.09
4.8538
3.5641
3.9289
3.666
3.0345
3.1951
4.0255
3.13
3.5989
4.2951
4.2529
4.3161
4.2113
4.2086
4.1833
3.2634
3.0186
3.6657
4.9541
4.2538
3.0582
4.2416
3.8363
3.0775
3.6159
3.0996
3.4999
3.6174
3.2293
4.3745
3.6942
3.2993
3.1399
5.2417
4.0405
3.6094
4.3198
3.7346
5.0602
4.2865
4.327
4.37
4.1831
4.3306
4.2942
3.7216
4.0019
3.1809
3.6935
4.4023
3.7118
4.2276
3.1318
3.7227
5.1622
4.1136
3.2856
3.1362
3.6488
4.3901
5.3592
3.7782
4.1822
3.6288
3.1364
3.3384
4.3734
3.116
3.7372
4.3195
4.4694
4.4692
4.4213
4.401
4.4692
3.2161
3.1806
3.7433
5.3964
4.4064
3.2482
4.373
4.1733
3.2164
3.7135
3.0692
3.7741
3.7638
3.2892
4.0642
3.4729
2.9963
2.842
5.4671
4.1911
3.6948
4.1207
3.4726
5.432
4.3192
4.1419
4.1174
4.351
4.3758
4.2923
3.6385
4.2012
3.0011
3.4199
4.3816
3.6754
4.3037
2.8963
3.5055
5.3956
4.1945
3.0585
3.0088
3.6345
4.4321
5.5541
3.7387
4.3077
3.4041
2.9012
3.0078
4.3867
3.0591
3.4179
4.137
4.5014
4.5548
4.0844
4.1302
4.4377
3.0645
2.8534
3.5372
5.6057
4.1483
3.0301
4.1233
4.2546
3.0415
3.3795
2.9207
3.6769
3.5147
3.0072
4.1289
3.4168
3.0865
2.9506
5.6489
4.4157
3.7477
4.1437
3.6371
5.668
4.5034
4.1908
4.2066
4.4533
4.5808
4.1719
3.7826
4.3609
3.1079
3.4019
4.567
3.563
4.5134
2.958
3.6416
5.6593
4.4486
3.0704
3.1099
3.7842
4.5511
5.6449
3.8036
4.4687
3.3353
2.9821
3.0622
4.5365
3.1527
3.3588
4.0673
4.5646
4.5689
4.0799
4.1803
4.5647
3.0548
2.9774
3.627
5.6418
4.16
3.1019
4.0751
4.4526
3.1344
3.3618
2.9777
3.7731
3.6142
3.0674
4.0124
3.3071
3.03
2.9482
5.5874
4.4901
3.8588
4.0541
3.618
5.6415
4.5986
4.1719
4.1923
4.5669
4.5874
4.1884
3.9128
4.4754
3.1855
3.2733
4.5769
3.6474
4.5639
3.0037
3.5929
5.6158
4.4858
3.0407
3.1673
3.8806
];
pd = pd';
 pd = 0.5*pd;
% pd = [pd(:,96*1+1:96*2) pd(:,96*5+1:96*6) pd(:,96*11+1:96*12)];
% pd = [pd(:,96*1+1:96*3) pd(:,96*4+1:96*6) ];
%%
%优化变量
p          = sdpvar(gennum,T,'full');%24时刻优化的机组实时功率p(i,t)
u           = binvar(gennum,T,'full');%24时刻优化状态变量
costH       = sdpvar(gennum,T,'full');%24时刻优化启动成本
costJ       = sdpvar(gennum,T,'full');%24时刻优化关停成本


%约束条件
st1  = [];

%机组出力上下限约束
for     t = 1:T
    for   i = 1:gennum
         st1 = [st1,u(i,t)*Pmin(i)<=p(i,t)];
         st1 = [st1,p(i,t)<=u(i,t)*Pmax(i)];
    end
end
%爬坡约束
% for t=2:T
%     for i=1:gennum
%         st1=st1+[(p(i,t)-p(i,t-1))<=Rud(i,1)*u(i,t-1)+(u(i,t)-u(i,t-1))*Su(i)+(1-u(i,t))*Pmax(i)];%上坡
%         st1=st1+[(p(i,t-1)-p(i,t))<=Rud(i,1)*u(i,t)+(u(i,t-1)-u(i,t))*Sd(i)+(1-u(i,t-1))*Pmax(i)];%下坡
%     end
% end
for t=2:T
    for i=1:gennum
        st1=st1+[-Ru(i)*Pmax(i)<=p(i,t)-p(i,t-1)];%上坡
        st1=st1+[p(i,t)-p(i,t-1)<=Rd(i)*Pmax(i)];%下坡
    end
end
%启动约束
for t=2:T
    for i=1:gennum
        indicator=u(i,t)-u(i,t-1);%启停时间约束的简化表达式（自己推导的）,indicator为1表示启动，为0表示停止
        range=t:min(T,t+lasttime(i)-1);
        st1=st1+[u(i,range)>=indicator];
    end
end
%停机约束
for t=2:T
    for i=1:gennum
        indicator=u(i,t-1)-u(i,t);%启停时间约束
        range=t:min(T,t+lasttime(i)-1);%特别限制时间上限
        st1=st1+[u(i,range)<=1-indicator];
    end
end
% %启停成本约束
% for t=1:T   %启停成本零限约束
%     for i=1:gennum
%         st1=st1+[costH(i,t)>=0];
%         st1=st1+[costJ(i,t)>=0];
%     end
% end
% for i=1:gennum  %启停成本条件约束
%     for t=2:T
%         st1=st1+[costH(i,t)>=H_start(i,1)*(u(i,t)-u(i,t-1))];
%         st1=st1+[costJ(i,t)>=J_stop(i,1)*(u(i,t-1)-u(i,t))];
%     end
% end
% 储能约束
%    C_sto = [C_sto; 0 <= P_s <= 0; 0<= E_s <= 0]; %建模时必须去掉这个约束；测试的时候才用。
bigM = 10^5;
for i = 1:N_s
    st1 = [st1; P_s(i)*H(i)*E_lower(i)<=E_init(i)<=P_s(i)*H(i);];
    E_s(i,1) = E_init(i);
    for t = 1:T
                st1 = [st1; 0<=P_c(i,t)<=P_s(i); 0<=P_dc(i,t)<=P_s(i)];
    end
    for t = 2:T
        E_s(i,t) = E_s(i,t-1)+ P_c(i,t-1)*eff_c(i)-P_dc(i,t-1)/eff_dc(i);
        st1 = [st1; P_s(i)*H(i)*E_lower(i)<=E_s(i,t)<=P_s(i)*H(i)];
    end
end

% 可再生能源发电约束
w_1 = w_ref(data_r);
w_0 = w_1(1:T);
% w_0 = 0.5*w_0;
for t = 1:T
    P_r_max(:,t) = 5*(data_r(:,2))*w_0(t);
    st1 = [st1; 0<=P_r(:,t)<=P_r_max(:,t)]; %新能源出力小于预测的波动出力
    cur(:,t) =  P_r_max(:,t)-P_r(:,t);
    %    C_r = [C_r; 0<=P_r(:,t)<=0];%测试的时候才用。
end
     st1 = [st1; sum(P_r(:,t))>=beta*sum(pd(:,t))]; %新能源渗透约束
     % st1 = [st1; 0<=beta<=0.5]; %新能源出力小于预测的波动出力
    beta_true = sum(P_r(:,t))/sum(sum(pd(:,t)));
    

%负荷平衡约束
        st1=[st1,sum(p)+P_r+P_dc-P_c==pd];

%目标函数
% obj_1 = 0;
% obj_2 = 0;
%     for  t = 1:T
%         for  i = 1:gennum
%             obj_2=obj_2+costH(i,t)+costJ(i,t);%加上机组启停产生的开停机成本
%             obj_1=obj_1+para(i,2)*(BaseMVA*p(i,t));%煤耗成本
%         end
%     end
% obj_3 = 10000*sum(cur);
% obj_4 = 12000000*P_s;
% obj_5 = 300*sum(P_dc+P_c);% 因为用标幺值计算 这里都乘了100
%      obj=obj_1+obj_2+obj_3+obj_4+obj_5;   

obj = 12000000*P_s;

%% 求解
ops1 = sdpsettings('solver', 'cplex','savesolveroutput',1);
% ops1.cplex= cplexoptimset('cplex');
% ops1.cplex.mip.tolerances.absmipgap = 0.01;
result1 = solvesdp(st1,obj,ops1);
solve1=double(obj) ;
p1_double=double(p);
u_double = double(u);
value(P_s)
% value(obj)
P_C = P_dc-P_c;

%% 参数规划
% plp = Opt (st1, totalcost1, beta, P_s); % 利用MPT3工具包建模
% solution = plp.solve; % 利用MPT3工具包求解多参数规划，结果存在solution里面
% 
% for i = 1:N_s
%     figure;
%     solution.xopt.fplot('primal', 'position', i);
%     xlabel('beta');
%     ylabel(sprintf('x_%d(beta)', i));
% end


%% 绘制出力曲线
p_P_r = sum(p) + P_r;
subplot(1,2,1)
bar(value(pd)','stack')%阶梯图
% legend('Unit 1','Unit 2','Unit 3','Unit 4','Unit 5','Unit 6');	%在坐标轴上添加图例
subplot(1,2,2)
bar(value(P_r)','stack')%阶梯图
% legend('Unit 1','Unit 2','Unit 3','Unit 4','Unit 5','Unit 6');	%在坐标轴上添加图例
% stairs(value(p)')
% legend('Unit 1','Unit 2','Unit 3','Unit 4','Unit 5','Unit 6');	%在坐标轴上添加图例
cur_double = double(cur);

net_demand = pd - P_r_max;
net_demand=net_demand';
plot(net_demand)


toc
disp(['运行时间: ',num2str(toc)]);


