%% CEFAS modelling work
% Golam Gause Jaman

clear all; close all; clc;

%%
% RLC Circuit system
R_=1;
L_=1;
C_=1;

A=[0, 1; -1/(L_*C_), -1/(R_*C_)];
B=[0; 1/(R_*L_*C_)];
C=[1,0];
D=[0];

sys_ss=ss(A,B,C,D)
[num,den]=ss2tf(sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D);
sys_tf=tf(num,den)

figure(1)
impulse(sys_ss);
figure(2);
step(sys_ss);

% Poles & Zeroes
eigens=eigs(A)
figure(3)
pzmap(sys_tf)
[p,z]=pzmap(sys_ss)

t=[0:0.1:10];
u=[ones(1,20), zeros(1,30), ones(1,10), zeros(1,5), ones(1,15), zeros(1,15), ones(1,6)];

y=lsim(sys_tf, u, t);

figure(4)
plot(t,y,'r',t,u,'g');
legend('response', 'input');

%%
% Transfer function estimation

%matrix0=readmatrix('sidCEFAS.xlsx');
matrix0=readmatrix('CEFAStest08072023.xlsx');

time=matrix0(:,2);
volt=matrix0(:,3);
amps=matrix0(:,4);
temp=matrix0(:,5);

u=u';
t=t';
Ts=0.1;
Ts_=mean(time(2:end)-time(1:end-1));

data=iddata(temp,volt,Ts_);

np=4;
modelTF=tfest(data,np)
num_=modelTF.Numerator;
den_=modelTF.Denominator;

modelDiscrete=c2d(tf(num_,den_), Ts_)
numD=cell2mat(modelDiscrete.Numerator);
denD=cell2mat(modelDiscrete.Denominator);

%%
% NARX model construction
net = narxnet(1:2,1:2,10); % Input delay/feedback delay/hidden neurons
% https://www.mathworks.com/help/deeplearning/ug/design-time-series-narx-feedback-neural-networks.html
% https://www.mathworks.com/help/deeplearning/ref/preparets.html

u_=num2cell(u');
y_=num2cell(y');

net.trainParam.min_grad = 1e-10;
net.trainFcn = 'trainlm';
[p_,Pi,Ai,t_] = preparets(net,u_,{},y_);

net = train(net,p_,t_,Pi);
net_closed = closeloop(net);

[p_,Pi,Ai,t_] = preparets(net_closed,u_,{},y_);
yp = net_closed(p_,Pi,Ai);
e = cell2mat(yp)-cell2mat(t_);
figure(5);
plot(e);
title('error');

view(net_closed)
% Open SID Application, Fuzzy, NNstart and PID tuner if GUI is relevant.