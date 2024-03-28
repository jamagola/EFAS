%% CEFAS SID : Golam Gause Jaman 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            CEFAS SID              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;
%% Load Data
T0=25;
gma=14.053;
beta=0.2;
zelta=0.1;
material=2; % 1: Graphite and 2: Aluminium
if material == 1
    m=0.3888;
    Cp=720;
    R=0.003867;
else
    m=0.4878;
    Cp=902;
    R=5.453*10^(-6);
end

commonTF=tf([1*zelta],[1 beta*gma/(m*Cp)]) % 1/(s + gamma/mC)

matrix0=readmatrix('sidCEFAS.xlsx');
time=matrix0(1:3369,2);
volt=matrix0(1:3369,3);
amps=matrix0(1:3369,4);

f_V=((volt.^2)/(m*Cp*R)) + (gma*T0/(m*Cp));
t=linspace(0,time(end),length(time));
t=t';
T=lsim(commonTF,f_V,t);

figure(1)
subplot(3,1,1);
plot(t,volt,'r');
xlabel('time (s)');
ylabel('voltage');

subplot(3,1,2);
plot(t,amps,'k');
xlabel('time (s)');
ylabel('amps');

subplot(3,1,3);
plot(t,T,'r');
xlabel('time (s)');
ylabel('temperature (celcius)');


Ts=mean(time(2:end,1)-time(1:end-1,1));
gamma=4;
%% SID Construction
u_=volt;
y_=T;
data_=iddata(y_,u_,Ts);

nx=[1:10];
sys_d=n4sid(data_, nx)
sys=d2c(sys_d)
[num,den]=ss2tf(sys.A, sys.B, sys.C, sys.D);
model0=tf(num,den)

poles_=length(eig(model0));
model1=tfest(data_, poles_)
model2=c2d(model0,Ts)

figure(2)
%compare(data_,sys_d)
compare(data_,model1)

% Best model (validation fit%) appeared : transfer function (4 pole, 3 zero)
% and NARX

model3=tfest(data_, 4)
model4=c2d(model3,Ts)
%%
% Apply SID GUI to find best model and apply in simulink defining the
% PID/LQG/RL controller.

systemIdentification
%nntart

%% Evaluate continuous plant
EIG=eig(model1)
% Impulse response
figure(3);
impulse(model1);
grid on;
% Step response
figure(4);
step(model1);
grid on;
% Margin (Bode)
figure(5);
margin(model1);
grid on;
% PZ-map
figure(6);
pzmap(model1);
grid on;
% Nyquist plot
figure(7);
nyquist(model1);
grid on;


Fs=1/Ts;
L=length(time);
Freq=(Fs/L)*(0:L-1);

Y_=fft(y_);
figure(8)
plot(Freq(500:end-500), abs(Y_(500:end-500)), 'k', 'linewidth', 1);
xlabel('Hz');
ylabel('|fft(y)|');
title('FFT of y')
grid on;

U_=fft(u_);
figure(9)
plot(Freq(500:end-500), abs(U_(500:end-500)), 'b', 'linewidth', 1);
xlabel('Hz');
ylabel('|fft(u)|');
title('FFT of u')
grid on;


model1_ss=ss(model1)
model2_ss=ss(model2)

control=rank(ctrb(model2_ss.A, model2_ss.B))
observe=rank(obsv(model2_ss.A, model2_ss.C))

test_ss=ss(model2)


% LQR/LQG work, assuming A,B,C,D creates observability and controllability
%Q=1*[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]; % state x state
%R=[5]; % 1 x 1 : (u x u)
%[K,S,P] = lqr(model4_ss,Q,R)
%[K,S,P] = lqr(test_ss,0.1,0.1)

% A=model4_ss.A;
% B=model4_ss.B;
% C=model4_ss.C;
% D=model4_ss.D;

A=test_ss.A;
B=test_ss.B;
C=test_ss.C;
D=test_ss.D;

%result=inv([A,B;C,D])*[0;0;0;0;1]
% result=inv([A,B;C,D])*[0;1]
% 
% Nx=result(1)
% Nu=result(2)
% Nbar=Nu+K*Nx

% Nx=result(1:4)
% Nu=result(5)
% Nbar=Nu+K*Nx
% sysLQR=ss((A-B*K),B,C,D,Ts)
