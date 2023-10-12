%% CEFAS SID : Golam Gause Jaman 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            CEFAS SID              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;
%% Load Data
matrix0=readmatrix('sidCEFAS.xlsx');
time=matrix0(:,2);
volt=matrix0(:,3);
amps=matrix0(:,4);
temp=matrix0(:,5);

figure(1)
subplot(3,1,1);
plot(time,volt,'r');
xlabel('time (s)');
ylabel('voltage');

subplot(3,1,2);
plot(time,amps,'k');
xlabel('time (s)');
ylabel('amps');

subplot(3,1,3);
plot(time,temp,'b');
xlabel('time (s)');
ylabel('temperature (celcius)');


Ts=mean(time(2:end,1)-time(1:end-1,1));
gamma=4;
%% SID Construction
u_=volt;
y_=temp;
data_=iddata(y_,u_,Ts);

nx=[1:10];
sys_d=n4sid(data_, nx)
sys=d2c(sys_d)
[num,den]=ss2tf(sys.A, sys.B, sys.C, sys.D);
model0=tf(num,den)

poles_=length(eig(model0));
model1=tfest(data_, poles_)
model2=c2d(model0,Ts)

% Best model (validation fit%) appeared : transfer function (4 pole, 3 zero)
% and NARX

model3=tfest(data_, 4)
figure(2)
%compare(data_,sys_d)
compare(data_,model3)
model4=c2d(model3,Ts)
%%
% Apply SID GUI to find best model and apply in simulink defining the
% PID/LQG/RL controller.

systemIdentification
%nntart

%% Evaluate continuous plant
EIG=eig(model3)
% Impulse response
figure(3);
impulse(model3);
grid on;
% Step response
figure(4);
step(model4);
grid on;
% Margin (Bode)
figure(5);
margin(model3);
grid on;
% PZ-map
figure(6);
pzmap(model3);
grid on;
% Nyquist plot
figure(7);
nyquist(model3);
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

sysN=nlarx(data_,[4,4,1]);
figure(10)
compare(data_,sysN)