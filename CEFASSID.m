%% CEFAS SID : Golam Gause Jaman 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            CEFAS SID              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;
%% Test plant

N=[1];
D=[1, 0.5, 1];
Ts=1; % Sampling rate in sec.
Tf=100;
t=[0:Ts:Tf];
u=[1,zeros(1,length(t)-1)]';
%%
% *Continuous domain*
%%
G=tf(N,D) % Dummy Plant
Gss=ss(G) % State-Space of the plant
EIG=eig(G)
% Impulse response
figure(1);
impulse(G);
grid on;
% Step response
figure(2);
step(G);
grid on;
% Margin (Bode)
figure(3);
margin(G);
grid on;
% PZ-map
figure(4);
pzmap(G);
grid on;
% Nyquist plot
figure(5);
nyquist(G);
grid on;


%%
% *Discrete domain*
%%
G_d=c2d(G, Ts)
Gss_d=c2d(Gss, Ts)
X0=[0,0]'; % Initial condition
y=lsim(Gss,u,t,X0,'foh');
figure(6);
plot(t',y,'k');
xlabel('time (s)');
ylabel('Response');
title('Ground truth');
grid on;

noise=0.02*randn(size(y));
y_=y+noise;
figure(7)
plot(t',y_,'r');
xlabel('time (s)');
ylabel('Response');
title('Ground truth - added noise');
grid on;

u_=[ones(1,10),zeros(1,10),round(rand(1,length(t)-20))]';
y_=(0.02*randn(size(t)))'+lsim(Gss,u_,t,X0,'foh');
y2=lsim(Gss,u_,t,X0,'foh');

figure(8)
plot(t',y_,'g');
xlabel('time (s)');
ylabel('Response');
title('Dummy Experiment Data');
grid on;

figure(9)
plot(t',u_,'m','linewidth',2);
xlabel('time (s)');
ylabel('Input');
title('Dummy Experiment Data');
grid on;

%% SID Construction
%%
Fs=1/Ts;
L=length(t);
Freq=(Fs/L)*(0:L-1);

Y_=fft(y_);
figure(10)
plot(Freq, abs(Y_), 'k', 'linewidth', 2);
xlabel('Hz');
ylabel('|fft(y)|');
title('FFT of y')
grid on;


U_=fft(u_);
figure(11)
plot(Freq, abs(U_), 'b', 'linewidth', 2);
xlabel('Hz');
ylabel('|fft(u)|');
title('FFT of u')
grid on;

data_=iddata(y_,u_,Ts);
data=iddata(y2,u_,Ts);
nx=[1:10];
sys_d=n4sid(data_, nx)
sys=d2c(sys_d)
[num,den]=ss2tf(sys.A, sys.B, sys.C, sys.D);
model0=tf(num,den)

figure(12)
compare(data_,sys_d)
poles_=length(eig(model0));
model1=tfest(data_, poles_)

%%
% Apply SID GUI to find best model and apply in simulink defining the
% PID/LQG/RL controller.

systemIdentification