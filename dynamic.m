%% Dynamic system : RLC
% Golam Gause Jaman

clc; clear all; close all;

% Differential Equation
% $ \frac{\partial^2 i_L}{\partial t^2} + \frac{1}{RC}\frac{\partial i_L}{\partial t} + \frac{1}{LC}i_L 
% = \frac{v}{RLC}

L_=1; % L - Inductor
C_=1; % C - Capacitor
R_=1; % R - Resistor

syms t R L C v(t) i_L(t)
Di_L = diff(i_L,t);
DDi_L = diff(i_L,t,t);
DEq = DDi_L + Di_L/(R*C) + i_L/(L*C) == v/(R*L*C)
IC=[i_L(0)==0, Di_L(0)==0]

soln=dsolve(DEq, IC)

% Placing R_ L_C_ and State Space presentation:

A=[0 1; -1/(L_*C_), -1/(R_*C_)];
B=[0; 1/(R_*L_*C_)];
C=[1,0];
D=[0];

num=[1/(R_*L_*C_)];
den=[1 1/(R_*C_) 1/(L_*C_)];

G=tf(num,den)
Gss=ss(A,B,C,D)

figure(1)
step(G)

figure(2)
pzmap(G)
grid on;
eig(G)

figure(3)
margin(G)
grid on;

figure(4)
nyquist(G)
grid on;


Ts=0.1;
Tf=10;
Gd=c2d(G, Ts)
t=[0:Ts:Tf];
amp=1;
u=[0, amp*ones(1,length(t)-1)];
y=lsim(Gd,u);
figure(5)
plot(t,y');
xlabel('time(s)');
ylabel('Response');
grid on;

% noise
sigma=1;
mu=0;
noise=mu+sigma*randn(1,length(y));

% Generate SID data and develop model - Test controller 