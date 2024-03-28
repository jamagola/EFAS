
%% CEFAS Reinforcement Learning Work
% *Golam Gause Jaman*

clc; clear all; close all;

%% Set parameter
Reward=readfis('Reward.fis')
load idpoly
% Observation info
% State >= 0
obsInfo = rlNumericSpec([3 1],'LowerLimit',[-1.5 -inf 0]','UpperLimit',[ inf  inf inf]');
%obsInfo = rlNumericSpec([3 1]);
% Name and description are optional and not used by the software
obsInfo.Name = "observations";
obsInfo.Description = "error, delta-error, state";

% Action info
actInfo = rlNumericSpec([1 1], 'LowerLimit',[0]', 'UpperLimit',[10]'); %%%%%%%%%%%%%%%%%%%%%%%%
actInfo.Name = "act";

%%
env = rlSimulinkEnv("cefasRL","cefasRL/RL Agent",...
    obsInfo,actInfo);

env.ResetFcn = @(in)localResetFcn(in);

% In seconds
Ts = 1.2273;
Tf = 1000;
gamma=4;
rng(0);

%% Create Critic
% Observation path
obsPath = [
    featureInputLayer(obsInfo.Dimension(1),'Name','obsInLyr'), ...
    fullyConnectedLayer(50, 'Name', 'obsHidden'), ...
    reluLayer('Name','ReLu'), ...
    fullyConnectedLayer(25,'Name','obsPathOutLyr'), ...
    ];

% Action path
actPath = [
    featureInputLayer(actInfo.Dimension(1),'Name','actInLyr'), ...
    fullyConnectedLayer(25,'Name','actPathOutLyr'), ...
    ];

% Common path
commonPath = [
    additionLayer(2,'Name','add'), ...
    reluLayer('Name','ReLu2'), ...
    fullyConnectedLayer(1,'Name','QValue'), ...
    ];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,obsPath);
criticNetwork = addLayers(criticNetwork,actPath);
criticNetwork = addLayers(criticNetwork,commonPath);

criticNetwork = connectLayers(criticNetwork, ...
    "obsPathOutLyr","add/in1");
criticNetwork = connectLayers(criticNetwork, ...
    "actPathOutLyr","add/in2");

figure(1)
plot(criticNetwork)

criticNetwork = dlnetwork(criticNetwork);
summary(criticNetwork) % Matlab 2022

critic = rlQValueFunction(criticNetwork, ...
    obsInfo,actInfo, ...
    'ObservationInputNames',"obsInLyr", ...
    'ActionInputNames',"actInLyr");

%% Create the Actor

actorNetwork = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(3)
    tanhLayer
    fullyConnectedLayer(actInfo.Dimension(1))
    ];

actorNetwork = dlnetwork(actorNetwork);
summary(actorNetwork)

figure(2)
plot(actorNetwork)

actor = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo);

%% DDPG AGENT
agent = rlDDPGAgent(actor,critic);

agent.SampleTime = Ts;

agent.AgentOptions.TargetSmoothFactor = 1e-3;
agent.AgentOptions.DiscountFactor = 1.0;
agent.AgentOptions.MiniBatchSize = 64;
agent.AgentOptions.ExperienceBufferLength = 1e6; 

agent.AgentOptions.NoiseOptions.Variance = 0.3;
agent.AgentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-03;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-04;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;

%% Train agent
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000, ...
    'MaxStepsPerEpisode',ceil(Tf/Ts), ...
    'ScoreAveragingWindowLength',20, ...
    'Verbose',false, ...
    'Plots',"training-progress",...
    'StopTrainingCriteria',"AverageReward",...
    'StopTrainingValue',50000); %%%%%%%%%%%%%%%%%%ATTENTION%%%%%%%%%%%%%%%%%%%%

doTraining = true; %%%%%%%%%%% TRUE %%%%%%%%%%%%%

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    % save('RLagent','agent')
    load("RLagen.mat","agent") %%% SAVE TRAINED AGENT FIRST %%%
end

%% Reset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function in = localResetFcn(in)

% Randomize reference signal
blk = sprintf("cefasRL/Desired");
value = 100+round(200*rand());
in = setBlockParameter(in,blk,'Value',num2str(value));
%disp(value)
% Randomize initial height
%y=0; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%blk = "cefasRL/Environment/y";
%in = setBlockParameter(in,blk,'InitialCondition',num2str(y));

end