% function kalman(duration, dt)
%
% Kalman filter simulation for a train's 1D postion and velocity utilising two sensors.
%
% E = covariance
%
close all;

% INPUTS
duration = 20;    %length of simulation (seconds)
dt = 0.1;          %step size (seconds)

% Measurement noise due to imperfect sensors.
positionnoise = 10; % laser height measurement noise (m)
velocitynoise = 10;  % GPS velocity measurement noise (m/s)
% External noise due to imperfect replication of instruction wrt the environment.
accelnoise = 10; % acceleration noise (m/sec^2)

% Computational matrices.
F = [1 dt; 
     0 1]; % transition matrix, used to extrapolate the new state from the previous position and velocity ignoring acceleration
B = [dt^2/2; dt]; % input matrix, used to apply the effect of acceleration on the new state when extrapolating from the old
H = [1 0; 
     0 1]; % measurement matrix, maps the measurements onto the state space

% Initial state.
x = [0; 0]; % initial state vector, [position; velocity]
xhat = x; % initial state estimate

R = [positionnoise^2 0; 0 velocitynoise^2];  % measurement error E
Qu = accelnoise^2 * (B * B'); % process noise E (of xhat); describes the effect of process noise on estimated position and velocity
P = Qu; % initial estimation E (of xhat)

% Initialize arrays for later plotting.
pos = [];       % true position array
poshat = [];    % estimated position array
posmeas = [];   % measured position array
vel = [];       % true velocity array
velmeas = [];   % measures velocity array
velhat = [];    % estimated velocity array

for t = 0 : dt: duration,
    % Use a constant commanded acceleration in m/sec^2.
    u = 1;
    % Simulate the linear system.
    ProcessNoise = accelnoise * [(dt^2/2)*randn; dt*randn];
    x = F * x + B * u + ProcessNoise;
    % Simulate the noisy measurements
    MeasNoise = [positionnoise * randn; velocitynoise * randn];   %randomly weighted measurement noise
    z = H * x + MeasNoise;   %actual position with added noise to create simulated (mean) measurement readings
    % Extrapolate the most recent state estimate to the present time; 
    % this is the new prediction based on extrapolation from the position 
    % and velocity of the previous state plus any compensation that needs 
    % to be made due to acceleration.
    xhat = F * xhat + B * u;
    % Form the Innovation vector; 
    % measured position - extrapolated position, describes the measurement 
    % residual.
    Inn = z - H * xhat;
    % Compute the E of the Innovation; 
    % Epp + measurement error E.
    InnE = H * P * H' + R;    
    % Form the Kalman Gain matrix;
    % this represents the proportional confidence we have in our 
    % measurements vs our extrapolated prediction.
    K = (((F * P * F') + Qu) * H') / InnE;
    % Update the state estimate;
    % the estimate plus the confidence weighted innovation.
    xhat = xhat + K * Inn;
    % Compute the E of the estimation error for next iteration.
    P = ((F * P * F') + Qu) - (K * H * (F * P * F' + Qu));
    % Save some parameters for plotting later.
    pos = [pos; x(1)];
    posmeas = [posmeas; z(1)];
    poshat = [poshat; xhat(1)];
    vel = [vel; x(2)];
    velmeas = [velmeas; z(2)];
    velhat = [velhat; xhat(2)];

end

% Plot the results
t = 0 : dt : duration;

subplot(2,2,1);
plot(t,pos, t,posmeas, t,poshat);
grid;
xlabel('Time (sec)');
ylabel('Position (m)');
title('Figure 1 - Vehicle Position (True, Measured, and Estimated)')
legend('True ','Measured','Estimated');

subplot(2,2,2);
plot(t,pos-posmeas, t,pos-poshat);
grid;
xlabel('Time (sec)');
ylabel('Position Error (m)');
title('Figure 2 - Position Measurement Error and Position Estimation Error');
legend('Measurement Error','Estimation Error');

subplot(2,2,3);
plot(t,vel, t,velmeas, t,velhat);
grid;
xlabel('Time (sec)');
ylabel('Velocity (m/sec)');
title('Figure 3 - Velocity (True, Measured, and Estimated)');
legend('True ','Measured','Estimated');

subplot(2,2,4);
plot(t,vel-velmeas, t,vel-velhat);
grid;
xlabel('Time (sec)');
ylabel('Velocity Error (m/sec)');
title('Figure 4 - Velocity Measurement Error and Velocity Estimation Error');
legend('Measurement Error','Estimation Error');