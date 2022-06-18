%% Data import
ds = datastore('C:\Users\Admin\term paper'); % Data import
data = readall(ds); % Read all data from the data store
Pool = zeros(120, 100); % Pool of 100 assets during 120 months
for k = 1:100
    Pool(:,k) = data{120*(k-1)+1:120*k,2};
end
r = diff(Pool)./Pool(2:end,:); % Assets return

%% Initializing variables for a loop (for speed)
n_replic = 5000; % Number of replications
gamma = [0.5 3 10]; % Values of gamma
delta_0_re = zeros(n_replic,1); % Replications of CE diff
delta_op_re = zeros(n_replic,1); % Replications of out-of-sample CE diff
penal1_re = zeros(n_replic,1); % Replications of penal due to estimation risk
penal2_re = zeros(n_replic,1); % Replications of penal due to out-of-sample risk
delta_0 = zeros(size(r,2),size(gamma,2)); % Average CE diff
penal1 = zeros(size(r,2),size(gamma,2)); % Average penal due to estimation risk
penal2 = zeros(size(r,2),size(gamma,2)); % Average penal due to out-of-sample risk
delta_op = zeros(size(r,2),size(gamma,2)); % Average out-of-sample CE diff
% Confidence Bounds (CB)
upperCB = zeros(size(r,2),size(gamma,2)); % Upper CB for CE diff
lowerCB = zeros(size(r,2),size(gamma,2)); % Lower CB for CE diff
upper_pen1 = zeros(size(r,2),size(gamma,2)); % Upper CB for penal #1
lower_pen1 = zeros(size(r,2),size(gamma,2)); % Lower CB for penal #1
upper_pen2 = zeros(size(r,2),size(gamma,2)); % Upper CB for penal #2
lower_pen2 = zeros(size(r,2),size(gamma,2)); % Lower CB for penal #2
upperCB_op = zeros(size(r,2),size(gamma,2)); % Upper CB for out-of-sample CE diff
lowerCB_op = zeros(size(r,2),size(gamma,2)); % Lower CB for out-of-sample CE diff
%ill = zeros(size(r,2),n_replic);

%% Evaluation
sz = size(r,1); % Total number of observations
T = ceil(0.9*sz); % Training sample
H = sz-T; % Test sample
rp_g = zeros(1,H); % Return of the GMVP
rp_e = zeros(1,H); % Return of the equally-weighted portfolio
for g = 1:size(gamma,2)
    for N = 1:size(r,2) % N is a potfolio size
        i = ones(N, 1); % Vector of ones
        w_e = 1/N*i; % Weights of the equally-weighted portfolio
        for k = 1:n_replic
            idx = datasample(1:size(r,2),N,'Replace',false); % index for selection of assets
            shares = r(:,idx); % Selected assets' return
            mu = mean(shares)'; % Mean of selected assets' return
            sigm = cov(shares); % Covariance matrix of selected assets' return
            invsigm = inv(sigm); % Inverse covariance matrix
            w_g = (sigm\i)/(i'*(sigm\i)); % Weights of the GMVP
            delta_0_re(k) = (w_g-w_e)'*mu + gamma(g)/2*(w_e'*sigm*w_e-w_g'*sigm*w_g);
            V = 1/(sz-N-1)/(i'*(sigm\i))*...
                (invsigm-(sigm\i*i'/sigm)/(i'*(sigm\i))); % Covariance of the GMVP weights
            penal1_re(k) = -gamma(g)/2*trace(sigm*V);
            penal2_re(k) = -gamma(g)/2*(mu'*V*mu);
            for t = 1:H
                shares = r(t:T+t-1,idx);
                mu = mean(shares)';
                sigm = cov(shares);
                w_g = (sigm\i)/(i'*(sigm\i));
                rp_g(t) = r(T+t,idx)*w_g;
                rp_e(t) = r(T+t,idx)*w_e;
            end
            CE_g = mean(rp_g)-gamma(g)/2*var(rp_g); % Out-of-sample CE of GMVP
            CE_e = mean(rp_e)-gamma(g)/2*var(rp_e); % Out-of-sample CE of 1/N-portfolio
            delta_op_re(k) = CE_g - CE_e;
        end
        delta_0(N,g) = mean(delta_0_re);
        upperCB(N,g) = delta_0(N,g) + 1.96*sqrt(var(delta_0_re)/n_replic);
        lowerCB(N,g) = delta_0(N,g) - 1.96*sqrt(var(delta_0_re)/n_replic);
        penal1(N,g) = mean(penal1_re);
        upper_pen1(N,g) = penal1(N,g) + 1.96*sqrt(var(penal1_re)/n_replic);
        lower_pen1(N,g) = penal1(N,g) - 1.96*sqrt(var(penal1_re)/n_replic);
        penal2(N,g) = mean(penal2_re);
        upper_pen2(N,g) = penal2(N,g) + 1.96*sqrt(var(penal2_re)/n_replic);
        lower_pen2(N,g) = penal2(N,g) - 1.96*sqrt(var(penal2_re)/n_replic);
        delta_op(N,g) = mean(delta_op_re);
        upperCB_op(N,g) = delta_op(N,g) + 1.96*sqrt(var(delta_op_re)/n_replic);
        lowerCB_op(N,g) = delta_op(N,g) - 1.96*sqrt(var(delta_op_re)/n_replic);
    end
end
%% Graphics
figure('Name','Результаты', 'NumberTitle','off');
subplot(2, 2, 1)
hold on
plot(1:N, delta_0,'LineWidth', 1.5)
plot(1:N, upperCB, 'k', 'LineStyle', ':')
plot(1:N, lowerCB, 'k', 'LineStyle', ':')
title('$\Delta_0(g, e)$', 'Interpreter',"latex")
xlabel('Размер портфеля')
ylabel(sprintf('Разность \n результативностей'))
grid on
legend({'$\gamma = 0.5$','$\gamma = 3$','$\gamma = 10$','Доверительный интервал',},...
    'Location', 'northwest', 'Interpreter',"latex")
hold off
subplot(2, 2, 2)
hold on
plot(1:N, delta_op, 'LineWidth',1.5)
plot(1:N, upperCB_op, 'k', 'LineStyle', ':')
plot(1:N, lowerCB_op, 'k', 'LineStyle', ':')
title('$\Delta_{op}(g, e)$', 'Interpreter',"latex")
xlabel('Размер портфеля')
ylabel(sprintf('Разность \n результативностей'))
grid on
legend({'$\gamma = 0.5$','$\gamma = 3$','$\gamma = 10$','Доверительный интервал',},...
    'Location', 'southwest', 'Interpreter',"latex")
hold off
subplot(2, 2, 3)
hold on
plot(1:N, penal1, 'LineWidth', 1.5)
plot(1:N, upper_pen1, 'k', 'LineStyle', ':')
plot(1:N, lower_pen1, 'k', 'LineStyle', ':')
title('$-\frac{\gamma}{2}tr[\Sigma V(\hat\omega)]$', 'Interpreter',"latex")
xlabel('Размер портфеля')
ylabel(sprintf('Штраф 1'))
grid on
legend({'$\gamma = 0.5$','$\gamma = 3$','$\gamma = 10$','Доверительный интервал',},...
    'Location', 'southwest', 'Interpreter',"latex")
hold off
subplot(2, 2, 4)
hold on
plot(1:N, penal2, 'LineWidth',1.5)
plot(1:N, upper_pen2, 'k', 'LineStyle', ':')
plot(1:N, lower_pen2, 'k', 'LineStyle', ':')
title('$-\frac{\gamma}{2}\mu^T V(\hat\omega)\mu$', 'Interpreter',"latex")
xlabel('Размер портфеля')
ylabel(sprintf('Штраф 2'))
grid on
legend({'$\gamma = 0.5$','$\gamma = 3$','$\gamma = 10$','Доверительный интервал',},...
    'Location', 'southwest', 'Interpreter',"latex")
hold off