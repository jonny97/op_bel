rng(0);

% number of realizations to generate
N = 2000;

% parameters for the Gaussian random field
gamma = 2;
tau = 5;
sigma = 5^(2);

% viscosity
visc = 1/1000;

% grid size
s = 8192;
num_cheb = 512;
steps = 200;


input = zeros(N, s);
output = zeros(N, s);
coef = zeros(N, num_cheb, 2);

tspan = linspace(0,0.1,steps+1);
x = linspace(0,1,s+1);
for j=1:N
    [u0,ab] = GRF2(num_cheb, 0, gamma, tau, sigma);

    u = burgers1(u0, tspan, s, visc);
    
    u0eval = u0(x);
    input(j,:) = u0eval(1:end-1);
    
    %for k=1:(steps+1)
    %    output(j,k,:) = u{k}.values; % these values are evaluated on
    %    chebychev points instead of linspace
    %end
    output(j,:) = u{end}(x(1:end-1));
    coef(j,:,:) = ab;
    disp(j);
end

%save('dataGRFshorttspan.mat', 'input', 'output', 'coef');
