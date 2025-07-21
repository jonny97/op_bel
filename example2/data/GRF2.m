%Radom function from N(m, C) on [0 1] where
%C = sigma^2(-Delta + tau^2 I)^(-gamma)
%with periodic, zero dirichlet, and zero neumann boundary.
%Dirichlet only supports m = 0.
%N is the # of Fourier modes, usually, grid size / 2.
% also get actual coeffcients for experimental purposes
function [u, coef] = GRF2(N, m, gamma, tau, sigma)

my_eigs = sqrt(2)*(abs(sigma).*((2*pi.*(1:N)').^2 + tau^2).^(-gamma/2));

xi_alpha = randn(N,1);
xi_beta = randn(N,1);

alpha = my_eigs.*xi_alpha;
beta = my_eigs.*xi_beta;

a = alpha/2;
b = -beta/2;

coef = [a,b];
c = [flipud(a) - flipud(b).*1i;m + 0*1i;a + b.*1i];

uu = chebfun(c, [0 1], 'trig', 'coeffs');
u = chebfun(@(t) uu(t - 0.5), [0 1], 'trig');
