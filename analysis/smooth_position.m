function Pstar = smooth_position( P, delta_t, alpha, lambda, eps )
%SMOOTH_POSITION smooth a sequence of positions

%see the following citation for details:
    
%"Noise Smoothing for VR Equipment in Quaternions," C.C. Hsieh,
%Y.C. Fang, M.E. Wang, C.K. Wang, M.J. Kim, S.Y. Shin, and
%T.C. Woo, IIE Transactions, vol. 30, no. 7, pp. 581-587, 1998
%    
% inputs
% ------
%
% P       positions (m by n matrix, m = dimensionality, n = number of samples)
% delta_t temporal interval, in seconds, between samples
% alpha   relative importance of acceleration versus position
% lambda  step size when descending gradient
% eps     termination criterion
%
% output
% ------
%
% Pstar   smoothed positions (same format as P)

h = delta_t;

Pstar = P;

err = 2*eps; % cycle at least once
while err > eps
    del_F = get_del_F( P, Pstar, h, alpha);
    err = sum(sum(del_F.^2));
    disp(sprintf('err=%e',err))
    Pstar = Pstar - lambda*del_F;
end
    
function F = objective_function( P, Pstar, h, alpha )
D = sum(sum( (P - Pstar).^2)); % distance (L2 norm)

d2p = (Pstar(:,3:end) - 2*Pstar(:,2:end-1) + Pstar(:,1:end-2))/ h^2; % 2nd derivative
E = sum(sum( d2p.^2 )); % "energy" (smoothness of 2nd derivative)

F = D + alpha*E; % eqn 22


function PDs = get_del_F( P, Pstar, h, alpha )

ndims = size(P,1);
npoints = size(P,2);

cached_F_Pstar = objective_function( P, Pstar, h, alpha );
PDs = [];
for j = 1:npoints
%    PDs(i) = eval_pd(i, P, Pstar, cached_F_Pstar, h, alpha);
    PDs = [ PDs eval_pd(j, P, Pstar, cached_F_Pstar, h, alpha) ];
end


function dFdP = eval_pd(j, P, Pstar, cached_F_Pstar, h, alpha)

PERTURB = 1e-6; % perturbation amount (in units of P)
ndims = size(P,1);

Phat = Pstar;
F_Pi=[];
for i=1:ndims
    P_i_j = Phat(i,j);
    % temporarily perturb Phat
    Phat(i,j) = P_i_j + PERTURB;
    F_Pi(i) = objective_function( P, Phat, h, alpha );
    Phat(i,j) = P_i_j; 
end

dFdP = ((F_Pi - cached_F_Pstar)/PERTURB)';