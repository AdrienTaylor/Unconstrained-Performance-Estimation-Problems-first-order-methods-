%   Author: A. Taylor; Universite catholique de Louvain.
%   Date:   March 11, 2015
%
%
%   Version: June 2, 2015
%
%
% ---- Performance Estimation Problem [THG][DT][KF] routine using YALMIP [Lof]
%
% [THG] A.B. Taylor, J.M. Hendrickx, F. Glineur. "Smooth Strongly Convex
%       Interpolation and Exact Worst-case Performance of First-order
%       Methods." (2015).
% [DT] Y. Drori, M. Teboulle. "Performance of first-order methods for
%      smooth convex minimization: a novel approach."
%      Mathematical Programming 145.1-2 (2014): 451-482.
% [KF] D. Kim, J.A. Fessler. "Optimized first-order methods for smooth
%      convex minimization" (2014).
% [Lof] J. Lofberg. "Yalmip A toolbox for modeling and optimization in
%       MATLAB". Proceedings of the CACSD Conference (2004).
%
%
%
%   inputs:
%         - P: problem class structure  (P.L, P.mu, P.R)
%         - A: algorithm structure      (A.name, A.stepsize, A.N)
%         - C: criterion structure      (C.name)
%         - S: solver and problem attributes (S.tol, S.verb, S.structure, S.relax)
%         (see below for detailed descriptions)
%
%   outputs:
%         - val: structure containing
%              val.primal: primal optimal value
%              val.dual:   dual optimal value
%              val.conj:   conjectured optimal value 
%                          (NaN if no conjecture is available)
%           Primal and dual values may correspond to non-feasible points
%           (up to numerical and solver precision), hence the interval
%           [primal dual] may not contain the conjectured result. 
%         - Sol: structure containing
%              Sol.G:      Gram matrix of gradients and x0 (primal PSD matrix)
%              Sol.f:      Function values (primal linear variables)
%              Sol.S :     Dual PSD matrix 
%              Sol.lambda: multipliers of linear inequalities
%              Sol.err:    error code returned by the solver 
%         - Prob: summary of the problem setting (to help diagnose problems
%           in case something went wrong in the input arguments)
%
%   Problem class structure:
%         - P.L:    Lipschitz constant (default L=1)
%         - P.mu:   strong convexity constant mu (default mu=0)
%         - P.R:    bound on distance to optimal solution R (default R=1)
%
%   Algorithm structure:
%         - A.name: name of the algorithm to be chosen among
%              'GM' for the gradient method GM
%              'FGM1' for the fast gradient method FGM (primary sequence)
%              'FGM2' for the fast gradient method FGM (secondary sequence)
%              'OGM1' for the optimized method OGM (primary* sequence)
%              'OGM2' for the optimized method OGM (secondary* sequence)
%              'Custom' for a custom fixed-step algorithm (see below)
%           (* note that [KF] presents OGM in terms of its secondary seq.)
%         - A.stepsize: stepsize coefficients
%              'GM' method: scalar stepsize (default h=1.5)
%              'FGM' or 'OGM' method: not applicable
%              'Custom' method: contains an NxN matrix of coefficients H
%               such that each step of the method corresponds to 
%               x_i = x_0 - 1/L * sum_{k=1}^{i-1} H(i,k) * g_{k-1}
%         - A.N: number of iteration (default N=1)
%
%   Criterion structure:
%         - C.name: criterion name to be chosen among
%              'Obj' for the objective value of the last iterate
%              'Grad' for the residual gradient norm of the last iterate
%              'MinGrad' for the smallest radient norm among all iterates
%              'Dist' for the distance between last iterate and opt. solution 
%              'AvgObj' for the objective value at the averaged iterate
%                       f((x1+x2+...+xN)/N)
%
%   Solver and problem attributes:
%         - S.tol: tolerance for SDP solver (default 1e-8)
%         - S.verb: verbose mode (0 or 1 ; default 1)
%         - S.relax: use relaxation proposed in [DT] (0 or 1 ; default 0)
%         - S.solver: 'sedumi' or 'mosek' (default: yalmip's default)
%
%
%   Examples:
%         (1) worst-case of the optimized gradient method with respect to
%             objective value at the best iterate, 20 iterations. Solver
%             set to Mosek with tolerance 1e-10.
%
%           clear P, A, C, S;
%           P.L=1; P.mu=0; P.R=1;
%           A.name='OGM2'; A.N=40; 
%           C.name='Obj'; S.solver='mosek'; S.tol=1e-10;
%           S.verb=0;
%           [val, Sol, Prob]=pep_yalmip(P,A,C,S); format long; val
%
%         (2) worst-case of the gradient method with h=1.5 with respect to
%             last gradient norm, 10 iterations. Default Yalmip solver and
%             tolerance.
%
%           clear P, A, C, S;
%           P.L=1; P.mu=0; P.R=1;
%           A.name='GM'; A.N=10; A.stepsize=1.5;
%           C.name='Grad';
% 
%           [val, Sol, Prob]=pep_yalmip(P,A,C); format long; val
%
%         (3) worst-case of the fast gradient method with respect to
%             best gradient norm, 5 iterations. Solver set to Sedumi
%             tolerance 1e-9.
%
%           clear P, A, C, S;
%           P.L=1; P.mu=0; P.R=1;
%           A.name='FGM1'; A.N=5;
%           C.name='MinGrad';
%           S.solver='sedumi'; S.tol=1e-9;
% 
%           [val, Sol, Prob]=pep_yalmip(P,A,C,S); format long; val
%
%         (4) worst-case of the unit-step gradient method with respect to
%             best gradient norm, 2 iterations. Solver set to Sedumi
%             with tolerance 1e-9. 2 ways of doing this: via the 'Custom'
%             and via the 'GM' options.
%
%           clear P, A, C, S;
%           P.L=1; P.mu=0; P.R=1;
%           A.name='Custom'; A.N=2; C.name='MinGrad';
%           S.solver='sedumi'; S.tol=1e-9; S.verb=0;
%           A.stepsize=[1 0 ; 1 1];
%           [val, Sol, Prob]=pep_yalmip(P,A,C,S); format long; val
%           
%           clear P, A, C, S;
%           P.L=1; P.mu=0; P.R=1;
%           A.name='GM'; A.N=2; C.name='MinGrad';
%           S.solver='sedumi'; S.tol=1e-9; S.verb=0;
%           A.stepsize=1;
%           [val, Sol, Prob]=pep_yalmip(P,A,C,S); format long; val
%
%         (5) worst-case of the gradient method with h=1.5 with respect to
%             objective value of the average of the iterates, 
%             10 iterations. Default Yalmip solver and tolerance.
%
%           clear P, A, C, S;
%           P.L=1; P.mu=0; P.R=1;
%           A.name='GM'; A.N=10; A.stepsize=1.5;
%           C.name='AvgObj';
%           [val, Sol, Prob]=pep_yalmip(P,A,C); format long; val
