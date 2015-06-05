%         (1) worst-case of the optimized gradient method with respect to
%             objective value at the best iterate, 20 iterations. Solver
%             set to Mosek with tolerance 1e-10.

clear P A C S;
P.L=1; P.mu=0; P.R=1;
A.name='OGM2'; A.N=40;
C.name='Obj'; S.solver='mosek'; S.tol=1e-10;
S.verb=0;
[val, Sol, Prob]=pep_yalmip(P,A,C,S); format long; val

%%
%         (2) worst-case of the gradient method with h=1.5 with respect to
%             last gradient norm, 10 iterations. Default Yalmip solver and
%             tolerance.

clear P A C S;
P.L=1; P.mu=0; P.R=1;
A.name='GM'; A.N=10; A.stepsize=1.5;
C.name='Grad';

[val, Sol, Prob]=pep_yalmip(P,A,C); format long; val

%%
%         (3) worst-case of the fast gradient method with respect to
%             best gradient norm, 5 iterations. Solver set to Sedumi
%             tolerance 1e-9.

clear P A C S;
P.L=1; P.mu=0; P.R=1;
A.name='FGM1'; A.N=5;
C.name='MinGrad';
S.solver='sedumi'; S.tol=1e-9;

[val, Sol, Prob]=pep_yalmip(P,A,C,S); format long; val

%%
%         (4) worst-case of the unit-step gradient method with respect to
%             best gradient norm, 2 iterations. Solver set to Sedumi
%             with tolerance 1e-9. 2 ways of doing this: via the 'Custom'
%             and via the 'GM' options.
%
clear P A C S;
P.L=1; P.mu=0; P.R=1;
A.name='Custom'; A.N=2; C.name='MinGrad';
S.solver='sedumi'; S.tol=1e-9; S.verb=0;
A.stepsize=[1 0 ; 1 1];
[val, Sol, Prob]=pep_yalmip(P,A,C,S); format long; val

clear P A C S;
P.L=1; P.mu=0; P.R=1;
A.name='GM'; A.N=2; C.name='MinGrad';
S.solver='sedumi'; S.tol=1e-9; S.verb=0;
A.stepsize=1;
[val, Sol, Prob]=pep_yalmip(P,A,C,S); format long; val

%%
%         (5) worst-case of the gradient method with h=1.5 with respect to
%             objective value of the average of the iterates,
%             10 iterations. Default Yalmip solver and tolerance.

clear P A C S;
P.L=1; P.mu=0; P.R=1;
A.name='GM'; A.N=10; A.stepsize=1.5;
C.name='AvgObj';
[val, Sol, Prob]=pep_yalmip(P,A,C); format long; val
