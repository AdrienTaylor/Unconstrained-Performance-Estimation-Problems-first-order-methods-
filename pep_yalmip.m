function [val, Sol, Prob]=pep_yalmip(P,A,C,S)
%
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

%% Parameter association:

if nargin>=1
    if isfield(P,'L')
        L=P.L;
    else
        L=1;
    end
    if isfield(P,'mu')
        mu=P.mu;
    else
        mu=0;
    end
    if isfield(P,'R')
        R=P.R;
    else
        R=1;
    end
else
    L=1;R=1;mu=0;
end

if nargin>=2
    if isfield(A,'name')
        method=A.name;
    else
        method='GM';
    end
    if isfield(A,'stepsize')
        h=A.stepsize;
    else
        h=1.5;
    end
    if isfield(A,'N')
        N=A.N;
    else
        N=1;
    end
else
    method='GM';h=1.5;N=1;
end
if nargin>=3
    if isfield(C,'name')
        criterion=C.name;
    else
        criterion='Obj';
    end
else
    criterion='Obj';
end
if nargin>=4
    if isfield(S,'tol')
        tol_spec=S.tol;
    else
        tol_spec=1e-8;
    end
    if isfield(S,'verb')
        verb_spec=S.verb;
    else
        verb_spec=1;
    end
    if isfield(S,'relax')
        relax=S.relax;
    else
        relax=0;
    end
    if isfield(S,'solver')
        solver=S.solver;
    else
        solver='Yalmip''s default';
    end
else
    tol_spec=1e-8;verb_spec=1;relax=0;
    solver='Yalmip''s default';
end
%% Method' choice

% Stepsize parameters:
%
%   x_{i+1}=x_{0} - sum_{k=0}^{i} step_h(i+1,k) * g_k/L
%
%   steps_h matrix (N+1) x N (no x0 nor gN entries columnwise, first row corresponds to x0)

steps_h=zeros(N+1,N);
switch method
    case 'FGM1'
        t=zeros(N-1,1);
        t(1,1)=1;
        for i=1:N-1
            t(i+1,1)=(1+sqrt(1+4*t(i,1)^2))/2;
        end
        steps_h(2,1)=1; %step for x1
        for i=2:N-1
            cur_step_param=(t(i,1)-1)/t(i+1,1);
            steps_h(i+1,:)=steps_h(i,:)+cur_step_param*(steps_h(i,:)-steps_h(i-1,:));
            steps_h(i+1,i)=1+cur_step_param;
            steps_h(i+1,i-1)=steps_h(i+1,i-1)-cur_step_param;
        end
        steps_h(end,:)=steps_h(end-1,:);
        steps_h(end,end)=1;
    case 'FGM2'
        t=zeros(N+1,1);
        t(1,1)=1;
        
        for i=1:N
            t(i+1,1)=(1+sqrt(1+4*t(i,1)^2))/2;
        end
        
        steps_h=zeros(N+1,N);
        steps_h(2,1)=1; %step for x1
        for i=2:N
            cur_step_param=(t(i,1)-1)/t(i+1,1);
            steps_h(i+1,:)=steps_h(i,:)+cur_step_param*(steps_h(i,:)-steps_h(i-1,:));
            steps_h(i+1,i)=1+cur_step_param;
            steps_h(i+1,i-1)=steps_h(i+1,i-1)-cur_step_param;
        end
    case 'StrCvxFGM1'
        steps_h(2,1)=1; %step for x1
        gamma=(1-sqrt(mu/L))/(1+sqrt(mu/L));
        for i=2:N-1
            cur_step_param=gamma;
            steps_h(i+1,:)=steps_h(i,:)+cur_step_param*(steps_h(i,:)-steps_h(i-1,:));
            steps_h(i+1,i)=1+cur_step_param;
            steps_h(i+1,i-1)=steps_h(i+1,i-1)-cur_step_param;
        end
        steps_h(end,:)=steps_h(end-1,:);
        steps_h(end,end)=1;
    case  'OGM2'
        t(1,1)=1;
        for i=1:N
            if (i<=N-1)
                t(i+1,1)=(1+sqrt(1+4*t(i,1)^2))/2;
            else
                t(i+1,1)=(1+sqrt(1+8*t(i,1)^2))/2;
            end
        end
        steps_h=zeros(N+2,N+1);
        for i=0:N-1
            cur_step_param=(t(i+1,1)-1)/t(i+2,1);
            cur_step_param2=(2*t(i+1,1)-1)/t(i+2,1);
            steps_h(i+3,:)=steps_h(i+2,:)+cur_step_param*(steps_h(i+2,:)-steps_h(i+1,:));
            steps_h(i+3,i+2)=1+cur_step_param2;
            steps_h(i+3,i+1)=steps_h(i+3,i+1)-cur_step_param;
        end
        steps_h=steps_h(2:end,2:end);
    case  'OGM1'
        t(1,1)=1;
        for i=1:N
            if (i<=N-1)
                t(i+1,1)=(1+sqrt(1+4*t(i,1)^2))/2;
            else
                t(i+1,1)=(1+sqrt(1+8*t(i,1)^2))/2;
            end
        end
        steps_h=zeros(N+2,N+1);
        for i=0:N-2
            cur_step_param=(t(i+1,1)-1)/t(i+2,1);
            cur_step_param2=(2*t(i+1,1)-1)/t(i+2,1);
            steps_h(i+3,:)=steps_h(i+2,:)+cur_step_param*(steps_h(i+2,:)-steps_h(i+1,:));
            steps_h(i+3,i+2)=1+cur_step_param2;
            steps_h(i+3,i+1)=steps_h(i+3,i+1)-cur_step_param;
        end
        steps_h(N+2,:)=steps_h(N+1,:);
        steps_h(N+2,N+1)=1;
        steps_h=steps_h(2:end,2:end);
    case 'Custom'
        if size(A.stepsize,1)==N && size(A.stepsize,2)==N
            steps_h=[zeros(1,N); A.stepsize];
        else
            error('Wrong use of the Custom option');
        end
    otherwise %GM
        steps_h(2,1)=h;
        for i=2:N
            steps_h(i+1,:)=steps_h(i,:);
            steps_h(i+1,i)=h;
        end
        method='GM';
end


%% General stepsize input: steps_c(i,:)=[-h(i,:) 0 1]  (supplementary entries
% corresponding to g_N and x_0)
if strcmp(criterion,'AvgObj')
    steps_c=[-steps_h/L zeros(N+1,2) ones(N+1,1)];
    steps_c=[steps_c;mean(steps_c(2:end,:),1)];
else
    steps_c=[-steps_h/L zeros(N+1,1) ones(N+1,1)];
end
if strcmp(criterion,'AvgObj')
    N=N+1;
end
%% Matrices Generation:
%
% G= [g0 g1 ... gN x0]^T[g0 g1 ... gN x0]
%
%

%Starting condition
AR=zeros(N+2,N+2);
AR(N+2,N+2)=1;

% Functional class (along with iterations)
% fi >= fj + gj^T(xi-xj) + 1/(2L) ||gi-gj||^2_2 (+str cvx)
% with j>i

AF=zeros(N+2,N+2,(N+1)*N/2);
BF=zeros(N+1,1,(N+1)*N/2);
count=0;
count_assoc=zeros((N+1)*N/2,2);
for j=2:N+1
    for i=1:j-1
        count=count+1;
        count_assoc(count,:)=[i j];
        BF(i,1,count)=-1;
        BF(j,1,count)=1;
        AF(j,j,count)=1/(4*(L-mu)); %True  value/2 because we sum with transpose afterwards
        AF(i,i,count)=1/(4*(L-mu)); %True  value/2 because we sum with transpose afterwards
        AF(i,j,count)=-1/(2*(L-mu));
        
        ci=steps_c(i,:);
        cj=steps_c(j,:);
        ei=zeros(N+2,1);
        ej=ei;
        ei(i)=1;
        ej(j)=1;
        
        AF(:,:,count)=AF(:,:,count)+(L/(L-mu))*1/2*(ej*(ci-cj));
        AF(:,:,count)=AF(:,:,count)+(mu/(L-mu))*1/2*(ei*(cj-ci));
        
        c=(ci-cj);
        AF(:,:,count)=AF(:,:,count)+L*mu/(2*(L-mu))*(c.'*c)/2;
        
        AF(:,:,count)=AF(:,:,count).'+AF(:,:,count);
    end
end

% fj >= fi + gi^T(xj-xi) + 1/(2L) ||gj-gi||^2_2 (+str cvx)
% with j>i
AF2=zeros(N+2,N+2,(N+1)*N/2);
BF2=zeros(N+1,1,(N+1)*N/2);
count=0;
for i=1:N
    for j=i+1:N+1
        count=count+1;
        BF2(i,1,count)=1;
        BF2(j,1,count)=-1;
        AF2(j,j,count)=1/(4*(L-mu)); %True  value/2 because we sum with transpose afterwards
        AF2(i,i,count)=1/(4*(L-mu)); %True  value/2 because we sum with transpose afterwards
        AF2(i,j,count)=-1/(2*(L-mu));
        
        ci=steps_c(i,:);
        cj=steps_c(j,:);
        ei=zeros(N+2,1);
        ej=ei;
        ei(i)=1;
        ej(j)=1;
        
        
        AF2(:,:,count)=AF2(:,:,count)+(L/(L-mu))*1/2*(ei*(cj-ci));
        AF2(:,:,count)=AF2(:,:,count)+(mu/(L-mu))*1/2*(ej*(ci-cj));
        
        c=(ci-cj);
        AF2(:,:,count)=AF2(:,:,count)+L*mu/(2*(L-mu))*(c.'*c)/2;
        
        AF2(:,:,count)=AF2(:,:,count).'+AF2(:,:,count);
    end
end

% f* >= fj + gj^T(x*-xj) + 1/(2L) ||g*-gj||^2_2 (+str cvx)

AFopt=zeros(N+2,N+2,N+1);
BFopt=zeros(N+1,1,N+1);
count=0;
for j=1:N+1
    count=count+1;
    BFopt(j,1,count)=1;
    AFopt(j,j,count)=1/(4*(L-mu));
    
    cj=steps_c(j,:);
    ej=zeros(N+2,1);
    ej(j)=1;
    
    AFopt(:,:,count)=AFopt(:,:,count)-(L/(L-mu))*1/2*(ej*cj);
    
    AFopt(:,:,count)=AFopt(:,:,count)+L*mu/(2*(L-mu))*(cj.'*cj)/2;
    AFopt(:,:,count)=AFopt(:,:,count).'+AFopt(:,:,count);
    
end

%fj >= f* + 1/(2L) ||g*-gj||^2_2  (+str cvx)

AFopt2=zeros(N+2,N+2,N+1);
BFopt2=zeros(N+1,1,N+1);
count=0;
for j=1:N+1
    count=count+1;
    BFopt2(j,1,count)=-1;
    AFopt2(j,j,count)=1/(4*(L-mu));
    
    cj=steps_c(j,:);
    ej=zeros(N+2,1);
    ej(j)=1;
    AFopt2(:,:,count)=AFopt2(:,:,count)-(mu/(L-mu))*1/2*(ej*cj);
    
    
    AFopt2(:,:,count)=AFopt2(:,:,count)+L*mu/(2*(L-mu))*(cj.'*cj)/2;
    
    AFopt2(:,:,count)=AFopt2(:,:,count).'+AFopt2(:,:,count);
    
end

%% Complete primal problem


G=sdpvar(N+2);
F=sdpvar(N+1,1);
cons=(G>=0);
cons=cons+(trace(AR*G)-R^2<=0);
count=0;
for j=2:N+1
    for i=1:j-1
        count=count+1;
        if (~relax || (j==i+1 && relax))
            cons=cons+(trace(AF(:,:,count)*G)+BF(:,:,count).'*F<=0);
        end
    end
end
count=0;
for i=1:N
    for j=i+1:N+1
        count=count+1;
        if (~relax)
            cons=cons+(trace(AF2(:,:,count)*G)+BF2(:,:,count).'*F<=0);
        end
    end
end
count=0;
for j=1:N+1
    count=count+1;
    cons=cons+(trace(AFopt(:,:,count)*G)+BFopt(:,:,count).'*F<=0);
end
count=0;
for j=1:N+1
    
    if (~relax)
        count=count+1;
        cons=cons+(trace(AFopt2(:,:,count)*G)+BFopt2(:,:,count).'*F<=0);
    end
end

switch solver
    case 'sedumi'
        ops = sdpsettings('verbose',verb_spec,'solver','sedumi','sedumi.eps',tol_spec);
        tolerance=tol_spec;
    case 'mosek'
        ops = sdpsettings('verbose',verb_spec,'solver','mosek','mosek.MSK_DPAR_INTPNT_CO_TOL_PFEAS',tol_spec);
        tolerance=tol_spec;
    otherwise
        ops=sdpsettings('verbose',verb_spec);
        solver='Yalmip default';
        tolerance='Yalmip default';
end
switch criterion
    case 'Grad'
        obj=-G(end-1,end-1);
    case 'MinGrad'
        tau_slack=sdpvar(1,1);
        obj=-tau_slack;
        for i=1:N+1
            cons=cons+(tau_slack<=G(i,i));
        end
    case 'Dist'
        c=steps_c(end,:);
        obj=-trace((c.'*c)*G);
    case 'AvgObj'
        obj=-F(end);
    otherwise %case 'Obj'
        obj=-F(end);
        criterion='Obj';
end
outth=NaN;
saveYMdetails=optimize(cons,obj,ops);
outp=-double(obj);
outd=dual(cons(2))*R^2;
err=saveYMdetails.problem;

%% Conjectures

switch method
    case 'FGM1'
        sumgamma=sum(steps_h(end,:));
        switch criterion
            case 'Obj'
                tau=1/(2*sumgamma+1);
                outth=L*R^2/2*tau;
            case 'Grad'
                outth=NaN;
            case 'MinGrad'
                outth=NaN;
            case 'Dist'
                outth=NaN;
        end
    case 'FGM2'
        sumgamma=sum(steps_h(end,:));
        switch criterion
            case 'Obj'
                tau=1/(2*sumgamma+1);
                outth=L*R^2/2*tau;
            case 'Grad'
                outth=NaN;
            case 'MinGrad'
                outth=NaN;
            case 'Dist'
                outth=NaN;
        end
    case 'StrCvxFGM1'
        outth=NaN;
    case  'OGM2'
        sumgamma=sum(steps_h(end,:));
        switch criterion
            case 'Obj'
                tau=1/(2*sumgamma+1);
                outth=L*R^2/2*tau;
            case 'Grad'
                outth=NaN;
            case 'MinGrad'
                outth=NaN;
            case 'Dist'
                outth=NaN;
        end
    case  'OGM1'
        sumgamma=sum(steps_h(end,:));
        switch criterion
            case 'Obj'
                tau=1/(2*sumgamma+1);
                outth=L*R^2/2*tau;
            case 'Grad'
                outth=NaN;
            case 'MinGrad'
                outth=NaN;
            case 'Dist'
                outth=NaN;
        end
    case 'GM'
        if (mu==0)
            switch criterion
                case 'Grad'
                    outth=(L*R*max(1/(N*h+1),abs(1-h)^(N)))^2;
                case 'MinGrad'
                    outth=(L*R*max(1/(N*h+1),abs(1-h)^(N)))^2;
                case 'Dist'
                    outth=R;
                case 'Obj'
                    outth=L*R^2/2*max(1/(2*N*h+1),(1-h)^(2*N));
                case 'AvgObj'
                    outth=NaN;
            end
        else
            kappa=mu/L;
            switch criterion
                case 'Grad'
                    outth=(L*R*max(kappa*(1-h*kappa)^N/((kappa-1)*(1-h*kappa)^N+1),abs(1-h)^(N)))^2;
                case 'MinGrad'
                    outth=(L*R*max(kappa*(1-h*kappa)^N/((kappa-1)*(1-h*kappa)^N+1),abs(1-h)^(N)))^2;
                case 'Dist'
                    outth=NaN;
                case 'Obj'
                    outth=L*R^2/2*max(kappa*(1-h*kappa)^(2*N)/((kappa-1)*(1-h*kappa)^(2*N)+1),(1-h)^(2*N));
                case 'AvgObj'
                    outth=NaN;
            end
        end
end

val.primal=outp;
val.dual=outd;
val.conj=outth;

Sol.G=double(G);
Sol.S=dual(cons(1));
Sol.lambda=dual(cons(2:end));
Sol.f=double(F);
Sol.err=err;

Prob.criterion=criterion;
Prob.solver=solver;
Prob.solvertolerance=tolerance;
Prob.method=method;
if strcmp(criterion,'AvgObj')
    N=N-1;
end
Prob.nbIter=N;
Prob.L=L;
Prob.R=R;
Prob.mu=mu;
Prob.relax=relax;
Prob.H=steps_c;

if verb_spec
    disp(sprintf('\nWorst-case estimation of criterion %s for method %s on an (L=%g,mu=%g)-function after %d iterations:', C.name, A.name, P.L, P.mu, A.N));
    if isnan(val.conj)
        conj_str = 'no conjectured value available.';
    else
        conj_str = sprintf('conjectured value = %g', val.conj);
    end
    disp(sprintf('-> primal-dual interval found = [%g %g] ; %s', val.primal, val.dual, conj_str));
end
