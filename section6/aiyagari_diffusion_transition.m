%TRANSITION DYNAMICS IN AIYAGARI ECONOMY 
%Written by SeHyoun Ahn
%The algorithm is based on a relaxation scheme to find K(t). The value function is
%found every round by solving the HJB equation through an upwind finite
%differences scheme. The distribution is found by also solving using finite
%differences the Fokker-Planck (Kolmogorov forward) equation
clear all; close all; format long;
tic;

%--------------------------------------------------
%PARAMETERS
ga = 2;       % CRRA utility with parameter gamma
alpha = 0.35; % Production function F = K^alpha * L^(1-alpha) 
delta = 0.1;  % Capital depreciation
zmean = 1.0;      % mean O-U process (in levels). This parameter has to be adjusted to ensure that the mean of z (truncated gaussian) is 1.
sig2 = (0.10)^2;  % sigma^2 O-U
Corr = exp(-0.3);  % persistence -log(Corr)  O-U
rho = 0.05;   % discount rate

K = 3.8;      % initial aggregate capital. It is important to guess a value close to the solution for the algorithm to converge
relax = 0.99; % relaxation parameter 
J=40;         % number of z points 
zmin = 0.5;   % Range z
zmax = 1.5;
amin = -1;    % borrowing constraint
amax = 30;    % range a
I=100;        % number of a points 

%simulation parameters
maxit  = 100;     %maximum number of iterations in the HJB loop
maxitK = 100;    %maximum number of iterations in the K loop
crit = 10^(-10); %criterion HJB loop
critK = 1e-7;   %criterion K loop
Delta = 1000;   %delta in HJB algorithm

%ORNSTEIN-UHLENBECK IN LEVELS
the = -log(Corr);
Var = sig2/(2*the);

%--------------------------------------------------
%VARIABLES 
a = linspace(amin,amax,I)';  %wealth vector
da = (amax-amin)/(I-1);      
z = linspace(zmin,zmax,J);   % productivity vector
dz = (zmax-zmin)/(J-1);
dz2 = dz^2;
aa = a*ones(1,J);
zz = ones(I,1)*z;

mu = the*(zmean - z);        %DRIFT (FROM ITO'S LEMMA)
s2 = sig2.*ones(1,J);        %VARIANCE (FROM ITO'S LEMMA)

%Finite difference approximation of the partial derivatives
Vaf = zeros(I,J);             
Vab = zeros(I,J);
Vzf = zeros(I,J);
Vzb = zeros(I,J);
Vzz = zeros(I,J);
c = zeros(I,J);

%CONSTRUCT MATRIX Aswitch SUMMARIZING EVOLUTION OF z
yy = - s2/dz2 - mu/dz;
chi =  s2/(2*dz2);
zeta = mu/dz + s2/(2*dz2);

%This will be the upperdiagonal of the matrix Aswitch
updiag=zeros(I,1); %This is necessary because of the peculiar way spdiags is defined.
for j=1:J
    updiag=[updiag;repmat(zeta(j),I,1)];
end

%This will be the center diagonal of the matrix Aswitch
centdiag=repmat(chi(1)+yy(1),I,1);
for j=2:J-1
    centdiag=[centdiag;repmat(yy(j),I,1)];
end
centdiag=[centdiag;repmat(yy(J)+zeta(J),I,1)];

%This will be the lower diagonal of the matrix Aswitch
lowdiag=repmat(chi(2),I,1);
for j=3:J
    lowdiag=[lowdiag;repmat(chi(j),I,1)];
end

%Add up the upper, center, and lower diagonal into a sparse matrix
Aswitch=spdiags(centdiag,0,I*J,I*J)+spdiags(lowdiag,-I,I*J,I*J)+spdiags(updiag,I,I*J,I*J);


%----------------------------------------------------
%%%%% COMPUTE THE LONG RUN STEADY STATE VALUE

TFP=1.1;        %Long run TFP value

%----------------------------------------------------
%INITIAL GUESS
r = alpha     * TFP * K^(alpha-1) -delta; %interest rates
w = (1-alpha) * TFP * K^(alpha);          %wages
v0 = (w*zz + r.*aa).^(1-ga)/(1-ga)/rho;
v = v0;
dist = zeros(1,maxit);

%-----------------------------------------------------
%SOLVING FOR LONG RUN STEADY STATE
for iter=1:maxitK

   % HAMILTON-JACOBI-BELLMAN EQUATION %
   for n=1:maxit
        V = v;
        % forward difference
        Vaf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
        Vaf(I,:) = (w*z + r.*amax).^(-ga); %will never be used, but impose state constraint a<=amax just in case
        % backward difference
        Vab(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
        Vab(1,:) = (w*z + r.*amin).^(-ga);  %state constraint boundary condition

        I_concave = Vab > Vaf;              %indicator whether value function is concave (problems arise if this is not the case)

        %consumption and savings with forward difference
        cf = Vaf.^(-1/ga);
        sf = w*zz + r.*aa - cf;
        %consumption and savings with backward difference
        cb = Vab.^(-1/ga);
        sb = w*zz + r.*aa - cb;
        %consumption and derivative of value function at steady state
        c0 = w*zz + r.*aa;
        Va0 = c0.^(-ga);

        % dV_upwind makes a choice of forward or backward differences based on
        % the sign of the drift
        If = sf > 0; %positive drift --> forward difference
        Ib = sb < 0; %negative drift --> backward difference
        I0 = (1-If-Ib); %at steady state
        %make sure backward difference is used at amax
        %     Ib(I,:) = 1; If(I,:) = 0;
        %STATE CONSTRAINT at amin: USE BOUNDARY CONDITION UNLESS sf > 0:
        %already taken care of automatically

        Va_Upwind = Vaf.*If + Vab.*Ib + Va0.*I0; %important to include third term

        c = Va_Upwind.^(-1/ga);
        u = c.^(1-ga)/(1-ga);

        %CONSTRUCT MATRIX A
        X = - min(sb,0)/da;
        Y = - max(sf,0)/da + min(sb,0)/da;
        Z = max(sf,0)/da;
        
        updiag=[0]; %This is needed because of the peculiarity of spdiags.
        for j=1:J
            updiag=[updiag;Z(1:I-1,j);0];
        end
        
        centdiag=reshape(Y,I*J,1);
        
        lowdiag=X(2:I,1);
        for j=2:J
            lowdiag=[lowdiag;0;X(2:I,j)];
        end
        
        AA=spdiags(centdiag,0,I*J,I*J)+spdiags([updiag;0],1,I*J,I*J)+spdiags([lowdiag;0],-1,I*J,I*J);
        
        A = AA + Aswitch;
        
        if max(abs(sum(A,2)))>10^(-9)
           disp('Improper Transition Matrix')
           break
        end
   
        B = (1/Delta + rho)*speye(I*J) - A;

        u_stacked = reshape(u,I*J,1);
        V_stacked = reshape(V,I*J,1);

        b = u_stacked + V_stacked/Delta;

        V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS

        V = reshape(V_stacked,I,J);

        Vchange = V - v;
        v = V;

        dist(n) = max(max(abs(Vchange)));
        if dist(n)<crit
            %disp('Value Function Converged, Iteration = ')
            %disp(n)
            break
        end
   end
    % FOKKER-PLANCK EQUATION %
    AT = A';
    b = zeros(I*J,1);

    %need to fix one value, otherwise matrix is singular
    i_fix = 1;
    b(i_fix)=.1;
    row = [zeros(1,i_fix-1),1,zeros(1,I*J-i_fix)];
    AT(i_fix,:) = row;

    %Solve linear system
    gg = AT\b;
    g_sum = gg'*ones(I*J,1)*da*dz;
    gg = gg./g_sum;

    g = reshape(gg,I,J);
    
    % Update aggregate capital
    S = sum(g'*a*da*dz);
   
    clear A AA AT B
    if abs(K-S)<critK
        break
    end
    
    %update prices
    K = relax*K +(1-relax)*S;           %relaxation algorithm (to ensure convergence)
    r = alpha     * TFP *  K^(alpha-1) -delta; %interest rates
    w = (1-alpha) * TFP * K^(alpha);          %wages
 
end

% Save Long term steady-state value to iterate backward from.
v_st=v;
g_st=sparse(g);
r_st=r;
K_st=K;

fprintf('Long run steady state found.\n    Long run equilibrium level of capital is %.5f\n',K_st);

%----------------------------------------------------
%%%%% FIND THE INITIAL CONDITION
%%%%% g is taken to be the steady state when TFP=1

%Parameters
TFP=1;
K = 3.8;

%Finite difference approximation of the partial derivatives
Vaf = zeros(I,J);             
Vab = zeros(I,J);
Vzf = zeros(I,J);
Vzb = zeros(I,J);
Vzz = zeros(I,J);
c = zeros(I,J);

%----------------------------------------------------
%INITIAL GUESS
r = alpha     * TFP * K^(alpha-1) -delta; %interest rates
w = (1-alpha) * TFP * K^(alpha);          %wages
v0 = (w*zz + r.*aa).^(1-ga)/(1-ga)/rho;
v = v0;
dist = zeros(1,maxit);

%-----------------------------------------------------
%SOLVING FOR INITIAL CONDITION
for iter=1:maxitK

    % HAMILTON-JACOBI-BELLMAN EQUATION %
    for n=1:maxit
        V = v;
        % forward difference
        Vaf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
        Vaf(I,:) = (w*z + r.*amax).^(-ga); %will never be used, but impose state constraint a<=amax just in case
        % backward difference
        Vab(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
        Vab(1,:) = (w*z + r.*amin).^(-ga);  %state constraint boundary condition

        I_concave = Vab > Vaf;              %indicator whether value function is concave (problems arise if this is not the case)

        %consumption and savings with forward difference
        cf = Vaf.^(-1/ga);
        sf = w*zz + r.*aa - cf;
        %consumption and savings with backward difference
        cb = Vab.^(-1/ga);
        sb = w*zz + r.*aa - cb;
        %consumption and derivative of value function at steady state
        c0 = w*zz + r.*aa;
        Va0 = c0.^(-ga);

        % dV_upwind makes a choice of forward or backward differences based on
        % the sign of the drift
        If = sf > 0; %positive drift --> forward difference
        Ib = sb < 0; %negative drift --> backward difference
        I0 = (1-If-Ib); %at steady state
        %make sure backward difference is used at amax
        %     Ib(I,:) = 1; If(I,:) = 0;
        %STATE CONSTRAINT at amin: USE BOUNDARY CONDITION UNLESS sf > 0:
        %already taken care of automatically

        Va_Upwind = Vaf.*If + Vab.*Ib + Va0.*I0; %important to include third term

        c = Va_Upwind.^(-1/ga);
        u = c.^(1-ga)/(1-ga);

        %CONSTRUCT MATRIX A
        X = - min(sb,0)/da;
        Y = - max(sf,0)/da + min(sb,0)/da;
        Z = max(sf,0)/da;
        
        updiag=[0]; %This is needed because of the peculiarity of spdiags.
        for j=1:J
            updiag=[updiag;Z(1:I-1,j);0];
        end
        
        centdiag=reshape(Y,I*J,1);
        
        lowdiag=X(2:I,1);
        for j=2:J
            lowdiag=[lowdiag;0;X(2:I,j)];
        end
        
        AA=spdiags(centdiag,0,I*J,I*J)+spdiags([updiag;0],1,I*J,I*J)+spdiags([lowdiag;0],-1,I*J,I*J);
        
        A = AA + Aswitch;
        
        if max(abs(sum(A,2)))>10^(-9)
           disp('Improper Transition Matrix')
           break
        end
        
        B = (1/Delta + rho)*speye(I*J) - A;

        u_stacked = reshape(u,I*J,1);
        V_stacked = reshape(V,I*J,1);

        b = u_stacked + V_stacked/Delta;

        V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS

        V = reshape(V_stacked,I,J);

        Vchange = V - v;
        v = V;

        dist(n) = max(max(abs(Vchange)));
        if dist(n)<crit
            %disp('Value Function Converged, Iteration = ')
            %disp(n)
            break
        end
    end
    %toc;
    % FOKKER-PLANCK EQUATION %
    AT = A';
    b = zeros(I*J,1);

    %need to fix one value, otherwise matrix is singular
    i_fix = 1;
    b(i_fix)=.1;
    row = [zeros(1,i_fix-1),1,zeros(1,I*J-i_fix)];
    AT(i_fix,:) = row;

    %Solve linear system
    gg = AT\b;
    g_sum = gg'*ones(I*J,1)*da*dz;
    gg = gg./g_sum;

    g = reshape(gg,I,J);
    
    % Update aggregate capital
    S = sum(g'*a*da*dz);

    clear A AA AT B
    if abs(K-S)<critK
        break
    end
    
    %update prices
    K = relax*K +(1-relax)*S;           %relaxation algorithm (to ensure convergence)
    r = alpha     * TFP *  K^(alpha-1) -delta; %interest rates
    w = (1-alpha) * TFP * K^(alpha);          %wages
 
end

g0=g;
gg0=gg;
r00=r;

fprintf('Initial conditions found.\n    Initial level of capital is %.5f\n',K);

%----------------------------------------------------
%%%%%FIND THE TRANSITION DYNAMICS

T=75;       %The time under consideration
N=500;      %Fineness of the grid
dt=T/N;
N1=N;       %By this time, the system should have converged to new steady state


%Initial guess of the change of capital over time
K_t = [K:(K_st-K)/(N-1):K_st];
Knew=K_t;       %This is just preallocation. The values will not be used.

%Preallocation
v = zeros(I,J,N);
gg = cell(N+1,1);
A_t=cell(N,1);
maxit = 1000;
convergence_criterion = 0.0005;


TFP=1.1;    %Set the TFP level back to the long run level.

v(:,:,N)= v_st;
relax=0.95;
%relax=relax(end:-1:1);
for it=1:maxit
    fprintf('ITERATION = %d\n',it);
    r_t = alpha     * TFP *  K_t.^(alpha-1) -delta; %interest rates
    w_t = (1-alpha) * TFP * K_t.^(alpha);          %wages

    
    V = v_st;
    
    for n=N1:-1:1
        v(:,:,n)=V;
        % forward difference
        dVf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
        dVf(I,:) = (w_t(n)*z + r_t(n).*amax).^(-ga); %will never be used, but impose state constraint a<=amax just in case
        % backward difference
        dVb(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
        dVb(1,:) = (w_t(n)*z + r_t(n).*amin).^(-ga); %state constraint boundary condition
        
        I_concave = dVb > dVf; %indicator whether value function is concave (problems arise if this is not the case)
        
        %consumption and savings with forward difference
        cf = dVf.^(-1/ga);
        ssf = w_t(n)*zz + r_t(n).*aa - cf;
        %consumption and savings with backward difference
        cb = dVb.^(-1/ga);
        ssb = w_t(n)*zz + r_t(n).*aa - cb;
        %consumption and derivative of value function at steady state
        c0 = w_t(n)*zz + r_t(n).*aa;
        dV0 = c0.^(-ga);
        
        % dV_upwind makes a choice of forward or backward differences based on
        % the sign of the drift
        If = ssf > 0; %positive drift --> forward difference
        Ib = ssb < 0; %negative drift --> backward difference
        I0 = (1-If-Ib); %at steady state
        %make sure backward difference is used at amax
        %Ib(I,:) = 1; If(I,:) = 0;
        %STATE CONSTRAINT at amin: USE BOUNDARY CONDITION UNLESS muf > 0:
        %already taken care of automatically
        
        dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0; %important to include third term
        c = dV_Upwind.^(-1/ga);
        u = c.^(1-ga)/(1-ga);
        
        %CONSTRUCT MATRIX
        X = - min(ssb,0)/da;
        Y = - max(ssf,0)/da + min(ssb,0)/da;
        Z = max(ssf,0)/da;
        
        updiag=[0];
        for j=1:J
            updiag=[updiag;Z(1:I-1,j);0];
        end
        
        centdiag = reshape(Y,I*J,1);
        
        lowdiag=X(2:I,1);
        for j=2:J
            lowdiag=[lowdiag;0;X(2:I,j)];
        end
        
        A=Aswitch+spdiags(centdiag,0,I*J,I*J)+spdiags([updiag;0],1,I*J,I*J)+spdiags([lowdiag;0],-1,I*J,I*J);
        if max(abs(sum(A,2)))>10^(-9)
               disp('Improper Transition Matrix')
               break
        end
   
        %%Note the syntax for the cell array
        A_t{n} = A;
        B = (1/dt + rho)*speye(I*J) - A;
        
        u_stacked = reshape(u,I*J,1);
        V_stacked = reshape(V,I*J,1);
        
        b = u_stacked + V_stacked/dt;
        V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS
        
        V = reshape(V_stacked,I,J);
        ss = w_t(n)*zz + r_t(n).*aa - c;
    end
    
    %plot(a,v(:,:,1),a,v(:,:,N))
    
    gg{1}=gg0;
    for n=1:N
        AT=A_t{n}';
        %Implicit method in Updating Distribution.
        gg{n+1}= (speye(I*J) - AT*dt)\gg{n};
        %gg{n+1}=gg{n}+AT*gg{n}*dt; %This is the explicit method.
        %check(n) = gg(:,n)'*ones(2*I,1)*da;
        g=reshape(gg{n},I,J);
        Knew(n)=sum(g'*a*da*dz);
    end
    
    %SS_it(:,it)=SS;
    
    
    
    fprintf('    Maximum change in capital is %.8f\n',max(abs(K_t-Knew)));
    if max(abs(K_t-Knew))<convergence_criterion
        break
    end
    
    K_t=relax.*K_t+(1-relax).*Knew;
    
    h=figure(1);
    if mod(it,30)==0
        clf;
    end
    plot(1:500,K_st*ones(500,1),'r--');
    hold on;
    
    plot(K_t);
    hold on;
    saveas(h,'figure.png');
end
toc;

save('aiyagari_transition')