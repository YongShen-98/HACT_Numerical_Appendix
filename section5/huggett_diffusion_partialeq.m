%Optimized for speed by SeHyoun Ahn

clear all; close all; clc;
tic;
ga = 2; %CRRA utility with parameter gamma
r = 0.04; %interest rate
rho = 0.05; %discount rate

% %ORNSTEIN-UHLENBECK PROCESS dlog(z) = -the*log(z)dt + sig2*dW
% %STATIONARY DISTRIBUTION IS log(z) ~ N(0,Var) WHERE Var = sig2/(2*the)
% sig2 = 0.05;
% % Corr = 0.85;
% Corr = 0.97;
% the = -log(Corr);
% Var = sig2/(2*the);
% logzmean = 0;
% zmean = logzmean + exp(Var/2); %MEAN OF LOG-NORMAL DISTRIBUTION N(0,Var)

%ORNSTEIN-UHLENBECK IN LEVELS
sig2 = 0.05;
Corr = 0.9;
the = -log(Corr);
Var = sig2/(2*the);
zmean = 1;

I=100;
amin = -0.1; %borrowing constraint
amax = 30;
a = linspace(amin,amax,I)';
da = (amax-amin)/(I-1);

J=40;
% zmin = exp(logzmean - 3 + Var/2);
% zmax = exp(logzmean + .3 + Var/2);
% zmin = 0.7;
% zmax = 1.25;
zmin = 0.4;
zmax = 1.5;
z = linspace(zmin,zmax,J);
dz = (zmax-zmin)/(J-1);
dz2 = dz^2;

% mu = (the*(logzmean - log(z)) + sig2/2).*z; %DRIFT (FROM ITO'S LEMMA)
% s2 = sig2.*z.^2; %VARIANCE (FROM ITO'S LEMMA)
%ORNSTEIN-UHLENBECK IN LEVELS
mu = the*(zmean - z); %DRIFT (FROM ITO'S LEMMA)
s2 = sig2.*ones(1,J); %VARIANCE (FROM ITO'S LEMMA)

% plot(log(z),normpdf(log(z),0,Var))
% plot(z,lognpdf(z,0,Var))

%plot(z,normpdf(z,zmean,Var))

aa = a*ones(1,J);
zz = ones(I,1)*z;

maxit= 100;
crit = 10^(-6);
Delta = 1000;

Vaf = zeros(I,J);
Vab = zeros(I,J);
Vzf = zeros(I,J);
Vzb = zeros(I,J);
Vzz = zeros(I,J);
c = zeros(I,J);

%CONSTRUCT MATRIX Bswitch SUMMARIZING EVOLUTION OF z
chi =  - min(mu,0)/dz + s2/(2*dz2);
yy =  min(mu,0)/dz - max(mu,0)/dz - s2/dz2;
zeta = max(mu,0)/dz + s2/(2*dz2);

%This will be the upperdiagonal of the B_switch
updiag=zeros(I,1); %This is necessary because of the peculiar way spdiags is defined.
for j=1:J
    updiag=[updiag;repmat(zeta(j),I,1)];
end

%This will be the center diagonal of the B_switch
centdiag=repmat(chi(1)+yy(1),I,1);
for j=2:J-1
    centdiag=[centdiag;repmat(yy(j),I,1)];
end
centdiag=[centdiag;repmat(yy(J)+zeta(J),I,1)];

%This will be the lower diagonal of the B_switch
lowdiag=repmat(chi(2),I,1);
for j=3:J
    lowdiag=[lowdiag;repmat(chi(j),I,1)];
end

%Add up the upper, center, and lower diagonal into a sparse matrix
Aswitch=spdiags(centdiag,0,I*J,I*J)+spdiags(lowdiag,-I,I*J,I*J)+spdiags(updiag,I,I*J,I*J);

%INITIAL GUESS
v0 = (zz + r.*aa).^(1-ga)/(1-ga)/rho;
v = v0;

maxit = 20;

for n=1:maxit
    V = v;
    % forward difference
    Vaf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
    Vaf(I,:) = (z + r.*amax).^(-ga); %will never be used, but impose state constraint a<=amax just in case
    % backward difference
    Vab(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
    Vab(1,:) = (z + r.*amin).^(-ga); %state constraint boundary condition
    
    I_concave = Vab > Vaf; %indicator whether value function is concave (problems arise if this is not the case)
    
    %consumption and savings with forward difference
    cf = Vaf.^(-1/ga);
    sf = zz + r.*aa - cf;
    %consumption and savings with backward difference
    cb = Vab.^(-1/ga);
    sb = zz + r.*aa - cb;
    %consumption and derivative of value function at steady state
    c0 = zz + r.*aa;
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
    X = -min(sb,0)/da;
    Y = -max(sf,0)/da + min(sb,0)/da;
    Z = max(sf,0)/da;
    
    updiag=0; %This is needed because of the peculiarity of spdiags.
    for j=1:J
        updiag=[updiag;Z(1:I-1,j);0];
    end
    
    centdiag=reshape(Y,I*J,1);
    
    lowdiag=X(2:I,1);
    for j=2:J
        lowdiag=[lowdiag;0;X(2:I,j)];
    end
    
    AA =spdiags(centdiag,0,I*J,I*J)+spdiags([updiag;0],1,I*J,I*J)+spdiags([lowdiag;0],-1,I*J,I*J);
    
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
        disp('Value Function Converged, Iteration = ')
        disp(n)
        break
    end
end
toc;


%%%%%%%%%%%%%%%%%%%%%%%%%%
% FOKKER-PLANCK EQUATION %
%%%%%%%%%%%%%%%%%%%%%%%%%%
AT = A';
b = zeros(I*J,1);

%need to fix one value, otherwise matrix is singular
i_fix = 1;
b(i_fix)=.001;
row = [zeros(1,i_fix-1),1,zeros(1,I*J-i_fix)];
AT(i_fix,:) = row;

%Solve linear system
gg = AT\b;
g_sum = gg'*ones(I*J,1)*da*dz;
gg = gg./g_sum;

g = reshape(gg,I,J);


%%%%%%%%%%
% GRAPHS %
%%%%%%%%%%
%SAVINGS POLICY FUNCTION
ss = zz + r.*aa - c;

icut = 90;
acut = a(1:icut);
sscut = ss(1:icut,:);
% set(gca,'FontSize',14)
% plot(a,ss,a,zeros(1,I),'--')
% xlabel('a')
% ylabel('s(a,z)')
% xlim([amin amax])

surf(acut,z,sscut')
view([45 35])
xlabel('Wealth, a','FontSize',14)
ylabel('Productivity, z','FontSize',14)
zlabel('Savings s(a,z)','FontSize',14)
xlim([amin amax-3])
ylim([zmin zmax])
print -depsc HJB_diffusion.eps

%WEALTH DISTRIBUTION
icut = 30;
acut = a(1:icut);
gcut = g(1:icut,:);

% figure; hold on;
% surf(acut,z,gcut')
% planeimg = abs(gcut');
% imgzposition = -0.002;
% amax2 = max(acut);
% surf([amin amax2],[zmin zmax],repmat(imgzposition, [2 2]),planeimg,'facecolor','texture')
% colormap(jet);
% view([55 35])
% xlabel('Wealth, a','FontSize',14)
% ylabel('Productivity, z','FontSize',14)
% zlabel('Density g(a,z)','FontSize',14)
% % print -depsc densities_diffusion.eps

surf(acut,z,gcut')
view([45 35])
xlabel('Wealth, a','FontSize',14)
ylabel('Productivity, z','FontSize',14)
zlabel('Density g(a,z)','FontSize',14)
xlim([amin max(acut)])
ylim([zmin zmax])
alpha(.7)
print -depsc densities_diffusion.eps