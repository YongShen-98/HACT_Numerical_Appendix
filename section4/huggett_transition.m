%Written by SeHyoun Ahn and Ben Moll
%NEEDS INPUT FROM huggett_initial.m AND huggett_terminal.m
clear all; close all; clc;

load huggett_initial.mat %start from equilibrium with low lambda2
g0 = sparse(g);
gg0 = sparse(gg);
r00 = r;
A0 = A;
AT0 = A';

load huggett_terminal.mat %terminal condition
g_st = sparse(g); v_st = v;
plot(a,v);
plot(a,g,a,g0);
xlim([amin 1]);
r_st = r;                      %% page 11 step 1 Given r(t),vsolve ..... 

clear r Delta;

%format long;

T = 20;
N = 100;
dt = T/N;

%initial guess of interest rate sequence
r0 = r_st*ones(N,1);

S = zeros(N,1);
SS = zeros(N,1);
dS = zeros(N,1);

v = zeros(I,2,N);
gg = cell(N+1,1);
v(:,:,N)= v_st;

rnew = r0;

A_t=cell(N,1);

maxit = 1000;
r_it=zeros(N,maxit);
Sdist=zeros(maxit,1);


convergence_criterion = 10^(-5);
%speed of updating the interest rate
xi = 20*(exp(-0.05*(1:N)) - exp(-0.05*N));

% maxit = 1;
for it=1:maxit
    r_t = rnew;
    r_it(:,it)=r_t;
    
    V = v_st;
    
    % step 1 compute the time path of v
    for n=N:-1:1
        v(:,:,n)=V;
        % forward difference
        dVf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
        dVf(I,:) = (z + r_t(n).*amax).^(-s); %will never be used, but impose state constraint a<=amax just in case
        % backward difference
        dVb(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
        dVb(1,:) = (z + r_t(n).*amin).^(-s); %state constraint boundary condition
        
        I_concave = dVb > dVf; %indicator whether value function is concave (problems arise if this is not the case)
        
        %consumption and savings with forward difference
        cf = dVf.^(-1/s);
        ssf = zz + r_t(n).*aa - cf;
        %consumption and savings with backward difference
        cb = dVb.^(-1/s);
        ssb = zz + r_t(n).*aa - cb;
        %consumption and derivative of value function at steady state
        c0 = zz + r_t(n).*aa;
        dV0 = c0.^(-s);
        
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
        c = dV_Upwind.^(-1/s);
        u = c.^(1-s)/(1-s);
        
        %CONSTRUCT MATRIX
        X = - min(ssb,0)/da;
        Y = - max(ssf,0)/da + min(ssb,0)/da;
        Z = max(ssf,0)/da;
        
        A1=spdiags(Y(:,1),0,I,I)+spdiags(X(2:I,1),-1,I,I)+spdiags([0;Z(1:I-1,1)],1,I,I);
        A2=spdiags(Y(:,2),0,I,I)+spdiags(X(2:I,2),-1,I,I)+spdiags([0;Z(1:I-1,2)],1,I,I);
        A = [A1,sparse(I,I);sparse(I,I),A2] + Aswitch;
        
        if max(abs(sum(A,2)))>10^(-9)
           disp('Improper Transition Matrix')
           break
        end
        
        %%Note the syntax for the cell array
        A_t{n} = A;
        B = (1/dt + rho)*speye(2*I) - A;
        
        u_stacked = [u(:,1);u(:,2)];
        V_stacked = [V(:,1);V(:,2)];
        
        b = u_stacked + V_stacked/dt;
        V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS
        
        V = [V_stacked(1:I),V_stacked(I+1:2*I)];
        ss = zz + r_t(n).*aa - c;
    end
    
    %plot(a,v(:,:,1),a,v(:,:,N))
    
    % step 2 and step 3
    % compute g. see page 12 eq.27 and page 13  
    gg{1}=gg0;
    for n=1:N
        AT=A_t{n}';
        %Implicit method in Updating Distribution.
        gg{n+1}= (speye(2*I) - AT*dt)\gg{n}; 
        %gg{n+1}=gg{n}+AT*gg{n}*dt; %This is the explicit method.
        %check(n) = gg(:,n)'*ones(2*I,1)*da;
        SS(n) = gg{n}'*aa(:)*da;
        dS(n) = gg{n+1}'*aa(:)*da - gg{n}'*aa(:)*da;
    end
    
    dS_it(:,it)=dS;
    SS_it(:,it)=SS;
    

    % step 4 update r
    %Update the interest rate to reduce aggregate saving
    rnew = r_t - xi'.*dS;
    
    %This is plotting things to show how things are changing. This can be
    %removed for speed.
    if mod(it,20)==0
        figure(1);
        clf;
        subplot(2,2,1);
        plot(r_st*ones(N,1),'r--');
        hold on;
        title('r_t');
        plot(r_t);
        subplot(2,2,2);
        plot(dS);
        title('dS');
        subplot(2,2,3);
        plot(zeros(1,21),'r--');
        hold on;
        plot(dS(1:20));
        title('dS(1:20)');
        subplot(2,2,4);
        plot(dS(N-20:N));
        hold on;
        title('dS(N-20:N)');
        pause(0.1);
    end
    
    Sdist(it) = max(abs(dS));
    disp(['ITERATION = ', num2str(it)])
    disp(['Convergence criterion = ', num2str(Sdist(it))])
    if Sdist(it)<convergence_criterion
        break
    end
    
end

time = (1:N)'*dt;

figure(1);
clf;
plot(Sdist(1:it))

figure(2);
plot(1:N,SS_it(:,1),1:N,SS_it(:,it))
legend('Iteration 1','Last Iteration')
ylabel('Excess Supply')
xlabel('Time Period')

figure(3);
plot(r_t);
hold on;
plot(r_st*ones(N,1),'r--');

N1 = 4;
T1 = -N1*dt;
time1 = T1 + (1:N1)'*dt;
time2 = [time1;time];
r_t2 = [r00*ones(N1,1);r_t];

close all
set(gca,'FontSize',16)
plot(time2,r_t2,time2,r_st*ones(N1+N,1),'--','LineWidth',2)
xlim([T1 10])
ylim([-0.05 0.035])
xlabel('Year')
title('Equilibrium Interest Rate, r(t)')
print -depsc transition.eps



amax1 = 0.5;
gmax = 3;
close all
set(gcf,'PaperPosition',[0 0 15 10])
n=2;
subplot(2,2,1)
set(gca,'FontSize',16)
h1 = plot(a,gg{n}(1:I),'b',a,gg{n}(I+1:2*I),'r','LineWidth',2)
legend(h1,'g_1(a)','g_2(a)')
hold on
plot(a,g0(:,1),'b--',a,g0(:,2),'r--','LineWidth',2)
xlim([amin amax1])
ylim([0 gmax])
xlabel('Wealth, $a$','interpreter','latex')
ylabel('Densities, $g_i(a)$','interpreter','latex')
title('t = 0.1')

t = 2;
n = t/dt;
subplot(2,2,2)
set(gca,'FontSize',16)
h1 = plot(a,gg{n}(1:I),'b',a,gg{n}(I+1:2*I),'r','LineWidth',2)
legend(h1,'g_1(a)','g_2(a)')
hold on
plot(a,g0(:,1),'b--',a,g0(:,2),'r--','LineWidth',2)
xlim([amin amax1])
ylim([0 gmax])
xlabel('Wealth, $a$','interpreter','latex')
ylabel('Densities, $g_i(a)$','interpreter','latex')
title('t = 2')

t = 5;
n = t/dt;
subplot(2,2,3)
set(gca,'FontSize',16)
h1 = plot(a,gg{n}(1:I),'b',a,gg{n}(I+1:2*I),'r','LineWidth',2)
legend(h1,'g_1(a)','g_2(a)')
hold on
plot(a,g0(:,1),'b--',a,g0(:,2),'r--','LineWidth',2)
xlim([amin amax1])
ylim([0 gmax])
xlabel('Wealth, $a$','interpreter','latex')
ylabel('Densities, $g_i(a)$','interpreter','latex')
title('t = 5')


t = T;
n = t/dt;
subplot(2,2,4)
set(gca,'FontSize',16)
h1 = plot(a,gg{n}(1:I),'b',a,gg{n}(I+1:2*I),'r','LineWidth',2)
legend(h1,'g_1(a)','g_2(a)')
hold on
plot(a,g0(:,1),'b--',a,g0(:,2),'r--','LineWidth',2)
xlim([amin amax1])
ylim([0 gmax])
xlabel('Wealth, $a$','interpreter','latex')
ylabel('Densities, $g_i(a)$','interpreter','latex')
title('t = \infty')
print -depsc transition_distribution.eps
