%% on initialise les variables

clear variables
close all
clc

%liste des omegas
Nomega = 5;
W_o = linspace(0,2,Nomega);
err_omega = zeros(1,Nomega);

%pour chaque omega on va calculer l'erreur
for j=1:Nomega
    
% Nombre de points sur la grille
Nb = 20;

%xi = rand(1,Nb);
xi = linspace(-1,1,Nb);

% Train data
Xt = xi;
Xt = [Xt;ones(1,Nb)];

% Fonction que l'on veut interpoler
%yi = xi.^2 + cos(2*pi*W_o(j)*xi);.1106
%yi = sin(2*pi*W_o(j)*xi);
%yi = randn(1,Nb);
yi = atan(W_o(j)*xi);
yt = yi;

%Rq: theoreme de Shannon respecte si omega<10.

%% on calcule la sortie scalaire


err_mean = 1 ;
N2 = 2 ;
e = 0 ;

% Nombre max d'iteration
itermax = 10000 ;


%on fait la moyenne sur plusieurs essais
for u=1:N2
    
% initialisation    
dims = 1000 ;
W = randn(dims,2) ;
W2 = 1/sqrt(dims)*randn(1,dims+1) ;
%W2 = randn(1,dims);

% Pas du gradient
mu = 0.001 ;

% exponential decay rates
beta_1 = 0.9 ;
beta_2 = 0.999 ; 

number = 0 ;

[M , N] = size(W) ;
[M_W2 , N_W2] = size(W2) ;

err_mean = 1 ;
N2 = 2 ;
number = 0 ;
e = 0 ;
eps = 1e-5 ;

m_t =  zeros(M , N) ;
v_t =  zeros(M , N) ;

m_t_2 = zeros(M_W2 , N_W2) ;
v_t_2 = zeros(M_W2 , N_W2) ;

    while (err_mean>eps) && (number<itermax)

        %mu = 0.1*rand(1); %test chelou
        if mod(number,100)==0
        fprintf('iter: %d, err_mean: %g\n',number,err_mean)
        end
        err_mean = 0 ;
        
        index_list = randperm(length(xi)) ;
        
        for i=index_list
            
            % Precomputations
            [N1,a1,a2]=NNforward(Xt(:,i),W,W2) ;
            err = a2-yt(i ) ;
            err_mean = err_mean + err^2 ;
            
            % Gradient computations
            g2 = err*[a1;1]' ;
            %g  = (W2(1:M)'.*a1.*(1-a1))*Xt(:,i)'*err ;
            g  = (W2(1:M)'.*(N1>0))*Xt(:,i)'*err ; % RELU
            
            % Update biases
            m_t_2 = beta_1 * m_t_2 + (1 - beta_1) .* g2 ;
            v_t_2 = beta_2 * v_t_2 + (1 - beta_2) .* g2 .* g2 ;
            
            m_hat_2 = m_t_2 / (1 - beta_1 ^ (number + 1)) ; 
            v_hat_2 = v_t_2 / (1 - beta_2 ^ (number + 1)) ;
            
            m_t = beta_1 * m_t + (1 - beta_1) .* g ;
            v_t = beta_2 * v_t + (1 - beta_2) .* g .* g ;
            
            m_hat = m_t / (1 - beta_1 ^ (number + 1)) ; 
            v_hat = v_t / (1 - beta_2 ^ (number + 1)) ;
            
            % Gradient steps
            W = W - mu * m_hat ./ (sqrt(v_hat) + eps) ;
            W2 = W2 - mu * m_hat_2 ./ (sqrt(v_hat_2) + eps) ;
            
        end % Fin for
        
        err_mean = err_mean / size(Xt,2) ;
        
        %if abs(err_mean-e) < 0.05*abs(err_mean-e)
        number = number + 1 ; % increment iteration
        %end
    
    end % Fin while
    


%% Post-processing

% % Sorties yt_est sur les données train
for w=1:length(Xt)
    [~,~,out] =NNforward(Xt(:,w),W,W2);
    yt_est(1,w)= out;
end

% 
% % Tracés
%     % 1/ Performances sur données train (normalement très bon)
%     figure
%     plot(Xt(1,:),yt,'+')
%     hold on
%     plot(Xt(1,:),yt_est,'*')
%     legend('True','NN')
%     title('Train')
        

    % Training error
    err_train(j,u) = err_mean ;


    % 3/ Sur une grille fine (test)
    N_grille = 1000;
    grille = linspace(-1,1,N_grille);
    %fun_real = grille.^2 + cos(2*pi*W_o(j)*grille);       
    %fun_real = sin(2*pi*W_o(j)*grille);       
    fun_real = atan(W_o(j)*grille);
    for l=1:1000
       [~,~, outnn] = NNforward([grille(l);1],W,W2);
       fun_est(l) = outnn;
       err_2(l) = (fun_est(l)-fun_real(l))^2;
    end
    err_plot(u) = mean(err_2);
end

%on stocke l'erreur pour un omega donné
std_omega(j) = std(err_plot);
err_omega(j) = mean(err_plot);

end

% Last Fitting
 figure
plot(xi,yi,'+')
hold on
plot(xi,yt_est,'*')
plot(grille,fun_est,'r')
plot(grille,fun_real,'b')
plot(grille,interp1(xi,yi,grille),'k')

    %on verifie que la courbe err train est relativement plate (on doit
    %pouvoir avoir toujours proche de 0 erreur de reconstruction)
    figure
    plot(W_o,mean(err_train,2))
    
    %on affiche la courbe err(omega)
    figure
    plot(W_o,err_omega);
    hold on
    plot(W_o,err_omega + std_omega);
    plot(W_o,err_omega - std_omega);
    

    
    % TODO: regarder l'influence de la taille interne des W
