%% on initialise les variables

clear variables
close all
clc

%liste des omegas
Nomega = 5;
W_o = linspace(0,0.5,Nomega);
err_omega = zeros(1,Nomega);

%pour chaque omega on va calculer l'erreur
for j=1:Nomega
    
% Nombre de points sur la grille
Nb = 10;

%xi = rand(1,Nb);
xi = linspace(0,1,Nb);

% Train data
Xt = xi;
Xt = [Xt;ones(1,Nb)];

% Fonction que l'on veut interpoler
%yi = xi.^2 + cos(2*pi*W_o(j)*xi);.1106
yi = sin(2*pi*W_o(j)*xi);
yt = yi;

%Rq: theoreme de Shannon respecte si omega<Nb/2 et si temps infini.

%% on calcule la sortie scalaire


err_mean = 1;
N2 = 5 ;
e = 0 ;

% Nombre max d'iteration
itermax = 100000;


%on fait la moyenne sur plusieurs essais
for u=1:N2
    
% initialisation    
dims = 10;
W = randn(dims,2);
W2 = randn(1,dims+1);
% Pas du gradient
mu = 0.00001;

number = 0 ;
[M,N] = size(W);
err_mean = 1;
N2 = 2;
number = 0 ;
e = 0 ;
mom = 0; mom2 = 0;
alpha = 0.5;
Y1 = W;
Y2 = W2;

    while (err_mean>5*10^(-5)) && (number<itermax)

        
        if mod(number,100)==0
        fprintf('iter: %d, err_mean: %g\n',number,err_mean)
        end
        err_mean = 0;
        
        index_list = randperm(length(xi));
%         
%         for i=index_list
%             
%             % Precomputations
%             [~,a1,a2]=NNforward(Xt(:,i),W,W2);
%             err = a2-yt(i);
%             err_mean = err_mean + err^2;
%             
%             % Gradient computations
%             g2 = err*[a1;1]';
%             g  = (W2(1:M)'.*a1.*(1-a1))*Xt(:,i)'*err;
%             
%             % Storing olds
%             Wold = W;
%             W2old = W2;
%             
%             % Gradient steps
%             W = W - mu*g + alpha*mom;
%             W2 = W2 - mu*g2 + alpha*mom2;
%             
%             % Momentum update
%             mom  =  W - Wold;
%             mom2 =  W2-W2old;
%             
%         end % Fin for
%         
        % Batch implementation
        
            
        % Precomputations
        for i=index_list
            [~,a1,a2]=NNforward(Xt(:,i),W,W2);
            a1_vec(:,i) = a1;
            err(i) = a2-yt(i);
        end    
        err_mean = mean(err.^2);
        a1_vec_ones = [a1_vec;ones(1,length(index_list))];
        
        % Gradient computations
        g2 = sum(err)*sum(a1_vec_ones,2)';
        g  = (repmat(W2(1:M)',1,length(index_list)).*a1_vec.*(1-a1_vec))...
           *diag(err)*Xt';    
        g  = (repmat(Y2(1:M)',1,length(index_list)).*a1_vec.*(1-a1_vec))...
            *diag(err)*Xt';    
    
        % Storing olds
        Wold = W;
        W2old = W2;
            
        % Gradient step
        W = Y1 - mu*g;
        W2 = Y2 - mu*g2; 
        
        % Black magic (q=0)
        alpha_old = alpha;
        alpha = 1/2*(-alpha_old^2 + sqrt(alpha_old^4 + 4*alpha_old^2));
        beta = alpha_old*(1-alpha_old)/(alpha_old^2 + alpha);
        
        % Auxiliary variables update
        Y1 = W + beta * (W - Wold);
        Y2 = W2 + beta * (W2 - W2old);
        
        %if abs(err_mean-e) < 0.05*abs(err_mean-e)
        number = number +1; % increment iteration
        %end
    
    end % Fin while
    


%% Post-processing

% % Sorties yt_est sur les données train
% for w=1:length(Xt)
%     [~,~,out] =NNforward(Xt(:,w),W,W2);
%     yt_est(1,w)= out;
% end
% 
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
    err_train(j,u) = err_mean;


    % 3/ Sur une grille fine (test)
    N_grille = 1000;
    grille = linspace(0,1,N_grille);
    %fun_real = grille.^2 + cos(2*pi*W_o(j)*grille);       
    fun_real = sin(2*pi*W_o(j)*grille);       
    for l=1:1000
       [~,~, outnn] = NNforward([grille(l);1],W,W2);
       fun_est(l) = outnn;
       err_2(l) = (fun_est(l)-fun_real(l))^2;
    end
    err_plot(u) = mean(err_2);
end

%on stocke l'erreur pour un omega donn�
std_omega(j) = std(err_plot);
err_omega(j) = mean(err_plot);

end

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
