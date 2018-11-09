%% on initialise les variables

clear variables
close all
clc

%liste des omegas
Nomega = 20;
W_o = linspace(0,11,Nomega);
err_omega = zeros(1,Nomega);

%pour chaque omega on va calculer l'erreur
for j=1:Nomega
    

% Nombre de points sur la grille
Nb = 20;

%xi = rand(1,Nb);
xi = linspace(0,1,Nb);

% Train data
Xt = xi;
Xt = [Xt;ones(1,Nb)];

% Fonction que l'on veut interpoler
%yi = xi.^2 + cos(2*pi*W_o(j)*xi);
yi = sin(2*pi*W_o(j)*xi);
yt = yi;

%Rq: theoreme de Shannon respecte si omega<10.

%% on calcule la sortie scalaire


err_mean = 1;
N2 = 5 ;
e = 0 ;
% Pas du gradient
mu = 0.01;
% Nombre max d'iteration
itermax = 1000;


%on fait la moyenne sur plusieurs essais
for u=1:N2
    
% initialisation    
dims = 10;
W = randn(dims,2);
W2 = randn(1,dims+1);
number = 0 ;
[M,N] = size(W);
err_mean = 1;

    while (err_mean>10^(-4)) && (number<itermax)
    
        if mod(number,10)==0
        fprintf('iter: %d, err_mean: %g\n',number,err_mean)
        end
        err_mean = 0;
        
        index_list = randperm(length(xi));
        
        for i=index_list
            
            % Precomputations
            [~,a1,a2]=NNforward(Xt(:,i),W,W2);
            err = a2-yt(i);
            err_mean = err_mean + err^2;
            W2_old = W2;
            
            % Gradient steps
            W2 = W2 - mu*err*[a1;1]';
            
            % hierarchical method
            % re-computation of error since W2 has changed
            %[n1,a1,a2]=NNforward(Xt(:,i),W,W2);
            %err = a2-yt(i);
            %W = W - mu*(W2(1:M)'.*a1.*(1-a1))*Xt(:,i)'*err;
            
            % Gradient method
            W = W - mu*(W2_old(1:M)'.*a1.*(1-a1))*Xt(:,i)'*err;
            
        end % Fin for
        
        err_mean = err_mean / size(Xt,2);
        
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
err_omega(j) = mean(err_plot);

end
    %on affiche la courbe err(omega)
    plot(W_o,err_omega);
%     figure    
%     plot(grille,fun_real,'--r')
%     hold on
%     plot(grille,fun_est,'k')
%     legend('True','NN')
%     title('Comparaison grille fine') 
     
    % TODO: Ajouter une couche pour tester l'expressivité
    % TODO: Regarder différentes fonctions (pas que x^2)
    % TODO: regarder l'influence de la taille interne des W
