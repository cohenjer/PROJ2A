%% on initialise les variables

clear variables
close all
clc

%liste des omegas
N = 5;
W_o = linspace(1,10,N);
err_plot = zeros(1,N);

%pour chaque omega on va calculer l'erreur
for j=1:N
    
% 05/11/2018
% j'ai un peu réarrangé le code pour que ce soit plus simple, en supprimant
% la boucle sur Xt. J'ai aussi changé xi pour qu'il soit directement une
% grille sur [0,1]

% Nombre de points sur la grille
Nb = 20;

%xi = rand(1,Nb);
xi = linspace(0,1,Nb);

% Train data
Xt = xi;
Xt = [Xt;ones(1,Nb)];

% Fonction carrée comme fonction que l'on veut interpoler

yi = xi.^2 + cos(W_o(j)*xi);
yt = yi;
% Ici les dimensions de W relèvent du choix de l'utilisateur. Pourquoi
% mettre 500 ? Ici avec 1 au lieu de 500, ca marche aussi...
% J'ai également mis des randn au lieu de ones.
dims = 100;
W = randn(dims,2);
W2 = randn(1,dims+1);

[M,N] = size(W);

% Pas du gradient
mu = 0.01;

%% on calcule la sortie scalaire


% Mes corrections:
%   Corrections Majeures
%---------------------------------
% - Erreur aussi dans la syntaxe pour W (calculs probablement OK mais
% implémentation bizare). J'ai changé en refaisant comme toi les calculs à
% la main.
% - J'ai changé l'ordre des boucles. Maintenant on passe sur toutes les
% données d'entrainement, puis on regarde si la moyenne des erreurs au
% carré est faible pour savoir si on recommence. 
% - Ajout d'un b2 directement dans W2 comme je t'avais demandé.
% - Choix aléatoire de l'ordre de parcours des données d'entrainement
%
%   Corrections Mineures
%---------------------------------
% - Changement de la sortie de NNforward pour avoir les sorties séparées
% (pas dans un gros vecteur). 
% - Précalcul des Xt² dans un yt pour plus de lisibilité
% - J'ai enlevé les facteurs 2 partout. Il suffit de multiplier mu par 2
% pour retrouver le même comportement, donc c'est inutile de les mettre.
% - J'ai réécris la mise à jour de W2 en une ligne, mais c'est pareil que
% ce que tu avais avant.

err_mean = 1;
N2 = 2 ;
number = 0 ;
e = 0 ;

%on fait la moyenne sur plusieurs essais
for u=1:N2
    while (err_mean>1e-4) && (number<1000)
    
    fprintf('err_mean: %d\n',err_mean)
    err_mean = 0;
    
    index_list = randperm(length(xi));
    
    for i=index_list
    
    [n1,a1,a2]=NNforward(Xt(:,i),W,W2);
    err = a2-yt(i);
    err_mean = err_mean + err^2;
    
    W2_old = W2;
    W2 = W2 - mu*err*[a1;1]'; 
    
            % Version correcte loop
    %    for j=1:N
    %        for k=1:M
    %        W(k,j) = W(k,j) - mu*Xt(j,i)*W2_old(k)*(a1(k)*(1-a1(k)))*err;        
    %        end
    %    end
        
   % Version vectorisee
    %W = W - mu*(W2_old(1:M)'.*a1.*(1-a1))*Xt(:,i)'*err;
    % Version hierarchique
    W = W - mu*(W2(1:M)'.*a1.*(1-a1))*Xt(:,i)'*err;
    
    end % Fin for
    
    err_mean = err_mean / size(Xt,2);
    
    if abs(err_mean-e) < 0.05*abs(err_mean-e)
        number =+ 1
    end
    
    e = err_mean ;
    
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
    fun_real = grille.^2 + cos(W_o(j)*grille);       
    for l=1:1000
       [~,~, y] = NNforward([grille(l);1],W,W2);
       fun_est(l) = y;
       err_2 = (fun_est(l)-fun_real(l))^2;
       err_mean_2 =+ err_2;
    end
    err_mean_2 = err_mean_2/N_grille;
    err_plot_mean =+ err_mean_2;
end

%on stocke l'erreur pour un omega donn�
err_plot_mean = err_plot_mean/N2;
err_plot(j) = err_plot_mean;

end
    %on affiche la courbe err(omega)
    plot(W_o,err_plot);
%     figure    
%     plot(grille,fun_real,'--r')
%     hold on
%     plot(grille,fun_est,'k')
%     legend('True','NN')
%     title('Comparaison grille fine') 
     
    % TODO: Ajouter une couche pour tester l'expressivité
    % TODO: Regarder différentes fonctions (pas que x^2)
    % TODO: regarder l'influence de la taille interne des W
