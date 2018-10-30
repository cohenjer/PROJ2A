%% on initialise les variables

xi = 10*rand(1,1000);
X = xi(1,1:2:end);
yi=zeros(1,500);

for k=1:length(xi)/2
    Xt(k) = xi(1,2*k);
end

yi=zeros(1,50);

W = ones(1000,1);
W2 = ones(1,1000);

[M,N] = size(W);
mu = 1e-7;

%% on calcule la sortie scalaire
e=1;
i=1;

while (e>1e-3)&&(i<=length(xi))
    K=NNforward(Xt(:,i),W,W2,0);
    
    a2 = K(end);
    a1 = K(101:end-1);
    n1 = K(1:100);

    e = ((Xt(i)^2)-a2)^2;
    for k=1:M
        W2(k) = W2(k) - 2*mu*a1(k)*e;
        for j=1:N
            E = -2*exp(-n1(k))/(1+exp(-n1(k))^2)*e*Xt(j,i);
            W(k,j) = W(k,j) + mu*E;
        end
    end
    i =i+1;
end

for w=1:length(X)
    K =NNforward(X(:,w),W,W2,0);
    yi(1,w)= K(end);
end




