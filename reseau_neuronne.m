%% on initialise les variables

xi = 10*rand(1,1000);
Xt = zeros(2,500);
X = xi(1,1:2:end);
X = [X;ones(1,500)];
yi=zeros(1,500);

for k=1:length(xi)/2
    Xt(1,k) = xi(1,2*k);
    Xt(2,k) = 1;
end
W = ones(500,2);
W2 = ones(1,500);

[M,N] = size(W);
mu = 1e-2;

%% on calcule la sortie scalaire
e=1;
i=1;

while (e>1e-3)
    K=NNforward(Xt(:,i),W,W2);
    
    a2 = K(end);
    a1 = K(501:end-1);
    n1 = K(1:1000);

    e = ((Xt(i)^2)-a2)^2;
    for k=1:M
        W2(k) = W2(k) + 2*mu*a1(k)*sqrt(e);
        for j=1:N
            E = -2*exp(-n1(k))/(1+exp(-n1(k)))^2*sqrt(e)*Xt(j,i);
            W(k,j) = W(k,j) - mu*E;
        end
    end
    i = mod(i,length(xi)/2)+1;
end

for w=1:length(X)
    K =NNforward(X(:,w),W,W2);
    yi(1,w)= K(end);
end

y=xi.^2;
X=sort(X(1,:));
yi=sort(yi);
xi=sort(xi);
y=sort(y);
plot(X,yi)
hold on 
plot(xi,y)



