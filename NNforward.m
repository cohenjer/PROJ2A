function [S]= NNforward(p,W,W2)

N = W*p;
k = length(N);
s = zeros(k,1);

for i=1:k
    s(i) = sigmoid(N(i));
end

y = W2*s;

S=[N;s;y];

end


