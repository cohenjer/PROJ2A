function [N,s,y]= NNforward(p,W,W2)

N = W*p;
k = length(N);
s = zeros(k,1);

for i=1:k
    s(i) = sigmoid(N(i));
end

% Adding an offset on the second layer as well
s = [s;1];

y = W2*s;


end


