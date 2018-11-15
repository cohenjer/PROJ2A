function [N,s,y]= NNforward(p,W,W2)

N = W*p;
k = length(N);
s = zeros(k,1);

for i=1:k
    %s(i) = sigmoid(N(i));
    s(i) = max(N(i),0); %RELU
end

% Adding an offset on the second layer as well
sb = [s;1];
%sb = s;

y = W2*sb;


end


