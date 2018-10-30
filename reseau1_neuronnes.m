%% on initialise les variables 
W = transpose([0 1 0])
b = 0
%critères = [forme, texture, poids]
p_1 = [1 -1 -1]
p_2 = [1 1 -1]
%% on calcule le réseau de neuronne

%on fait le produit matriciel WP
cross_1 = p_1*W
cross_2 = p_2*W
N=[cross_1, cross_2]
%on utilise hardlims pour déterminer les outputs a1 et a2
s=zeros(1,length(N))

for i=1:length(N)
    s(i) = hardlims(N(i))
end

%on donne le nom du fruit en question
for k=1:length(s)
    if s(k)==1
        disp("pomme")
    elseif s(k)==-1
        disp('poire')
    end
end


