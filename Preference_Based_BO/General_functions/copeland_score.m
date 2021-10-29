function C = copeland_score(c, normalized)
DEFAULT('normalized', 'yes')

C = sum(c>0.5,1); 

if strcmp(normalized, 'yes') %compute the normalized Copeland score
    C = C/size(c,1); 
end
return
