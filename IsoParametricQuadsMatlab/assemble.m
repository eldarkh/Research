function A=assemble(A,M,ind)

 eldof = length(ind);
 for k=1:eldof
    i=ind(k);
        for l=1:eldof
            j=ind(l);
            A(i,j)=A(i,j)+M(k,l);
        end
 end
