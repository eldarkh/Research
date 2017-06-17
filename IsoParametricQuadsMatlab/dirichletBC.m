function [A,f]=dirichletBC(A,f,dirichlet,func)

u_D(:,1) = unique(dirichlet);
u_D(:,2) = func(u_D(:,1));

 for k = 1:length(dirichlet)
    for j = 1:size(A)
        A(dirichlet(k),j) = 0;
    end

    A(dirichlet(k),dirichlet(k))=1;
    f(dirichlet(k))=u_D(k,2);
 end