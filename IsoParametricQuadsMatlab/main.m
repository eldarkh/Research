load coordinates.dat; 
coordinates(:,1)=[];
load elements.dat; 
elements(:,1)=[];
load neumann.dat;
neumann(:,1) = [];
load dirichlet.dat; 
dirichlet(:,1) = [];

% Parameters:
nelem    = size(elements, 1);
nelnodes = size(elements, 2);
ndof     = 1;
nnodes   = size(coordinates,1);
totaldof = nnodes*ndof;
neldof   = nelnodes*ndof;
nx       = 2;
ny       = 2;

% Initialization
FreeNodes=setdiff(1:size(coordinates,1),unique(dirichlet));
A = sparse(size(coordinates,1),size(coordinates,1));
b = sparse(size(coordinates,1),1);

% Assemble of element matrices
[point, weight] = glq2d(nx,ny);
for el = 1:nelem
    for k = 1:nelnodes
        nodes(k) = elements(el, k); % determine nodes on this element
        xcoord(k) = coordinates(nodes(k), 1);
        ycoord(k) = coordinates(nodes(k), 2);
    end
    
    % Initialize element matrix
    M = zeros(neldof, neldof);
    
    % Numerical integration
    for intx = 1:nx
        x = point(intx,1);
        wx = weight(intx,1);
        for inty = 1:ny
            y = point(inty,2);
            wy = weight(inty,2);
            
            % Compute basis functions
            [basis, d_ksi, d_eta] = basisfcn(x,y);
            
            % Compute Jacobian, its determinant and inverse
            jcbian = jacob(nelnodes, d_ksi, d_eta, xcoord, ycoord);
            det_jacobian = det(jcbian);
            inverse_jacobian = inv(jcbian);
            
            % Compute physical derivatives
            [dx, dy] = deriv(nelnodes, d_ksi, d_eta, inverse_jacobian);
            
            % Compute element matrix
            for k = 1:neldof
                for l = 1:neldof
                    M(k,l) = M(k,l) + (dx(k)*dx(l) + dy(k)*dy(l))*wx*wy* ...
                    det_jacobian;
                end
            end
        end
    end
    
    A = assemble(A, M, nodes);
end

% Volume Forces
for j = 1:size(elements,1)
    b(elements(j,:)) = b(elements(j,:)) + ...
        0.5*(abs(det([1 1 1; coordinates(elements(j,[1 2 4]),:)'])) + ... 
        abs(det([1 1 1; coordinates(elements(j,[2 3 4]),:)']))) * ...
        f(sum(coordinates(elements(j,:),:))/4)/4;
end

% Boundary conditions
% Neumann conditions
for j = 1 : size(neumann,1)
    b(neumann(j,:))=b(neumann(j,:)) + ...
        norm(coordinates(neumann(j,1),:) - ...
        coordinates(neumann(j,2),:)) * ...
        g(sum(coordinates(neumann(j,:),:))/2)/2;
end

% Dirichlet conditions
u = sparse(size(coordinates,1),1);
u(unique(dirichlet)) = u_d(coordinates(unique(dirichlet),:));
b = b - A * u;

u(FreeNodes) = A(FreeNodes,FreeNodes) \ b(FreeNodes);
show(elements,coordinates,full(u))
    