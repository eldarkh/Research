function [point, weight] = glq2d(nx, ny)

if nx > ny
    n = nx;
else 
    n = ny;
end

point = zeros(n,2);
weight = zeros(n,2);

[pointx, weightx] = glq1d(nx);
[pointy, weighty] = glq1d(ny);

for intx = 1:nx
    point(intx,1)  = pointx(intx);
    weight(intx,1) = weightx(intx);
end

for inty = 1:ny
    point(inty,2)  = pointy(inty);
    weight(inty,2) = weighty(inty);
end
