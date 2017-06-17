function [point, weight] = glq1d(n)

point = zeros(n,1);
weight = zeros(n,1);

if n==1
    point(1)  = 0.0;
    weight(1) = 2.0;
elseif n==2
    point(1)  = -0.577350269189626;
    point(2)  = -point(1);
    weight(1) = 1.0;
    weight(2) = 1.0;
elseif n==3
    point(1)  = -0.774596669241483;
    point(2)  = 0.0;
    point(3)  = -point(1);
    weight(1) = 0.555555555555556;
    weight(2) = 0.888888888888889;
    weight(3) = 0.555555555555556;
elseif n==4
    point(1)  = -0.861136311594053;
    point(2)  = -0.339981043584856;
    point(3)  = -point(2);
    point(4)  = -point(1);
    weight(1) = 0.347854845137454;
    weight(2) = 0.652145154862546;
    weight(3) = 0.652145154862546;
    weight(4) = 0.347854845137454;
else 
    error('Higher order quadrature rules are not available')
end