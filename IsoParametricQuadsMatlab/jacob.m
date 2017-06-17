function jcbian = jacob(nnodes, d_ksi, d_eta, x, y)

jcbian = zeros(2,2);
for k = 1:nnodes
    jcbian(1,1) = jcbian(1,1) + d_ksi(k)*x(k);
    jcbian(1,2) = jcbian(1,2) + d_ksi(k)*y(k);
    jcbian(2,1) = jcbian(2,1) + d_eta(k)*x(k);
    jcbian(2,2) = jcbian(2,2) + d_eta(k)*y(k);
end
