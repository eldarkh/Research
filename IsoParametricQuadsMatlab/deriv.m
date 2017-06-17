function [dx,dy] = deriv(nnodes, d_ksi, d_eta, inverse_jacob)

for k = 1:nnodes
    dx(k) = inverse_jacob(1,1)*d_ksi(k) + inverse_jacob(1,2)*d_eta(k);
    dy(k) = inverse_jacob(2,1)*d_ksi(k) + inverse_jacob(2,2)*d_eta(k);
end

