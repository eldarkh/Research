function [basis4, d_ksi, d_eta] = basisfcn(ksi,eta)

% Basis functions
basis4(1) = 0.25*(1-ksi)*(1-eta);
basis4(2) = 0.25*(1+ksi)*(1-eta);
basis4(3) = 0.25*(1+ksi)*(1+eta);
basis4(4) = 0.25*(1-ksi)*(1+eta); 

% Derivatives w.r.t ksi and eta
d_ksi(1) = -0.25*(1-eta);
d_ksi(2) = 0.25*(1-eta);
d_ksi(3) = 0.25*(1+eta);
d_ksi(4) = -0.25*(1+eta);

d_eta(1) = -0.25*(1-ksi);
d_eta(2) = -0.25*(1+ksi);
d_eta(3) = 0.25*(1+ksi);
d_eta(4) = 0.25*(1-ksi);