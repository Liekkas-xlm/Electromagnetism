clc;
clear;
lambda_wave = 2*pi;
L = 5*lambda_wave;
m = 1000;
x = 0:L/m:L;
epsilon = 4+(2-0.1i)*(1-x./L).^2;
miu = (2-0.1i).*ones(1,m);
epsilon = [epsilon(1:m),1];
miu = [miu(1:m),1];
theta = 0;
k0 = 2*pi/lambda_wave;

kxm = k0*(miu.*epsilon-sin(theta)^2).^0.5;

R = -1;
R_record = R;
lambda_Record = [];
for i  = 1:1:m
    lambda = (-miu(i+1)*kxm(i)+miu(i)*kxm(i+1))/(miu(i+1)*kxm(i)+miu(i)*kxm(i+1));
    lambda_Record = [lambda_Record,lambda];
    R = (lambda+R*exp(-2i*kxm(i)*x(i+1)))/(1+lambda*R*exp(-2i*kxm(i)*x(i+1)))...
        *exp(2i*kxm(i+1)*x(i+1));
    R_record = [R_record,R];
end

disp(abs(R))
plot(abs(R_record))
figure()
plot(abs(kxm))
% disp(lambda_Record)