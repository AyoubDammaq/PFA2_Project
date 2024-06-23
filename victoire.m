close all;
AA=load('bd.txt');
n=max(length(AA));
for i=1:50 %parcours i=1:50  puis i=51:100 ....i=3101:3135
xx=[0 AA(i,7) AA(i,9) AA(i,11) AA(i,13) AA(i,15) AA(i,16)];
yy=[AA(i,5) AA(i,6) AA(i,8) AA(i,10) AA(i,12) AA(i,14) 0];
figure;
plot(xx,yy);hold on;
plot(-xx,yy);
end