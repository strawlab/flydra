t = linspace(0, 10*pi, 400);
Porig = sin(t);
P = Porig + randn( size(Porig) );
Pstar = smooth_position( P, 0.01, 0.5, 1e-9, 1e12 );
plot(t,Porig,'r');
hold on;
plot(t,P,'g');
plot(t,Pstar,'b');