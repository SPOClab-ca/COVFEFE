% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [KFD] = Katz_FD(serie)
%{
Script for computing the Katz Fractal Dimension (KFD).

INPUT:
    serie: is the temporal series that one wants to analyze by KFD. 
    It must be a row vector.

OUTPUT:
    KFD: the KFD of the temporal series.

TIP: the KFD of a straight line must be exactly 1. Otherwise, the
implementation of the algorithm is wrong. 

PROJECT: Research Master in signal theory and bioengineering - University of Valladolid

DATE: 08/07/2014

AUTHOR: Jess Monge lvarez
%}
%% Checking the ipunt parameters:
control = ~isempty(serie);
assert(control,'The user must introduce a series (first inpunt).');


%% Processing:
% Computing 'L':
% 'L' is the total length of the curve, that is to say, the sum of the
% distance between succesive points. The distance between two points of the
% waveform is defined as the Ecludian distance: dist(s1,s2) = sqrt[(x1-x2)^2 + (y1 - y2)^2];
% For this case: (x1 - x2) = 1 for all samples. 
L = 0;
N = length(serie);
n = N - 1; %'n' is the number of steps in the waveform. 
for i = 1:(N - 1)
    aux = sqrt(1 + ((serie(i) - serie(i+1))^2));
    L = L + aux;
    clear('aux');
end

% Computing 'd':
% 'd' is the planar extent or diameter of the waveform. It is estimated as
% the distance between the first point of the sequence and the point of the 
% sequence that provides the farthest distance: d = max(dist(1,i)); i=2...N.
dist = NaN(1,N-1); %Predifinition variable for computatinoal efficiency.
for i = 2:N
    dist(i) = sqrt(((1 - i)^2) + ((serie(1) - serie(i))^2));
end
d = max(dist);

% Computing of KFD:
% The KFD is computed as follows: KFD = log10(n) / [log10(n) + log10(d/L)];
KFD = log10(n)/(log10(n) + log10(d/L));
