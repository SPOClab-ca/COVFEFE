function [feature, feature_names] = extract_acoustics( wav_fn, outFile )
% extractFeatures
%
%     for wav_fn, this function extracts
%
%% http://www.mathworks.com/help/signal/ug/estimating-fundamental-frequency-with-the-complex-cepstrum.html
%

addpath(genpath('/p/spoclab/tools/SpeechAnalysis/Straight/'));
addpath(genpath('/p/spoclab/tools/SilenceRemoval/'));
addpath(genpath('/p/spoclab/tools/crptool2'));
addpath('/p/spoclab/tools/rpde/');
addpath('/p/spoclab/tools/');
addpath('/p/spoclab/tools/SpeechAnalysis/voicebox');
addpath('/p/spoclab/tools/SpeechAnalysis');

RPDEm = 10;
RPDEtau = 1;
RPDEepsilon = 0.1;

featMatrix = [];

filenames = {};
  
feature = [];
feature_names = {};

[x,fs]=audioread( wav_fn );

[segments,sr,Limits]=detectVoiced( wav_fn ); % speech/unspeech
pauses = [];

if ~isempty(Limits)
  pauses = [1 Limits(1, 1)-1];
end

for l=2:(size(Limits,1))
  pauses = [pauses; Limits(l-1,2)+1 Limits(l,1)-1];
end

allSegs = cell2mat(segments');

if (length(allSegs) == 0 )
  return;
end

% FORMANTS
F123 = zeros(length(segments),3);
for s=1:length(segments)
  x1 = segments{s}.*hamming(length(segments{s}));
  preemph = [1 0.63]; % pre-emphasis
  x1 = filter(1, preemph, x1);
  A = lpc(x1, 8);
  rts = roots(A);
  rts = rts(imag(rts)>=0);
  angz = atan2(imag(rts),real(rts));
  [frqs,indices] = sort(angz.*(fs/(2*pi)));
  bw = -1/2*(fs/(2*pi))*log(abs(rts(indices)));
  nn = 1;
  formants = [0 0 0];
  for kk = 1:length(frqs)
    if (frqs(kk) > 90 && bw(kk) <400) % formant criteria
      formants(nn) = frqs(kk);
      nn = nn+1;
    end
  end
  
  F123(s,1) = formants(1);
  F123(s,2) = formants(2);
  F123(s,3) = formants(3);
end

feature_names = [feature_names {'filename'}];
filenames = [filenames {wav_fn}];

% 1. Phonation rate
if ~isempty(Limits)
  feature(length(feature)+1) = sum(Limits(:,2)-Limits(:,1))/length(x);
else
  feature(length(feature)+1) = -1;
end
feature_names = [feature_names {'phon_rate'}];

% 2. mean duration of pauses
if ~isempty(pauses)
  feature(length(feature)+1) = sum(pauses(:,2)-pauses(:,1))/size(pauses,1);
else
  feature(length(feature)+1) = -1;
end
feature_names = [feature_names {'mean_dur_pauses'}];

% 3. pause-to-word ratio (word??) 
%    ratio of non-silent segments, excluding filled pauses, to silent segments longer 
%    than 150 ms
if ~isempty(pauses)
  feature(length(feature)+1) = size(segments,1)/length(find((pauses(:,2)-pauses(:,1))/sr>0.15));
else
  feature(length(feature)+1) = -1;
end
feature_names = [feature_names {'pause-to-word_ratio'}];

% 5. total duration of speech
if ~isempty(Limits)
  feature(length(feature)+1) = sum(Limits(:,2)-Limits(:,1));
else
  feature(length(feature)+1) = -1;
end
feature_names = [feature_names {'tot_dur'}];

% 6. Long pause count (raw)
if ~isempty(pauses)
  feature(length(feature)+1) = length(find((pauses(:,2)-pauses(:,1))/sr>0.4));
else
  feature(length(feature)+1) = -1;
end
feature_names = [feature_names {'long_pause_count'}];

% 7. Short pause count (raw)
if ~isempty(pauses)
  feature(length(feature)+1) = length(intersect( find((pauses(:,2)-pauses(:,1))/sr>0.15), ...
                                                 find((pauses(:,2)-pauses(:,1))/sr<0.4)));
else
  feature(length(feature)+1)=-1;
end
feature_names = [feature_names {'short_pause_count'}];

% 11. Skewness
feature(length(feature)+1) = skewness( allSegs, 0); % adjusted for bias
feature_names = [feature_names {'skewness'}];

% 12. Kurtosis
feature(length(feature)+1) = kurtosis( allSegs, 0); % adjusted for bias
feature_names = [feature_names {'kurtosis'}];

% 13. ZCR
feature(length(feature)+1) = ZCR( allSegs);
feature_names = [feature_names {'ZCR'}];

% Average F1, F2, F3
feature(length(feature)+1:length(feature)+3) = mean(F123,1);
feature_names = [feature_names {'mean_f1', 'mean_f2', 'mean_f3'}];

% Variance F1, F2, F3
feature(length(feature)+1:length(feature)+3) = var(F123,0,1);
feature_names = [feature_names {'var_f1', 'var_f2', 'var_f3'}];

% kurtosis, skewness of LPC
[lpcAR, lpcE] = lpcauto( allSegs ); % 12th order, default; a covariance-based method is available
feature(length(feature)+1) = kurtosis( lpcAR, 0 );
feature(length(feature)+1) = skewness( lpcAR, 0 );
feature(length(feature)+1) = lpcE;
feature_names = [feature_names {'lpc_kurtosis', 'lpc_skewness', 'lpc_E'}];

% xcorr
xxXcorr = xcorr( allSegs, allSegs );
feature(length(feature)+1) = mean( allSegs );
feature(length(feature)+1) = max( allSegs );
feature(length(feature)+1) = min( allSegs );
feature_names = [feature_names {'mean_xcorr', 'max_xcorr', 'min_xcorr' } ];


% crqa
% FR's params
if (length(allSegs) > 500 )
  y = crqa (allSegs, 1,1,2,  500,250,'euc', 'nogui')
  % remove rows with NaNs in them
  y(any(isnan(y),2),:)=[];
else
  y = -1*ones(1,13);
end

feature = [feature mean(y,1)];
feature_names = [feature_names {'recurrence_rate', 'determinism', 'mean_diag_line_length', 'max_diag_line_length', 'entropy_of_diag_line_length', 'laminarity', 'trapping_time', 'max_vert_line_length', 'recurr_time_1st_type', 'recurr_time_2nd_type', 'rpde', 'clustering_coef', 'transitivity'}];

%transitivity added above to feature_names
   
%TODO: add wavelets, filterbanks...

featMatrix = [ featMatrix ; feature ];

fid = fopen( outFile, 'w');

for i=1:length(feature_names)-1
  fprintf(fid, [feature_names{i} ',']);
end
fprintf(fid, [feature_names{end} '\n']);
for i=1:size(featMatrix,1)
    fprintf(fid, [filenames{i} ',']);
    for j=1:size(featMatrix,2)-1
        fprintf(fid, [num2str(featMatrix(i,j)) ',']);
    end
    fprintf(fid, [num2str(featMatrix(i,end)) '\n']);
end




