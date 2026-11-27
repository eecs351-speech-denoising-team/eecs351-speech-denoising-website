function [output, info] = denoise_wiener_dd(y, fs, options)
% Standard STFT-domain Wiener filter with a decision-directed a priori SNR.
% Only the noisy signal is an input. Analysis, noise initialization and
% overlap-add match denoise_speech.m so outputs are directly comparable.
%
% For each frame m and bin k, with noise power N(k) and noisy power P = |Y|^2:
%   gamma = P / N                                   (a posteriori SNR)
%   xi    = a*|S_prev|^2/N + (1-a)*max(gamma-1, 0)  (decision-directed a priori SNR)
%   xi    = max(xi, xiMin)
%   G     = xi / (1 + xi),   S = G*Y
% With ddAlpha=0 and xiMinDb=-Inf this reduces to denoise_speech 'wiener'
% with gainFloor=0. Kept in its own file so the frozen denoise_speech.m hash
% (config/frozen_denoisers.json) is unaffected.
%
% Options (defaults): windowLength (25 ms), hop (10 ms), nfft, noiseFraction (0.2),
% ddAlpha (0.98), xiMinDb (-25). Defaults are literature starting points, not tuned.
if nargin < 3, options=struct; end
validateattributes(y,{'double','single'},{'column','real','finite','nonempty'});
validateattributes(fs,{'numeric'},{'scalar','real','finite','integer','positive'});
assert(isstruct(options) && isscalar(options),'EECS351:Options','Options must be a scalar struct.');
y=double(y); n=numel(y);
defaults=struct('windowLength',round(0.025*fs),'hop',round(0.010*fs), ...
    'nfft',2^nextpow2(round(0.025*fs)),'noiseFraction',0.2,'ddAlpha',0.98,'xiMinDb',-25);
unknown=setdiff(fieldnames(options),fieldnames(defaults));
assert(isempty(unknown),'EECS351:Options','Unknown options: %s',strjoin(unknown,', '));
names=fieldnames(defaults);
for k=1:numel(names)
    if ~isfield(options,names{k}), options.(names{k})=defaults.(names{k}); end
end
L=options.windowLength; hop=options.hop; K=options.nfft;
validateattributes(L,{'numeric'},{'scalar','integer','>=',2,'<=',n});
validateattributes(hop,{'numeric'},{'scalar','integer','positive','<',L});
validateattributes(K,{'numeric'},{'scalar','integer','even','>=',L});
validateattributes(options.noiseFraction,{'numeric'},{'scalar','real','>',0,'<=',1});
validateattributes(options.ddAlpha,{'numeric'},{'scalar','real','>=',0,'<',1});
validateattributes(options.xiMinDb,{'numeric'},{'scalar','real','nonnan','<',Inf});
w=hann(L,'periodic');

% Fixed noise estimate from the lowest-energy original full frames (as denoise_speech).
originalStarts=1:hop:n-L+1;
originalIndices=(0:L-1)'+originalStarts;
energy=mean(y(originalIndices).^2,1);
[~,rank]=sort(energy,'ascend');
chosen=rank(1:max(1,ceil(options.noiseFraction*numel(rank))));
noiseFFT=fft(y(originalIndices(:,chosen)).*w,K,1);
noisePower=mean(abs(noiseFFT(1:K/2+1,:)).^2,2);

frameCount=ceil((n+L)/hop)+1;
paddedLength=(frameCount-1)*hop+L;
padded=[zeros(L,1);y;zeros(paddedLength-L-n,1)];
starts=1:hop:paddedLength-L+1;
indices=(0:L-1)'+starts;
spectrum=fft(padded(indices).*w,K,1);
spectrum=spectrum(1:K/2+1,:);
power=abs(spectrum).^2;

% Decision-directed recursion, frame by frame in time order.
a=options.ddAlpha; xiMin=10^(options.xiMinDb/10);
N=max(noisePower,realmin);
gain=zeros(size(power));
previous=zeros(size(N)); % |S_hat(m-1)|^2
for m=1:size(power,2)
    gamma=power(:,m)./N;
    xi=max(a*previous./N+(1-a)*max(gamma-1,0),xiMin);
    G=xi./(1+xi);
    gain(:,m)=G;
    previous=G.^2.*power(:,m);
end
enhanced=gain.*spectrum;

twoSided=[enhanced;conj(enhanced(end-1:-1:2,:))];
frames=real(ifft(twoSided,K,1));
frames=frames(1:L,:).*w;
reconstructed=zeros(paddedLength,1); weights=zeros(paddedLength,1);
for k=1:numel(starts)
    ix=indices(:,k);
    reconstructed(ix)=reconstructed(ix)+frames(:,k);
    weights(ix)=weights(ix)+w.^2;
end
keep=L+(1:n);
assert(all(weights(keep)>0),'EECS351:Reconstruction','Uncovered output samples.');
output=reconstructed(keep)./weights(keep);
assert(isequal(size(output),size(y)) && all(isfinite(output)), ...
    'EECS351:DenoiserOutput','Denoiser returned invalid samples or length.');
info=struct('method','wiener_dd','fs',fs,'validSamples',[1 n], ...
    'causalFilterDelaySamples',0,'options',options, ...
    'noiseEstimator','fixed-low-energy-noisy-frames', ...
    'noiseFrameStarts',originalStarts(chosen),'noisePower',noisePower, ...
    'meanGain',mean(gain(:)));
end
