function [output, info] = denoise_speech(y, fs, method, options)
% Offline single-channel baselines; only the noisy signal is an input.
% Methods: identity, fir, spectral_subtraction, wiener.
% Returns an aligned, same-length column vector without gain normalization.
% info.validSamples gives the inclusive interval unaffected by FIR zero padding.
% Spectral methods use a fixed estimate from low-energy, unpadded noisy frames.
% Defaults are implementation candidates, not tuned or validated optima.
if nargin < 4, options=struct; end
validateattributes(y,{'double','single'},{'column','real','finite','nonempty'});
validateattributes(fs,{'numeric'},{'scalar','real','finite','integer','positive'});
assert(isstruct(options) && isscalar(options),'EECS351:Options','Options must be a scalar struct.');
method=validatestring(method,{'identity','fir','spectral_subtraction','wiener'});
y=double(y); n=numel(y);
info=struct('method',method,'fs',fs,'validSamples',[1 n], ...
    'causalFilterDelaySamples',0,'options',struct);
switch method
    case 'identity'
        checkOptions(options,{});
        output=y;
    case 'fir'
        defaults=struct('order',128,'cutoffHz',200,'upperCutoffHz',[]);
        options=completeOptions(options,defaults);
        validateattributes(options.order,{'numeric'},{'scalar','integer','positive','even'});
        validateattributes(options.cutoffHz,{'numeric'},{'scalar','real','finite','positive','<',fs/2});
        if isempty(options.upperCutoffHz)
            b=fir1(options.order,options.cutoffHz/(fs/2),'high',hamming(options.order+1),'scale');
        else
            validateattributes(options.upperCutoffHz,{'numeric'}, ...
                {'scalar','real','finite','>',options.cutoffHz,'<',fs/2});
            b=fir1(options.order,[options.cutoffHz options.upperCutoffHz]/(fs/2), ...
                'bandpass',hamming(options.order+1),'scale');
        end
        delay=options.order/2;
        filtered=conv(y,b(:),'full');
        output=filtered(delay+(1:n));
        info.causalFilterDelaySamples=delay;
        info.validSamples=[delay+1 n-delay]; % Start > end means no fully supported samples.
        info.coefficients=b;
        info.options=options;
    otherwise
        defaults=struct('windowLength',round(0.025*fs),'hop',round(0.010*fs), ...
            'nfft',2^nextpow2(round(0.025*fs)),'noiseFraction',0.2);
        if strcmp(method,'spectral_subtraction')
            defaults.alpha=1; defaults.beta=0.01;
        else
            defaults.gainFloor=0.05;
        end
        options=completeOptions(options,defaults);
        L=options.windowLength; hop=options.hop; K=options.nfft;
        validateattributes(L,{'numeric'},{'scalar','integer','>=',2,'<=',n});
        validateattributes(hop,{'numeric'},{'scalar','integer','positive','<',L});
        validateattributes(K,{'numeric'},{'scalar','integer','even','>=',L});
        validateattributes(options.noiseFraction,{'numeric'},{'scalar','real','>',0,'<=',1});
        w=hann(L,'periodic');

        % Estimate noise from original full frames, excluding artificial padding.
        originalStarts=1:hop:n-L+1;
        originalIndices=(0:L-1)'+originalStarts;
        energy=mean(y(originalIndices).^2,1);
        [~,rank]=sort(energy,'ascend');
        chosen=rank(1:max(1,ceil(options.noiseFraction*numel(rank))));
        noiseFFT=fft(y(originalIndices(:,chosen)).*w,K,1);
        noisePower=mean(abs(noiseFFT(1:K/2+1,:)).^2,2);

        % Pad both ends; normalize by actual overlap weights (hop need not be COLA).
        frameCount=ceil((n+L)/hop)+1;
        paddedLength=(frameCount-1)*hop+L;
        padded=[zeros(L,1);y;zeros(paddedLength-L-n,1)];
        starts=1:hop:paddedLength-L+1;
        indices=(0:L-1)'+starts;
        spectrum=fft(padded(indices).*w,K,1);
        spectrum=spectrum(1:K/2+1,:);
        power=abs(spectrum).^2;
        if strcmp(method,'spectral_subtraction')
            validateattributes(options.alpha,{'numeric'},{'scalar','real','finite','nonnegative'});
            validateattributes(options.beta,{'numeric'},{'scalar','real','>=',0,'<=',1});
            speechPower=max(power-options.alpha*noisePower,options.beta*noisePower);
            enhanced=sqrt(speechPower).*exp(1i*angle(spectrum));
        else
            validateattributes(options.gainFloor,{'numeric'},{'scalar','real','>=',0,'<=',1});
            speechPower=max(power-noisePower,0);
            denominator=speechPower+noisePower;
            denominator(denominator==0)=1;
            gain=max(options.gainFloor,speechPower./denominator);
            enhanced=gain.*spectrum;
        end
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
        info.options=options;
        info.noiseEstimator='fixed-low-energy-noisy-frames';
        info.noiseFrameStarts=originalStarts(chosen);
        info.noisePower=noisePower;
end
assert(isequal(size(output),size(y)) && all(isfinite(output)), ...
    'EECS351:DenoiserOutput','Denoiser returned invalid samples or length.');
end

function options=completeOptions(options,defaults)
checkOptions(options,fieldnames(defaults));
names=fieldnames(defaults);
for k=1:numel(names)
    if ~isfield(options,names{k}), options.(names{k})=defaults.(names{k}); end
end
end

function checkOptions(options,allowed)
unknown=setdiff(fieldnames(options),allowed);
assert(isempty(unknown),'EECS351:Options','Unknown options: %s',strjoin(unknown,', '));
end
