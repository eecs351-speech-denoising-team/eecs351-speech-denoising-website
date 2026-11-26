function tests=test_noise_updates
tests=functiontests(localfunctions);
end

function testUpdateRecurrenceAndHold(testCase)
[y,fs]=changingSpectrum;
options=struct('noiseMode','pause_updated','storeNoiseHistory',true);
[~,info]=denoise_speech(y,fs,'wiener',options);
tracking=info.noiseTracking; history=info.noisePowerHistory;
verifyGreaterThan(testCase,tracking.updateCount,0);
verifyFalse(testCase,any(tracking.updateMask & ~tracking.fullFrames));
previous=[info.noisePower history(:,1:end-1)];
verifyEqual(testCase,history(:,~tracking.updateMask),previous(:,~tracking.updateMask),'AbsTol',1e-14);
for k=find(tracking.updateMask)
    verifyTrue(testCase,all(tracking.pauseCandidates(k-2:k)));
end
k=find(tracking.updateMask,1);
ix=tracking.frameStarts(k)+(0:399)';
observed=fft(y(ix).*hann(400,'periodic'),512);
expected=0.9*previous(:,k)+0.1*abs(observed(1:257)).^2;
verifyEqual(testCase,history(:,k),expected,'AbsTol',1e-12);
% The second noise-only interval changes spectrum but has the same energy.
% Its final tracked power must follow 2 kHz rather than the earlier 500 Hz.
f=(0:256)'*fs/512;
latePower=sum(tracking.finalNoisePower(abs(f-2000)<100));
earlyPower=sum(tracking.finalNoisePower(abs(f-500)<100));
verifyGreaterThan(testCase,latePower,5*earlyPower);
end

function testLambdaOneMatchesFixed(testCase)
[y,fs]=changingSpectrum;
for method={'spectral_subtraction','wiener'}
    fixed=denoise_speech(y,fs,method{1});
    [updated,info]=denoise_speech(y,fs,method{1}, ...
        struct('noiseMode','pause_updated','noiseLambda',1));
    verifyGreaterThan(testCase,info.noiseTracking.updateCount,0);
    verifyEqual(testCase,updated,fixed,'AbsTol',1e-12);
end
end

function testNoContrastFallsBackToFixed(testCase)
y=0.1*ones(4001,1);
for method={'spectral_subtraction','wiener'}
    fixed=denoise_speech(y,16000,method{1});
    [updated,info]=denoise_speech(y,16000,method{1},struct('noiseMode','pause_updated'));
    verifyEqual(testCase,updated,fixed,'AbsTol',1e-12);
    verifyEqual(testCase,info.noiseTracking.updateCount,0);
    verifyEqual(testCase,info.noiseTracking.fallbackReason,'insufficient-energy-contrast');
end
end

function testNoConfirmedPauseHoldsInitialEstimate(testCase)
[y,fs]=changingSpectrum;
[~,info]=denoise_speech(y,fs,'wiener',struct('noiseMode','pause_updated', ...
    'minPauseFrames',1000,'storeNoiseHistory',true));
verifyEqual(testCase,info.noiseTracking.updateCount,0);
verifyEqual(testCase,info.noiseTracking.fallbackReason,'no-confirmed-pause');
verifyEqual(testCase,info.noisePowerHistory, ...
    repmat(info.noisePower,1,size(info.noisePowerHistory,2)),'AbsTol',1e-14);
end

function testUpdatedUnityAndSilence(testCase)
[y,fs]=changingSpectrum;
y=[y;0];
sub=denoise_speech(y,fs,'spectral_subtraction', ...
    struct('noiseMode','pause_updated','alpha',0,'beta',0));
wiener=denoise_speech(y,fs,'wiener',struct('noiseMode','pause_updated','gainFloor',1));
verifyEqual(testCase,sub,y,'AbsTol',1e-12);
verifyEqual(testCase,wiener,y,'AbsTol',1e-12);
for method={'spectral_subtraction','wiener'}
    output=denoise_speech(zeros(4001,1),fs,method{1},struct('noiseMode','pause_updated'));
    verifyEqual(testCase,output,zeros(4001,1));
end
end

function [y,fs]=changingSpectrum
fs=16000; t=(0:7999)'/fs;
low=0.02*sin(2*pi*500*t); high=0.02*sin(2*pi*2000*t);
speechLike=0.25*sin(2*pi*180*t);
y=[low;speechLike+low;high;speechLike+high];
end
