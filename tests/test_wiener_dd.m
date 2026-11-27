function tests=test_wiener_dd
tests=functiontests(localfunctions);
end

function setupOnce(~)
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))),'code'));
end

function testReducesToBasicWiener(testCase)
% ddAlpha=0 and no SNR floor is the maximum-likelihood Wiener gain in denoise_speech.
y=makeSignal(3);
basic=denoise_speech(y,16000,'wiener',struct('gainFloor',0,'noiseFraction',0.2));
dd=denoise_wiener_dd(y,16000,struct('ddAlpha',0,'xiMinDb',-Inf,'noiseFraction',0.2));
verifyEqual(testCase,dd,basic,'AbsTol',1e-12);
end

function testUnityGainReconstructs(testCase)
% A very high SNR floor makes G ~ 1, so analysis/synthesis must return the input.
y=makeSignal(4);
out=denoise_wiener_dd(y,16000,struct('xiMinDb',80));
verifyEqual(testCase,out,y,'AbsTol',1e-7*max(abs(y)));
end

function testSilenceAndShape(testCase)
out=denoise_wiener_dd(zeros(4000,1),16000);
verifyEqual(testCase,size(out),[4000 1]);
verifyEqual(testCase,out,zeros(4000,1));
end

function testSuppressesStationaryNoise(testCase)
% Speech-like bursts in white noise: SNR improves and the noise-only lead-in is attenuated.
fs=16000; n=2*fs; t=(0:n-1)'/fs;
rng(351); noise=0.05*randn(n,1);
s=zeros(n,1); on=t>0.5 & t<1.5;
s(on)=0.3*sin(2*pi*220*t(on)).*(1+0.5*sin(2*pi*3*t(on)));
y=s+noise;
out=denoise_wiener_dd(y,fs);
snrIn=10*log10(sum(s.^2)/sum((y-s).^2));
snrOut=10*log10(sum(s.^2)/sum((out-s).^2));
verifyGreaterThan(testCase,snrOut,snrIn+3);
lead=1:round(0.4*fs);
verifyLessThan(testCase,rms(out(lead)),0.3*rms(y(lead)));
end

function testSmoothingReducesGainFluctuation(testCase)
% The decision-directed recursion should make noise-only gains steadier than ML gains.
fs=16000; rng(7); y=0.05*randn(fs,1);
[~,ml]=denoise_wiener_dd(y,fs,struct('ddAlpha',0,'xiMinDb',-Inf));
[~,dd]=denoise_wiener_dd(y,fs,struct('ddAlpha',0.98,'xiMinDb',-Inf));
verifyLessThan(testCase,dd.meanGain,ml.meanGain);
end

function testRejectsUnknownOption(testCase)
verifyError(testCase,@() denoise_wiener_dd(ones(1000,1),16000,struct('gainFloor',0.1)),'EECS351:Options');
end

function y=makeSignal(seed)
rng(seed); n=6000; t=(0:n-1)'/16000;
y=0.2*sin(2*pi*300*t).*(t>0.1)+0.02*randn(n,1);
end
