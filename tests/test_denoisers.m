function tests=test_denoisers
tests=functiontests(localfunctions);
end

function testSpectralIdentityIncludesBoundaries(testCase)
stream=RandStream('mt19937ar','Seed',351);
y=0.1*randn(stream,4301,1); % Deliberately not a whole number of frames.
sub=denoise_speech(y,16000,'spectral_subtraction',struct('alpha',0,'beta',0));
wiener=denoise_speech(y,16000,'wiener',struct('gainFloor',1));
verifyEqual(testCase,sub,y,'AbsTol',1e-12);
verifyEqual(testCase,wiener,y,'AbsTol',1e-12);
end

function testFirDelayAndFrequencyResponse(testCase)
y=zeros(2001,1); y(1001)=1;
[output,info]=denoise_speech(y,16000,'fir');
verifyEqual(testCase,info.causalFilterDelaySamples,64);
verifyEqual(testCase,info.validSamples,[65 1937]);
verifyEqual(testCase,output(937:1065),info.coefficients(:),'AbsTol',1e-14);
[response,f]=freqz(info.coefficients,1,8192,16000);
verifyLessThan(testCase,abs(response(1)),0.1);
verifyGreaterThan(testCase,abs(response(find(f>=1000,1))),0.95);
end

function testFixedEstimatorRejectsArtificialSilence(testCase)
y=0.1*ones(4001,1);
[~,info]=denoise_speech(y,16000,'wiener');
verifyGreaterThanOrEqual(testCase,min(info.noiseFrameStarts),1);
verifyLessThanOrEqual(testCase,max(info.noiseFrameStarts)+399,numel(y));
verifyGreaterThan(testCase,sum(info.noisePower),0);
end

function testSilenceAndStationaryNoise(testCase)
stream=RandStream('mt19937ar','Seed',351);
noise=0.1*randn(stream,16000,1);
for method={'identity','fir','spectral_subtraction','wiener'}
    silence=denoise_speech(zeros(16001,1),16000,method{1});
    verifyEqual(testCase,silence,zeros(16001,1));
    output=denoise_speech(noise,16000,method{1});
    verifySize(testCase,output,size(noise));
    verifyTrue(testCase,all(isfinite(output)));
    if ismember(method{1},{'spectral_subtraction','wiener'})
        verifyLessThan(testCase,mean(output.^2),mean(noise.^2));
    end
end
end

function testUnknownParameterRejected(testCase)
verifyError(testCase,@() denoise_speech(ones(1000,1),16000,'wiener',struct('alpha',2)), ...
    'EECS351:Options');
end
