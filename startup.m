% add the list of the paths for the startup
addpath(genpath('tools/caffe'));
addpath(genpath('./experiments/train/'));
addpath('./experiments/demo/');

% following are the set of options that need to be 
% set to use the train code --
options.cachepath = ['./cachedir/'];
options.datapath = ['./experiments/train/data/'];
options.cnn_input_size = 224;
options.segimbatch = 5;
options.segsamplesize = 2000;
options.segbatchsize = (options.segimbatch)*(options.segsamplesize);
options.trainFlip = 1;
options.seed = 1989;
options.segepoch = 80;
options.saveEpoch = 1;
options.meanvalue = [102.9801, 115.9465, 122.7717];
