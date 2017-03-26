% this is an example code for training a PixelNet model using caffe
% here we consider a 224x224 image 
% uniform sampling of pixels in an image, used for segmentation
function trainSeg(gpuid, options)

	method = 'seg';
	cachepath = [options.cachepath, method, '/'];
	if(~isdir(cachepath))
		mkdir(cachepath);
	end

	% --
	caffe.reset_all;
	caffe.set_device(gpuid);
	caffe.set_mode_gpu;
        
	% train the network --
	trainNet(options, cachepath);
	caffe.reset_all;
end

function trainNet(options,  cachepath)

	% check if it has already been trained
	trainfolder = [cachepath, 'TRAIN/'];
	if(~isdir(trainfolder))
		mkdir(trainfolder);
	end


	% check if the dataset is ready
	% this would be available in the download_data.sh script
	datasetfile = [options.datapath,...
			'/train/seg-voc2012-aug_trainval.mat'];
	if ~(exist(datasetfile, 'file'))
    		error('Dataset is not prepared!');
	end
	load(datasetfile, 'imagelist', 'seglist');
	
	imagelist = strcat(options.datapath, '/',imagelist);
	seglist = strcat(options.datapath, '/',seglist);

	% load the network --
	solverpath = [options.datapath, '/train/solver.prototxt'];
	initmodelpath = [options.datapath,'/train/VGG16_fconv.caffemodel'];
	trainModel(options,trainfolder,imagelist,seglist,solverpath,initmodelpath);
end

function trainModel(options,trainfolder,imagelist,seglist,solverpath,initmodelpath)


	% Though this is not required -- 
	imagelist = repmat(imagelist, 10,1);
	seglist = repmat(seglist,10,1);
	rand_ids = randperm(length(imagelist));
	imagelist = imagelist(rand_ids);
	seglist = seglist(rand_ids);

	%
	li = length(imagelist);
	caffe.reset_all;

	% initialize network
	solver = caffe.get_solver(solverpath);
	% load the model
	solver.net.copy_from(initmodelpath);

	% --
	maxsize = options.cnn_input_size + 200;
	input_data = zeros(maxsize, maxsize, 3, options.segimbatch, 'single');
	input_index = zeros(3,options.segbatchsize, 'single');
	input_label = zeros(1,options.segbatchsize, 'single');
	solver.net.blobs('data').reshape([maxsize, maxsize, 3, options.segimbatch]);
	solver.net.blobs('pixels').reshape([3,options.segbatchsize]);
	solver.net.blobs('labels').reshape([1,options.segbatchsize]);

	% set up memory
	solver.net.forward_prefilled();
	% then start training
	oldrng = rng;
	rng(options.seed, 'twister');

	for epoch = 1:options.segepoch
   	 index = randperm(li);
   	 for i = 1:options.segimbatch:li
        	j = i+options.segimbatch-1;
        	if j > li
            	continue;
        	end

	        st = 1;
       		ed = options.segsamplesize;
        	im = 0;
        	for k=i:j
            		ik = index(k);

            		if options.trainFlip
		                flip = rand(1) > 0.5;
            		end


	    		im_data = seg_image_provider(options, imagelist{ik}, flip);
	    		[sampled, label] = seg_label_provider(...
				options, seglist{ik}, options.segsamplesize, flip);

            		% notice the zero-index vs. the one-index
            		sampled(1,:) = im;
            		sampled(2,:) = sampled(2,:) + 100;
            		sampled(3,:) = sampled(3,:) + 100;

            		im = im + 1;
            		input_data(101:options.cnn_input_size+100,....
		      	 101:options.cnn_input_size+100, :, im) = im_data;
            		input_index(:, st:ed) = sampled;
            		input_label(st:ed) = label;
           		st = st + options.segsamplesize;
            		ed = ed + options.segsamplesize;
        	end

	        solver.net.blobs('data').set_data(input_data);
        	solver.net.blobs('pixels').set_data(input_index);
        	solver.net.blobs('labels').set_data(input_label);
        	solver.step(1);
        	% clean up everything
        	input_data(:) = 0;
    	end
    	% add another condition
    	if mod(epoch, options.saveEpoch) == 0 && epoch < options.segepoch
        	epochfile = [trainfolder, sprintf('(%02d).caffemodel',epoch)];
        	solver.net.save(epochfile);
    	end

	% -- 
      end

	targetfile = [trainfolder,'final_model.caffemodel'];
	solver.net.save(targetfile);
	caffe.reset_all;
	rng(oldrng);

end
