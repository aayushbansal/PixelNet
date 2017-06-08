% demo code to use the edge detection models --
% no thining and NMS are applied to the outputs
clc; clear all;

%
conv_cache = ['./cachedir/demo_results/edges/'];
if(~isdir(conv_cache))
        mkdir(conv_cache);
end

% initialize caffe
NET_FILE_PATH = ['./experiments/demo/demo_edges/'];
net_file     = [NET_FILE_PATH, 'vgg16_edges.caffemodel'];
deploy_file  = [NET_FILE_PATH, 'deploy_edges.prototxt']; 

% set the gpu --
% if not using GPU, set it to CPU mode.
gpu_id = 0;
caffe.reset_all;
caffe.set_device(gpu_id);
caffe.set_mode_gpu;
net = caffe.Net(deploy_file, net_file, 'test');

% this is random size I took, as the example images
% are of size 96x96
cnn_input_size = 96;
crop_height = 96; crop_width = 96;
image_mean = cat(3,  103.9390*ones(cnn_input_size),...
		     116.7700*ones(cnn_input_size),...
		     123.6800*ones(cnn_input_size));

% read the image set 
img_data = {'img_000001.png','img_000002.png'};

% for each image in the img_set
for i = 1:length(img_data)

	display(['Image : ', img_data{i}]);
	ith_Img = im2uint8(imread(['./experiments/demo/demo_edges/', img_data{i}]));

	%
        save_file_name = [conv_cache, img_data{i}];
        if(exist([save_file_name], 'file'))
                continue;
        end
	 
        j_tmp = single(ith_Img(:,:,[3 2 1]));
        j_tmp = j_tmp - image_mean;
        ims(:,:,:,1) = permute(j_tmp, [2 1 3]);	


        %
        net.blobs('data').reshape([crop_height+200, crop_width+200, 3, 1]);
	net.blobs('pixels').reshape([3,crop_height*crop_width]);
        h = crop_height;
        w = crop_width;
        hw = h * w;

        xs = reshape(repmat(0:w-1,h,1), 1, hw) + 100;
        ys = reshape(repmat(0:h-1,w,1)', 1, hw)+ 100;


	% set the image data --
        input_data = zeros(crop_height+200,crop_width+200,3,1);
        input_data(101:crop_width+100, 101:crop_width+100, :, 1) = ims;
        net.blobs('data').set_data(input_data);
	
	% set the pixels --
        input_index = zeros(3, crop_height*crop_width);
        input_index(1,:) = 0;
        input_index(2,:) = xs;
        input_index(3,:) = ys;
        net.blobs('pixels').set_data(input_index);

	% feed forward the values --
        net.forward_prefilled();
        out = net.blobs('cls_prob').get_data();

        % reshape the data --
        f2 = out';
        f2 = reshape(f2, [crop_height, crop_width,1]);
        f2 = permute(f2, [2,1,3]);

	% this step is without any thinning or NMS -- 
	% just to generate the output --
	a = uint8(255 - uint8(255*f2));
        imwrite(a, [save_file_name, '.png']);

end

% reset caffe
caffe.reset_all;
