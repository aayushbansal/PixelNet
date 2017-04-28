% demo code to use the surface normal mode --
clc; clear all;

%
conv_cache = ['./cachedir/demo_results/'];
if(~isdir(conv_cache))
        mkdir(conv_cache);
end

% initialize caffe
NET_FILE_PATH = ['./experiments/demo/demo/'];
net_file     = [NET_FILE_PATH, 'vgg16_surface_normals.caffemodel'];
deploy_file  = [NET_FILE_PATH, 'deploy.prototxt']; 

% set the gpu --
% if not using GPU, set it to CPU mode.
gpu_id = 0;
caffe.reset_all;
caffe.set_device(gpu_id);
caffe.set_mode_gpu;
net = caffe.Net(deploy_file, net_file, 'test');

cnn_input_size = 224;
crop_height = 224; crop_width = 224;
image_mean = cat(3,  103.9390*ones(cnn_input_size),...
		     116.7700*ones(cnn_input_size),...
		     123.6800*ones(cnn_input_size));

% read the image set for NYU
img_data = {'img_000001.jpg','img_000002.jpg'};

% for each image in the img_set
for i = 1:length(img_data)

	display(['Image : ', img_data{i}]);
	ith_Img = im2uint8(imread(['./experiments/demo/', img_data{i}]));

	%
        save_file_name = [conv_cache, strrep(img_data{i}, '.jpg', '')];
        if(exist([save_file_name, '.mat'], 'file'))
                continue;
        end
	 
        j_ims = single(ith_Img(:,:,[3 2 1]));
        j_tmp = imresize(j_ims, [cnn_input_size, cnn_input_size], ...
                           'bilinear', 'antialiasing', false);
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
        input_index = zeros(3, 224*224);
        input_index(1,:) = 0;
        input_index(2,:) = xs;
        input_index(3,:) = ys;
        net.blobs('pixels').set_data(input_index);

	% feed forward the values --
        net.forward_prefilled();
        out = net.blobs('fc8_new').get_data();

        % reshape the data --
        f2 = out';
        f2 = reshape(f2, [224, 224,3]);
        f2 = permute(f2, [2,1,3]);

        % normalize
        nx = f2(:,:,1); ny = f2(:,:,2); nz = f2(:,:,3);
        N = (nx.^2 + ny.^2 + nz.^2).^0.5 + eps;
        nx = nx./N; ny = ny./N; nz = nz./N;

        predns = cat(3, nx, ny, nz);
        predns = imresize(predns,...
                 [size(ith_Img,1), size(ith_Img,2)]);
        pred_N = (predns(:,:,1).^2 + predns(:,:,2).^2 +...
                                 predns(:,:,3).^2).^0.5 + eps;
        predns(:,:,1) = predns(:,:,1)./pred_N;
        predns(:,:,2) = predns(:,:,2)./pred_N;
        predns(:,:,3) = predns(:,:,3)./pred_N;

        predns_vis = uint8(255*(max(min(predns,1),-1)+1)/2);

        % dump the nx/ny/nz
        predns = single(predns);
        imwrite(predns_vis, [save_file_name, '.png']);
        save([save_file_name, '.mat'], 'predns')

end

% reset caffe
caffe.reset_all;
