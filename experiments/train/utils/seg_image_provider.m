function [im_data] = seg_image_provider(options, impath, flip)

% --
im = color(imread(impath));
if(flip)
	im = flipdim(im,2);
end

%
color_code = (1.3-0.8)*rand(3,1) + 0.8;
for c = 1:3
	 im(:,:,c) = color_code(c)*im(:,:,c); 
end

%
cnn_input_size = options.cnn_input_size;
im = imresize(im, [cnn_input_size, cnn_input_size], ...
                           'bilinear', 'antialiasing', false);

% setting the im-data
im_data = single(im(:, :, [3, 2, 1]));  
im_data = permute(im_data, [2, 1, 3]);  
im_data = single(im_data);
% subtract mean_data (already in W x H x C, BGR)
im_data(:,:,1) = im_data(:,:,1) - options.meanvalue(1);  
im_data(:,:,2) = im_data(:,:,2) - options.meanvalue(2);
im_data(:,:,3) = im_data(:,:,3) - options.meanvalue(3);

end
