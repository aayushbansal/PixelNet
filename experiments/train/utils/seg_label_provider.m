function [sampled, label] = seg_label_provider(options,  segpath, sample_size, flip)

% first get the mask
try,

	cnn_input_size = options.cnn_input_size;
	bmask = imread(segpath);

	if(flip)
		bmask = flipdim(bmask,2);
	end

	bmask = imresize(bmask,...
	 [cnn_input_size, cnn_input_size], 'nearest');

	bmask = reshape(bmask, 1, cnn_input_size*cnn_input_size);
	tmask = find(bmask < 255);

	if((length(tmask)>0) && (length(tmask) < sample_size))
		tmask = repmat(tmask, 1, ceil(sample_size/length(tmask)));
	end

	% then sample from each
	sampled = zeros(3, sample_size, 'single');
	label = zeros(1, sample_size, 'single');
	tmask = randsample(tmask,sample_size);
	[y,x] = ind2sub([cnn_input_size, cnn_input_size],tmask);
	label(:) = bmask(tmask);

	%
	sampled(2,:) = y - 1;
	sampled(3,:) = x - 1;
catch,
	keyboard;
end


end

