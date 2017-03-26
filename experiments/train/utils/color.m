function im = color(input)
% Convert input image to color.
%   im = color(input)

if size(input, 3) == 1
  im(:,:,1) = input;
  im(:,:,2) = input;
  im(:,:,3) = input;
elseif size(input, 3) == 4 % CMYK
  h = size(input,1);
  w = size(input,2);
  im = reshape(input,[h*w,4]);
  im = rgb2cmyk(im);
  im = reshape(im,[h,w,3]);
else
  im = input;
end

end