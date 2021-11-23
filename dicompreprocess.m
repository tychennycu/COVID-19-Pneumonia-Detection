function output = dicompreprocess(filename)

% Code for resnet50 and densenet201
%dcm = im2uint8(dcm);
dcm = dicomread(filename);
dcm = histeq(dcm);
%dcm = adapthisteq(dcm);
%dcm = imadjust(dcm,[],[],1);
dcm_resize = imresize(dcm,[224 224]);
output = cat(3,dcm_resize,dcm_resize,dcm_resize);

end
