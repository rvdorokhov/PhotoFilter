function [distort_I] = imdist_generator(im, dist_type, dist_level)
% given the image, distortion type id and distortion level, generate
% distorted image

im = mapmm(im);
switch dist_type
    case 1
        levels = [0.1, 0.5, 1, 2, 5];
        distort_I = imblurgauss(im, levels(dist_level));
    case 2
        levels = [1, 2, 4, 6, 8];
        distort_I = imblurlens(im, levels(dist_level));
    case 3
        levels = [1, 2, 4, 6, 10];
        distort_I = imblurmotion(im, levels(dist_level));
    case 16
        levels = [0.1, 0.2, 0.4, 0.7, 1.1];
        distort_I = imbrighten(im, levels(dist_level));
    case 17
        levels = [0.05, 0.1, 0.2, 0.4, 0.8];
        distort_I = imdarken(im, levels(dist_level));
    otherwise
            warning('Skipping unsupported distortion type: %d', dist_type);
            distort_I = [];
            return;
end
distort_I = mapmm(distort_I);

end