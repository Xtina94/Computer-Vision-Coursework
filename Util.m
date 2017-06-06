classdef Util
    methods (Static)
        function showkeys(image, locs)

            disp('Drawing SIFT keypoints ...');

            % Draw image with keypoints
            figure('Position', [50 50 size(image,2) size(image,1)]);
            colormap('gray');
            imagesc(image);
            hold on;
            imsize = size(image);
            for i = 1: size(locs,1)
                % Draw an arrow, each line transformed according to keypoint parameters.
                TransformLine(imsize, locs(i,:), 0.0, 0.0, 1.0, 0.0);
                TransformLine(imsize, locs(i,:), 0.85, 0.1, 1.0, 0.0);
                TransformLine(imsize, locs(i,:), 0.85, -0.1, 1.0, 0.0);
            end
            hold off;
        end
        
        %projects the images on a cylindrical surface
        % image : planar image
        % imageC : image projected on the cylindrical surface
        % angle: half FOV of the camera 
        function [imageC] = projectIC(image,angle)
            ig = rgb2gray(image);
            [h w] = size(ig);
            imageC = uint8(zeros(h,w));

            alpha = angle/180*pi;
            d = (w/2)/tan(alpha);
            r = d/cos(alpha);

            for x = -w/2+1:w/2
                for y = -h/2+1:h/2

                   x1 = d * tan(x/r);
                   y1 = y * (d/r) /cos(x/r);

                   if x1 < w/2 && x1 > -w/2+1 && y1 < h/2 && y1 > -h/2+1 
                        imageC(y+(h/2), x+(w/2) ) = ig(round(y1+(h/2)),round(x1+(w/2)));
                    end
                end
            end
        end
        
        % This function reads two images, finds their SIFT features, and
        %   displays lines connecting the matched keypoints.  A match is accepted
        %   only if its distance is less than distRatio times the distance to the
        %   second closest match.
        % It returns the number of matches displayed.
        %
        % Example: match('scene.pgm','book.pgm');
        function [match, num] = match2(im1, des1, loc1, im2, des2, loc2, threshold)
            % Find SIFT keypoints for each image
            %[im1, des1, loc1] = sift(image1);
            %[im2, des2, loc2] = sift(image2);

            % For efficiency in Matlab, it is cheaper to compute dot products between
            %  unit vectors rather than Euclidean distances.  Note that the ratio of 
            %  angles (acos of dot products of unit vectors) is a close approximation
            %  to the ratio of Euclidean distances for small angles.
            %
            % distRatio: Only keep matches in which the ratio of vector angles from the
            %   nearest to second nearest neighbor is less than distRatio.
            distRatio = threshold;   

            % For each descriptor in the first image, select its match to second image.
            des2t = des2';                          % Precompute matrix transpose
            for i = 1 : size(des1,1)
               dotprods = des1(i,:) * des2t;        % Computes vector of dot products
               [vals,indx] = sort(acos(dotprods));  % Take inverse cosine and sort results

               % Check if nearest neighbor has distance less than distRatio times 2nd.
               if (vals(1) < distRatio * vals(2))
                  match(i) = indx(1);
               else
                  match(i) = 0;
               end
            end

            % Create a new image showing the two images side by side.
            im3 = appendimages(im1,im2);

            % Show a figure with lines joining the accepted matches.
            figure('Position', [100 100 size(im3,2) size(im3,1)]);
            colormap('gray');
            imagesc(im3);
            hold on;
            cols1 = size(im1,2);
            for i = 1: size(des1,1)
              if (match(i) > 0)
                line([loc1(i,2) loc2(match(i),2)+cols1], ...
                     [loc1(i,1) loc2(match(i),1)], 'Color', 'c');
              end
            end
            hold off;
            num = sum(match > 0);
            fprintf('Found %d matches.\n', num);
        end
        function [best_M, best_T] = compute_M(z, num_iteration, num_match, matches, loc)
            for j = 1:num_iteration
                B = []; % Object matrix
                U = []; % Scene coordinates vector
                for i=1:3 %pick random 3 features
                    pick(j,i) = randi(num_match(z));

                    A = [loc{z,1}(matches{z,1}(pick(j,i),1),2), loc{z,1}(matches{z,1}(pick(j,i),1),1), 0, 0; 0, 0, loc{z,1}(matches{z,1}(pick(j,i),1),2), loc{z,1}(matches{z,1}(pick(j,i),1),1)];
                    A = [A eye(2)];
                    B = [B; A];
                    U = [U; loc{z+1,1}(matches{z,1}(pick(j,i),2), 2); loc{z+1,1}(matches{z,1}(pick(j,i),2), 1)];
                end
                M = pinv(B)*U;

                T = M(5:6);
                M = [M(1,1), M(2,1); M(3,1), M(4,1)];

                %Count of the goodness of the affine matrix
                count = 0;
                for i=1:num_match(z)
                    uv = M*[loc{z,1}(matches{z,1}(i,1),2); loc{z,1}(matches{z,1}(i,1),1)] + T;
                    uv_real = [loc{z+1,1}(matches{z,1}(i,2),2); loc{z+1,1}(matches{z,1}(i,2),1)];
                    delta_u = abs(uv(1) - uv_real(1));
                    delta_v = abs(uv(2) - uv_real(2));
                    if(delta_u + delta_v < 3)
                            count = count +1;
                    end
                end
                pick(j,4) = count;
            end

            [Y,I] = max(pick(:,4));

            %Compute best M
            B = []; % Object matrix
            U = []; % Scene coordinates vector
            for i=1:3 %pick random 3 features
                A = [loc{z,1}(matches{z,1}(pick(I,i),1),2), loc{z,1}(matches{z,1}(pick(I,i),1),1), 0, 0; 0, 0, loc{z,1}(matches{z,1}(pick(I,i),1),2), loc{z,1}(matches{z,1}(pick(I,i),1),1)];
                A = [A eye(2)];
                B = [B; A];
                U = [U; loc{z+1,1}(matches{z,1}(pick(I,i),2), 2); loc{z+1,1}(matches{z,1}(pick(I,i),2), 1)];
            end
            best_M = pinv(B)*U;

            best_T = best_M(5:6);
            best_M = [best_M(1,1), best_M(2,1); best_M(3,1), best_M(4,1)];
        end
    end
end




% ------ Subroutine: TransformLine -------
% Draw the given line in the image, but first translate, rotate, and
% scale according to the keypoint parameters.
%
% Parameters:
%   Arrays:
%    imsize = [rows columns] of image
%    keypoint = [subpixel_row subpixel_column scale orientation]
%
%   Scalars:
%    x1, y1; begining of vector
%    x2, y2; ending of vector
function TransformLine(imsize, keypoint, x1, y1, x2, y2)
    % The scaling of the unit length arrow is set to approximately the radius
    %   of the region used to compute the keypoint descriptor.
    len = 6 * keypoint(3);

    % Rotate the keypoints by 'ori' = keypoint(4)
    s = sin(keypoint(4));
    c = cos(keypoint(4));

    % Apply transform
    r1 = keypoint(1) - len * (c * y1 + s * x1);
    c1 = keypoint(2) + len * (- s * y1 + c * x1);
    r2 = keypoint(1) - len * (c * y2 + s * x2);
    c2 = keypoint(2) + len * (- s * y2 + c * x2);

    line([c1 c2], [r1 r2], 'Color', 'c');
end