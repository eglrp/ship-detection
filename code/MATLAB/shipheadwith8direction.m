%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate positive samples for ship head classification
% Here we divide ship heads with different directions into 8 bins 
% Please specify the path according to your configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the path of the training images
ImagePath = fullfile('D:','trainimage');
% the path of the samples for ship head classification
patchPath = fullfile('D:','shiphead');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mkdir(patchPath);

folderTxt=fopen( fullfile(patchPath,'shiphead.txt','a') );

len = 2400;
radius = 20;
offset = floor(radius*0.6);
totalSample = 0;
for i = 1:len
    imagePath  = fullfile(ImagePath, [num2str(i),'.jpg']);
    recordPath = fullfile(ImagePath, [num2str(i),'.txt']);

    if( exist(recordPath, 'file') && exist(imagePath, 'file'))
        [data1,data2,data3,data4,data5,data6,data7,data101,data102,data103,data104]= textread(recordPath,'%n%n%n%n%n%n%n%n%n%n%n');
        shipNumber = length(data1);
        I = imread(imagePath);
        II = padarray(I,[2*radius 2*radius],'symmetric','both');
        [imageHeight, imageWidth, ~] = size(II);
        data1 = data1 + 2*radius;
        data2 = data2 + 2*radius; 
        data3 = data3 + 2*radius;
        data4 = data4 + 2*radius;
        
        for j = 1:shipNumber
            if ( (data5(j) == 2) && (data1(j) >radius*3) && (data1(j) <imageWidth-radius*3)  && (data2(j) >radius*3) && (data2(j) <imageHeight-radius*3) )
                imOriginPatch = II( data2(j)-radius*3 : data2(j)+radius*3, data1(j)-radius*3 : data1(j)+radius*3 ,:);
                xdis = data3(j) - data1(j);
                ydis = data4(j) - data2(j);
                theta = atan2(xdis, ydis) /3.1416*180;
                for n = 0:45:315
                    for k = 1:1
                        th = -theta + 90 - n + 45 * (rand() - 0.5);
                        if( rand()>0.7 )
                            scale = (rand()-0.5)*0.4+1;
                            impatch = imrotate(imOriginPatch, th ,'bilinear','loose');
                            impatch = imresize(impatch, scale, 'bilinear');
                            center = floor( size(impatch, 1)/2 )+1;
                        else
                            impatch = imrotate(imOriginPatch, th ,'bilinear','loose');
                            center = floor( size(impatch, 1)/2 )+1;
                        end
                        for m = 1:2
                            switch n
                                case 0
                                    newm = center + randi( [0, offset*2] );
                                    newn = center + randi( [-offset, offset] );
                                case 45
                                    newm = center + randi( [0, offset*2] );
                                    newn = center + randi( [0, offset*2] );
                                case 90
                                    newm = center + randi( [-offset, offset] );
                                    newn = center + randi( [0, offset*2] );
                                case 135
                                    newm = center + randi( [-offset*2, 0] );
                                    newn = center + randi( [0, offset*2] );
                                case 180
                                    newm = center + randi( [-offset*2, 0] );
                                    newn = center + randi( [-offset, offset] );
                                case 225
                                    newm = center + randi( [-offset*2, 0] );
                                    newn = center + randi( [-offset*2, 0] );
                                case 270
                                    newm = center + randi( [-offset, offset] );
                                    newn = center + randi( [-offset*2, 0] );
                                case 315
                                    newm = center + randi( [0, offset*2] );
                                    newn = center + randi( [-offset*2, 0] ); 
                                otherwise
                                    warning('Unexpected switch.');
                            end
                            try
                                imageShipHead = impatch( newn-radius:newn+radius-1, newm-radius:newm+radius-1 ,:);
                                totalSample  = totalSample + 1;
                                shipHeadName = [num2str(totalSample + int32(n/45 + 1)*1000000),'.jpg'];
                                shipHeadPath = fullfile(patchPath,shipHeadName);
%                                 imshow(imageShipHead); 
                                imwrite(imageShipHead,shipHeadPath); 
                                fprintf(folderTxt,'%s %s\r\n',shipHeadName,num2str(int32(n/45)+1));
                            catch
                                q=1;
                            end
                        end
                    end
                end
            end
        end
        disp(i)
    end
end
fclose('all');