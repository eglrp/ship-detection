%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate negative samples for ship head classification
% Please specify the path according to your configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the path of the training images
ImagePath = fullfile('D:','trainimage');
% the path of the negative samples for ship head classification
patchPath = fullfile('D:','noshiphead');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mkdir(patchPath);

folderTxt=fopen( fullfile(patchPath,'noshiphead.txt','a') );

len = 2400;
imageWidth  = 1024;
imageHeight = 768;
radius = 20;
randiLength = floor(radius/2*2.0);
offset = floor(radius*1.0);
totalNoShipHead = 0;
numberperimage = 50;

for i = 1:len
    imagePath  = fullfile(ImagePath, [num2str(i),'.jpg']);
    recordPath = fullfile(ImagePath, [num2str(i-1200),'.txt']);
    if( exist(recordPath, 'file') && exist(imagePath, 'file'))
        A = load(recordPath);
        [data1,data2,data3,data4,data5,data6,data7,data101,data102,data103,data104]= textread(recordPath,'%n%n%n%n%n%n%n%n%n%n%n');
        I = imread(imagePath);
     	[imageHeight, imageWidth, ~] = size(I);
        [harrPoints, harrmap, ~, max_local, harrthresh] = ExtractHarris(I, 1.2);
        prowPoints = [harrPoints(:,2), harrPoints(:,1)];
        harrisNumber = size(prowPoints,1);
        newx1 = data1;
        newy1 = data1;        
        newx2 = data1;        
        newy2 = data1;
        for k = 1:size(data1,1)
                    if( data5(k) == 2 )
                    	xdis = data3(k) - data1(k);
                        ydis = data4(k) - data2(k);
                        theta = atan2(xdis, ydis) /3.1416*180 + 180;
                        if( (theta>=180+157.5  && theta<=360)   || (theta>=0  && theta<22.5)   )
                            newx1(k) = data1(k)-offset*0.8;
                            newy1(k) = data2(k)-offset*1.2;
                            newx2(k) = data1(k)+offset*0.8;        
                            newy2(k) = data2(k)+offset*0.4;
                        elseif( theta>=22.5  && theta<67.5  )
                            newx1(k) = data1(k)-offset*1.2;
                            newy1(k) = data2(k)-offset*1.2;        
                            newx2(k) = data1(k)+offset*0.4;        
                            newy2(k) = data2(k)+offset*0.4;
                        elseif( theta>=67.5  && theta<112.5  )
                            newx1(k) = data1(k)-offset*1.2;
                            newy1(k) = data2(k)-offset*0.8;        
                            newx2(k) = data1(k)+offset*0.4;        
                            newy2(k) = data2(k)+offset*0.8;
                        elseif( theta>=112.5  && theta<157.5  )
                            newx1(k) = data1(k)-offset*1.2;
                            newy1(k) = data2(k)-offset*0.4;        
                            newx2(k) = data1(k)+offset*0.4;        
                            newy2(k) = data2(k)+offset*1.2;
                        elseif( theta>=180-22.5  && theta<180+22.5  )
                            newx1(k) = data1(k)-offset*0.8;
                            newy1(k) = data2(k)-offset*0.4;        
                            newx2(k) = data1(k)+offset*0.8;        
                            newy2(k) = data2(k)+offset*1.2;
                        elseif( theta>=180+22.5  && theta<180+67.5  )
                            newx1(k) = data1(k)-offset*0.4;
                            newy1(k) = data2(k)-offset*0.4;        
                            newx2(k) = data1(k)+offset*1.2;        
                            newy2(k) = data2(k)+offset*1.2;
                        elseif( theta>=180+67.5  && theta<180+112.5  )
                            newx1(k) = data1(k)-offset*0.4;
                            newy1(k) = data2(k)-offset*0.8;        
                            newx2(k) = data1(k)+offset*1.2;        
                            newy2(k) = data2(k)+offset*0.8;
                        elseif( theta>=180+112.5  && theta<180+157.5  )
                            newx1(k) = data1(k)-offset*0.4;
                            newy1(k) = data2(k)-offset*1.2;        
                            newx2(k) = data1(k)+offset*1.2;        
                            newy2(k) = data2(k)+offset*0.4;
                        else
                            warning('Unexpected switch.');
                        end
                end
        end    
        start = randi( [1, 10] );
        stride = int16(harrisNumber/numberperimage);
        for j = start:stride:harrisNumber
            newj = prowPoints(j,2) + randi( [-randiLength, randiLength] );
            newk = prowPoints(j,1) + randi( [-randiLength, randiLength] );
            savedata = 0;
            if (( newk >radius) && ( newk <imageWidth-radius)  &&  ( newj >radius) && ( newj <imageHeight-radius))
                savedata = 1;
                for k = 1:size(data1,1)
                    if( data5(k) == 2 )
                        if( (savedata == 1) &&   (~( newk+radius<newx1(k) ||  newk-radius>newx2(k) || newj+radius<newy1(k) ||  newj-radius>newy2(k) )))
                            savedata = 0;
                        end
                    end
                end
                if savedata == 1
                    try
                      	image = I(  newj-radius:newj+radius-1, newk-radius:newk+radius-1,:);
                        totalNoShipHead = totalNoShipHead + 1;
                        nonshipheadname = [num2str(totalNoShipHead),'.jpg'];
                      	nonshipheadPath = fullfile(patchPath,nonshipheadname);
                      	imwrite(image,nonshipheadPath);
                        fprintf(foldertxt,'%s %s\r\n',nonshipheadname,num2str(0)); 
                    catch
                        q=1;
                    end
                end
            end
        end
    disp(i)       
    end
end
fclose('all');


numberperimage = 50;
for i = 1:len
    imagePath  = fullfile(ImagePath, [num2str(i),'.jpg']);
    recordPath = fullfile(ImagePath, [num2str(i-1200),'.txt']);
    
    if( exist(recordPath, 'file') && exist(imagePath, 'file') )
        A = load(recordPath);
        [data1,data2,data3,data4,data5,data6,data7,data101,data102,data103,data104]= textread(recordPath,'%n%n%n%n%n%n%n%n%n%n%n');
        I = imread(imagePath);
        [imageHeight, imageWidth, ~] = size(I);
        number = 1;
        newx1 = data1;
        newy1 = data1;        
        newx2 = data1;        
        newy2 = data1;
        for k = 1:size(data1,1)
                    if( data5(k) == 2 )
                    	xdis = data3(k) - data1(k);
                        ydis = data4(k) - data2(k);
                        theta = atan2(xdis, ydis) /3.1416*180 + 180;
                        if( (theta>=180+157.5  && theta<=360)   || (theta>=0  && theta<22.5)   )
                            newx1(k) = data1(k)-offset*0.8;
                            newy1(k) = data2(k)-offset*1.2;        
                            newx2(k) = data1(k)+offset*0.8;        
                            newy2(k) = data2(k)+offset*0.4;
                        elseif( theta>=22.5  && theta<67.5  )
                            newx1(k) = data1(k)-offset*1.2;
                            newy1(k) = data2(k)-offset*1.2;        
                            newx2(k) = data1(k)+offset*0.4;        
                            newy2(k) = data2(k)+offset*0.4;
                        elseif( theta>=67.5  && theta<112.5  )
                            newx1(k) = data1(k)-offset*1.2;
                            newy1(k) = data2(k)-offset*0.8;        
                            newx2(k) = data1(k)+offset*0.4;        
                            newy2(k) = data2(k)+offset*0.8;
                        elseif( theta>=112.5  && theta<157.5  )
                            newx1(k) = data1(k)-offset*1.2;
                            newy1(k) = data2(k)-offset*0.4;        
                            newx2(k) = data1(k)+offset*0.4;        
                            newy2(k) = data2(k)+offset*1.2;
                        elseif( theta>=180-22.5  && theta<180+22.5  )
                            newx1(k) = data1(k)-offset*0.8;
                            newy1(k) = data2(k)-offset*0.4;        
                            newx2(k) = data1(k)+offset*0.8;        
                            newy2(k) = data2(k)+offset*1.2;
                        elseif( theta>=180+22.5  && theta<180+67.5  )
                            newx1(k) = data1(k)-offset*0.4;
                            newy1(k) = data2(k)-offset*0.4;        
                            newx2(k) = data1(k)+offset*1.2;        
                            newy2(k) = data2(k)+offset*1.2;
                        elseif( theta>=180+67.5  && theta<180+112.5  )
                            newx1(k) = data1(k)-offset*0.4;
                            newy1(k) = data2(k)-offset*0.8;        
                            newx2(k) = data1(k)+offset*1.2;        
                            newy2(k) = data2(k)+offset*0.8;
                        elseif( theta>=180+112.5  && theta<180+157.5  )
                            newx1(k) = data1(k)-offset*0.4;
                            newy1(k) = data2(k)-offset*1.2;        
                            newx2(k) = data1(k)+offset*1.2;        
                            newy2(k) = data2(k)+offset*0.4;
                        else
                            warning('Unexpected switch.');
                        end
                end
        end
        while( number < numberperimage)
            number = number + 1;
            newj = randi( [radius+1, imageHeight-radius-1] );
            newk = randi( [radius+1, imageWidth-radius-1] );
            savedata = 1;
            for k = 1:size(data1,1)
                if( data5(k) == 2 )
                	if( (savedata == 1) &&   (~( newk+radius<newx1(k) ||  newk-radius>newx2(k) || newj+radius<newy1(k) ||  newj-radius>newy2(k) )))
                    	savedata = 0;
                	end
                end
            end
            if( savedata == 1)
                try
            	image = I( newj-radius:newj+radius-1, newk-radius:newk+radius-1, :);
            	if( mean( mean(mean(image)))>25 && std2(image)>16 )
                    totalNoShipHead = totalNoShipHead + 1;
                    noshipheadname = [num2str(totalNoShipHead),'.jpg'];
                    noshipheadPath = fullfile(patchPath,noshipheadname);
                	imwrite(image,noshipheadPath);
                  	fprintf(foldertxt,'%s %s\r\n',noshipheadname,num2str(0)); 
                else
                	if( randi([1, 2]) > 1 )
                        totalNoShipHead = totalNoShipHead + 1;
                    	noshipheadname = [num2str(totalNoShipHead),'.jpg'];
                    	noshipheadPath = fullfile(patchPath,noshipheadname);
                    	imwrite(image,noshipheadPath);
                    	fprintf(foldertxt,'%s %s\r\n',noshipheadname,num2str(0)); 
                    end
                end
                catch
                    number = number - 1;
                end
            end
        end
    end
	disp(i)    
end
fclose('all');