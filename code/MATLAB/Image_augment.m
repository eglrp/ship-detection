%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data augmentation
% Here we do flipping and scaling
% Further augmentation, please see imageAugmen.ipynb
% Please specify the path according to your configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the path of the original images
oriImagePath = fullfile('D:','trainset');
% the path of the augmented images
detImagePath = fullfile('D:','trainsetnew');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mkdir(detImagePath);

len = 200;
for i = 1:len
    imagePath  = fullfile(oriImagePath, [num2str(i),'.jpg']);    
    recordPath = fullfile(oriImagePath, [num2str(i),'.txt']);
    if( exist(recordPath, 'file')&& exist(imagePath, 'file') )
        %data1,   data2,  data3,  data4,  data5,    data6,   data7,   data101,   data102,  data103  data104
        %起点x    起点y    终点x  终点y    类别      中点x    中点y     框左上x    框左上y    框右下x  框右下y
        [data1,data2,data3,data4,data5,data6,data7,data101,data102,data103,data104]= textread(recordPath,'%n%n%n%n%n%n%n%n%n%n%n');
        shipNumber = length(data1);
        I = imread(imagePath);
        %flipping : left-right
        I2 = I(:, end:-1:1, :);
        shipName = [num2str(i + len),'.jpg'];
        imwrite(I2,fullfile(detImagePath,shipName) ); 
        shipName = [num2str(i + len),'.txt'];
        a=fopen( fullfile(detImagePath,shipName),'a');
        for j=1:shipNumber
            fprintf(a,'%s %s %s %s %s %s %s %s %s %s %s\r\n',num2str( 1025-data1(j) ),num2str( data2(j) ), num2str( 1025-data3(j) ), num2str( data4(j) ),...
                num2str( data5(j) ), num2str(1025-data6(j) ), num2str( data7(j) ), ...
                num2str( min(1025-data101(j),1025-data103(j)) ), num2str( min(data102(j),data104(j)) ), num2str(max(1025-data101(j),1025-data103(j))), num2str( max(data102(j),data104(j) )) );
        end      
        %flipping : top-down
        I3 = I(end:-1:1, :, :);
        shipName = [num2str(i + len*2),'.jpg'];
        imwrite(I3,fullfile(detImagePath,shipName) );
        shipName = [num2str(i + len*2),'.txt'];
        b=fopen( fullfile(detImagePath,shipName),'a');
        for j=1:shipNumber
            fprintf(b,'%s %s %s %s %s %s %s %s %s %s %s\r\n',num2str( data1(j) ),num2str( 769-data2(j) ), num2str( data3(j) ), num2str( 769-data4(j) ),...
                num2str( data5(j) ), num2str(data6(j) ), num2str( 769-data7(j) ),...
                num2str( min(data101(j),data103(j)) ), num2str( min(769-data102(j),769-data104(j) ) ), num2str( max(data101(j),data103(j))), num2str(max(769-data102(j),769-data104(j) ) ));
        end
    end
    disp(i)
    fclose('all');
end

%scaling 
len = 600;
for i = 1:len
    imagePath  = fullfile(detImagePath, [num2str(i),'.jpg']);
    recordPath = fullfile(detImagePath, [num2str(i),'.txt']);
    if( exist(recordPath, 'file') && exist(imagePath, 'file'))
        %data1,   data2,  data3,  data4,  data5,    data6,   data7,   data101,   data102,  data103  data104
        %起点x    起点y    终点x  终点y    类别      中点x    中点y     框左上x    框左上y    框右下x  框右下y
        [data1,data2,data3,data4,data5,data6,data7,data101,data102,data103,data104]= textread(recordPath,'%n%n%n%n%n%n%n%n%n%n%n');
        shipNumber = length(data1);
        I = imread(imagePath);
        [imageHeight, imageWidth, ~] = size(I);
        % 0.8 ～ 1.2
        scale = (rand()-0.5)*0.4+1;
        II = imresize(I, scale, 'bilinear');
        data1(:) =  data1(:)*scale;
        data2(:) =  data2(:)*scale;
        data3(:) =  data3(:)*scale;
        data4(:) =  data4(:)*scale;
        data101(:) =  data101(:)*scale;
        data102(:) =  data102(:)*scale;
        data103(:) =  data103(:)*scale;
        data104(:) =  data104(:)*scale;

            radius = 300;
            data1(:) =  data1(:) + radius;
            data2(:) =  data2(:) + radius;
            data3(:) =  data3(:) + radius;
            data4(:) =  data4(:) + radius;
            data101(:) =  data101(:) + radius;
            data102(:) =  data102(:) + radius;
            data103(:) =  data103(:) + radius;
            data104(:) =  data104(:) + radius;
            II = padarray(II,[radius radius],'replicate','both');
            [imageHeightnew, imageWidthnew, ~] = size(II);
            centerx = floor( imageWidthnew/2 );
            centery = floor( imageHeightnew/2 );
            III = II(centery-384:centery+384-1,  centerx-512:centerx+512-1, :);
            offsetx = centerx - 512;
            offsety = centery - 384;
            data1(:) =  data1(:) - offsetx;
            data2(:) =  data2(:) - offsety;
            data3(:) =  data3(:) - offsetx;
            data4(:) =  data4(:) - offsety;
            data101(:) =  data101(:) - offsetx;
            data102(:) =  data102(:) - offsety;
            data103(:) =  data103(:) - offsetx;
            data104(:) =  data104(:) - offsety;

        shipName = [num2str(i + len),'.jpg'];
        imwrite(III,fullfile(detImagePath,shipName) ); 
        shipName = [num2str(i + len),'.txt'];
        c = fopen( fullfile(detImagePath,shipName),'a');
        for j=1:shipNumber
            fprintf(c,'%s %s %s %s %s %s %s %s %s %s %s\r\n',num2str( floor(data1(j)) ),num2str( floor(data2(j)) ), num2str( floor(data3(j)) ), num2str( floor(data4(j)) ),...
                num2str( data5(j) ), num2str( floor( (data101(j)+data103(j))/2) ), num2str(  floor( (data102(j)+data104(j))/2) ), ...
                num2str( floor(data101(j)) ), num2str( floor(data102(j)) ), num2str( floor(data103(j)) ), num2str( floor(data104(j))) );
        end
    end
    disp(i)
    fclose('all');
end