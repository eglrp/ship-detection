%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate training samples for ship localization
% Please specify the path according to your configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the path of the training images (after augmentation)
ImagePath = fullfile('D:','trainimage');
% the path to store the position of the samples
samplePath = fullfile('D:','shipbody');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mkdir(samplePath);
foldertxt = fopen( fullfile(samplePath, 'shipbody.txt'), 'a');

len = 2400;
imageWidth  = 1024;
imageHeight = 768;
total = 0;
boxradius = 100;
k_threshold = 50;
%data1,   data2,  data3,  data4,  data5,    data6,   data7,   data101,   data102,  data103  data104
%起点x    起点y    终点x   终点y    类别      中点x    中点y     框左上x    框左上y   框右下x   框右下y
for i = 1:len
    imagePath  = fullfile(ImagePath, [num2str(i),'.jpg']);
    if ( i>1200 && i<2401)
        recordPath = fullfile(ImagePath, [num2str(i-1200),'.txt']);
    else
        recordPath = fullfile(ImagePath, [num2str(i),'.txt']);
    end
    if( exist(recordPath, 'file') && exist(imagePath, 'file'))
        [data1,data2,data3,data4,data5,data6,data7,data101,data102,data103,data104] = textread(recordPath,'%n%n%n%n%n%n%n%n%n%n%n');
        Image = imread(imagePath);
        data201 = data104;
        data201(1:end) = 1000000;
        if (  isempty( find(data5>1, 1) ))
                continue;
        else       
        for j = 1:length(data1)
            k = 0;
            itera = 0;
            if( data5(j) == 2)
                while( k<k_threshold )
                    if(itera>k_threshold*10)
                        break;
                    else
                        itera = itera + 1;
                    end
            	newkx = randi( [ min(data101(j),data103(j)), max(data101(j),data103(j)) ] );
                newky = randi( [ min(data102(j),data104(j)), max(data102(j),data104(j)) ] );
                x1 = max(1,newkx-boxradius+1);
                x2 = min(newkx+boxradius,imageWidth);
                y1 = max(1,newky-boxradius+1);
                y2 = min(newky+boxradius,imageHeight);
                newkx = floor( (x1+x2)*0.5 );
                newky = floor( (y1+y2)*0.5 );  
                
                for i1 = 1:length(data201)
                    if (  data5(i1) == 2  )
                        data201(i1) = ( newkx-data6(i1) )*( newkx-data6(i1) ) + ( newky-data7(i1) )*( newky-data7(i1) );
                    end
                end
                    [C,I] = min(data201);
                    savedata = 0;
                    interminx = max(  x1, min(data101(j),data103(j)) );
                    interminy = max(  y1, min(data102(j),data104(j)) );
                    intermaxx = min(  x2, max(data101(j),data103(j)) );
                    intermaxy = min(  y2, max(data102(j),data104(j)) );                  
                    if( interminx<intermaxx && interminy<intermaxy )
                        interS = (intermaxx-interminx)*(intermaxy-interminy);
                        ratio = interS/( abs((data104(I)-data102(I))*(data103(I)-data101(I))) );
                        if( ratio>0.6 && (  ( x1 < data1(I) && x2 > data1(I) && y1 < data3(I) && y2 > data3(I) )||...
                                            ( x1 < data3(I) && x2 > data3(I) && y1 < data4(I) && y2 > data4(I) )  ) )
                               savedata = 1;
                        end
                    end
                    if (savedata == 1)
                    	try
                        	image = Image( y1:y2, x1:x2,: );
                        	total = total + 1;
                            k = k+1;
%                            shipbodyname = [num2str(total),'.jpg'];
%                            shipbodyPath = fullfile('D:','shipbody',shipbodyname);
%                          	 imwrite(image,shipbodyPath);
% roi ith,  in ith ship image,   patch在图像左上x,  左上y,  右下x,  右下y,  groundtruth 在图像左上x,  左上y,   右下x,   右下y,   类别( 是否船 ) 
                           	fprintf(foldertxt,'%s %s %s %s %s %s %s %s %s %s %s\r\n', num2str( 0 ), num2str( i ), num2str( x1 ), num2str( y1 ), num2str( x2 ) ,num2str( y2 ),...
                                    num2str(  min(data101(j),data103(j)) ), num2str( min(data102(j),data104(j)) ), num2str( max(data101(j),data103(j)) ) ,num2str( max(data102(j),data104(j)) ), num2str( 1 ) ); 
                        catch
                            q = 1;
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