IOUsum = 0;
trueNumber = 0;
detectNumber = 0;
totalNumber = 0;
totalImage = 0;
sumX1 = 0;
sumX2 = 0;
sumY1 = 0;
sumY2 = 0;
summidx = 0;
summidy = 0;
summidw = 0;
summidh = 0;

vectortrueNumber = 0;
vectordetectNumber = 0;
vectortotalNumber = 0;
vectorindex = 0; 
vectorIOU = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the IoU, precision, recall
% Please specify the path according to your configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the path of the ground truth
% we put the position of ships in txt
groundtruthPath = fullfile('D:','shipPosition');
% the path of the result
% we put the result (the predicted box of ship) in txt
resultPath = fullfile('D:','myresultv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mkdir(detImagePath);


for i = 1:100
    imageNumber = num2str(i);
    truthPath  = fullfile(groundtruthPath,[imageNumber,'.txt']);
    resultPath = fullfile(resultPath, [imageNumber,'.txt']);
    imagePath  = fullfile(resultPath, [imageNumber,'.jpg']);     
    
    if( exist(imagePath,'file') && exist(resultPath,'file') && exist(truthPath,'file') ) 
        [data1,data2,data3,data4,data5,data6,data7,data101,data102,data103,data104] = textread(truthPath,'%n%n%n%n%n%n%n%n%n%n%n');
        [trut101,trut102,trut103,trut104]= textread(resultPath,'%n%n%n%n');
        totalImage = totalImage + 1;
        IndexVector  = trut101;
        IoUvector    = trut101;
        IndexVector(:) = 0;
        IoUvector(:)   = 0;
        tempVector = data1;
        temp    = data101;
        temp(:) = 0;
        detectNumber = detectNumber + length(trut101);

        pertotalNumber = 0;
        perIOU = 0;
        for iiii = 1:length(data101)
            if( data5(iiii)==2 )
                pertotalNumber = pertotalNumber + 1;
                totalNumber = totalNumber + 1;
            end
        end

        pertrueNumber = 0;        
        for j = 1:length(trut101)
            tempVector(:) = 0.0;
            for k = 1:length(data101)
            	if( data5(k)==2 && temp(k)==0 && ( ~( trut101(j)>max(data101(k),data103(k)) || trut103(j)<min(data101(k),data103(k)) || trut102(j)>max(data102(k),data104(k)) || trut104(j)<min(data102(k),data104(k)) ) ) )
                	ix1 = max( trut101(j), min(data101(k),data103(k)) );
                	iy1 = max( trut102(j), min(data102(k),data104(k)) );
                	ix2 = min( trut103(j), max(data101(k),data103(k)) );
                	iy2 = min( trut104(j), max(data102(k),data104(k)) );
                	IoU = ( iy2 - iy1 )*( ix2 - ix1 )/( abs( data102(k)-data104(k) )*abs( data101(k)-data103(k) ) + abs( trut104(j)-trut102(j) )*abs( trut103(j)-trut101(j) ) -  ( iy2 - iy1 )*( ix2 - ix1 ) );
                	tempVector(k) = IoU;
            	end
            end
            [C,I] = max( tempVector );
            if( C>0.5 )
                temp(I) = 1;
                IndexVector(j) = I;
                IoUvector(j) = C;
                perIOU = perIOU + C;
                trueNumber = trueNumber + 1;
                pertrueNumber = pertrueNumber + 1;
            end
        end
                
        vectordetectNumber = [vectordetectNumber; pertrueNumber/length(trut101)*1.0 ];
   
        vectortrueNumber   = [vectortrueNumber;   pertrueNumber/pertotalNumber*1.0 ];

        vectorindex = [vectorindex; i];
        
        vectortotalNumber  = [vectortotalNumber;  pertotalNumber];
        
        vectorIOU = [vectorIOU;  perIOU/pertrueNumber*1.0 ];
        
        temp1 = zeros(size(trut101));
        temp2 = temp1;
        temp3 = temp1;
        temp4 = temp1; 
        
        for m = 1:length( trut101 )
            if( IoUvector(m)>0.5 )
                temp1(m) = data101( floor( IndexVector(m) ) );
                temp2(m) = data102( floor( IndexVector(m) ) );
                temp3(m) = data103( floor( IndexVector(m) ) );
                temp4(m) = data104( floor( IndexVector(m) ) );
            end
        end
        M = [trut101,trut102,trut103,trut104,temp1,temp2,temp3,temp4,IoUvector];
        imageNumber = num2str(i);
%         pairPath = fullfile('D:','result',[imageNumber,'.txt']);
%         dlmwrite(pairPath, M, 'delimiter', '\t','newline','pc');
        
        IOUsum = IOUsum + sum( IoUvector );
        for m = 1:length( trut101 )
        	if( IoUvector(m)>0.5 )
                sumX1 = sumX1 + abs( trut101(m)-temp1(m) );
                sumX2 = sumX2 + abs( trut102(m)-temp2(m) );
                sumY1 = sumY1 + abs( trut103(m)-temp3(m) );
                sumY2 = sumY2 + abs( trut104(m)-temp4(m) );
                summidx = summidx + abs(  abs(trut101(m)+trut103(m))*0.5 - abs(temp1(m)+temp3(m))*0.5 );
                summidy = summidy + abs(  abs(trut102(m)+trut104(m))*0.5 - abs(temp2(m)+temp4(m))*0.5 );
                summidw = summidw + abs(  abs(trut101(m)-trut103(m)) - abs(temp1(m)-temp3(m)) );
                summidh = summidh + abs(  abs(trut102(m)-trut104(m)) - abs(temp2(m)-temp4(m)) );
            end
        end
    end
end
fclose('all');
a = IOUsum*1.0/trueNumber;
b = sumX1*1.0/trueNumber;
c = sumX2*1.0/trueNumber;
d = sumY1*1.0/trueNumber;
e = sumY2*1.0/trueNumber;
f = summidx*1.0/trueNumber;
g = summidy*1.0/trueNumber;
h = summidw*1.0/trueNumber;
l = summidh*1.0/trueNumber;

disp(a);
disp(b);
disp(c);
disp(d);
disp(e);
disp(f);
disp(g);
disp(h);
disp(l);
disp(trueNumber);
disp(detectNumber);
disp(totalNumber);
disp(totalImage);
disp(trueNumber/detectNumber*1.0);
disp(trueNumber/totalNumber*1.0);