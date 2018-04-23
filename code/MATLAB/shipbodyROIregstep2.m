%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate the regression target and classification target
% Please specify the path according to your configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the path to store the regression target and classification target for samples
samplePath = fullfile('D:','shipbody');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
step2txt = fopen( fullfile(samplePath, shipbodystep2roi.txt),'a');

step = 2;
% roi ith,   in ith ship image,   patch在图像左上x,  左上y,   右下x,   右下y, groundtruth 在图像左上x,   左上y,    右下x,    右下y,    类别( 是否船 ), 权重
[data0,data1,patchx1,patchy1,patchx2,patchy2,gtx1,gty1,gtx2,gty2,data3] = textread( fullfile(samplePath, 'shipbody.txt'), '%n%n%n%n%n%n%n%n%n%n%n');
len = length(data0);

% Ground truth x1 y1 x2 y2   step1
x1step1 = gtx1(:)*1.0/2.0 + patchx1(:)*1.0/2.0;
y1step1 = gty1(:)*1.0/2.0 + patchy1(:)*1.0/2.0;
x2step1 = gtx2(:)*1.0/2.0 + patchx2(:)*1.0/2.0;
y2step1 = gty2(:)*1.0/2.0 + patchy2(:)*1.0/2.0;
% Ground truth x1 y1 x2 y2   step2
x1step2 = gtx1(:);
y1step2 = gty1(:);
x2step2 = gtx2(:);
y2step2 = gty2(:);

% proposal midx midy width height  step1
midxstep0 = patchx1(:)*0.5 + patchx2(:)*0.5;
midystep0 = patchy1(:)*0.5 + patchy2(:)*0.5;
widstep0  = patchx2(:) - patchx1(:);
heistep0  = patchy2(:) - patchy1(:);
% Ground truth midx midy width height  step1   /  proposal step2
midxstep1 = x1step1(:)*0.5 + x2step1(:)*0.5;
midystep1 = y1step1(:)*0.5 + y2step1(:)*0.5;
widstep1  = x2step1(:) - x1step1(:);
heistep1  = y2step1(:) - y1step1(:);
% Ground truth midx midy width height  step2
midxstep2 = x1step2(:)*0.5 + x2step2(:)*0.5;
midystep2 = y1step2(:)*0.5 + y2step2(:)*0.5;
widstep2  = x2step2(:) - x1step2(:);
heistep2  = y2step2(:) - y1step2(:);

M = [data0,data0,data0,data0,data0];

% N1 :the 1st iteration bounding-box regression
N1 = M;
N1(:,2:5)=0;
NN1 = N1;
for i=1:len
    N1(i,2) =    ( midxstep1(i) - midxstep0(i) )/widstep0(i);
    N1(i,3) =    ( midystep1(i) - midystep0(i) )/heistep0(i);
    N1(i,4) = log( widstep1(i)/widstep0(i) );
    N1(i,5) = log( heistep1(i)/heistep0(i) );
end

% N2 :the 2nd iteration bounding-box regression
N2 = M;
N2(:,2:5)=0;
NN2 = N2;
for i=1:len
    N2(i,2) =    ( midxstep2(i) - midxstep1(i) )/widstep1(i);
    N2(i,3) =    ( midystep2(i) - midystep1(i) )/heistep1(i);
    N2(i,4) = log( widstep2(i)/ widstep1(i) );
    N2(i,5) = log( heistep2(i)/ heistep1(i) );
end

% N3 :the 3rd iteration bounding-box regression(groundtruth to groundtruth)
 N3 = M;
 N3(:,2:5)=0;
 NN3 = N3;
 for i=1:len
     N3(i,2) =  ( midxstep2(i) - midxstep2(i) )/widstep2(i);
     N3(i,3) =  ( midystep2(i) - midystep2(i) )/heistep2(i);
     N3(i,4) =  log( widstep2(i)/ widstep2(i) );
     N3(i,5) =  log( heistep2(i)/ heistep2(i) );
 end

[width, height] = size( N1 );

XYZ = [N1; N2; N3];
% XYZ = [N1; N2];
[width1, height1] = size( XYZ );

std2data2 = std2( XYZ(:,2)  );
meandata2 = mean( XYZ(:,2)  );        
std2data3 = std2( XYZ(:,3) );
meandata3 = mean( XYZ(:,3) );        
std2data4 = std2( XYZ(:,4) );
meandata4 = mean( XYZ(:,4) );        
std2data5 = std2( XYZ(:,5) );
meandata5 = mean( XYZ(:,5) );      
	
NN1(:,2) =  (  N1(:,2) -  meandata2)/ std2data2;
NN1(:,3) =  (  N1(:,3) -  meandata3)/ std2data3;
NN1(:,4) =  (  N1(:,4) -  meandata4)/ std2data4;
NN1(:,5) =  (  N1(:,5) -  meandata5)/ std2data5;  

NN2(:,2) =  (  N2(:,2) -  meandata2)/ std2data2;
NN2(:,3) =  (  N2(:,3) -  meandata3)/ std2data3;
NN2(:,4) =  (  N2(:,4) -  meandata4)/ std2data4;
NN2(:,5) =  (  N2(:,5) -  meandata5)/ std2data5; 

NN3(:,2) =  (  N3(:,2) -  meandata2)/ std2data2;
NN3(:,3) =  (  N3(:,3) -  meandata3)/ std2data3;
NN3(:,4) =  (  N3(:,4) -  meandata4)/ std2data4;
NN3(:,5) =  (  N3(:,5) -  meandata5)/ std2data5;

for j=1:len
% 	fprintf(step2txt,'%s %s %s %s %s %s %s %s %s %s\r\n',num2str( floor( data0(j) ) ),num2str( floor( data1(j) ) ), ...
%         num2str( patchx1(j) ), num2str( patchy1(j) ), num2str( patchx2(j) ), num2str( patchy2(j) ), num2str( NN1(j,2) ), num2str( NN1(j,3) ), num2str( NN1(j,4) ), num2str( NN1(j,5) ) );
% 	fprintf(step2txt,'%s %s %s %s %s %s %s %s %s %s\r\n',num2str( floor( data0(j) ) ),num2str( floor( data1(j) ) ), ...
%         num2str( x1step1(j) ), num2str( y1step1(j) ), num2str( x2step1(j) ), num2str( y2step1(j) ), num2str( NN2(j,2) ), num2str( NN2(j,3) ), num2str( NN2(j,4) ), num2str( NN2(j,5) ) );
% 	fprintf(step2txt,'%s %s %s %s %s %s %s %s %s %s\r\n',num2str( floor( data0(j) ) ),num2str( floor( data1(j) ) ), ...
%         num2str( x1step2(j) ), num2str( y1step2(j) ), num2str( x2step2(j) ), num2str( y2step2(j) ), num2str( NN3(j,2) ), num2str( NN3(j,3) ), num2str( NN3(j,4) ), num2str( NN3(j,5) ) );    

    ix1 = max( patchx1(j), x1step2(j) );
    iy1 = max( patchy1(j), y1step2(j) );
    ix2 = min( patchx2(j), x2step2(j) );
    iy2 = min( patchy2(j), y2step2(j) );
    IOU =  ( iy2 - iy1 )*( ix2 - ix1 )/( ( y2step2(j) - y1step2(j) )*( x2step2(j) - x1step2(j) ) + ( patchy2(j) - patchy1(j) )*( patchx2(j) - patchx1(j) ) -  ( iy2 - iy1 )*( ix2 - ix1 ) );
	if( IOU > 0.7)
        fprintf(step2txt,'%s %s %s %s %s %s %s %s %s %s %s %s\r\n',num2str( 0 ),num2str( floor( data1(j) ) ),...
                num2str( patchx1(j) ), num2str( patchy1(j) ), num2str( patchx2(j) ), num2str( patchy2(j) ), num2str( NN1(j,2) ), num2str( NN1(j,3) ), num2str( NN1(j,4) ), num2str( NN1(j,5) ), num2str( 1 ), num2str( 1 ) );
    else
        fprintf(step2txt,'%s %s %s %s %s %s %s %s %s %s %s %s\r\n',num2str( 0 ),num2str( floor( data1(j) ) ),...
                num2str( patchx1(j) ), num2str( patchy1(j) ), num2str( patchx2(j) ), num2str( patchy2(j) ), num2str( NN1(j,2) ), num2str( NN1(j,3) ), num2str( NN1(j,4) ), num2str( NN1(j,5) ), num2str( 0 ), num2str( 1 ) );
	end

    ix1 = max( x1step1(j), x1step2(j) );
    iy1 = max( y1step1(j), y1step2(j) );
    ix2 = min( x2step1(j), x2step2(j) );
    iy2 = min( y2step1(j), y2step2(j) );
    IOU =  ( iy2 - iy1 )*( ix2 - ix1 )/( ( y2step2(j) - y1step2(j) )*( x2step2(j) - x1step2(j) ) + ( y2step1(j) - y1step1(j) )*( x2step1(j) - x1step1(j) ) -  ( iy2 - iy1 )*( ix2 - ix1 ) );
	if( IOU > 0.7)
        fprintf(step2txt,'%s %s %s %s %s %s %s %s %s %s %s %s\r\n',num2str( 0 ),num2str( floor( data1(j) ) ), ...
                num2str( x1step1(j) ), num2str( y1step1(j) ), num2str( x2step1(j) ), num2str( y2step1(j) ), num2str( NN2(j,2) ), num2str( NN2(j,3) ), num2str( NN2(j,4) ), num2str( NN2(j,5) ), num2str( 1 ), num2str( 1 ) );
    else
        fprintf(step2txt,'%s %s %s %s %s %s %s %s %s %s %s %s\r\n',num2str( 0 ),num2str( floor( data1(j) ) ), ...
                num2str( x1step1(j) ), num2str( y1step1(j) ), num2str( x2step1(j) ), num2str( y2step1(j) ), num2str( NN2(j,2) ), num2str( NN2(j,3) ), num2str( NN2(j,4) ), num2str( NN2(j,5) ), num2str( 0 ), num2str( 1 ) );
    end

    
  	fprintf(step2txt    ,'%s %s %s %s %s %s %s %s %s %s %s %s\r\n',num2str( 0 ),num2str( floor( data1(j) ) ), ...
          num2str( x1step2(j) ), num2str( y1step2(j) ), num2str( x2step2(j) ), num2str( y2step2(j) ),  num2str( NN3(j,2) ), num2str( NN3(j,3) ), num2str( NN3(j,4) ), num2str( NN3(j,5) ),num2str( 1),num2str( 1) );    
end
fclose('all');