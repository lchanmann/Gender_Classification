%% Face Detection...
clear ; close all; clc
display('_________________________________________________________');
display('                                                         ');
display('                LFW Dataset Prepearing                   ');
display('                    Face Detection                       ');
display('_________________________________________________________');
display(' ');
fprintf(['\n LFW faces Detection and Image Cropping ... \n' ...
         '(This mght take a few minute ...)\n\n']);
% set direction for reading 
cd ('LFW_dataset/Training/male/');
D = dir('*.jpg');
for i = 1:numel(D)
    I = (imread(D(i).name));
%% Face Detection ...
    FDetect = vision.CascadeObjectDetector;
    % Returns Bounding Box values based on number of objects
    BB = step(FDetect,I);
    
   % checking if the boundary box out of index
    [c ~]=size(BB);

    % Returns Bounding Box values based on number of objects
    figure,imshow(I); hold on
    for j = 1:size(BB,1)
        rectangle('Position',BB(j,:),'LineWidth',3,'LineStyle','-','EdgeColor','r')  
    end
    if (c==0)
        BB1=80;BB2=50;BB3=120;BB4=150; 
    elseif (c==2)
        %BB1=BB(k,1),BB2=BB(k,2),BB3=BB(k,3),BB4=BB(k,4)
        [x1 y1]  = size(imcrop(I,[BB(1,1) BB(1,2) BB(1,3) BB(1,4)]));
        [x2 y2]  = size(imcrop(I,[BB(2,1) BB(2,2) BB(2,3) BB(2,4)]));
       % if (BB3<100)||(BB4<100)
           if ((x1>=x2) || (y1>=y2))
                 % fprintf('\n Face image rejected ... \n');
                 BB1=BB(1,1);BB2=BB(1,2);BB3=BB(1,3);BB4=BB(1,4);      
           else
                 BB1=BB(2,1);BB2=BB(2,2);BB3=BB(2,3);BB4=BB(2,4); 
           end
    elseif (c>2)
       %BB1=BB(k,1),BB2=BB(k,2),BB3=BB(k,3),BB4=BB(k,4)
        [x1 y1]  = size(imcrop(I,[BB(1,1) BB(1,2) BB(1,3) BB(1,4)]));
        [x2 y2]  = size(imcrop(I,[BB(2,1) BB(2,2) BB(2,3) BB(2,4)]));
        [x3 y3]  = size(imcrop(I,[BB(3,1) BB(3,2) BB(3,3) BB(3,4)]));
       % if (BB3<100)||(BB4<100)
       if ((x1>=x2) &&(x1>=x3))||((y1>=y2) &&(y1>=y3))
             % fprintf('\n Face image rejected ... \n');
             BB1=BB(1,1);BB2=BB(1,2);BB3=BB(1,3);BB4=BB(1,4);      
       elseif ((x2>=x1) &&(x2>=x3))||((y2>=y1) &&(y2>=y3))
             BB1=BB(2,1);BB2=BB(2,2);BB3=BB(2,3);BB4=BB(2,4);
       else
           BB1=BB(3,1);BB2=BB(3,2);BB3=BB(3,3);BB4=BB(3,4);
       end 
     else
        BB1=BB(1,1);BB2=BB(1,2);BB3=BB(1,3);BB4=BB(1,4); 
    end
    % Checking is the boundry box out of index
    temp  = imcrop(I,[BB1 BB2 BB3 BB4]);
    % copy of the croped face that we want to save
    figure,imshow(temp);
    close all;
    Output_folder = 'C:\Users\Adil\Desktop\New_LFW_Cropping\male\';
    imwrite(temp, [Output_folder D(i).name],'jpg');
end;
fprintf('\n Done with male faces ... \n');
clear temp;
%% Female face images reading...
% set the direction
cd ('Final Project\LFW_dataset\Training\female\');
D = dir('*.jpg');
for i = 1:numel(D)
    I = (imread(D(i).name));
    %imshow(I);
%% Face Detection ...
    FDetect = vision.CascadeObjectDetector;
    % Returns Bounding Box values based on number of objects
    BB = step(FDetect,I);
    
   % checking if the boundary box out of index
    [c ~]=size(BB);
    
    % Returns Bounding Box values based on number of objects
    figure,imshow(I); hold on
    for j = 1:size(BB,1)
        rectangle('Position',BB(j,:),'LineWidth',3,'LineStyle','-','EdgeColor','r')  
    end
    if (c==0)
        BB1=80;BB2=50;BB3=120;BB4=150; 
    elseif (c==2)
        %BB1=BB(k,1),BB2=BB(k,2),BB3=BB(k,3),BB4=BB(k,4)
        [x1 y1]  = size(imcrop(I,[BB(1,1) BB(1,2) BB(1,3) BB(1,4)]));
        [x2 y2]  = size(imcrop(I,[BB(2,1) BB(2,2) BB(2,3) BB(2,4)]));
       % if (BB3<100)||(BB4<100)
           if ((x1>=x2) || (y1>=y2))
                 % fprintf('\n Face image rejected ... \n');
                 BB1=BB(1,1);BB2=BB(1,2);BB3=BB(1,3);BB4=BB(1,4);      
           else
                 BB1=BB(2,1);BB2=BB(2,2);BB3=BB(2,3);BB4=BB(2,4); 
           end
    elseif (c>2)
       %BB1=BB(k,1),BB2=BB(k,2),BB3=BB(k,3),BB4=BB(k,4)
        [x1 y1]  = size(imcrop(I,[BB(1,1) BB(1,2) BB(1,3) BB(1,4)]));
        [x2 y2]  = size(imcrop(I,[BB(2,1) BB(2,2) BB(2,3) BB(2,4)]));
        [x3 y3]  = size(imcrop(I,[BB(3,1) BB(3,2) BB(3,3) BB(3,4)]));
       % if (BB3<100)||(BB4<100)
       if ((x1>=x2) &&(x1>=x3))||((y1>=y2) &&(y1>=y3))
             % fprintf('\n Face image rejected ... \n');
             BB1=BB(1,1);BB2=BB(1,2);BB3=BB(1,3);BB4=BB(1,4);      
       elseif ((x2>=x1) &&(x2>=x3))||((y2>=y1) &&(y2>=y3))
             BB1=BB(2,1);BB2=BB(2,2);BB3=BB(2,3);BB4=BB(2,4);
       else
           BB1=BB(3,1);BB2=BB(3,2);BB3=BB(3,3);BB4=BB(3,4);
       end 
     else
        BB1=BB(1,1);BB2=BB(1,2);BB3=BB(1,3);BB4=BB(1,4); 
    end
    % Checking is the boundry box out of index
    temp  = imcrop(I,[BB1 BB2 BB3 BB4]);
    % copy of the croped face that we want to save
    figure,imshow(temp);
    close all;
    Output_folder = 'C:\Users\Adil\Desktop\LFW_Cropping\Training\female\';
    imwrite(temp, [Output_folder D(i).name],'jpg');
end;
fprintf('\n Done with famale faces ... \n');
