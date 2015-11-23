function [confusion_matrix]=k_nearest_neighbor(k)
% this function calculates the classification of
% vectors using K-nearest neighbor algorithm. Only two features are used 
% at a time to do the classification. values of k is input for this
% function
load('adil');  %data given in project is saved in 'patdata'
y=adil;
train_f=y(51:200,:);
train_s=y(201:end,:);
train=[train_f;train_s];
test_f=y(1:200,:);
test_s=y(201:400,:);
test=[test_f;test_s];
class_one_one=0; class_one_two=0;class_two_one=0;class_two_two=0;
for j=1:400;
count=0;
    for i=1:200
     dist(i)=sqrt(sum((test(j,1:2)-train(i,1:2)).^2));
     %Lines 17 to 29 find k
     if count<k                                   
          count=count+1; 
          % Nearest neighbor out of 200
          neighbor(count,:)=train(i,1:2);       
          % Test vectors
          distance(count)=dist(i);
          class(count)=train(i,5);
    elseif count==k  
         [maxd,ind]=max(distance);   
         if maxd>=dist(i)
               neighbor(ind,:)=train(i,3:4);
               distance(ind)=dist(i);
               class(ind)=train(i,5);   
         end
       end
      end
 ones=find(class==1);
 twos=find(class==2);
 if size(ones,2)>size(twos,2) &&  test(j,5)==1
     class_one_one=class_one_one+1;
 elseif size(ones,2)>size(twos,2) &&  test(j,5)==2
     class_one_two=class_one_two+1;
 elseif size(ones,2)<size(twos,2)  && test(j,5)==1   
     %lineas 33 to 68 find the class of test vector
     class_two_one=class_two_one+1;                  
     %and subsequently confusion matrix is formed.
 elseif size(ones,2)<size(twos,2)  && test(j,5)==2   
     class_two_two=class_two_two+1;
 elseif size(ones,2)==size(twos,2)
     tobsum_f=distance(ones);
     first=sum(tobsum_f,2);
     tobsum_s=distance(twos);
     second=sum(tobsum_s,2);
       if first<second && test(j,5)==1
          class_one_one=class_one_one+1;
       elseif first<second && test(j,5)==2
            class_one_two=class_one_two+1;
       elseif first>second && test(j,5)==1
            class_two_one=class_two_one+1;
       elseif first>second && test(j,5)==2
            class_two_two=class_two_two+1;
       elseif first==second 
           if class(ind)==1 && test(j,5)==1
               class_one_one=class_one_one+1;
           elseif class(ind)==1 && test(j,5)==2
               class_one_two=class_one_two+1;
           elseif class(ind)==2 && test(j,5)==1
               class_two_one=class_two_one+1;
           elseif class(ind)==2 && test(j,5)==2
               class_two_two=class_two_two+1;
           end               
       end
 end
end
 confusion_matrix=[class_one_one class_one_two;class_two_one class_two_two];
 