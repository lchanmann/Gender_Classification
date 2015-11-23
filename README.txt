STEP 1
------
Extract faces by running Face_detection_image_cropping.m on all the images in the folder. Resulting 250×250 Images are saved in ./LFW_Cropping/Training/
Requires Image processing toolbox
Requires ./LFW/ directory containing male images
Requires ./LFW_dataset/Training/female directory containing female images
- Creates ./LFW_Cropping/Training/male/ directory containing the extracted male faces
- Creates ./LFW_Cropping/Training/female/ directory containing the extracted female faces

STEP 2
------
Preprocess each image by running Face_image_reading_and_preprocessing.m
Requires Image processing toolbox
Requires folder ./LFW_Cropping/Training/, which it assumes contains subdirectores male/ and female/
Preprocessing consists of
1. Conversion to Grayscale
2. Histogram equalization (normalization)
3. Scale to 80×80
4. convert to vectors of 6400 columns
5. Each vector saved as row of matrix X
6. Labels are stored in linear vector Y
	Y[i] = 1 if X[i,:] corresponds to a male image
	Y[i] = 2 if X[i,:] corresponds to a female image
7. Creates ./LFW_faces.mat containing X and Y

STEP 3
------
run compute_PCA
Requires ./LFW_faces.mat 
Creates ./LFW_Face Detection/