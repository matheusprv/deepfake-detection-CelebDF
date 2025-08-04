#1 - separate_train_val_test_videos
<p>Given the list of test videos provided from the dataset, these videos will be the test videos that is going to be used, being 178 real and 340 with deepfakes.</p>
<p>To select the total of videos from train and validation, it was decided that the validation videos would be 326 videos with deepfake and 2*163 real videos, where each real video would have two subclips.</p>
<p>And the remaining videos would go to the training data, being 4973 fake and 548*9 real videos.</p>
<p>Every subclip has 24 frames. This implicates that some videos for training would not have enough frames, that's why they were automatically moved to validatio, while the other were reandomly selected.</p>


#2 - extract_faces_and_generate_subclips


#3 - check_wrong_videos
<p>This is not a pre-processing step. This is supposed to happen after the training process is completed</p>
<p>This file will go through each of the test videos and check what are the videos that are being mislabeled by the model</p>