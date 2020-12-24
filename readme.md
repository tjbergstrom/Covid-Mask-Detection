This was for a software engineering course project in fall semester 2020. All projects had to be covid related, so I wanted to detect face masks in video and build an app for that. The idea is to detect masks in real time video, and report the mask usage for the duration of the stream. For example, you can upload a webcam stream at the front entrance of a building, and get an estimate of mask usage there.

The ```machine_learning/``` directory is all of my own original work that I have been doing. It's the machine learning project and the backend computer vision for testing it out and getting everything to work.

The model that I built has also been deployed in the web application, found [here](https://github.com/tjbergstrom/facemask-detection-app)

The following is a brief overview of my machine learning and computer vision project.

<br>

![alt text](https://raw.githubusercontent.com/tjbergstrom/Covid-Mask-Detection/master/machine_learning/data/demo.png)

<br>

## How it works:

### Machine Learning:

Make a prediction: is a person wearing a face mask or not? Train a model with a dataset of images. Here are some quick notes.

- #### Data collection:

I manually collected thousands of images of people wearing face masks, and not wearing masks. I only skimmed from two datasets that I could find, but they were not good quality and had some bias.

- #### Data processing:

I have a really great processing pipeline with some scripts. The data gets cleaned up, bad files are rejected, filepaths are renamed, and so on. There's also a script to find and remove duplicate images (by calculating hashes), and some other cool stuff.

Then another script uses face detection to extract all faces from all images and saves them into a processed directory. Next, you need to go through this directory and manually remove images that were false positives (because face detection doesn't always get the face cropped exactly). There's another script to help find the originals so you can process them manually later. Also, images that did not detect any faces (by the face detection cropping script) are saved for manually processing later.

So now there is a good dataset, but they are high quality images, and do not resemble what you'd expect to see in low quality video. So I have another script to go through and process the images to lower their quality. It's pretty cool, because they come out looking pretty close to blurry pixelated video quality. This really helps improve accuracy by giving the models better data that's closer to what they will see in video.

I tried out building different models with different levels of lower quality processing. So then, during the stage of testing out the models on actual video, I was able to adjust the accuracy by trying out models trained with different types of processing.

- #### Training

I have another great training pipeline. I use scripts to build many models with different combinations of hypertunings, such as epochs, batch sizes, image sizes, learning rates, and so much more. After training for a few days I can pick out the best models, but I usually know pretty quickly what parameters are the best.

- #### Testing

You can say that a model comes out to ~99% accuracy based on training metrics, but how good is it on video for the intended use? First I had to actually deploy the models in video, as described below.

### Computer Vision:

The goal of the project was to detect whether a person is wearing a face mask in real time video, and calculate the mask usage for the duration of the stream.

- #### Part 1, detect and classify:

This is the easy part. It's just a script that loads an input video stream. Then you read every frame of the stream, and for every frame use face detection to find faces, and for every face get a prediction from my models, and then display the prediction on the frame.

- #### Part 2, tracking:

This is the hard part. If you want to count how many masks you have detected, you can't just start counting, because every frame in the stream is like a new image, so for example one person in a hundred frames gets counted as one hundred people, and when they step away there are a total of zero people counted in the current frame.

So you have to use object tracking. When you detect a face, add it to a list of trackers, and update its location from frame to frame. And of course, make the prediction - mask our without mask - and add up the counts of each.

- #### Testing

I had to spend a lot of time with a webcam seeing how it works and "calibrating" everything just right. Here are some of the more interesting notes.

When you first detect a face, don't make the mask prediction on the first frame. It might not be the best angle or something, so wait for 16 frames to go by. If it was a false positive and not even a face, then you will know that the trackers haven't updated a location change for the last ~8 frames, so reject and do not classify.

You only need to attempt detecting faces every few frames, but still need to update the trackers locations every frame.

If a person removes or puts on a mask, you need to know about it. So for every detected face that already has a classification, attempt to reclassify every ~128 frames or so, and only change if it's a strong probability.

If the trackers have lost track of a face for about ~32 frames, then this person has left the frame and this one can be removed from the trackers. In case a reclassification was about to happen, do not attempt if the trackers have not detected movement over the last ~16 frames.

- #### More testing

After calibrating everything just right, I wanted to do more testing for the models. So I collected videos - I screen recorded a lot of clips. And then manually labeled the expected number of detected masks, the number without masks detected, and the final usage percentage for each video. Then I just let another script to run and save the output for each video, and then check the results against the expected output for the whole batch. I was able to get a better idea of the accuracy, and test out different models again, and was able to choose the best.

### WebApp

The web app development is split into another repository. It's a slightly different project. It uses the same models, but uses different tools to deploy them and stream the video in a client browser. One additional difference is that the web app implementation uses pose detection to better track the location of faces across frames.

