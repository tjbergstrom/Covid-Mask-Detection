What's in here:

`processing/`
- You can run `bash preprocess.sh` to save a processed dataset of faces cropped from the original dataset.
- You can run `bash preprocess_lowerq.sh` to save a new copy processed dataset with the image quality reduced to match video stream quality.

`training/`
- You can run `python3 -W ignore train_a_model.py` to train a default model.
- You can run `bash auto_train.sh` and compare models with different hypertunings.

`testing/`
- You can run `python3 tests.py` to test a model on video and see how well it performs.

`computer_vision/`
- You can run `python3 -W ignore detect_and_track.py -d tru` to load your webcam and see the detection and tracking/counting in real time.


