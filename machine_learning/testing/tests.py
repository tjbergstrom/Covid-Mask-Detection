# Ty Bergstrom
# tests.py
# CSCE A401
# November 2020
# Software Engineering Project
#
# Testing for the main mask detection model fully integrated with tracking statitics
# This will run the detect_and_track.py with different input video files
# The output results of for each video will be saved to args["testfile"]
# These are compared with the expected results located in "expected.csv"
# The output results are:
#		number of masks counted, number without masks counted, usage percent
#
# python3 tests.py -t test1.csv
#
# Sample output from this testing looks like this:
'''
	Test [1] passed
	Test [2] passed
	Test [3] passed
	Test [4] passed
	Test [5] failed: 
	 Expected: ['1', '0', '100.00'], Actual: ['2', '0', '100.00']
	Test [6] failed: 
	 Expected: ['1', '0', '100.00'], Actual: ['0', '0', '0.00']
'''




import os
import sys
import argparse
import filecmp

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testfile", required=True)
args = vars(ap.parse_args())

test_file = args["testfile"]
test_file_path = "../testing/" + test_file

os.system("touch " + test_file)
os.system("echo \"filename,mask,without_mask,usage\" > " + test_file)

cmd = "python3 -W ignore ../computer_vision/detect_and_track.py "
cmd += " -t " + test_file_path + " "

vid_dir = "../computer_vision/uploads/"

for vid in os.listdir(vid_dir):
	vid_path = vid_dir + vid
	os.system(cmd + " -f " + vid_path)

expected = "expected.csv"

if filecmp.cmp(test_file, expected):
	print("All tests passed")
	sys.exit(0)

print("Failed some tests")
expected = open(expected, "r")
correct_tests = [i for i in expected]
outputs = open(test_file)
output_tests = [i for i in outputs]
passed = 0
i = 1
while i < len(correct_tests):
	if correct_tests[i] == output_tests[i]:
		print("Test [{}] passed".format(i))
		passed += 1
	else:
		print("Test [{}] failed: ".format(i))
		correct = [j.replace("\n", "") for j in correct_tests[i].split(",")[1:] ]
		actual = [j.replace("\n", "") for j in output_tests[i].split(",")[1:] ]
		print(" Expected: {}, Actual: {}".format(
			correct, actual
		))
	i += 1
print("Passed {} out of {} tests".format(passed, len(output_tests)-1))



##
