# IntelligentSystemsProject
Sign Language translator using a neural network completed as a project for my Intelligent Systems class.

The running demo is the file sign-to-speech_demo.py.To run it, cd into the file's directory, then use the following command:

python sign-to-speech_demo.py --training_set train

The program runs the neural network training first. It does this for a few minutes, then asks for an input. The input is one of the test folder (test1, test2, test3, test4). *This input is not validated and will break if the wrong input is given.

Test 1 is the word "SATA".
Test 2 is the word "MOTHERBOARD".
Test 3 is the phrase "PETG TUBING".
Test 4 is the phrase "ARTIFICIAL INTELLIGENCE".

Here is a link to the tutorials used to setup the working environment if needed.
http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
http://www.pyimagesearch.com/2016/11/14/installing-keras-with-tensorflow-backend/
