
# Computer Vision take-home: Astronomy

**Problem:**
You are given a data synthesizer which generates images and labels. Your goal is to train a model with at most 4.5 million trainable parameters which determines whether each image has a star and, if so, finds a rotated bounding box that bounds the star.

More precisely, the labels contain the following five numbers, which your model should predict:
* the x and y coordinates of the center
* yaw
* width and height.

If there is no star, the label consists of 5 `np.nan`s. The height of the star is always noticeably larger than its width, and the yaw points in one of the height directions. The yaw is always in the interval `[0, 2 * pi)`, oriented counter-clockwise and with zero corresponding to the upward direction.
For your reference, train.py contains a basic CNN architecture (and training code) which performs poorly and. You can extend this model/training or start over on your own.

**Evaluation:**
Your submission will be graded quantitatively using `main.py`. The metric is the percent of correctly identified stars based on an IOU threshold of 0.7 (for 1024 random samples). Furthermore, we will look specifically at the following qualitative things:

* overall approach
* model architecture
* loss function
* code quality (please reference any outside code that you used or built on top of)


**Deliverables:**
1. Final score
1. Model weights and summary (e.g. `model.summary()` or `torchsummary`)
1. Filled-out `main.py` file that reproduces the reported score, and a `train.py` script that reproduces a model which produces the final score
1. A `requirements.txt` file that includes all python dependencies and their versions
1. (Optional) A brief explanation of your approach


