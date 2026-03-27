This project handles object detection of ct scans.
Each image comes as a .tif file, and contains a cross-section stem of a plant, with some constructions around it, on a relatively constant background.
The goal is to calculate the cross-section area of the stem in that image.

first a CSV containing a rough estimation of the middle of the stem is created.
It is created this way:
1. The image is read as a numpy array.
2. The image is blurred using a Gaussian filter, to remove noise.
3. the image presented using matplotlib, and the user is asked to click first on the middle of the stem, then on a point on the edge of the stem.
4. The coordinates of the two click are saved in a CSV file, along with the name of the image.
   1. The csv is saved between each image, so that if the user wants to stop the process, they can continue later without losing any data.


The second part of the project is to use the CSV file to create a mask of the stem, and then calculate the area of the stem in pixels.
This is done by:
1. The image is read as a numpy array.
2. The image is blurred using a Gaussian filter, to remove noise.
3. A mask is created using the coordinates from the CSV file, by drawing a circle around the middle of the stem
4. The image is converted to polar coordinates, using the middle of the stem as the origin.
5. The image is converted to a binary image of the contrast in the radial direction, by calculating the difference between each pixel and the pixel in the row column (the previous radius), and then converting to binary. both positive contrast and negative contrast are accounted for.
6. The image is rotated (shifted circularly in the polar coordinates) so that the clicked point of the edge of the stem is the first column (it becomes \theta=0)
   1. The rotation is saved so that the process can be reversed later.
7. A hidden markov model is used to extract the edge of the stem in polar coordinates:
   1. The states of the HMM are the possible radii of the stem, and the observations are the contrast in the radial direction.
   2. The transition probabilities from the radius of one angle (column) to the next is assumed to be a gaussian distribution centered around the same radius, with a standard deviation of 3 pixels.
      1. This value of 3 pixels is saved as a global variable, so that it can be changed later if needed.
   3. The probability of obseving a contrast pixel (the contast image is binary, so pixels are either 0 or 1) given a state (radius) is calculated as follows:
      1. If the pixel is a pixel of the edge of the stem, then the probability of it being one is 0.9.
      2. If the pixel is not a pixel of the edge of the stem, then the probability of it being one is 0.5.
         1. Those numbers (0.9 and 0.5) are saved as global variables, so that they can be changed later if needed.
   4. The Viterbi algorithm is used to find the most likely sequence of states (radii) given the observations (contrast in the radial direction).
      1. In the Viterbi algorith, the radii in the first column (the first angle) is assumed to be the radius of the clicked point on the edge of the stem, with a probability of 1, and all other radii in the first column have a probability of 0.
      2. In the last iteration of the algorith, where the path that maximizes the probability is found for each final state, the path that ends in the original radius is assumed to be the most likely path, with a probability of 1, and all other paths have a probability of 0.
8. The area of the stem is calculated by summing the radii of the edge of the stem (the most likely sequence of states) and multiplying by the angle step (the angle between each column in the polar coordinates).
9. The mask is being converted back to cartesian coordinates (after rotating back) and is presented on top of the original image, to visually check the results.
10. The area of the stem in pixels is saved in a CSV file, along with the name of the image.

