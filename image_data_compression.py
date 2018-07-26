
# Import needed functions for transformation
from PIL import Image
import numpy as np
import matplotlib.pylab as plt
from numpy import linalg as LA


# Import helper functions
from helpers import *



#################


# Set name of image and number of singular values used for image.
IMAGE_NAME = 'Saltmarsh_svd'
FLAG = True
NUMBER_OF_SINGULAR_VALUES = 2

# IMAGE PRE-PROCESSING:
#   The decomposition pipeline starts with the pre-processing of the colour
#   image into a greyscale image. The follow three lines converts a colour
#   image to greyscale, reads the converted greyscale image and finally stores
#   the greyscale image in a list of lists as 'n_by_n_image_matrix'.

# Convert the image to greyscale
convert_image_to_greyscale(IMAGE_NAME)
# Read in the converted greyscale image
image_matrix = plt.imread(IMAGE_NAME + '_greyscale.png')
# CHECK IMAGE HAS CORRECTLY BEEN CONVERTED TO GREYSCALE.
print(check_format_of_greyscale(image_matrix))
# Convert image to a list of lists
n_by_n_image_matrix = convert_gs_image_to_matrix(image_matrix)



# LINEAR ALGEBRA:
#   Below is the crux of the data compression with use of the singular value decomposition (SVD).
#   Founded from the same principals as the eigenvalue decomposition (EVD), the SVD relaxes a
#   condition of the EVD, that is requirements for a full rank, square matrix. The SVD allows
#   for a matrix to have rectangular dimensions to work. However, the algebra for the decomposition
#   is similar to EVD where the lemma (A - \lambda I )X = 0 is rearranged to represent A as the
#   eigenvalues and eigenvectors as \lambda and X, respectively.
U, sigma, V = np.linalg.svd(n_by_n_image_matrix)
u, sig, v = calc_singular_value_decomposition(n_by_n_image_matrix)
rank_reduced_matrix = get_top_k_singular_projections(u, sig, v, 5)


# VIEWING COMPRESSED IMAGE:
#   Once the singular value decomposition matrices have been calculated the top 5
#   singular values are extracted. Using matrix multiplication on these top 5 singular
#   values a greyscale image can be constructed. NOTICE: there is details which is
#   lost from the original gray scale image though the size of the data has shrunken
#   from 320x320 to 5x320 + 5 + 5x320.
reconstructed_image = np.matrix(U[:, :5]) * np.diag(sigma[:5]) * np.matrix(V[:5, :])
plt.imshow(reconstructed_image, cmap='gray')
plt.show()
