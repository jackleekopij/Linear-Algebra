from PIL import Image
import numpy as np
import matplotlib.pylab as plt
from numpy import linalg as LA



def convert_image_to_greyscale(image_name):
    '''
    Takes a string as a reference to a colour image and converts to greyscale saving with the suffix
    '_greyscale.png'.
    :param image_name: a string which is the name of the image which is to be converted to grey scale.
    '''
    img = Image.open(image_name + '.jpg').convert('LA')
    img.save(image_name + '_greyscale.png')





def create_diag_matrix(Sigma_vector):
    '''
    create_diag_matrix takes a vector of singular values creating a diagonal matrix where the diagonal
    take singular values of the initial matrix.
    :param Sigma_vector: A list containing the singular values of a matrix.
    :return: A returned is a diagonal matrix of singular values of a matrix.
    '''
    num_sigma_entries = len(Sigma_vector)
    diag_matrix = np.zeros((num_sigma_entries, num_sigma_entries))
    for i in range(num_sigma_entries):
        for j in range(num_sigma_entries):
            if i == j:
                diag_matrix[i,j] = Sigma_vector[i]
    return diag_matrix





def check_format_of_greyscale(image):
    '''
    Checks image format to ensure .png image can be reduced to a an nxn matrix. This function is
    included to ensure the image has decomposed correctly into a greyscale image.
    :param image: is a list of list which is the grayscale image.
    :return: A string indicating whether an image is corrupt.
    '''
    for i in range(len(image)):
        for j in range(len(image[i])):
            assert image[i][j][0] == image[i][j][1], "Greyscale values are not consistent across two dimensions!"
            assert image[i][j][3] == 1, "Final value of the iamge is not equal to 1!"
    return "Image is not corrupt and can be converted to nxn matrix"





def convert_gs_image_to_matrix(image):
    '''
    convert_gs_image_to_matrix will take an image object then decompose to a list of lists.
    This function extracts the first value from the greyscale image storing it in a list of
    lists thus reducing the dimensions.
    :param image: an image object to be converted to a list of lists of the greyscale image.
    :return: a list of lists of the greyscale image.
    '''
    image_list_of_list_matrix = []
    for i in range(len(image)):
        row_of_image = [ ]
        for j in range(len(image[i])):
            row_of_image.append(image[i][j][0])
        image_list_of_list_matrix.append(row_of_image)
    print("Image is of pixexl dimensions: ({0},{1})".format(len(image_list_of_list_matrix),len(image_list_of_list_matrix[0])))
    return image_list_of_list_matrix




# start by calculating the (Image_matrix^t)(Image_matrix)
def calc_singular_value_decomposition(n_by_n_image_matrix):
    '''
    Calculate the singular value decomposition matrix by calculating the three matrices, U, Sigma, V
    :param n_by_n_image_matrix:
    :return: Returns the three components of the the singular value decomposition
    '''
    symmetric_matrix = np.matmul(np.transpose(n_by_n_image_matrix), n_by_n_image_matrix)
    evalues1, evectors1 = LA.eig(symmetric_matrix)

    symmetric_matrix = np.matmul(n_by_n_image_matrix, np.transpose(n_by_n_image_matrix))
    _, evectors2 = LA.eig(symmetric_matrix)

    V_matrix = evectors1
    Sigma_matrix = np.sqrt(evalues1)
    U_matrix = evectors2

    return U_matrix, Sigma_matrix, V_matrix



def get_top_k_singular_projections(U_matrix, Sigma_vector, V_matrix, k):
    Sigma_matrix = create_diag_matrix(Sigma_vector[0:k])

    top_k_u = np.transpose(U_matrix[0:k])
    top_k_vt = np.transpose(V_matrix)[0:k]

    U_by_sigma = np.matmul(top_k_u, Sigma_matrix)

    U_by_sigma_by_Vt = np.matmul(U_by_sigma, top_k_vt)

    #print("Top k ")
    #print(top_k_u)
    #print(create_diag_matrix(Sigma_matrix[0:k]))
    #for i in len(top_k_u[0]):

    return U_by_sigma_by_Vt


