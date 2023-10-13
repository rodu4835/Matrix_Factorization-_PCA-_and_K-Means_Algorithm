import matplotlib.pyplot as plt
import numpy as np


def show_decision_surface(model, X, y, ax=None):
    """
    Helper function to visualize the decision surface of the trained model
    :param model with predict method
    :return: None
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    x_grid, y_grid = np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05)
    xx, yy = np.meshgrid(x_grid, y_grid)
    r1, r2 = xx.reshape(-1, 1), yy.reshape(-1, 1)
    grid = np.hstack((r1, r2))
    y_hat = model.predict(grid).reshape(-1, )
    zz = y_hat.reshape(xx.shape)

    if ax is None:
        plt.contourf(xx, yy, zz, cmap='PiYG')
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()
    else:
        ax.contourf(xx, yy, zz, cmap='PiYG')
        ax.scatter(X[:, 0], X[:, 1], c=y)


class Tester(object):
    def __init__(self):
        self.questions = {}

    def add_test(self, question, test_function):
        self.questions[question] = test_function

    def run(self):
        for question in self.questions:
            success, comment = self.questions[question]()
            if success:
                print("Question %s: [PASS]" % question)
            else:
                print("Question %s: [FAIL]" % question, comment)

                
def testLU(luclass):
    tester = Tester()
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    test_multiplication = np.array([[ 30,  36,  42],
       [ 66,  81,  96],
       [102, 126, 150]])
    test_pivot = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    test_L = np.array([[1.,0.,0.],[0.14285714,1.,0.],[0.57142857,0.5,1.]])
    test_U = np.array([[7.,8.,9.],[0.,0.85714286,1.71428571],[0.,0.,0.]])
    ins = "\n A:" + str(A) + ",\nB:" + str(B)
    topic = "Testing LU Factorization "

    def test_matrix_multiplication():
        outs = test_multiplication
        comment = topic + "Matrix Multiplication" + ins + "\n expected output: " + str(outs)
        lu = luclass(matrix)
        obtained = lu.Matrix_Multiplication(A,B)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment
                       
    def test_pivot_calculation():
        outs = test_pivot
        comment = topic + "Pivot Calculation" + "\n Matrix:" + str(matrix) + "\n expected output:BBBB " + str(outs)
        lu = luclass(matrix)
        obtained = lu.Pivot_Calculation()
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    def test_lu_factorizer():
        outs = test_pivot
        comment = topic + "LU Factorizer" + "\n Matrix:" + str(matrix) + "\n expected output: " + "\n P" + str(outs) + "\n L" + str(test_L) + "\n U" + str(test_U)
        lu = luclass(matrix)
        obtained_P, obtained_L, obtained_U = lu.LU_Factorizer()
        comment = comment + "\n obtained: " + "\n P" + str(obtained_P) + "\n L" + str(obtained_L) + "\n U" + str(obtained_U)
        if (np.allclose(obtained_P, outs, atol=1e-5) and np.allclose(obtained_L, test_L, atol=1e-5) and np.allclose(obtained_U, test_U, atol=1e-5)):
            return True, comment
        return False, comment

    tester.add_test("1.11", test_matrix_multiplication)
    tester.add_test("1.12", test_pivot_calculation)
    tester.add_test("1.13", test_lu_factorizer)
    tester.run()                                                
                       
                       
def testQR(qrclass):
    tester = Tester()
    matrix = np.array([[60, 91, 26], [60, 3, 75], [45, 90, 31]])
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    test_norm = 181.3201588351389
    test_multiplication = np.array([[ 30,  36,  42],
       [ 66,  81,  96],
       [102, 126, 150]])
    test_Q = np.array([[0.62469505,-0.35495981,-0.69552831],[0.62469505,0.76160078,0.17239591],[0.46852129,-0.54218796,0.69750987]])
    test_R = np.array([[96.04686356,100.88825018,77.61835966],[0.,-78.81345682,31.08327672],[0.,0.,16.46876292]])
    ins = "\n A:" + str(A) + ",\nB:" + str(B)
    topic = "Testing QR Factorization "
    
                       
    def test_norm_calculation():
        outs = test_norm
        comment = topic + "Norm Calculation" + "\n Matrix:" + str(matrix) + "\n expected output: " + str(outs)
        qr = qrclass(matrix)
        obtained = qr.Norm_Calculation(matrix)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    def test_matrix_multiplication():
        outs = test_multiplication
        comment = topic + "Matrix Multiplication" + ins + "\n expected output: " + str(outs)
        qr = qrclass(matrix)
        obtained = qr.Matrix_Multiplication(A,B)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    def test_qr_factorizer():
        comment = topic + "QR Factorizer" + "\n Matrix:" + str(matrix) + "\n expected output: " + "\n Q" + str(test_Q) + "\n R" + str(test_R)
        qr = qrclass(matrix)
        obtained_Q, obtained_R = qr.QR_Factorizer()
        comment = comment + "\n obtained: " + "\n Q" + str(obtained_Q) + "\n R" + str(obtained_R)
        if (np.allclose(obtained_Q, test_Q, atol=1e-5) and np.allclose(obtained_R, test_R, atol=1e-5)):
            return True, comment
        return False, comment

    tester.add_test("1.21", test_norm_calculation)
    tester.add_test("1.22", test_matrix_multiplication)
    tester.add_test("1.23", test_qr_factorizer)
    tester.run()                
                       

                       

def testSVD(SVDclass):
    tester = Tester()
    test_matrix = np.array([[1,2],[3,4],[5,6]])
    U = np.array([[-0.2298477,0.88346102,0.40824829],[-0.52474482,0.24078249,-0.81649658],[-0.81964194,-0.40189603,0.40824829]])
    S = np.array([9.52551809,0.51430058])
    VT = np.array([[-0.61962948,-0.78489445],[-0.78489445,0.61962948]])
    shape = [3,2]
    topic = "Testing SVD "
    
    def test_reconstruct_SVD():
        comment = topic + "SVD" + "\n U:" + str(U) + "\n S:" + str(S) + "\n VT:" + str(VT) + "\n expected output: " + "\n Matrix" + str(test_matrix)
        svd = SVDclass(U,S,VT,shape)
        obtained_mat = svd.Reconstruct_SVD()
        comment = comment + "\n obtained: " + "\n Matrix" + str(obtained_mat) 
        if (np.allclose(obtained_mat, test_matrix, atol=1e-5)):
            return True, comment
        return False, comment

    tester.add_test("1.41", test_reconstruct_SVD)
    tester.run()                
                       
def testCholesky(choleskyclass):
    tester = Tester()
    matrix = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    test_L = np.array([[1.41421356,0.70710678,0.70710678],[0.,1.22474487,0.40824829],[0.,0.,1.15470054]])
    topic = "Testing Cholesky Factorization "
    
    def test_cholesky_factorizer():
        comment = topic + "Cholesky Factorizer" + "\n Matrix:" + str(matrix) + "\n expected output: " + "\n L" + str(test_L)
        ch = choleskyclass(matrix)
        obtained_L = ch.Cholesky_Factorizer()
        
        comment = comment + "\n obtained: " + "\n L" + str(obtained_L) 
        if (np.allclose(obtained_L, test_L, atol=1e-5) or np.allclose(obtained_L, np.transpose(test_L), atol=1e-5)):
            return True, comment
        return False, comment

    tester.add_test("1.31", test_cholesky_factorizer)
    tester.run()                
                       










