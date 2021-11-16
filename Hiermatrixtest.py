"""Compressed matrix.

Hierarchical matrix implementation.

"""

import numpy as np
import scipy.sparse.linalg as lg
from random import randrange

import string
import random

class CMatrix():
    """CMAtrix class that represents a matrix or a block"""

    def __init__(self, mat,type):
        """
        CMAtrix constructor.

        Args:
            mat (ndarray) : Matrix to compess.
        """

        # Store the number of rows and columns
        self.nr = mat.shape[0]
        self.nc = mat.shape[1]

        self.type = type

        # Make small blocks dense (smaller than the default SVD rank in SciPy)
        if min(self.nr, self.nc) <= 6:
            self.type = 0

        #For the Hierarchical part, we use typ==0 to indicate the dense matrix condition
        #when length no more than 6. typ==3 indicates SVD when the matrix is far from diagonal.
        #typ equals 2,1 and -1 respectively indicates the diagonal, upper and lower blocks.
            
            
        if self.type == 0:
            self.mat = mat

        
        elif self.type == 3:
            u, s, vt = lg.svds(mat) # SVD rank is 6 by default
            self.u  = u
            self.s  = s
            self.vt = vt

       
        elif self.type == 2:
            i = self.nr // 2
            j = self.nc // 2
            self.b11 = CMatrix(mat[:i,:j],2)
            self.b12 = CMatrix(mat[:i,j:],1)
            self.b21 = CMatrix(mat[i:,:j],-1)
            self.b22 = CMatrix(mat[i:,j:],2)

        elif self.type == 1:
            i = self.nr // 2
            j = self.nc // 2
            self.b11 = CMatrix(mat[:i,:j],3)
            self.b12 = CMatrix(mat[:i,j:],3)
            self.b21 = CMatrix(mat[i:,:j],1)
            self.b22 = CMatrix(mat[i:,j:],3)

        elif self.type == -1:
            i = self.nr // 2
            j = self.nc // 2
            self.b11 = CMatrix(mat[:i,:j],3)
            self.b12 = CMatrix(mat[:i,j:],-1)
            self.b21 = CMatrix(mat[i:,:j],3)
            self.b22 = CMatrix(mat[i:,j:],3)
            


    def dot(self,x):
        """
        Matrix-vector multiplication.

        Args:
            x (ndarray) : Initial vector.

        Returns:
            y (ndarray) : Resulting vector.
        """

        # Check if matrix and vector sizes mismatch
        if self.nc != len(x):
            print('Matrix-vector size mismatch')
            sys.exit(1)

        # Dense multiplication
        if self.type == 0:
            y = self.mat.dot(x)

        # Or multiplication using SVD decomposition
        elif self.type == 3:
            sigma = np.diagflat(self.s)  # Form a diagonal matrix from vector S
            y = self.u.dot(sigma.dot(self.vt.dot(x)))

        # Or delegate to sub-blocks and combine pieces
        else:
            j  = self.nc // 2
            y1 = self.b11.dot(x[:j]) + self.b12.dot(x[j:])
            y2 = self.b21.dot(x[:j]) + self.b22.dot(x[j:])
            y  = np.concatenate([y1,y2])

        return y


    def bstr(self):
        """
        Computes a character representation of the matrix.
         - Dense block is shown with a letter
         - Decomposed block is shown with a digit

        Different sub-blocks use different letters and digits
        so that the overall structure can be easily seen.

        Returns:
            s (ndarray) : 2D array of characters, where one character is one element.
        """

        # Return a block filled with the same random letter if dense
        if self.type == 0:
            char = random.choice(string.ascii_letters)
            s = np.full((self.nr, self.nc), char)

        # Or return a block filled with the same random digit if decomposed
        elif self.type == 3:
            digit = random.choice(string.digits)
            s = np.full((self.nr, self.nc), digit)

        # Or delegate to sub-blocks and combine pieces
        else:
            s1 = np.concatenate((self.b11.bstr(), self.b12.bstr()), axis=1)
            s2 = np.concatenate((self.b21.bstr(), self.b22.bstr()), axis=1)
            s  = np.concatenate((s1, s2), axis=0)

        return s


    def __str__(self):
        """
        Returns a string representation of the matrix.

        Useful for debugging.
        """

        s  = ' - Dense block is shown with a letter\n'
        s += ' - Decomposed block is shown with a digit\n\n'
        for i in self.bstr():
            s += ''.join(j for j in i)
            s += '\n'

        return s

    def memory(self):
        """
        Computes the size of memory used.

        Returns:
            k (int) : The number of doubles stored.
        """

        # Return the number of elements if dense
        if self.type == 0:
            k = self.nr * self.nc

        # Or the number of doubles in SVD decomposition
        elif self.type == 3:
            k  = self.u.shape[0] * self.u.shape[1]
            k += self.s.shape[0]
            k += self.vt.shape[0] * self.vt.shape[1]

        # Or sum over sub-blocks
        else:
            k1 = self.b11.memory() + self.b12.memory()
            k2 = self.b21.memory() + self.b22.memory()
            k  = k1 + k2

        return k


    def error(self,mat):
        """
        Computes matrix error.

        Generates a number of random vectors, multiplies by the matrix
        and computes the residual norms. The error is relative and is defined as
        a ratio of the residual norm and the norm of the exact solution. 
        The final error is averaged over all random vectors.

        Args:
            mat (ndarray) : The initial full matrix that is approximated.

        Returns:
            e (double): Error.
        """
    
        count = 100
        e = 0

        for i in range(count):
            x  = np.random.rand(n)
            yd =  mat.dot(x)
            yc = self.dot(x)
            dt = yd - yc
            e += np.linalg.norm(dt) / np.linalg.norm(yd)
    
        e /= count
    
        return e


if __name__ == "__main__":

    # Matrix size
    n = 80

    # Generate a random symmetric matrix
    mat = np.random.rand(n,n)
    mat = (mat + mat.T) / 2
    for i in range(n):
        for j in range(n):
            mat[i][j]=mat[i][j]*10/(abs(i-j)+1)

    print('Given matrix:')
    print(mat)
    print()

    # Generate a random vector
    print('Given vector:')
    x = np.random.rand(n)
    print(x)
    print()

    # Dense matrix vector multiplication
    print('Matrix vector product [Dense]:')
    y = mat.dot(x)
    print(y)
    print()

    # Generate a compressed matrix that approximates the given matrix
    print('CMatrix:')
    cmat = CMatrix(mat,2)
    print(cmat)
    print()

    # Compressed matrix vector multiplication
    print('Matrix vector product [CMatrix]:')
    y = cmat.dot(x)
    print(y)
    print()

    # Format strings for printing
    fs = '{0:10s} {1:10d} {2:10.5f}'
    fd = '{0:10d} {1:10d} {2:10.5f}'

    # Print the information about both matrices
    print('      Name     Memory   RelError')
    kd = n * n
    kc = cmat.memory()
    ed = 0
    ec = cmat.error(mat)
    print(fs.format('Dense', kd, ed))
    print(fs.format('CMatrix', kc, ec))
    print()
    print()


    

