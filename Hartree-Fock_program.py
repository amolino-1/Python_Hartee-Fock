"""
# Hartree-Fock Equation

FC = ESC

F = Fock Operator
C = Coefficient Matrix (Rows = Basis Functions, Columns = Molecular Orbitals)
E = Energy Eigenvalues of Fock Operator
S = Overlap Matrix

The Fock Matrix is the sum of the Kinetic Energy Matrix (T)
and the Nuclear-Electron Repulsion Matrix (V_NE) and the Electron-Electron
Repulsion Matrix (V_EE)

This Aim is to compute the energy eigenvalues and eigenvectors of H2

"""

import sys
import numpy as np


# Create a primitive gaussian class to access the components of the primitive later
# coeff = contraction coefficient
# coordinates = atomic centres of the Gaussian
# l1, l2, l3 = angular momentum of the Gaussian (for s orbital l1 = l2 = l3 = 0)
# A = normalisation constant


class primitive_gaussian:
    # Initialise the class
    def __init__(self, alpha, coeff, coordinates, l1, l2, l3):
        # Assign the variables
        # Assign the exponent
        self.alpha = alpha
        # Assign the contraction coefficient
        self.coeff = coeff
        # Convert the coordinates to a numpy array
        self.coordnates = np.array(coordinates)
        # Create the normalisation constant
        self.A = (2.0 * alpha/np.pi) ** 0.75  # + other terms if l1, l2, l3 > 0
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3


def overlap_matrix(molecule):
    """
    The Overlap Matrix (S)
    """
    # Get the number of basis functions
    nbasis = len(molecule)
    # Initialise the overlap matrix
    S = np.zeros([nbasis, nbasis])

    # Loop over the basis functions
    for i in range(nbasis):
        for j in range(nbasis):

            # Get the number of primitive gaussians in each basis function
            nprimitive_i = len(molecule[i])
            nprimitive_j = len(molecule[j])

            # Loop over the primitive gaussians
            for k in range(nprimitive_i):
                for l in range(nprimitive_j):

                    # Calculate the overlap matrix element
                    N = molecule[i][k].A * molecule[j][l].A
                    # Gaussian product theorem
                    p = molecule[i][k].alpha + molecule[j][l].alpha
                    q = molecule[i][k].alpha * molecule[j][l].alpha / p
                    Q = molecule[i][k].coordnates - molecule[j][l].coordnates
                    Q2 = np.dot(Q, Q)

                    # Calculate the overlap matrix element
                    S[i, j] += N * molecule[i][k].coeff * \
                        molecule[j][l].coeff * \
                        np.exp(-q * Q2) * (np.pi/p)**(3/2)
    # Return the overlap matrix
    return S


def kinetic(molecule):
    """
    The Kinetic Energy Matrix, T (Rows = Number of Basis Functions, Columns = Number of Basis Functions)
    """
    nbasis = len(molecule)

    T = np.zeros((nbasis, nbasis))

    # Loop over all the rows
    for i in range(nbasis):
        # Loop over all the columns
        for j in range(nbasis):

            nprimitives_i = len(molecule[i])
            nprimitives_j = len(molecule[j])

            for k in range(nprimitives_i):
                for l in range(nprimitives_j):

                    c1c2 = molecule[i][k].coeff * molecule[j][l].coeff

                    # Calculate the overlap matrix element
                    N = molecule[i][k].A * molecule[j][l].A
                    # Gaussian product theorem
                    p = molecule[i][k].alpha + molecule[j][l].alpha
                    q = molecule[i][k].alpha * molecule[j][l].alpha / p
                    Q = molecule[i][k].coordnates - molecule[j][l].coordnates
                    Q2 = np.dot(Q, Q)

                    P = molecule[i][k].alpha * molecule[i][k].coordnates + \
                        molecule[j][l].alpha * molecule[j][l].coordnates
                    Pp = P/p
                    PG = Pp - molecule[j][l].coordnates
                    PGx2 = PG[0] * PG[0]
                    PGy2 = PG[1] * PG[1]
                    PGz2 = PG[2] * PG[2]

                    # Calculate the overlap matrix element
                    s = N * c1c2 * np.exp(-q * Q2) * (np.pi/p)**(3/2)

                    T[i][j] += 3*molecule[j][l].alpha*s
                    T[i][j] -= 2*molecule[j][l].alpha * \
                        molecule[j][l].alpha*s * (PGx2 + 0.5/p)
                    T[i][j] -= 2*molecule[j][l].alpha * \
                        molecule[j][l].alpha*s * (PGy2 + 0.5/p)
                    T[i][j] -= 2*molecule[j][l].alpha * \
                        molecule[j][l].alpha*s * (PGz2 + 0.5/p)

    return T

# -----------------------------------------------------------------------------

# Get basis set from https://www.basissetexchange.org/basis


# STO-3G basis for 1s orbital on Hydrogen
# Create the primitive gaussians for Hydrogen 1 according the the class primitive_gaussian
H1_pg1a = primitive_gaussian(
    0.3425250914E+01, 0.1543289673E+00, [0.0, 0.0, 0.0], 0, 0, 0)
H1_pg1b = primitive_gaussian(
    0.6239137298E+00, 0.5353281423E+00, [0.0, 0.0, 0.0], 0, 0, 0)
H1_pg1c = primitive_gaussian(
    0.1688554040E+00, 0.4446345422E+00, [0.0, 0.0, 0.0], 0, 0, 0)

# Create the primitive gaussians for Hydrogen 2 according the the class primitive_gaussian
H2_pg1a = primitive_gaussian(
    0.3425250914E+01, 0.1543289673E+00, [1.4, 0.0, 0.0], 0, 0, 0)
H2_pg1b = primitive_gaussian(
    0.6239137298E+00, 0.5353281423E+00, [1.4, 0.0, 0.0], 0, 0, 0)
H2_pg1c = primitive_gaussian(
    0.1688554040E+00, 0.4446345422E+00, [1.4, 0.0, 0.0], 0, 0, 0)

# 1s basis functions for Hydrogen 1 is a list of 3 primitive gaussians
H1_1s = [H1_pg1a, H1_pg1b, H1_pg1c]

# 1s basis functions for Hydrogen 2 is a list of 3 primitive gaussians
H2_1s = [H2_pg1a, H2_pg1b, H2_pg1c]

# Create a list of the basis functions for the molecule
# 2 Basis functions for H2 using STO-3G
molecule = [H1_1s, H2_1s]

print("\nSTO-3G Basis for H2:\n")
# Print the overlap matrix
# Diagonal elements should be 1
print("Overlap Matrix (Columns = Orbitals):\n\n", "\tH1_1s\t\tH2_1s\n",
      overlap_matrix(molecule), "\n")
# Print the kinetic energy matrix
print("Kinetic Energy Matrix (Rows and Columns = Number of basis function):\n\n",
      kinetic(molecule), "\n")
# sys.exit(0)


# ------------

# 6-31G basis for 1s AND 2s orbitals on Hydrogen
# Create the primitive gaussians for Hydrogen 1 according the the class primitive_gaussian
# Hydrogen 1s orbital
H1_pg1a = primitive_gaussian(
    0.1873113696E+02, 0.3349460434E-01, [0.0, 0.0, 0.0], 0, 0, 0)
H1_pg1b = primitive_gaussian(
    0.2825394365E+01, 0.2347269535E+00, [0.0, 0.0, 0.0], 0, 0, 0)
H1_pg1c = primitive_gaussian(
    0.6401216923E+00, 0.8137573261E+00, [0.0, 0.0, 0.0], 0, 0, 0)
# Hydrogen 2s orbital
H1_pg2a = primitive_gaussian(
    0.1612777588E+00, 1.00000000000000, [0.0, 0.0, 0.0], 0, 0, 0)

# Create the primitive gaussians for Hydrogen 2 according the the class primitive_gaussian
# Hydrogen 1s orbital
H2_pg1a = primitive_gaussian(
    0.1873113696E+02, 0.3349460434E-01, [1.4, 0.0, 0.0], 0, 0, 0)
H2_pg1b = primitive_gaussian(
    0.2825394365E+01, 0.2347269535E+00, [1.4, 0.0, 0.0], 0, 0, 0)
H2_pg1c = primitive_gaussian(
    0.6401216923E+00, 0.8137573261E+00, [1.4, 0.0, 0.0], 0, 0, 0)
# Hydrogen 2s orbital
H2_pg2a = primitive_gaussian(
    0.1612777588E+00, 1.00000000000000, [1.4, 0.0, 0.0], 0, 0, 0)


# 1s basis functions for Hydrogen 1 is a list of 3 primitive gaussians
H1_1s = [H1_pg1a, H1_pg1b, H1_pg1c]
# 2s basis functions for Hydrogen 1 is a list of 1 primitive gaussians
H1_2s = [H1_pg2a]

# 1s basis functions for Hydrogen 2 is a list of 3 primitive gaussians
H2_1s = [H2_pg1a, H2_pg1b, H2_pg1c]
# 2s basis functions for Hydrogen 2 is a list of 1 primitive gaussians
H2_2s = [H2_pg2a]

# Create a list of the basis functions for the molecule
# 4 Basis functions for H2 using 6-31G
molecule = [H1_1s, H1_2s, H2_1s, H2_2s]

print("\n6-31G Basis for H2:\n")
# Print the overlap matrix
# Diagonal elements should be 1
print("Overlap Matrix (Columns = Orbitals):\n\n", "\tH1_1s\t\tH1_2s\t\tH2_1s\t\tH2_2s\n",
      overlap_matrix(molecule), "\n")
# Print the kinetic energy matrix
print("Kinetic Energy Matrix (Rows and Columns = Number of basis function):\n\n",
      kinetic(molecule), "\n")
