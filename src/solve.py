from numpy import *
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def solve (mat, a0,b0,a1,b1):
    mat = array(mat)
    print "NO KNOWLEDGE"
    (s0,v0) = solve_game_matrix_cvxopt (mat)
    print "Player 0:"
    print s0.transpose()
    print "Value:", v0
    (s1,v1) = solve_game_matrix_cvxopt (-mat.transpose())
    print "player 1:"
    print s1.transpose()
    print "Value:", v1
    print "P0 KNOWS B1 BEFORE CHOOSING B0"
    (s0,v0) = solve_for_player0_with_b1_known (mat, a0, b0, a1, b1)
    print "Player 0:"
    print s0.transpose()
    print "Value:", v0
    (s1,v1) = solve_for_player0_with_b0_known (-mat.transpose(), a1, b1, a0, b0)
    print "player 1:"
    print s1.transpose()
    print "Value:", v1
    print "P1 KNOWS B0 BEFORE CHOOSING B1"
    (s0,v0) = solve_for_player0_with_b0_known (mat, a0, b0, a1, b1)
    print "Player 0:"
    print s0.transpose()
    print "Value:", v0
    (s1,v1) = solve_for_player0_with_b1_known (-mat.transpose(), a1, b1, a0, b0)
    print "player 1:"
    print s1.transpose()
    print "Value:", v1

def solve_game_matrix (mat):
    # make matrix positive
    min_mat = float(mat.min())
    mat = mat - (min_mat - 1)
    n_var = mat.shape[0]
    # target is simple sum
    c = matrix([1.0 for i in range(n_var)])
    # multiply matrix inequalities by -1 to make them [lesser then]
    # and add single variable positivity constraints (again, x(-1) )
    G = [[0.0 for i in range(n_var)] + list(row)  for row in mat]
    for i in range(n_var):
        G[i][i] = 1.0
    G = -matrix(G)
##    print "G rank: ", linalg.matrix_rank(array(G))
    h = matrix([0.0 for i in range(n_var)] + [-1.0 for i in range(mat.shape[1])])
    sol = solvers.lp (c,G,h)
    solution = array(sol['x']).transpose()[0]
    objective = sol['primal objective']
    solution = solution / objective # normalize solution
    value = 1.0 / objective + (min_mat - 1)
    return (solution, value)

# each player's strategy is made of decision A and decision B
# player 0 learns of player 1's decision A before making his own decision B
# a0,b0,a1,b1 are sizes of decision sets for each stage for each player
        
def solve_for_player0_with_b1_known (mat, a0, b0, a1, b1):
    # make matrix positive
    min_mat = float(mat.min())
    mat = mat - (min_mat - 1)
    # player 0 has b1*a0*b0 + a0 variables:
    # for each opponent decision b1, a0*b0 distribution
    # and a0 sum-variables for a-priori a0 distribution

    # target is simple sum of last a variables
    c = matrix ([0.0 for i in range (b1*a0*b0)] + \
                [1.0 for i in range (a0)])
    # a1*a0*b0 single variable positivity constraints
    # a1*b1 mat-based constraints.  each set of b1 constraints only affects
    # the variables corresponding to its a1
    G = [[0.0 for j in range (b1*a0*b0+a1*b1)] for i in range (b1*a0*b0+a0)]
    for i in range (b1*a0*b0):
        G[i][i] = 1.0
    for a1i in range (a1):
        for b1i in range (b1):
            for ab0i in range (a0*b0):
                G[b1i*a0*b0+ab0i][b1*a0*b0+a1i*b1+b1i] = \
                    mat[ab0i][a1i*b1+b1i]
    G = -matrix (G)
##    print "G rank: ", linalg.matrix_rank(array(G))
    h = matrix([0.0 for i in range(b1*a0*b0)] + [-1.0 for i in range(a1*b1)])
    # equality constraints:
    # for each a0,b1: sum of vars over b0 is equal to a0 apriori var
    A = [[0.0 for j in range (a0*b1)] for i in range (b1*a0*b0+a0)]
    for b1a0i in range(b1*a0):
        for b0i in range(b0):
            A[b1a0i*b0+b0i][b1a0i] = 1.0
    for b1i in range(b1):
        for a0i in range(a0):
            A[b1*a0*b0+a0i][b1i*a0+a0i] = -1.0
    A = matrix (A)
##    print "A rank: ", linalg.matrix_rank(array(A))
##    print "AG rank: ", linalg.matrix_rank(array(matrix([A,G])))
    b = matrix ([0.0 for i in range (b1*a0)])
    sol = solvers.lp (c,G,h,A,b)
    solution = array(sol['x']).transpose()[0]
    objective = sol['primal objective']
    solution = solution / objective # normalize solution
    value = 1.0 / objective + (min_mat - 1)
    return (solution, value)

# same, but player 1 learns a0 
def solve_for_player0_with_b0_known (mat, a0, b0, a1, b1):
    # make matrix positive
    min_mat = float(mat.min())
    mat = mat - (min_mat - 1)
    # player 0 has a0*b0 + b0*a1 variables
    # a0*b0 vars for distribution
    # b0*a1 auxiliary variables

    # target is simple sum of distribution variables
    c = matrix ([1.0 for i in range (a0*b0)] + \
                [0.0 for i in range (b0*a1)])
                        
    # a0*b0 single variable positivity constraints
    # a1 constraints of type: sum of auxiliary variable > 1.0
    # a0*b0*a1 mat based constraints between real and auxiliary variables
    G = [[0.0 for j in range (a0*b0+a1+b0*a1*b1)] for i in range (a0*b0+b0*a1)]
    for i in range (a0*b0):
        G[i][i] = 1.0
    for b0i in range(b0):
        for a1i in range(a1):
            G[a0*b0+b0i*a1+a1i][a0*b0+a1i] = 1.0
    for a0i in range(a0):
        for b0i in range(b0):
            for ab1i in range(a1*b1):
                G[a0i*b0+b0i][a0*b0+a1+b0i*a1*b1+ab1i] = mat [a0i*b0+b0i][ab1i]
    for b0a1i in range(b0*a1):
        for b1i in range(b1):
            G[a0*b0+b0a1i][a0*b0+a1+b0a1i*b1+b1i] = -1.0
    G = -matrix(G)
##    print "G rank: ", linalg.matrix_rank(array(G))
    h = matrix ([0.0 for i in range(a0*b0)] + \
                [-1.0 for i in range(a1)] + \
                [0.0 for i in range(b0*a1*b1)])
    sol = solvers.lp (c,G,h)
    solution = array(sol['x']).transpose()[0]
    solution = solution [0:a0*b0]
    objective = sol['primal objective']
    solution = solution / objective # normalize solution
    value = 1.0 / objective + (min_mat - 1)
    return (solution, value)
