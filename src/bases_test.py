import numpy
import solve

def percentify (fraction):
    if fraction > 0.05:
        return str (int (fraction*100+.5)) + '%'
    else:
        return str(int(fraction*1000+.5)/10.0)+'%'
    
bases = ['Strike',
         'Shot',
         'Drive',
         'Burst',
         'Grasp',
         'Counter',
         'Wave',
         'Force',
         'Spike',
         'Throw']

results = [ [0,4,1,-3,-2,-1,-3,0,0,-2],
            [-4,0,-3,3,1,-2,-3,0,-3,1],
            [-1,3,0,3,-2,-2,0,3,3,-2],
            [3,-3,-3,0,3,0,-3,3,3,3],
            [2,-1,2,-3,0,-3,2,0,-3,0],
            [1,2,2,0,3,0,2,-2,-2,3],
            [3,3,0,3,-2,-2,0,3,3,-2],
            [0,0,-3,-3,0,2,-3,0,3,0],
            [0,3,-3,-3,3,2,-3,-3,0,3],
            [2,-1,2,-3,0,-3,2,0,-3,0]]

mat = numpy.array(results)
(mix, value) = solve.solve_game_matrix(mat)
mix = [percentify(m) for m in mix]
print "ALL", value
for bm in zip(bases,mix):
    print bm
    
mat = numpy.array([row[:5] for row in results[:5]])
(mix, value) = solve.solve_game_matrix(mat)
mix = [percentify(m) for m in mix]
print "ALPHA", value
for bm in zip(bases[:5],mix):
    print bm

mat = numpy.array([row[5:] for row in results[5:]])
(mix, value) = solve.solve_game_matrix(mat)
mix = [percentify(m) for m in mix]
print "BETA", value
for bm in zip(bases[5:],mix):
    print bm

mat = numpy.array([row[5:] for row in results[:5]])
(mix, value) = solve.solve_game_matrix(mat)
mix = [percentify(m) for m in mix]
print "ALPHA vs BETA", value
for bm in zip(bases[:5],mix):
    print bm

mat = numpy.array([row[:5] for row in results[5:]])
(mix, value) = solve.solve_game_matrix(mat)
mix = [percentify(m) for m in mix]
print "BETA vs ALPHA", value
for bm in zip(bases[5:],mix):
    print bm

