#Aimport pandas as pd
#
#df = pd.read_csv("/Users/alexanderwood/Documents/TDI/wood-clickbait/data/webis-clickbait-17/tweets-clean.csv")
#print(df.shape)

#print(df.value_counts('username'))
'''
from tdp2 import Hydrator

h = Hydrator()

h.bearer_token='abc'
print(h.header)
ids=[]
for i in range(100):
    ids.append('a{}'.format(i))
h.hydrate_list(ids=ids+['b00']*20, args={'aa':['1', '2', '2'], 'bb':['x', 'y']} )
'''
from pathlib import Path
from math import sqrt
def solve(n):
    total = 1
    counter = 1
    while total < n:
        counter += 1
        total += counter

    if total != n:
        counter = -1


    counter2 = (sqrt(8 * n +1) - 1) / 2

    #if int(counter2) == counter2:
    if total == n:
        return "Go On Bob {}".format(int(counter2))
    else:
        return "Better Luck Next Time"


fname = Path("/Users/alexanderwood/Documents/TDI/wood-clickbait/data/news-outlets/tester.txt")
with open(fname) as f:
    lines = f.readlines()
lines = lines[1:]

fname = Path("/Users/alexanderwood/Documents/TDI/wood-clickbait/data/news-outlets/tester-ans.txt")
with open(fname) as f:
    solutions = f.readlines()
solutions = [sol.rstrip() for sol in solutions]

lines = [int(line) for line in lines]
for i in range(len(lines)):
    soln = solve(lines[i])
    if soln != solutions[i]:
        print(line, sol, solutions[i])
