import subprocess
from subprocess import Popen, PIPE, STDOUT

p = subprocess.Popen(['./a.out'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
p.stdin.write('1\n')
p.stdin.write('2\n')
print p.communicate()[0]
p.stdin.close()