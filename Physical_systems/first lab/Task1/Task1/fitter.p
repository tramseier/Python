f1(x) = a*x+b
fit f1(x) "output" using ($1):($2*$2+$3*$3) via a, b

plot "" u 1:2 with lp ls 1, \
  "" u 1:3 with lp ls 2, \
  "" u ($1):($2*$2+$3*$3) w p ls 3, \
  a*x+b

pause -1
