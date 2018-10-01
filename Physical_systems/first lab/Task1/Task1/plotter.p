plot "output" u 1:2 title "x" with lp ls 1, \
     "" u 1:3 title "v" with lp ls 2, \
     "" u 1:4 title "x anal." with lp ls 3, \
     "" u 1:5 title "v anal." with lp ls 4, \
     "" u ($1):($2*$2+$3*$3) title "Energy" with p ls 5

pause -1
