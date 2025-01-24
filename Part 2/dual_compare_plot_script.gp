# plot_matrix.gp: Plot MatrixSize vs vers_map and vers_memcpy

set title "Performance Comparison"
set xlabel "Matrix Size"
set ylabel "Runtime (seconds)"
set grid
set key left top
set xrange [0:1100]

# Specify output as an image
# set terminal pngcairo size 600,400
set terminal pngcairo enhanced
set output "dual_comparison.png"

# Plot the data
plot "dual_compare.dat" using 1:2 with linespoints title "standard offload" pt 1 ps 2 linecolor rgb "navy" , \
     "dual_compare.dat" using 1:3 with linespoints title "dual offload" pt 3 ps 2 linecolor rgb "dark-green"



# plot_matrix.gp: Plot MatrixSize vs vers_map and vers_memcpy

set title "Performance Comparison"
set xlabel "Matrix Size"
set ylabel "Runtime (seconds)"
set grid
set key left top
set logscale y
set xrange [0:1100]


# Specify output as an image
# set terminal pngcairo size 700,500
set terminal pngcairo enhanced
set output "dual_comparison_logscale.png"

# Plot the data
plot "dual_compare.dat" using 1:2 with linespoints title "standard offload" pt 1 ps 2 linecolor rgb "navy" , \
     "dual_compare.dat" using 1:3 with linespoints title "dual offload" pt 3 ps 2 linecolor rgb "dark-green"
