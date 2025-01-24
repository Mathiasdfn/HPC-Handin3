# plot_matrix.gp: Plot MatrixSize vs vers_map and vers_memcpy

set title "Performance Comparison"
set xlabel "Matrix Size"
set ylabel "Runtime (seconds)"
set grid
set key left top

# Specify output as an image
set terminal pngcairo size 800,600
set output "map_memcpy_comparison.png"

# Plot the data
plot "map_memcpy_compare.dat" using 1:2 with linespoints title "version using map" pt 3 ps 3 linecolor rgb "dark-pink", \
     "map_memcpy_compare.dat" using 1:3 with linespoints title "version using memcpy" pt 1 ps 3 linecolor rgb "navy"
