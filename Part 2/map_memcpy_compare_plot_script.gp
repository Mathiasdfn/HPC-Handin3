# plot_matrix.gp: Plot MatrixSize vs vers_map and vers_memcpy

set title "Performance Comparison"
set xlabel "Matrix Size"
set ylabel "Time (seconds)"
set grid
set key left top

# Specify output as an image
set terminal pngcairo size 800,600
set output "map_memcpy_comparison.png"

# Plot the data
plot "map_memcpy_compare.dat" using 1:2 with linespoints title "vers\\_map", \
     "map_memcpy_compare.dat" using 1:3 with linespoints title "vers\\_memcpy"
