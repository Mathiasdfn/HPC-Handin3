# plot_matrix.gp: Plot MatrixSize vs vers_map and vers_memcpy

set title "Performance Comparison"
set xlabel "Matrix Size"
set ylabel "Runtime (seconds)"
set grid
set key left top

# Specify output as an image
set terminal pngcairo size 800,600
set output "gpu_cpu_comparison.png"

# Plot the data
plot "all_compare.dat" using 1:2 with linespoints title "CPU version " pt 4 ps 3 linecolor rgb "orange-red", \
     "all_compare.dat" using 1:3 with linespoints title "GPU version using map" pt 3 ps 3 linecolor rgb "dark-pink" \
     "all_compare.dat" using 1:4 with linespoints title "GPU version using memcpy" pt 1 ps 3 linecolor rgb "navy" \
     "all_compare.dat" using 1:5 with linespoints title "GPU version using two GPUs" pt 5 ps 3 linecolor rgb "dark-green"


