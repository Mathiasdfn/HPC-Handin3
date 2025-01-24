# plot_matrix.gp: Plot MatrixSize vs vers_map and vers_memcpy

set title "Performance Comparison"
set xlabel "Matrix Size"
set ylabel "Runtime (seconds)"
set grid
set key left top
# set logscale y

# Specify output as an image
set terminal pngcairo size 800,600
set output "all_comparison.png"

# Plot the data
plot "all_compare.dat" using 1:2 with linespoints title "CPU version (jacobi) " pt 4 ps 2 linecolor rgb "orange-red", \
     "all_compare.dat" using 1:3 with linespoints title "GPU version using map (jacobi\\_offload\\_map)" pt 3 ps 3 linecolor rgb "dark-pink", \
     "all_compare.dat" using 1:4 with linespoints title "GPU version using memcpy (jacobi\\_offload)" pt 1 ps 3 linecolor rgb "navy", \
     "all_compare.dat" using 1:5 with linespoints title "GPU version using two GPUs (jacobi\\_offload\\_dual)" pt 5 ps 2 linecolor rgb "dark-green"



# plot_matrix.gp: Plot MatrixSize vs vers_map and vers_memcpy

set title "Performance Comparison"
set xlabel "Matrix Size"
set ylabel "Runtime (seconds)"
set grid
set key left top
# set logscale y

# Specify output as an image
set terminal pngcairo size 800,600
set output "all_comparison_w_o_cpu.png"

# Plot the data
plot "all_compare.dat" using 1:3 with linespoints title "GPU version using map (jacobi\\_offload\\_map)" pt 3 ps 3 linecolor rgb "dark-pink", \
     "all_compare.dat" using 1:4 with linespoints title "GPU version using memcpy (jacobi\\_offload)" pt 1 ps 3 linecolor rgb "navy", \
     "all_compare.dat" using 1:5 with linespoints title "GPU version using two GPUs (jacobi\\_offload\\_dual)" pt 5 ps 2 linecolor rgb "dark-green"


# plot_matrix.gp: Plot MatrixSize vs vers_map and vers_memcpy

set title "Performance Comparison"
set xlabel "Matrix Size"
set ylabel "Runtime (seconds) - logarithmic scale"
set grid
set key left top
set logscale y

# Specify output as an image
set terminal pngcairo size 800,600
set output "all_comparison_w_o_cpu_logscale.png"

# Plot the data
plot "all_compare.dat" using 1:3 with linespoints title "GPU version using map (jacobi\\_offload\\_map)" pt 3 ps 3 linecolor rgb "dark-pink", \
     "all_compare.dat" using 1:4 with linespoints title "GPU version using memcpy (jacobi\\_offload)" pt 1 ps 3 linecolor rgb "navy", \
     "all_compare.dat" using 1:5 with linespoints title "GPU version using two GPUs (jacobi\\_offload\\_dual)" pt 5 ps 2 linecolor rgb "dark-green"


