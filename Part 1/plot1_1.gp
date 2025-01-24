set terminal pngcairo size 800,600
set output 'performance_plot.png'

set title "Performance in MFLOP/s vs Memory"
set xlabel "Memory (bytes)"
set ylabel "Performance (MFLOP/s)"
set grid
set logscale x

# Define the data file and the columns to use
datafile = 'mnk_omp_cpu_23843589.out'

# Plot the data
plot '< head -n 6 ' . datafile using 1:2 title 'matmult\_mkn\_omp' with linespoints lw 2 lc rgb 'blue', \
     '< tail -n +7 ' . datafile . ' | head -n 6' using 1:2 title 'matmult\_lib' with linespoints lw 2 lc rgb 'red'