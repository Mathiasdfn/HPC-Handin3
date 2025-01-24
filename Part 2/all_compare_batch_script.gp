#BSUB -J all_compare_batch
#BSUB -o all_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"


module load nvhpc/24.11


# Loop over matrix sizes
for size in 100 200 300 400 500 600 700 800 
do
    # Run the map_memcpu_compare_batch.sh script
    echo "Size $size"
    OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0,1 ./poisson $size 1000 -1.0 10.0 0 1 1 1 1
done

# End of file