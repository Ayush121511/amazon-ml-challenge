# Request an interactive GPU job on IIT Delhi HPC.
# Run tmux/screen first so job survives connection drops.
#
# tmux new -s amlc
#
# Then request job:
qsub -I -P scai -q scai_q -N amlc_interactive -lselect=1:ncpus=1:ngpus=1 -lwalltime=4:00:00

# scai_q gives NVIDIA A100 80GB PCIe
# Never run compute directly on login node.
# Verify GPU once job starts: nvidia-smi
