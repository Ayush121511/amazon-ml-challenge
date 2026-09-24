# Request an interactive GPU job on IIT Delhi HPC.
# Run tmux/screen first so job survives connection drops.
#
# tmux new -s amlc
#
# Then request job:
qsub -I -P scai -q standard -N amlc_interactive -lselect=1:ncpus=1:ngpus=1:centos=icelake -lwalltime=4:00:00

# NODE options: icelake, skylake, haswell (default haswell)
# QUEUE options: standard (default), high, scai_q
# Never run compute directly on login node.
