universe        = vanilla

# This defines the path of the executable we want to run.
executable      = /vol/bitbucket/tcwong/individual_project/proof-repair/batching_scripts/trace_directory/condor_instructions.sh 
arguments       = $(Dirname)
output          = /vol/bitbucket/tcwong/individual_project/proof-repair/logging/output_files/$(Dirname).out
error           = /vol/bitbucket/tcwong/individual_project/proof-repair/logging/error_files/$(Dirname).err

log             = /vol/bitbucket/tcwong/individual_project/proof-repair/logging/process_files/$(Dirname).log

request_cpus            = 2
request_memory          = 8G

requirements = (OpSysVer == 2204)

queue