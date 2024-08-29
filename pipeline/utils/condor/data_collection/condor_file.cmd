universe        = vanilla

# This defines the path of the executable we want to run.
executable      = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/data_collection/condor_instructions.sh 
arguments       = $(Filename)
output          = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/logging/output_files/$(Filename).out
error           = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/logging/error_files/$(Filename).err

log             = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/logging/process_files/$(Filename).log

request_cpus            = 4
request_memory          = 8G

requirements = (OpSysVer == 2204)

queue