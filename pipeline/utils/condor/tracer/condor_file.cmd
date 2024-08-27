universe        = vanilla

# This defines the path of the executable we want to run.
executable      = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/tracer/condor_instructions.sh 
arguments       = $(Dirname)
output          = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/logging/output_files/$(Dirname).out
error           = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/logging/error_files/$(Dirname).err

log             = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/logging/process_files/$(Dirname).log

request_cpus            = 2
request_memory          = 8G

requirements = (OpSysVer == 2204)

queue