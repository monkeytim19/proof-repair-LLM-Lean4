universe        = vanilla

# This defines the path of the executable we want to run.
executable      = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/verifier/condor_instructions.sh 
arguments       = $(Datapath) $(Indexpath) $(Job_num) $(Job_name)
output          = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/logging/output_files/$(Job_name).out
error           = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/logging/error_files/$(Job_name).err

log             = /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/utils/condor/logging/process_files/$(Job_name).log

request_cpus            = 8
request_memory          = 8G

requirements = (OpSysVer == 2204)

queue