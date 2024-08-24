import Lean
import Mathlib.Lean.CoreM

open Lean Meta Parser
open Lean.Environment

def getModuleData (filepath : Name) : CoreM (Option ModuleData) := do
  let env ← getEnv
  let moduleIdx := Lean.Environment.getModuleIdx? env filepath
  match moduleIdx with
  | some idx =>
    let moduleData := Array.get! env.header.moduleData idx
    pure moduleData
  | none =>
    pure none
