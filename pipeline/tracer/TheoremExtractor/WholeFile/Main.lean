import Lean
import Mathlib.Lean.CoreM
import Batteries.Lean.Util.Path

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


def findTheoremRange (theoremName : Name) : CoreM Unit := do
  let ranges ← findDeclarationRangesCore? theoremName
  match ranges with
  | some hello => IO.println #[hello.range.pos, hello.range.endPos]
  | none => pure ()


def AllTheoremNames (filepath : Name) : CoreM Unit := do
  let moduleData ← getModuleData filepath
  match moduleData with
  | some module =>
    for constinfo in module.constants do
      match constinfo with
      | ConstantInfo.thmInfo thm =>
        let ranges ← findDeclarationRangesCore? thm.name
        match ranges with
        | some info => IO.println s!"{thm.name}&{info.selectionRange.pos}&{info.selectionRange.endPos}"
        | none => pure ()
      | _ =>
        pure ()
  | none =>
    pure ()


def main (path : List String) : IO Unit := do
  searchPathRef.set compile_time_search_path%
  let pathName := (path.get! 0).toName
  CoreM.withImportModules #[pathName] (AllTheoremNames pathName)
