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


def TheoremNamePredicate (name : Name) (info : ConstantInfo) : Bool :=
  match info with
  | ConstantInfo.thmInfo thm =>
    match thm.name with
    | thmName => thmName = name
  | _ => false


def handleTheoremInfo (name : Name) (filepath : Name) (handler : TheoremVal → CoreM Unit) : CoreM Unit := do
  let maybeModuleData ← getModuleData filepath
  match maybeModuleData with
  | some moduleData =>
    match Array.find? moduleData.constants (TheoremNamePredicate name) with
      | some info =>
        match info with
        | ConstantInfo.thmInfo thm => handler thm
        | _ => pure ()
      | _ => pure ()
  | none => pure ()


def moduleTheoremStatement (name : Name) (filepath : Name) : CoreM Unit :=
  handleTheoremInfo name filepath fun thm => IO.println (repr thm.type)


def moduleTheoremProof (name : Name) (filepath : Name) : CoreM Unit :=
  handleTheoremInfo name filepath fun thm => IO.println (repr thm.value)


def findTheoremRange (theoremName : Name) : CoreM Unit := do
  let thmRanges ← findDeclarationRangesCore? theoremName
  match thmRanges with
  | some thmRange => IO.println s!"{thmRange.range.pos}&{thmRange.range.endPos}"
  | none => pure ()


def findTheoremSelectionRange (theoremName : Name) : CoreM Unit := do
  let thmRanges ← findDeclarationRangesCore? theoremName
  match thmRanges with
  | some thmRange => IO.println s!"{thmRange.rangeSelection.pos}&{thmRange.rangeSelection.endPos}"
  | none => pure ()


def main (path : List String) : IO Unit := do
  searchPathRef.set compile_time_search_path%
  let flag := (path.get! 0)
  let theoremName := (path.get! 1).toName
  let pathName := (path.get! 2).toName
  match flag with
  | "-position" =>
    CoreM.withImportModules #[pathName] (findTheoremRange theoremName)
  | "-selection-range" =>
    CoreM.withImportModules #[pathName] (findTheoremSelectionRange theoremName)
  | "-statement" =>
    CoreM.withImportModules #[pathName] (moduleTheoremStatement theoremName pathName)
  | "-proof" =>
    CoreM.withImportModules #[pathName] (moduleTheoremProof theoremName pathName)
  | _ => pure ()
