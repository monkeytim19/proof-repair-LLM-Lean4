-- tactic mode
theorem sub_add_cancel_tactic {n m : Nat} (h : m ≤ n) : n - m + m = n := by
  rw [Nat.add_comm, Nat.add_sub_of_le h]

-- term mode
theorem sub_add_cancel_term : ∀ {n m : Nat}, m ≤ n → n - m + m = n := fun {n m} h =>
  Eq.mpr (id (congrArg (fun _a => _a = n) (Nat.add_comm (n - m) m)))
    (Eq.mpr (id (congrArg (fun _a => _a = n) (Nat.add_sub_of_le h))) (Eq.refl n))
