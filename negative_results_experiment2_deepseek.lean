-- From theorem MeromorphicOn.pow in Mathlib/Analysis/Analytic/Meromorphic.lean (Reference Theorem & Proof)
lemma pow {f : 𝕜 → 𝕜} {x : 𝕜} (hf : MeromorphicAt f x) (n : ℕ) : MeromorphicAt (f ^ n) x := by
  induction' n with m hm
  · simpa only [Nat.zero_eq, pow_zero] using MeromorphicAt.const 1 x
  · simpa only [pow_succ] using hm.mul hf

-- DeepSeek-Prover-V1.5-SFT repair attempt after fine-tuning on random split
 := by
  induction' n with m hm
  · simpa only [Nat.zero_eq, pow_zero] using MeromorphicAt.const 1 x
  · simpa only [pow_succ'] using hm.mul hf

Repaired Proof:
lemma zpow {f : 𝕜 → 𝕜} {x : 𝕜} (hf : MeromorphicAt f x) (n : ℤ) : MeromorphicAt (f ^ n) x := by
  induction' n with m m
  · simpa only [Int.ofNat_eq_coe, zpow_natCast] using hf.pow m
  · simpa only [zpow_negSucc, inv_iff] using hf.pow (m + 1)

Repaired Proof:
lemma zpow {f : 𝕜 → 𝕜} {x : 𝕜} (hf : MeromorphicAt f x) (n : ℤ) : MeromorphicAt (f ^ n) x := by
  induction' n using Int.induction_on with m _ m _
  · simpa only [zpow_natCast] using hf.pow m
  · simpa only [zpow_negSucc, inv_iff] using hf.pow (m + 1)
  · simpa only [zpow_sub, zpow_negSucc, zpow_natCast, inv_iff] using (hf.pow (m + 1)).inv


-- From theorem Finsupp.univ_sum_single_apply in Mathlib/Algebra/BigOperators/Finsupp.lean (Reference Theorem & Proof)
theorem univ_sum_single_apply [AddCommMonoid M] [Fintype α] (i : α) (m : M) :
    ∑ j : α, single i m j = m := by
  classical rw [single, coe_mk, Finset.sum_pi_single']
  simp

-- DeepSeek-Prover-V1.5-Base repair attempt after fine-tuning on by-file split
↔ m = 0 ∨ i = default := by
  classical rw [Finsupp.single, DFinsupp.single, coe_mk', Finset.sum_pi_single']
  simp [eq_comm]

Repaired Proof:
theorem univ_sum_single_apply [AddCommMonoid M] [Fintype α] (i : α) (m : M) :
    ∑ j : α, single i m j = m ↔ m = 0 ∨ i = default := by
  classical rw [Finsupp.single, DFinsupp.single, coe_mk', Finset.sum_pi_single']
  simp [eq_comm]

Repaired Proof:
theorem univ_sum_single_apply [AddCommMonoid M] [Fintype α] (i : α) (m : M) :
    ∑ j : α, single i m j = m ↔ m = 0 ∨ i = default := by
  classical rw [Finsupp.single, DFinsupp.single, coe_mk', Finset.sum_pi_single']
  simp [eq_comm]
