#pragma once

// Fixture batteries for the safe-math suite (seventh boundary contract,
// the component execution contract).
//
// Each battery is the unmodified body of the family's original fixture probe,
// exposed as a callable so the suite can run all six in one process — one
// binary, one SHA, one launch. Battery semantics are byte-identical to the
// fast-math-era probes by construction; the archived logs are the comparison
// baseline, so a battery must never be "improved" here.
//
// Return values are each probe's original exit codes, unchanged. The suite
// exits 0 on a clean sweep and 64 + family_index otherwise; it prints the raw
// code beside the family name, because these codes are NOT a namespace and
// cannot be made into one -- two batteries already return values above 99, and
// a process exit is truncated to 8 bits regardless.

int run_kernel_battery();
int run_gdn_battery();
int run_attention_battery();
int run_moe_battery();
int run_head_battery();
int run_composition_battery();
