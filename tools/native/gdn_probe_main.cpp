#include "fixture_batteries.h"

// Standalone entry point for the gdn fixture battery. The battery body lives
// in gdn_fixture_probe.cpp; this exists so the family keeps its own binary
// and therefore its own SHA, which is what lets it be launched once on its own
// terms under the one-launch-per-SHA rule.

int main() {
    return run_gdn_battery();
}
