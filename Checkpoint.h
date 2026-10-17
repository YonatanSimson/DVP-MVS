#ifndef _CHECKPOINT_H_
#define _CHECKPOINT_H_

#include <boost/filesystem.hpp>

// Crash-resume support for main()'s iteration loop.
//
// ProcessProblem() overwrites the same per-problem output files (depths.dmb,
// APD_normals.dmb, ...) on every iteration, so file presence alone can't tell
// us which iteration a given problem last completed -- the only safe resume
// granularity is "the last iteration that finished for every problem". After
// each iteration completes across all problems, main() records that index;
// on the next launch we fast-forward past every iteration already done and
// pick back up from there, re-doing at most the one iteration that was
// interrupted mid-run (e.g. by the WSL GPU-paravirtualization segfault --
// see dxgkio_query_adapter_info in dmesg -- rather than a real APD bug).

// Returns the last iteration index fully completed and recorded, or -1 if
// there is no checkpoint yet (fresh run).
int ReadCheckpoint(const boost::filesystem::path& dense_folder);

// Records that `iteration_index` (and everything before it) is fully done.
void WriteCheckpoint(const boost::filesystem::path& dense_folder, int iteration_index);

#endif // !_CHECKPOINT_H_
