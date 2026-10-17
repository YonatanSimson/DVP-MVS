#include "Checkpoint.h"

#include <fstream>

using namespace boost::filesystem;

namespace {

path CheckpointPath(const path& dense_folder) {
	return dense_folder / path("APD") / path(".progress");
}

} // namespace

int ReadCheckpoint(const path& dense_folder) {
	path checkpoint_path = CheckpointPath(dense_folder);
	std::ifstream file(checkpoint_path.string());
	if (!file.good()) {
		return -1;
	}
	int last_completed_iteration = -1;
	file >> last_completed_iteration;
	if (!file) {
		return -1;
	}
	return last_completed_iteration;
}

void WriteCheckpoint(const path& dense_folder, int iteration_index) {
	path checkpoint_path = CheckpointPath(dense_folder);
	// Write to a temp file and rename over the real one so a crash mid-write
	// (the exact failure mode we're guarding against) can't leave a
	// half-written, unreadable checkpoint behind.
	path tmp_path = checkpoint_path.string() + ".tmp";
	{
		std::ofstream file(tmp_path.string(), std::ios::trunc);
		file << iteration_index;
	}
	rename(tmp_path, checkpoint_path);
}
