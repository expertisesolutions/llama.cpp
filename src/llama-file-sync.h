#pragma once

#include <cstdio>
#include <stdexcept>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

// Close even after a flush or sync failure. Never retry a failed close.
inline void llama_file_sync_close(FILE * file) {
    bool ok = std::fflush(file) == 0;
#ifdef _WIN32
    if (_commit(_fileno(file)) != 0) {
#else
    if (fsync(fileno(file)) != 0) {
#endif
        ok = false;
    }
    if (std::fclose(file) != 0) {
        ok = false;
    }
    if (!ok) {
        throw std::runtime_error("failed to flush, sync or close state file");
    }
}
