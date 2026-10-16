#include "llama-mmap.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

static std::string fault;
static int closes = 0;
extern "C" {
size_t __real_fwrite(const void *, size_t, size_t, FILE *);
int __real_fflush(FILE *);
int __real_fclose(FILE *);
int __real_fsync(int);
size_t __wrap_fwrite(const void * p, size_t size, size_t n, FILE * f) {
    return fault == "write" ? 0 : __real_fwrite(p, size, n, f);
}
int __wrap_fflush(FILE * f) {
    const int result = __real_fflush(f);
    return fault == "flush" ? EOF : result;
}
int __wrap_fclose(FILE * f) {
    ++closes;
    const int result = __real_fclose(f);
    return fault == "close" ? EOF : result;
}
int __wrap_fsync(int fd) {
    return fault == "sync" ? -1 : __real_fsync(fd);
}
}

int main(int argc, char ** argv) {
    if (argc != 2) { return 2; }
    int failures = 0;
    for (const auto & mode : {"", "write", "flush", "sync", "close"}) {
        fault = mode;
        closes = 0;
        bool threw = false;
        try {
            llama_file file(argv[1], "wb");
            file.write_u32(1234);
#ifndef BASELINE
            file.sync_close();
#endif
        } catch (const std::exception &) {
            threw = true;
        }
        const bool ok = threw == !fault.empty() && closes == 1;
        std::cout << (ok ? "PASS" : "FAIL") << " base-file/" << (fault.empty() ? "success" : fault)
                  << " closes=" << closes << '\n';
        failures += !ok;
    }
    return failures ? 1 : 0;
}
