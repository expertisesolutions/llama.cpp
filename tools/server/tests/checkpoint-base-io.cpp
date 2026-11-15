#include "llama-mmap.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

static std::string fault;
static int closes = 0;
static int flushes = 0;
static int syncs = 0;
extern "C" {
size_t __real_fwrite(const void *, size_t, size_t, FILE *);
int __real_fflush(FILE *);
int __real_fclose(FILE *);
int __real_fsync(int);
size_t __wrap_fwrite(const void * p, size_t size, size_t n, FILE * f) {
    return fault == "write" ? 0 : __real_fwrite(p, size, n, f);
}
int __wrap_fflush(FILE * f) {
    ++flushes;
    const int result = __real_fflush(f);
    return fault == "flush" ? EOF : result;
}
int __wrap_fclose(FILE * f) {
    ++closes;
    const int result = __real_fclose(f);
    return fault == "close" ? EOF : result;
}
int __wrap_fsync(int fd) {
    ++syncs;
    return fault == "sync" ? -1 : __real_fsync(fd);
}
}

#include "checkpoint-sequence-api.h"

int main(int argc, char ** argv) {
    if (argc != 2) { return 2; }
    int failures = 0;
    for (bool durable : {false, true}) {
        for (const auto & mode : {"", "write", "flush", "sync", "close"}) {
            fault = mode;
            closes = flushes = syncs = 0;
            bool threw = false;
            try {
                llama_file file(argv[1], "wb");
                file.write_u32(1234);
                if (durable) {
                    file.sync_close();
                } else {
                    file.flush_close();
                }
            } catch (const std::exception &) {
                threw = true;
            }
            const bool expected_throw = !fault.empty() && (durable || fault != "sync");
            bool ok = threw == expected_throw && closes == 1 &&
                      flushes == (fault == "write" ? 0 : 1) &&
                      syncs == (durable && fault != "write" ? 1 : 0);
            if (!threw) {
                llama_file input(argv[1], "rb");
                ok = ok && input.read_u32() == 1234;
            }
            std::cout << (ok ? "PASS" : "FAIL") << " base-file/" << (durable ? "durable/" : "cache/")
                      << (fault.empty() ? "success" : fault) << '\n';
            failures += !ok;
        }
    }
    llama_context ctx;
    const llama_token tokens[] = {17, 23, 42};
    const std::vector<uint8_t> state = {1, 3, 5, 7};
    for (bool durable : {false, true}) {
        for (const auto & mode : {"", "write", "flush", "sync", "close"}) {
            ctx.state = state;
            fault = mode;
            closes = flushes = syncs = 0;
            const size_t saved = durable ? llama_state_seq_save_file(&ctx, argv[1], 0, tokens, 3) :
                                           llama_state_seq_save_file_cache(&ctx, argv[1], 0, tokens, 3);
            const bool expected_failure = !fault.empty() && (durable || fault != "sync");
            bool ok = (saved == 0) == expected_failure && closes == 1 &&
                      flushes == (fault == "write" ? 0 : 1) &&
                      syncs == (durable && fault != "write" ? 1 : 0);
            const std::string name = fault.empty() ? "success" : fault;
            fault.clear();
            if (!expected_failure) {
                ctx.state.clear();
                llama_token restored[3] = {};
                size_t count = 0;
                const size_t loaded = llama_state_seq_load_file(&ctx, argv[1], 0, restored, 3, &count);
                ok = ok && loaded == saved && count == 3 &&
                     std::memcmp(tokens, restored, sizeof(tokens)) == 0 && ctx.state == state;
            }
            std::cout << (ok ? "PASS" : "FAIL") << " sequence-api/" << (durable ? "durable/" : "cache/")
                      << name << '\n';
            failures += !ok;
        }
    }
    return failures ? 1 : 0;
}
