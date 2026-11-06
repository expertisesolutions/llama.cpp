#!/usr/bin/env python3
"""Compile actual storage methods with model-free types and injected libc I/O faults."""
import pathlib
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).resolve().parents[3]
source = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'tools/server/server-context.cpp'
s = source.read_text()
start = s.index('    static constexpr uint32_t SLOT_CKPT_MAGIC')
end = s.index('    // returns false to decline the task', start)
methods = s[start:end]
preamble = r'''
#include <cstdio>
#include <unistd.h>
#include <fstream>
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <filesystem>
static std::string fault;
static int flushes = 0, closes = 0, syncs = 0;
namespace injected {
FILE * fopen(const char * p, const char * m) { if (fault == "open") return nullptr; return ::fopen(p,m); }
size_t fwrite(const void * p, size_t a, size_t b, FILE * f) {
    if (fault == "write") return 0;
    return ::fwrite(p,a,b,f);
}
int fflush(FILE * f) { ++flushes; int r = ::fflush(f); return fault == "flush" ? EOF : r; }
int fclose(FILE * f) { ++closes; int r = ::fclose(f); return fault == "close" ? EOF : r; }
}
int injected_sync(int fd) { ++syncs; return fault == "sync" ? -1 : ::fsync(fd); }
namespace std { namespace injected = ::injected; }
#define fopen injected::fopen
#define fwrite injected::fwrite
#define fflush injected::fflush
#define fclose injected::fclose
#define fsync injected_sync
#include "src/llama-file-sync.h"
#define SRV_WRN(...) ((void)0)
#define SRV_INF(...) ((void)0)
#define SLT_INF(...) ((void)0)
#define SLT_WRN(...) ((void)0)
constexpr int LLAMA_STATE_SEQ_FLAGS_NONE = 0;
constexpr int LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY = 1;
size_t llama_state_seq_get_size_ext(void *, int, int) { return 1; }
size_t llama_state_seq_get_data_ext(void *, uint8_t * p, size_t n, int, int) { *p = 4; return n; }
size_t llama_state_seq_set_data_ext(void *, const uint8_t *, size_t n, int, int) { return n; }
void * llama_get_memory(void * p) { return p; }
void llama_memory_seq_rm(void *, int, int, int) {}
void common_speculative_get_state(int *, int, std::vector<uint8_t> & data) { data = {5,6}; }
void common_speculative_set_state(int *, int, const std::vector<uint8_t> &) {}
struct common_prompt_checkpoint {
    int id_task = -1; int n_tokens = 0; int pos_min = 0; int pos_max = 0;
    std::vector<uint8_t> data_tgt, data_dft, data_spec;
    bool load_dft(void *, int, int) const { return true; }
    void clear_dft() { data_dft.clear(); }
};
struct server_slot_ckpt_info { int n_tokens, pos_min, pos_max; size_t offset, size; };
struct server_slot { int id = 0; struct { std::list<common_prompt_checkpoint> checkpoints; } prompt; };
struct harness {
    void * ctx_dft = this;
    std::unique_ptr<int> spec;
    struct { int n_ctx_checkpoints = 4; } params_base;
'''
main = r'''
};
#undef fopen
#undef fwrite
#undef fflush
#undef fclose
#undef fsync
int main(int argc, char ** argv) {
    const std::string path = argv[1];
    harness h;
    server_slot slot;
    common_prompt_checkpoint c; c.n_tokens=7; c.pos_min=1; c.pos_max=7;
    c.data_tgt={1,2,3}; c.data_dft={4}; c.data_spec={5,6};
    slot.prompt.checkpoints.push_back(c);
    std::vector<server_slot_ckpt_info> info;
    int failures=0;
    auto check = [&](bool ok, const std::string & name) {
        std::cout << (ok ? "PASS " : "FAIL ") << name << '\n'; failures += !ok;
    };
    size_t bytes = 0;
    for (bool draft : {false, true}) {
        for (const auto & f : {"", "open", "write", "flush", "sync", "close"}) {
            { std::ofstream reset(path, std::ios::binary); }
            fault=f; bool threw=false;
            flushes = closes = syncs = 0;
            try {
                if (draft) { h.save_slot_draft(path, slot, bytes); }
                else { h.save_slot_checkpoints(path, slot, 0, bytes, info); }
            } catch (const std::exception &) { threw=true; }
            const bool expected = !fault.empty() && fault != "sync";
            check(threw == expected && syncs == 0 &&
                  closes == (fault == "open" ? 0 : 1) &&
                  flushes == (fault == "open" || fault == "write" ? 0 : 1),
                  std::string(draft ? "draft/" : "append/") + (fault.empty() ? "success" : fault));
            fault.clear();
            if (!threw) {
                server_slot restored;
                const size_t read = draft ? h.load_slot_draft(path, 0, restored) :
                                            h.load_slot_checkpoints(path, 0, restored, info);
                check(read == bytes && bytes > 0, "visible after close");
            }
        }
    }
    { std::ofstream reset(path, std::ios::binary); }
    check(h.save_slot_checkpoints(path,slot,0,bytes,info), "checkpoint success");
    check(info.size()==1 && info[0].offset==12 && info[0].size==bytes-12, "metadata offsets");
    server_slot loaded;
    check(h.load_slot_checkpoints(path,0,loaded,info)==bytes && loaded.prompt.checkpoints.front().data_tgt==c.data_tgt, "round trip");
    std::ifstream input(path,std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(input)),{}); input.close();
    for (size_t n=4;n<data.size();++n) {
        { std::ofstream out(path,std::ios::binary); out.write(data.data(),n); }
        loaded.prompt.checkpoints.clear();
        check(h.load_slot_checkpoints(path,0,loaded,info)==0 && loaded.prompt.checkpoints.empty(),
              "recognized truncation/"+std::to_string(n));
    }
    { std::ofstream out(path,std::ios::binary); }
    check(h.load_slot_checkpoints(path,0,loaded,info)==0,"legacy/no appendix");
    server_slot empty;
    check(h.save_slot_checkpoints(path,empty,0,bytes,info) && bytes==0,"empty checkpoint success");
    return failures ? 1 : 0;
}
'''
# Compile the production API wrappers and file methods; only model state is synthetic.
context = (root / 'src/llama-context.cpp').read_text()
context_header = (root / 'src/llama-context.h').read_text()
api_preamble = r'''
#include "llama.h"
#include "llama-io.h"
#include "ggml-backend.h"
#include <vector>
#define LLAMA_LOG_ERROR(...) ((void)0)
'''
file_io = context[context.index('class llama_io_write_file :'):context.index('class llama_io_write_device :')]
declarations = context_header[context_header.index('    size_t state_seq_load_file('):context_header.index('    //\n    // perf')]
api_context = r'''
struct llama_context {
    std::vector<uint8_t> state;
    void synchronize() {}
    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id, uint32_t) {
        io.write(state.data(), state.size());
        return io.n_bytes();
    }
    size_t state_seq_read_data(llama_io_read_i & io, llama_seq_id, uint32_t) {
        state.resize(4);
        io.read(state.data(), state.size());
        return io.n_bytes();
    }
'''
file_methods = context[context.index('size_t llama_context::state_seq_load_file('):context.index('size_t llama_context::state_write_data(')]
api_wrappers = context[context.index('size_t llama_state_seq_save_file('):context.index('// compat: llama_batch ->')]
with tempfile.TemporaryDirectory(prefix='checkpoint-test-') as tmp:
    tmp = pathlib.Path(tmp)
    cpp = tmp / 'test.cpp'
    cpp.write_text(preamble + methods + main)
    subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Wno-unused-parameter', '-I', str(root), str(cpp), '-o', str(tmp / 'test')], check=True)
    result = subprocess.run([str(tmp / 'test'), str(tmp / 'state.bin')])
    if result.returncode:
        sys.exit(result.returncode)
    (tmp / 'checkpoint-sequence-api.h').write_text(
        api_preamble + file_io + api_context + declarations + '};\n' + file_methods + api_wrappers)
    subprocess.run([
        'g++', '-std=c++17', '-Wall', '-Wextra', '-ffunction-sections', '-fdata-sections',
        '-I', str(tmp), '-I', str(root / 'src'), '-I', str(root / 'include'),
        '-I', str(root / 'ggml/include'),
        str(root / 'tools/server/tests/checkpoint-base-io.cpp'), str(root / 'src/llama-mmap.cpp'),
        '-L', str(root / 'build-page-cache/bin'), '-Wl,-rpath,' + str(root / 'build-page-cache/bin'),
        '-lllama', '-lggml-base', '-Wl,--gc-sections',
        '-Wl,--wrap=fwrite,--wrap=fflush,--wrap=fclose,--wrap=fsync',
        '-o', str(tmp / 'base-test')], check=True)
    result = subprocess.run([str(tmp / 'base-test'), str(tmp / 'base-state.bin')])
    sys.exit(result.returncode)
