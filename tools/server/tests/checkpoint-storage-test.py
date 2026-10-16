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
static std::string fault;
namespace injected {
FILE * fopen(const char * p, const char * m) { if (fault == "open") return nullptr; return ::fopen(p,m); }
size_t fwrite(const void * p, size_t a, size_t b, FILE * f) {
    if (fault == "write") return 0;
    return ::fwrite(p,a,b,f);
}
int fflush(FILE * f) { int r = ::fflush(f); return fault == "flush" ? EOF : r; }
int fclose(FILE * f) { int r = ::fclose(f); return fault == "close" ? EOF : r; }
}
int injected_sync(int fd) { return fault == "sync" ? -1 : ::fsync(fd); }
namespace std { namespace injected = ::injected; }
#define fopen injected::fopen
#define fwrite injected::fwrite
#define fflush injected::fflush
#define fclose injected::fclose
#define fsync injected_sync
#include "src/llama-file-sync.h"
#define SRV_WRN(...) ((void)0)
#define SRV_INF(...) ((void)0)
struct common_prompt_checkpoint {
    int id_task = -1; int n_tokens = 0; int pos_min = 0; int pos_max = 0;
    std::vector<uint8_t> data_tgt, data_dft, data_spec;
};
struct server_slot_ckpt_info { int n_tokens, pos_min, pos_max; size_t offset, size; };
struct server_slot { struct { std::list<common_prompt_checkpoint> checkpoints; } prompt; };
struct harness {
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
    for (const auto & f : {"open", "write", "flush", "sync", "close"}) {
        { std::ofstream reset(path, std::ios::binary); }
        fault=f; bool threw=false;
        try { h.save_slot_checkpoints(path, slot, 0, info); }
        catch (const std::exception &) { threw=true; }
        fault.clear(); check(threw, std::string("append/")+f);
    }
    { std::ofstream reset(path, std::ios::binary); }
    const size_t bytes=h.save_slot_checkpoints(path,slot,0,info);
    check(info.size()==1 && info[0].offset==12 && info[0].size==bytes-12, "metadata offsets");
    server_slot loaded;
    check(h.load_slot_checkpoints(path,0,loaded,info)==bytes && loaded.prompt.checkpoints.front().data_tgt==c.data_tgt, "round trip");
    std::ifstream input(path,std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(input)),{}); input.close();
    for (size_t n=4;n<data.size();++n) {
        { std::ofstream out(path,std::ios::binary); out.write(data.data(),n); }
        loaded.prompt.checkpoints.clear(); bool threw=false;
        try { h.load_slot_checkpoints(path,0,loaded,info); } catch(const std::exception &) { threw=true; }
        check(threw && loaded.prompt.checkpoints.empty(), "recognized truncation/"+std::to_string(n));
    }
    { std::ofstream out(path,std::ios::binary); }
    check(h.load_slot_checkpoints(path,0,loaded,info)==0,"legacy/no appendix");
    server_slot empty;
    check(h.save_slot_checkpoints(path,empty,0,info)==0,"empty checkpoint success");
    return failures ? 1 : 0;
}
'''
with tempfile.TemporaryDirectory(prefix='checkpoint-test-') as tmp:
    tmp = pathlib.Path(tmp)
    cpp = tmp / 'test.cpp'
    cpp.write_text(preamble + methods + main)
    subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Wno-unused-parameter', '-I', str(root), str(cpp), '-o', str(tmp / 'test')], check=True)
    result = subprocess.run([str(tmp / 'test'), str(tmp / 'state.bin')])
    sys.exit(result.returncode)
