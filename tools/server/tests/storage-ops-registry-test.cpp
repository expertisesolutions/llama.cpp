// model-free test of server_storage_ops bounds, built and run by unit/test_storage_operations.py
#include "../server-storage-ops.h"

#include <iostream>
#include <string>

static int failures = 0;

static void check(bool ok, const std::string & name) {
    std::cout << (ok ? "PASS " : "FAIL ") << name << '\n';
    failures += !ok;
}

static server_storage_op_key save_key(int id_slot, const std::string & filename) {
    return { id_slot, "save", filename };
}

int main() {
    int64_t now = 1000;

    {
        server_storage_ops ops;
        ops.now_ms = [&] { return now; };
        const auto k = save_key(0, "a.bin");
        check(!ops.get(k).has_value(), "unknown before admit");
        check(ops.admit(k) == SERVER_STORAGE_OP_ADMIT_OK, "admit");
        check(ops.get(k)->status == SERVER_STORAGE_OP_QUEUED, "queued");
        server_storage_op_status active = SERVER_STORAGE_OP_COMPLETED;
        check(ops.admit(k, &active) == SERVER_STORAGE_OP_ADMIT_CONFLICT && active == SERVER_STORAGE_OP_QUEUED, "conflict while queued");
        ops.set_running(k);
        check(ops.get(k)->status == SERVER_STORAGE_OP_RUNNING, "running");
        check(ops.admit(k, &active) == SERVER_STORAGE_OP_ADMIT_CONFLICT && active == SERVER_STORAGE_OP_RUNNING, "conflict while running");
        ops.set_failed(k, "checkpoint_write_failed");
        check(ops.get(k)->status == SERVER_STORAGE_OP_FAILED && ops.get(k)->error_code == "checkpoint_write_failed", "failed keeps error code");
        ops.set_completed(k, "{}");
        check(ops.get(k)->status == SERVER_STORAGE_OP_FAILED, "terminal state is final");
        check(ops.admit(k) == SERVER_STORAGE_OP_ADMIT_OK && ops.get(k)->status == SERVER_STORAGE_OP_QUEUED, "terminal record replaced");
        ops.set_completed(k, "{\"n_saved\":3}");
        check(ops.get(k)->result_json == "{\"n_saved\":3}", "completed keeps result");
        check(!ops.get({ 0, "restore", "a.bin" }).has_value(), "action is part of the identity");
        check(!ops.get(save_key(1, "a.bin")).has_value(), "slot is part of the identity");
    }

    // retention: 1 hour for terminal records, outstanding records never expire
    {
        server_storage_ops ops;
        ops.now_ms = [&] { return now; };
        const auto done = save_key(0, "done.bin");
        const auto pend = save_key(0, "pending.bin");
        ops.admit(done);
        ops.set_completed(done, "{}");
        ops.admit(pend);
        now += server_storage_ops::RETENTION_MS - 1;
        check(ops.get(done).has_value(), "terminal kept before retention");
        now += 1;
        check(!ops.get(done).has_value(), "terminal expired after retention");
        check(ops.get(pend).has_value() && ops.get(pend)->status == SERVER_STORAGE_OP_QUEUED, "outstanding never expires");
    }

    // at most 128 terminal records per slot, oldest terminal evicted first, outstanding never evicted
    {
        server_storage_ops ops;
        ops.now_ms = [&] { return now; };
        const auto pend = save_key(0, "pending.bin");
        ops.admit(pend);
        for (size_t i = 0; i < server_storage_ops::MAX_TERMINAL_PER_SLOT + 2; ++i) {
            const auto k = save_key(0, "t" + std::to_string(i));
            ops.admit(k);
            ops.set_completed(k, "{}");
        }
        check(!ops.get(save_key(0, "t0")).has_value() && !ops.get(save_key(0, "t1")).has_value(), "oldest terminal evicted");
        check(ops.get(save_key(0, "t2")).has_value(), "newer terminal kept");
        check(ops.get(save_key(0, "t129")).has_value(), "newest terminal kept");
        check(ops.get(pend).has_value(), "outstanding not evicted");
        // other slots are independent
        ops.admit(save_key(1, "x"));
        ops.set_completed(save_key(1, "x"), "{}");
        check(ops.get(save_key(1, "x")).has_value(), "per-slot bound");
    }

    // at most 128 outstanding per slot, rejected before admission
    {
        server_storage_ops ops;
        ops.now_ms = [&] { return now; };
        for (size_t i = 0; i < server_storage_ops::MAX_OUTSTANDING_PER_SLOT; ++i) {
            ops.admit({ 0, "restore", "op" + std::to_string(i) });
        }
        check(ops.admit(save_key(0, "over")) == SERVER_STORAGE_OP_ADMIT_LIMIT, "outstanding limit");
        check(!ops.get(save_key(0, "over")).has_value(), "rejected operation not recorded");
        check(ops.admit(save_key(1, "other")) == SERVER_STORAGE_OP_ADMIT_OK, "limit is per slot");
        ops.set_failed({ 0, "restore", "op0" }, "restore_failed");
        check(ops.admit(save_key(0, "over")) == SERVER_STORAGE_OP_ADMIT_OK, "terminal frees outstanding quota");
    }

    return failures ? 1 : 0;
}
