#pragma once

// bounded, thread-safe registry of slot storage operations (save / restore / erase)
//
// a client that lost the HTTP response of a storage request can look up the terminal result here
// the HTTP status endpoint reads only this registry, so it never waits behind a running disk operation
//
// identity:
//   save:            (id_slot, "save",    filename)
//   restore / erase: (id_slot, "restore" | "erase", operation_id)
//
// note: only uses the standard library, so that it can be compiled and tested without a model

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

enum server_storage_op_status {
    SERVER_STORAGE_OP_QUEUED,
    SERVER_STORAGE_OP_RUNNING,
    SERVER_STORAGE_OP_COMPLETED,
    SERVER_STORAGE_OP_FAILED,
};

static inline const char * server_storage_op_status_str(server_storage_op_status status) {
    switch (status) {
        case SERVER_STORAGE_OP_QUEUED:    return "queued";
        case SERVER_STORAGE_OP_RUNNING:   return "running";
        case SERVER_STORAGE_OP_COMPLETED: return "completed";
        case SERVER_STORAGE_OP_FAILED:    return "failed";
    }
    return "unknown";
}

enum server_storage_op_admit {
    SERVER_STORAGE_OP_ADMIT_OK,
    SERVER_STORAGE_OP_ADMIT_CONFLICT, // same identity is queued or running
    SERVER_STORAGE_OP_ADMIT_LIMIT,    // too many outstanding operations on the slot
};

struct server_storage_op_key {
    int         id_slot;
    std::string action;   // "save", "restore" or "erase"
    std::string identity; // filename for save, operation_id for restore / erase
};

struct server_storage_op_record {
    server_storage_op_status status = SERVER_STORAGE_OP_QUEUED;
    std::string result_json; // completed: the original success response, serialized
    std::string error_code;  // failed: short, bounded error code
    int64_t     t_terminal_ms = 0;
    uint64_t    seq_terminal  = 0;
};

struct server_storage_ops {
    // internal constants, not deployment knobs
    static constexpr int64_t RETENTION_MS             = 60 * 60 * 1000; // terminal records are kept for 1 hour
    static constexpr size_t  MAX_TERMINAL_PER_SLOT    = 128;
    static constexpr size_t  MAX_OUTSTANDING_PER_SLOT = 128;

    // replaceable for tests
    std::function<int64_t()> now_ms = [] {
        return (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };

    // records the operation as queued; must be called before the task is posted
    // a terminal record with the same identity is replaced, a queued / running one is a conflict
    server_storage_op_admit admit(const server_storage_op_key & key, server_storage_op_status * active = nullptr) {
        std::lock_guard<std::mutex> lock(mtx);
        slot_ops & s = slots[key.id_slot];
        prune(s);
        auto it = s.records.find({ key.action, key.identity });
        if (it != s.records.end() && is_outstanding(it->second.status)) {
            if (active) {
                *active = it->second.status;
            }
            return SERVER_STORAGE_OP_ADMIT_CONFLICT;
        }
        if (s.n_outstanding >= MAX_OUTSTANDING_PER_SLOT) {
            return SERVER_STORAGE_OP_ADMIT_LIMIT;
        }
        if (it != s.records.end()) {
            s.records.erase(it);
            s.n_terminal--;
        }
        s.records[{ key.action, key.identity }] = server_storage_op_record{};
        s.n_outstanding++;
        return SERVER_STORAGE_OP_ADMIT_OK;
    }

    void set_running(const server_storage_op_key & key) {
        std::lock_guard<std::mutex> lock(mtx);
        server_storage_op_record * rec = find_outstanding(key);
        if (rec) {
            rec->status = SERVER_STORAGE_OP_RUNNING;
        }
    }

    void set_completed(const server_storage_op_key & key, std::string result_json) {
        finish(key, SERVER_STORAGE_OP_COMPLETED, std::move(result_json), "");
    }

    void set_failed(const server_storage_op_key & key, std::string error_code) {
        finish(key, SERVER_STORAGE_OP_FAILED, "", std::move(error_code));
    }

    // nullopt means unknown: never admitted, expired or evicted
    std::optional<server_storage_op_record> get(const server_storage_op_key & key) {
        std::lock_guard<std::mutex> lock(mtx);
        auto its = slots.find(key.id_slot);
        if (its == slots.end()) {
            return std::nullopt;
        }
        prune(its->second);
        auto it = its->second.records.find({ key.action, key.identity });
        if (it == its->second.records.end()) {
            return std::nullopt;
        }
        return it->second;
    }

private:
    struct slot_ops {
        std::map<std::pair<std::string, std::string>, server_storage_op_record> records;
        size_t n_outstanding = 0;
        size_t n_terminal    = 0;
    };

    std::mutex mtx;
    std::map<int, slot_ops> slots;
    uint64_t seq = 0;

    static bool is_outstanding(server_storage_op_status status) {
        return status == SERVER_STORAGE_OP_QUEUED || status == SERVER_STORAGE_OP_RUNNING;
    }

    // caller must hold mtx
    server_storage_op_record * find_outstanding(const server_storage_op_key & key) {
        auto its = slots.find(key.id_slot);
        if (its == slots.end()) {
            return nullptr;
        }
        auto it = its->second.records.find({ key.action, key.identity });
        if (it == its->second.records.end() || !is_outstanding(it->second.status)) {
            return nullptr;
        }
        return &it->second;
    }

    // caller must hold mtx; drops expired terminal records, outstanding records are never dropped
    void prune(slot_ops & s) {
        const int64_t now = now_ms();
        for (auto it = s.records.begin(); it != s.records.end(); ) {
            if (!is_outstanding(it->second.status) && now - it->second.t_terminal_ms >= RETENTION_MS) {
                it = s.records.erase(it);
                s.n_terminal--;
            } else {
                ++it;
            }
        }
    }

    void finish(const server_storage_op_key & key, server_storage_op_status status, std::string result_json, std::string error_code) {
        std::lock_guard<std::mutex> lock(mtx);
        server_storage_op_record * rec = find_outstanding(key);
        if (!rec) {
            return;
        }
        rec->status        = status;
        rec->result_json   = std::move(result_json);
        rec->error_code    = std::move(error_code);
        rec->t_terminal_ms = now_ms();
        rec->seq_terminal  = ++seq;

        slot_ops & s = slots[key.id_slot];
        s.n_outstanding--;
        s.n_terminal++;
        prune(s);
        // evict the oldest terminal records first
        while (s.n_terminal > MAX_TERMINAL_PER_SLOT) {
            auto oldest = s.records.end();
            for (auto it = s.records.begin(); it != s.records.end(); ++it) {
                if (!is_outstanding(it->second.status) &&
                    (oldest == s.records.end() || it->second.seq_terminal < oldest->second.seq_terminal)) {
                    oldest = it;
                }
            }
            s.records.erase(oldest);
            s.n_terminal--;
        }
    }
};
