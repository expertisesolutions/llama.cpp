#pragma once

// bounded registry of prefix pins (immutable, retained token prefixes)
//
// each pin owns one reserved sequence id from the pool [n_parallel, n_parallel + max_pins), never exposed as a slot
// a child pin extends its parent: it shares the parent's cells, so it accounts only for its own tokens
// capacity refusals never evict another pin
//
// note: this only accounts; it never calls llama (memory operations are done by the server context)
// note: only touched from the server main loop, so it is not thread-safe on purpose
// note: only uses the standard library, so that it can be compiled and tested without a model

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

enum server_prefix_pin_error {
    SERVER_PREFIX_PIN_OK,
    SERVER_PREFIX_PIN_CAPACITY_SEQS,   // no free reserved sequence
    SERVER_PREFIX_PIN_CAPACITY_TOKENS, // the sum of retained tokens would exceed max_tokens
    SERVER_PREFIX_PIN_HAS_CHILDREN,
    SERVER_PREFIX_PIN_IN_USE,
    SERVER_PREFIX_PIN_NOT_FOUND,
    SERVER_PREFIX_PIN_BAD_PARENT,
    SERVER_PREFIX_PIN_EMPTY_TOKENS,
};

struct server_prefix_pin {
    std::string              id;
    std::string              parent;   // empty for a root pin
    std::vector<int32_t>     tokens;   // full prefix: parent tokens followed by own tokens
    size_t                   n_own    = 0; // tokens added beyond the parent
    int32_t                  seq_id   = -1;
    std::vector<std::string> children; // in creation order
    int32_t                  n_in_use = 0; // requests currently holding the pin
    std::vector<float>       lora_scales;  // adapter scales used at prefill, index = adapter id

    size_t n_tokens() const { return tokens.size(); }
    size_t kv_cells() const { return n_own; }
};

struct server_prefix_pin_result {
    server_prefix_pin_error error = SERVER_PREFIX_PIN_OK;
    std::string             id; // set only on success
};

class server_prefix_pins {
public:
    server_prefix_pins(int32_t n_parallel, int32_t max_pins, size_t max_tokens)
        : n_parallel(n_parallel), max_tokens(max_tokens), seq_used(max_pins, false), rng(std::random_device{}()) {}

    // tokens are the tokens added after the parent (the whole prefix for a root pin)
    server_prefix_pin_result create(const std::string & parent, const std::vector<int32_t> & tokens,
                                    const std::vector<float> & lora_scales = {}) {
        if (tokens.empty()) {
            return { SERVER_PREFIX_PIN_EMPTY_TOKENS, "" };
        }
        server_prefix_pin * par = nullptr;
        if (!parent.empty()) {
            par = find(parent);
            if (!par) {
                return { SERVER_PREFIX_PIN_BAD_PARENT, "" };
            }
        }
        const auto free_seq = std::find(seq_used.begin(), seq_used.end(), false);
        if (free_seq == seq_used.end()) {
            return { SERVER_PREFIX_PIN_CAPACITY_SEQS, "" };
        }
        if (used_tokens + tokens.size() > max_tokens) {
            return { SERVER_PREFIX_PIN_CAPACITY_TOKENS, "" };
        }

        server_prefix_pin pin;
        pin.id     = new_id();
        pin.parent = parent;
        if (par) {
            pin.tokens = par->tokens;
            par->children.push_back(pin.id);
        }
        pin.tokens.insert(pin.tokens.end(), tokens.begin(), tokens.end());
        pin.n_own  = tokens.size();
        pin.lora_scales = lora_scales;
        pin.seq_id = n_parallel + (int32_t) (free_seq - seq_used.begin());

        *free_seq    = true;
        used_tokens += pin.n_own;
        pins.push_back(std::move(pin));
        return { SERVER_PREFIX_PIN_OK, pins.back().id };
    }

    server_prefix_pin_error remove(const std::string & id) {
        const auto it = std::find_if(pins.begin(), pins.end(), [&](const server_prefix_pin & p) { return p.id == id; });
        if (it == pins.end()) {
            return SERVER_PREFIX_PIN_NOT_FOUND;
        }
        if (!it->children.empty()) {
            return SERVER_PREFIX_PIN_HAS_CHILDREN;
        }
        if (it->n_in_use > 0) {
            return SERVER_PREFIX_PIN_IN_USE;
        }
        if (server_prefix_pin * par = find(it->parent)) {
            par->children.erase(std::find(par->children.begin(), par->children.end(), id));
        }
        seq_used[it->seq_id - n_parallel] = false;
        used_tokens -= it->n_own;
        pins.erase(it);
        return SERVER_PREFIX_PIN_OK;
    }

    // nullptr if unknown; invalidated by create / remove
    const server_prefix_pin * get(const std::string & id) const {
        for (const auto & p : pins) {
            if (p.id == id) {
                return &p;
            }
        }
        return nullptr;
    }

    // true if the pin tokens are an exact prefix of the prompt tokens
    bool match(const std::vector<int32_t> & prompt, const std::string & id) const {
        const server_prefix_pin * p = get(id);
        return p && prompt.size() >= p->tokens.size() && std::equal(p->tokens.begin(), p->tokens.end(), prompt.begin());
    }

    bool acquire(const std::string & id) {
        server_prefix_pin * p = find(id);
        if (!p) {
            return false;
        }
        p->n_in_use++;
        return true;
    }

    // false if unknown or not held (guards against unbalanced release)
    bool release(const std::string & id) {
        server_prefix_pin * p = find(id);
        if (!p || p->n_in_use == 0) {
            return false;
        }
        p->n_in_use--;
        return true;
    }

    // in creation order
    const std::vector<server_prefix_pin> & list() const { return pins; }

    size_t n_used()        const { return pins.size(); }
    size_t n_used_tokens() const { return used_tokens; }
    size_t n_in_use()      const {
        return (size_t) std::count_if(pins.begin(), pins.end(), [](const server_prefix_pin & p) { return p.n_in_use > 0; });
    }

private:
    int32_t                        n_parallel;
    size_t                         max_tokens;
    size_t                         used_tokens = 0;
    std::vector<bool>              seq_used; // index i is seq_id n_parallel + i
    std::vector<server_prefix_pin> pins;
    std::mt19937_64                rng;

    server_prefix_pin * find(const std::string & id) {
        return const_cast<server_prefix_pin *>(static_cast<const server_prefix_pins *>(this)->get(id));
    }

    std::string new_id() {
        while (true) {
            char buf[19];
            snprintf(buf, sizeof(buf), "p_%016llx", (unsigned long long) rng());
            if (!get(buf)) {
                return buf;
            }
        }
    }
};
