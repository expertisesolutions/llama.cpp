// model-free test of server_prefix_pins accounting
#include "../server-prefix-pins.h"

#include <cctype>
#include <iostream>
#include <set>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool ok, const std::string & name) {
    std::cout << (ok ? "PASS " : "FAIL ") << name << '\n';
    failures += !ok;
}

static bool is_pin_id(const std::string & id) {
    if (id.size() != 18 || id.compare(0, 2, "p_") != 0) {
        return false;
    }
    for (size_t i = 2; i < id.size(); ++i) {
        const char c = id[i];
        if (!(std::isdigit((unsigned char) c) || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

using tokens_t = std::vector<int32_t>;

// get() or an empty pin, so that a missing pin fails the check instead of crashing
static server_prefix_pin at(const server_prefix_pins & pins, const std::string & id) {
    const server_prefix_pin * p = pins.get(id);
    return p ? *p : server_prefix_pin{};
}

int main() {
    // n_parallel = 4, max pins = 2, max tokens = 10
    {
        server_prefix_pins pins(4, 2, 10);
        check(pins.n_used() == 0 && pins.n_used_tokens() == 0 && pins.n_in_use() == 0, "empty registry");

        const auto root = pins.create("", { 1, 2, 3 });
        check(root.error == SERVER_PREFIX_PIN_OK && is_pin_id(root.id), "create root, id format p_<16 hex>");
        const server_prefix_pin * r = pins.get(root.id);
        check(r != nullptr && r->parent.empty() && r->seq_id == 4 && r->tokens == tokens_t({ 1, 2, 3 }), "root fields");
        check(r != nullptr && r->n_tokens() == 3 && r->kv_cells() == 3, "root n_tokens and kv_cells");

        const auto child = pins.create(root.id, { 4, 5 });
        check(child.error == SERVER_PREFIX_PIN_OK && is_pin_id(child.id) && child.id != root.id, "create child, unique id");
        const server_prefix_pin * c = pins.get(child.id);
        check(c != nullptr && c->parent == root.id && c->seq_id == 5, "child fields");
        check(c != nullptr && c->tokens == tokens_t({ 1, 2, 3, 4, 5 }), "child tokens = parent tokens + own tokens");
        check(c != nullptr && c->n_tokens() == 5 && c->kv_cells() == 2, "child kv_cells = own tokens only");
        check(pins.n_used() == 2 && pins.n_used_tokens() == 5, "token sum counts only own tokens");
        check(at(pins, root.id).children == std::vector<std::string>({ child.id }), "parent lists child");

        const auto full = pins.create("", { 9 });
        check(full.error == SERVER_PREFIX_PIN_CAPACITY_SEQS && full.id.empty(), "refuse when no free seq");
        check(pins.n_used() == 2 && pins.n_used_tokens() == 5, "refusal changes nothing");

        check(pins.remove(root.id) == SERVER_PREFIX_PIN_HAS_CHILDREN, "remove with children refused");
        check(pins.get(root.id) != nullptr, "refused remove keeps pin");

        check(pins.acquire(child.id), "acquire");
        check(at(pins, child.id).n_in_use == 1 && pins.n_in_use() == 1, "acquire counted");
        check(pins.acquire(child.id) && at(pins, child.id).n_in_use == 2 && pins.n_in_use() == 1, "acquire twice");
        check(pins.remove(child.id) == SERVER_PREFIX_PIN_IN_USE, "remove in use refused");
        check(pins.release(child.id) && pins.release(child.id), "release twice");
        check(!pins.release(child.id) && at(pins, child.id).n_in_use == 0 && pins.n_in_use() == 0, "release below zero refused");
        check(!pins.acquire("p_0000000000000000") && !pins.release("p_0000000000000000"), "acquire/release unknown");

        check(pins.remove(child.id) == SERVER_PREFIX_PIN_OK && pins.get(child.id) == nullptr, "remove child");
        check(pins.n_used() == 1 && pins.n_used_tokens() == 3, "remove decrements counters");
        check(at(pins, root.id).children.empty(), "remove unlinks from parent");
        check(pins.remove(child.id) == SERVER_PREFIX_PIN_NOT_FOUND, "remove unknown");

        const auto reuse = pins.create("", { 7 });
        check(reuse.error == SERVER_PREFIX_PIN_OK && at(pins, reuse.id).seq_id == 5, "freed seq_id reused");
    }

    // token limit
    {
        server_prefix_pins pins(1, 4, 5);
        const auto a = pins.create("", { 1, 2, 3 });
        check(a.error == SERVER_PREFIX_PIN_OK, "create within token limit");
        check(pins.create("", { 1, 2, 3 }).error == SERVER_PREFIX_PIN_CAPACITY_TOKENS, "refuse by token limit");
        check(pins.create(a.id, { 4, 5, 6 }).error == SERVER_PREFIX_PIN_CAPACITY_TOKENS, "refuse child by token limit");
        const auto b = pins.create(a.id, { 4, 5 });
        check(b.error == SERVER_PREFIX_PIN_OK && pins.n_used_tokens() == 5, "child fills limit exactly (parent not counted again)");
        check(pins.n_used() == 2, "refusals do not consume seqs");
    }

    // input errors
    {
        server_prefix_pins pins(0, 2, 100);
        check(pins.create("", {}).error == SERVER_PREFIX_PIN_EMPTY_TOKENS, "empty tokens refused");
        check(pins.create("p_0123456789abcdef", { 1 }).error == SERVER_PREFIX_PIN_BAD_PARENT, "unknown parent refused");
        const auto a = pins.create("", { 1 });
        check(pins.create(a.id, {}).error == SERVER_PREFIX_PIN_EMPTY_TOKENS, "empty child tokens refused");
        check(at(pins, a.id).seq_id == 0, "pool starts at n_parallel");
        check(pins.n_used() == 1, "input errors change nothing");
    }

    // match
    {
        server_prefix_pins pins(1, 2, 100);
        const auto a = pins.create("", { 1, 2, 3 });
        const auto b = pins.create(a.id, { 4 });
        check(pins.match({ 1, 2, 3, 4, 5 }, a.id), "match exact prefix");
        check(pins.match({ 1, 2, 3, 4, 5 }, b.id), "match child prefix");
        check(pins.match({ 1, 2, 3 }, a.id), "match equal length");
        check(!pins.match({ 1, 2 }, a.id), "no match for shorter prompt");
        check(!pins.match({ 1, 9, 3, 4 }, a.id), "no match on divergence");
        check(!pins.match({ 1, 2, 3, 5, 6 }, b.id), "no match on divergence in child tokens");
        check(!pins.match({ 1, 2, 3, 4 }, "p_0123456789abcdef"), "no match for unknown id");
    }

    // list order and id uniqueness
    {
        server_prefix_pins pins(2, 16, 1000);
        std::vector<std::string> created;
        std::set<std::string> ids;
        for (int i = 0; i < 16; ++i) {
            const auto r = pins.create("", { i });
            created.push_back(r.id);
            ids.insert(r.id);
        }
        check(ids.size() == 16, "16 unique ids");
        pins.remove(created[3]);
        created.erase(created.begin() + 3);
        const auto again = pins.create("", { 99 });
        created.push_back(again.id);
        std::vector<std::string> listed;
        for (const auto & p : pins.list()) {
            listed.push_back(p.id);
        }
        check(listed == created, "list in creation order");
        check(at(pins, again.id).seq_id == 5, "lowest free seq_id reused");
    }

    return failures ? 1 : 0;
}
