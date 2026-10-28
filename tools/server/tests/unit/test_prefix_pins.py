import time

import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()

PIN_TEXT = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her dog"
CHILD_TEXT = " and her best friend Tom"
SUFFIX_1 = ". One day, she saw a big red ball"
SUFFIX_2 = ". The sun was shining and the birds were singing"

GEN = {"temperature": 0.0, "seed": 42, "n_predict": 16, "cache_prompt": True}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.kv_unified = True
    server.n_slots = 2
    server.cache_ram = 0
    server.prefix_pins = 4
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


def tokenize(text):
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": False})
    assert res.status_code == 200
    return res.body["tokens"]


def create_pin(tokens, parent=None):
    data = {"tokens": tokens}
    if parent is not None:
        data["parent"] = parent
    return server.make_request("POST", "/prefixes", data=data)


def complete(prompt, id_slot=0, prefix_id=None, **extra):
    data = {**GEN, "prompt": prompt, "id_slot": id_slot, **extra}
    if prefix_id is not None:
        data["prefix_id"] = prefix_id
    return server.make_request("POST", "/completion", data=data)


def error_type(res):
    return res.body["error"]["type"]


def list_pins():
    res = server.make_request("GET", "/prefixes")
    assert res.status_code == 200
    return {p["prefix_id"]: p for p in res.body}


def baseline_contents(prompts):
    """contents for the same token prompts on a server started without pins"""
    global server
    server.stop()
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.temperature = 0.0
    server.start()
    contents = []
    for prompt in prompts:
        res = complete(prompt, cache_prompt=False)
        assert res.status_code == 200
        contents.append(res.body["content"])
    return contents


def test_create_and_list():
    server.start()
    T = tokenize(PIN_TEXT)
    res = create_pin(T)
    assert res.status_code == 201
    assert res.body["n_tokens"] == len(T)
    assert res.body["kv_cells"] == len(T)
    assert res.body["parent"] is None
    pin = list_pins()[res.body["prefix_id"]]
    assert pin["n_tokens"] == len(T)
    assert pin["kv_cells"] == len(T)
    assert pin["in_use"] is False
    assert pin["children"] == []
    props = server.make_request("GET", "/props").body["prefix_pins"]
    assert props == {"enabled": True, "max": 4, "used": 1, "max_tokens": 8192, "used_tokens": len(T)}


def test_completion_with_pin_matches_no_pin():
    server.start()
    T = tokenize(PIN_TEXT)
    U = tokenize(SUFFIX_1)
    pid = create_pin(T).body["prefix_id"]
    res = complete(T + U, prefix_id=pid)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] >= len(T)
    with_pin = res.body["content"]
    assert baseline_contents([T + U]) == [with_pin]


def test_pin_stays_intact_across_suffixes_and_slots():
    server.start()
    T = tokenize(PIN_TEXT)
    U1 = tokenize(SUFFIX_1)
    U2 = tokenize(SUFFIX_2)
    pid = create_pin(T).body["prefix_id"]
    contents = []
    for id_slot, U in [(0, U1), (0, U2), (0, U1), (1, U1), (0, U2), (1, U2), (0, U1)]:
        res = complete(T + U, id_slot=id_slot, prefix_id=pid)
        assert res.status_code == 200
        assert res.body["timings"]["cache_n"] >= len(T)
        contents.append(res.body["content"])
    first_u1, first_u2 = contents[0], contents[1]
    assert first_u1 != first_u2
    assert contents[2] == first_u1
    assert contents[3] == first_u1
    assert contents[4] == first_u2
    assert contents[5] == first_u2
    assert contents[6] == first_u1


def test_child_pin():
    server.start()
    T = tokenize(PIN_TEXT)
    P = tokenize(CHILD_TEXT)
    U = tokenize(SUFFIX_1)
    parent = create_pin(T).body["prefix_id"]
    res = create_pin(P, parent=parent)
    assert res.status_code == 201
    child = res.body["prefix_id"]
    assert res.body["n_tokens"] == len(T) + len(P)
    assert res.body["kv_cells"] == len(P)
    assert res.body["parent"] == parent
    assert list_pins()[parent]["children"] == [child]

    res = complete(T + P + U, id_slot=1, prefix_id=child)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] >= len(T) + len(P)

    res = server.make_request("DELETE", f"/prefixes/{parent}")
    assert res.status_code == 409
    assert error_type(res) == "prefix_has_children"
    assert server.make_request("DELETE", f"/prefixes/{child}").status_code == 204
    assert server.make_request("DELETE", f"/prefixes/{parent}").status_code == 204
    assert list_pins() == {}


def test_prefix_mismatch():
    server.start()
    T = tokenize(PIN_TEXT)
    U = tokenize(SUFFIX_1)
    pid = create_pin(T).body["prefix_id"]
    other = tokenize(SUFFIX_2)[:4] + T[4:] + U
    res = complete(other, prefix_id=pid)
    assert res.status_code == 400
    assert error_type(res) == "prefix_mismatch"
    assert "content" not in res.body
    assert list_pins()[pid]["in_use"] is False


def test_capacity_seqs():
    server.start()
    T = tokenize(PIN_TEXT)
    U = tokenize(SUFFIX_1)
    pids = []
    for _ in range(4):
        res = create_pin(T)
        assert res.status_code == 201
        pids.append(res.body["prefix_id"])
    res = create_pin(T)
    assert res.status_code == 507
    assert error_type(res) == "prefix_capacity"
    assert set(list_pins()) == set(pids)
    for i, pid in enumerate(pids):
        res = complete(T + U, id_slot=i % 2, prefix_id=pid)
        assert res.status_code == 200
        assert res.body["timings"]["cache_n"] >= len(T)


def test_capacity_tokens():
    server.prefix_pin_max_tokens = 8
    server.start()
    T = tokenize(PIN_TEXT)
    assert len(T) > 8
    res = create_pin(T)
    assert res.status_code == 507
    assert error_type(res) == "prefix_capacity"
    res = create_pin(T[:8])
    assert res.status_code == 201
    res = create_pin(T[:1])
    assert res.status_code == 507
    assert error_type(res) == "prefix_capacity"
    props = server.make_request("GET", "/props").body["prefix_pins"]
    assert props["max_tokens"] == 8
    assert props["used_tokens"] == 8


def test_cancel_releases_pin():
    server.start()
    T = tokenize(PIN_TEXT)
    U = tokenize(SUFFIX_1)
    pid = create_pin(T).body["prefix_id"]
    url = f"http://{server.server_host}:{server.server_port}/completion"
    data = {**GEN, "prompt": T + U, "id_slot": 0, "prefix_id": pid, "n_predict": 400, "stream": True}
    with requests.post(url, json=data, stream=True, timeout=30) as res:
        assert res.status_code == 200
        for line in res.iter_lines():
            if line.startswith(b"data: "):
                break
    # connection closed: the main loop cancels the task and releases the slot (and the pin)
    res = complete(T + U, prefix_id=pid)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] >= len(T)
    deadline = time.time() + 3
    while True:
        res = server.make_request("DELETE", f"/prefixes/{pid}")
        if res.status_code != 409:
            break
        assert error_type(res) == "prefix_in_use"
        assert time.time() < deadline, "pin stayed in use after cancel"
        time.sleep(0.1)
    assert res.status_code == 204


def test_slot_erase_keeps_pin():
    server.start()
    T = tokenize(PIN_TEXT)
    U = tokenize(SUFFIX_1)
    pid = create_pin(T).body["prefix_id"]
    res = complete(T + U, prefix_id=pid)
    assert res.status_code == 200
    first = res.body["content"]
    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200
    pin = list_pins()[pid]
    assert pin["kv_cells"] == len(T)
    res = complete(T + U, prefix_id=pid)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] >= len(T)
    assert res.body["content"] == first


def test_disabled():
    server.prefix_pins = None
    server.kv_unified = False
    server.cache_ram = None
    server.start()
    for method, path in [("GET", "/prefixes"), ("POST", "/prefixes"), ("DELETE", "/prefixes/p_x")]:
        res = server.make_request(method, path, data={"tokens": [1, 2, 3]} if method == "POST" else None)
        assert res.status_code == 503
        assert error_type(res) == "prefix_pins_disabled"
    T = tokenize(PIN_TEXT)
    res = complete(T, prefix_id="p_x")
    assert res.status_code == 503
    assert error_type(res) == "prefix_pins_disabled"
    assert complete(T).status_code == 200
    props = server.make_request("GET", "/props").body["prefix_pins"]
    assert props == {"enabled": False, "max": 0, "used": 0, "max_tokens": 0, "used_tokens": 0}


@pytest.mark.parametrize("setup,message", [
    (lambda s: setattr(s, "enable_ctx_shift", True), "incompatible with --context-shift"),
    (lambda s: setattr(s, "sleep_idle_seconds", 30), "incompatible with --sleep-idle-seconds"),
    (lambda s: setattr(s, "kv_unified", False), "requires --kv-unified"),
    (lambda s: setattr(s, "cache_ram", 8192), "requires --cache-ram 0"),
])
def test_startup_refused(setup, message):
    server.prefix_pins = 2
    server.log_path = "./tmp/prefix_pins_refused.log"
    setup(server)
    with pytest.raises(RuntimeError, match="died with return code"):
        server.start(timeout_seconds=30)
    assert server.process.returncode != 0
    server.stop()
    with open(server.log_path) as f:
        assert message in f.read()


def test_usage_contract():
    server.start()
    T = tokenize(PIN_TEXT)
    U = tokenize(SUFFIX_1)
    pid = create_pin(T).body["prefix_id"]

    res = server.make_request("POST", "/completion", data={**GEN, "prompt": T + U, "prefix_id": pid})
    assert res.status_code == 400
    assert error_type(res) == "prefix_requires_slot"

    res = complete(T + U, prefix_id=pid, cache_prompt=False)
    assert res.status_code == 400
    assert error_type(res) == "prefix_requires_cache_prompt"

    res = complete(T, prefix_id=pid)
    assert res.status_code == 400
    assert error_type(res) == "prefix_prompt_too_short"

    res = complete(T + U, prefix_id="p_unknown")
    assert res.status_code == 404
    assert error_type(res) == "prefix_not_found"

    res = server.make_request("DELETE", "/prefixes/p_unknown")
    assert res.status_code == 404
    assert error_type(res) == "prefix_not_found"

    res = create_pin(U, parent="p_unknown")
    assert res.status_code == 404
    assert error_type(res) == "prefix_not_found"

    res = create_pin([])
    assert res.status_code == 400
    assert error_type(res) == "invalid_tokens"

    assert list_pins()[pid]["in_use"] is False


LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"


def test_pin_bound_to_lora_config():
    global server
    server = ServerPreset.stories15m_moe()
    server.lora_files = [download_file(LORA_FILE_URL)]
    server.kv_unified = True
    server.cache_ram = 0
    server.prefix_pins = 2
    server.start()
    T = tokenize(PIN_TEXT)
    U = tokenize(SUFFIX_1)
    # the pin is prefilled with the global adapter config (scale 1.0)
    pid = create_pin(T).body["prefix_id"]
    assert complete(T + U, prefix_id=pid).status_code == 200

    # per-request scale that differs from the pin
    res = complete(T + U, prefix_id=pid, lora=[{"id": 0, "scale": 0.0}])
    assert res.status_code == 400
    assert error_type(res) == "prefix_lora_mismatch"
    assert list_pins()[pid]["in_use"] is False
    # per-request scale equal to the pin
    assert complete(T + U, prefix_id=pid, lora=[{"id": 0, "scale": 1.0}]).status_code == 200

    # global default changed after the pin was created
    assert server.make_request("POST", "/lora-adapters", data=[{"id": 0, "scale": 0.0}]).status_code == 200
    res = complete(T + U, prefix_id=pid)
    assert res.status_code == 400
    assert error_type(res) == "prefix_lora_mismatch"
    # a child would be prefilled with another config than its parent
    res = create_pin(U, parent=pid)
    assert res.status_code == 400
    assert error_type(res) == "prefix_lora_mismatch"
    # a request without pin is not affected
    assert complete(T + U).status_code == 200


def test_pins_count_in_context_of_every_completion():
    server.start()
    T = tokenize(PIN_TEXT)
    filler = (T * 20)[:100]
    for _ in range(4):
        assert create_pin(filler).status_code == 201
    assert server.make_request("GET", "/props").body["prefix_pins"]["used_tokens"] == 400
    # 512 cells - 400 held by pins: a 150 token prompt fits in the slot context alone but not in the pool
    long_prompt = (T * 20)[:150]
    res = complete(long_prompt, n_predict=4)
    assert res.status_code == 400
    assert error_type(res) == "exceed_context_size_error"
    pid = list(list_pins())[0]
    res = complete(filler + long_prompt, prefix_id=pid, n_predict=4)
    assert res.status_code == 400
    assert error_type(res) == "prefix_context_exceeded"
    # a prompt that fits still works, with and without pin
    assert complete(long_prompt[:40], n_predict=4).status_code == 200
    res = complete(filler + long_prompt[:40], prefix_id=pid, n_predict=4)
    assert res.status_code == 200
