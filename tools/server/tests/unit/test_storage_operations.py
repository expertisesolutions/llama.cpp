import os
import pathlib
import shutil
import subprocess
import tempfile
import threading
import time

import pytest
import requests
from utils import *

# the save / restore handler blocks while <file>.hold exists, and a save fails before the checkpoint
# appendix while <file>.fail-checkpoint exists; both only with LLAMA_SERVER_TEST_STORAGE_HOOKS set
os.environ["LLAMA_SERVER_TEST_STORAGE_HOOKS"] = "1"

SAVE_KEYS = {"id_slot", "filename", "n_saved", "n_written", "n_state_bytes", "checkpoints", "timings"}
RESTORE_KEYS = {"id_slot", "filename", "n_restored", "n_read", "checkpoints", "timings"}
ERASE_KEYS = {"id_slot", "n_erased"}

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


def url(path):
    return f"http://{server.server_host}:{server.server_port}{path}"


def slot_path(filename):
    return os.path.join(server.slot_save_path, filename)


def op_status(slot_id, action, **identity):
    params = {"slot_id": slot_id, "action": action, **identity}
    t0 = time.time()
    res = requests.get(url("/storage-operations"), params=params, timeout=2)
    elapsed = time.time() - t0
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["slot_id"] == slot_id
    assert body["action"] == action
    for k, v in identity.items():
        assert body[k] == v
    return body, elapsed


def wait_status(slot_id, action, expected, timeout=20, **identity):
    deadline = time.time() + timeout
    while True:
        body, _ = op_status(slot_id, action, **identity)
        if body["status"] in expected:
            return body
        assert time.time() < deadline, f"status stayed {body['status']}, expected {expected}"
        time.sleep(0.05)


def fill_slot(id_slot):
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": id_slot,
        "cache_prompt": True,
    })
    assert res.status_code == 200


class Hold:
    def __init__(self, filename):
        self.path = slot_path(filename) + ".hold"

    def __enter__(self):
        pathlib.Path(self.path).touch()
        return self

    def release(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def __exit__(self, *args):
        self.release()


def post_async(path, data, timeout=60):
    out = {}

    def run():
        try:
            r = requests.post(url(path), json=data, timeout=timeout)
            out["status_code"] = r.status_code
            out["body"] = r.json()
        except requests.exceptions.RequestException as e:
            out["exception"] = e

    t = threading.Thread(target=run)
    t.start()
    return t, out


def test_registry_bounds_model_free():
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    src = pathlib.Path(__file__).resolve().parents[1] / "storage-ops-registry-test.cpp"
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "registry-test")
        subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(src), "-o", exe], check=True)
        res = subprocess.run([exe], capture_output=True, text=True)
        print(res.stdout)
        assert res.returncode == 0, res.stdout


def test_unknown_lookup():
    server.start()
    body, _ = op_status(0, "save", filename="never-sent.bin")
    assert body["status"] == "unknown"
    body, _ = op_status(1, "restore", operation_id="never-sent")
    assert body["status"] == "unknown"
    body, _ = op_status(1, "erase", operation_id="never-sent")
    assert body["status"] == "unknown"


def test_lookup_bad_requests():
    server.start()
    for params in [
        {"action": "save", "filename": "a"},
        {"slot_id": "x", "action": "save", "filename": "a"},
        {"slot_id": 0, "action": "bogus", "filename": "a"},
        {"slot_id": 0, "action": "save"},
        {"slot_id": 0, "action": "restore", "filename": "a"},
    ]:
        res = requests.get(url("/storage-operations"), params=params, timeout=2)
        assert res.status_code == 400, params


def test_save_completed_retains_original_response():
    server.start()
    fill_slot(1)
    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "op-save-1.bin"})
    assert res.status_code == 200
    assert set(res.body.keys()) == SAVE_KEYS
    assert res.body["filename"] == "op-save-1.bin"
    assert res.body["n_written"] == os.path.getsize(slot_path("op-save-1.bin"))

    body, _ = op_status(1, "save", filename="op-save-1.bin")
    assert body["status"] == "completed"
    assert body["result"] == res.body
    assert "error" not in body
    assert "operation_id" not in body


def test_restore_and_erase_tracked_by_operation_id():
    server.start()
    fill_slot(1)
    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "op-rs.bin"})
    assert res.status_code == 200
    saved = res.body

    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "op-rs.bin", "operation_id": "restore-0001"})
    assert res.status_code == 200
    assert set(res.body.keys()) == RESTORE_KEYS
    assert res.body["n_restored"] == saved["n_saved"]
    body, _ = op_status(0, "restore", operation_id="restore-0001")
    assert body["status"] == "completed"
    assert body["result"] == res.body
    assert "filename" not in body

    res = server.make_request("POST", "/slots/0?action=erase", data={"operation_id": "erase-0001"})
    assert res.status_code == 200
    assert set(res.body.keys()) == ERASE_KEYS
    body, _ = op_status(0, "erase", operation_id="erase-0001")
    assert body["status"] == "completed"
    assert body["result"] == res.body

    # a failed tracked restore keeps only a bounded error code
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "missing.bin", "operation_id": "restore-0002"})
    assert res.status_code == 400
    body, _ = op_status(0, "restore", operation_id="restore-0002")
    assert body["status"] == "failed"
    assert body["error"] == "restore_failed"
    assert "result" not in body


def test_legacy_requests_unchanged_and_untracked():
    server.server_slots = True
    server.start()
    fill_slot(1)
    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "legacy.bin"})
    assert res.status_code == 200
    assert set(res.body.keys()) == SAVE_KEYS
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "legacy.bin"})
    assert res.status_code == 200
    assert set(res.body.keys()) == RESTORE_KEYS
    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200
    assert set(res.body.keys()) == ERASE_KEYS

    # plain GET /slots keeps its shape
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    assert isinstance(res.body, list) and len(res.body) == 2
    for s in res.body:
        assert "id" in s and "is_processing" in s
        assert "storage_operations" not in s


def test_polling_reads_running_while_save_blocked_then_queued_behind_it():
    server.start()
    fill_slot(1)
    with Hold("blocked.bin") as hold:
        t1, out1 = post_async("/slots/1?action=save", {"filename": "blocked.bin"})
        body = wait_status(1, "save", {"running"}, filename="blocked.bin")
        assert body["status"] == "running"

        # the main loop is now blocked in the save handler, a second operation stays queued
        t2, out2 = post_async("/slots/0?action=erase", {"operation_id": "erase-behind"})
        body = wait_status(0, "erase", {"queued"}, operation_id="erase-behind")

        # status reads stay fast while the disk operation is blocked
        for _ in range(10):
            body, elapsed = op_status(1, "save", filename="blocked.bin")
            assert body["status"] == "running"
            assert elapsed < 0.5
            body, _ = op_status(0, "erase", operation_id="erase-behind")
            assert body["status"] == "queued"
            time.sleep(0.05)

        hold.release()
        t1.join(30)
        t2.join(30)

    assert out1["status_code"] == 200
    body = wait_status(1, "save", {"completed"}, filename="blocked.bin")
    assert body["result"] == out1["body"]
    assert out2["status_code"] == 200
    body = wait_status(0, "erase", {"completed"}, operation_id="erase-behind")
    assert body["result"] == out2["body"]


def test_response_loss_terminal_result_queryable():
    server.start()
    fill_slot(1)
    with Hold("lost.bin") as hold:
        t, out = post_async("/slots/1?action=save", {"filename": "lost.bin"}, timeout=0.5)
        t.join(10)
        assert "exception" in out  # the client gave up, the response is lost
        assert wait_status(1, "save", {"running"}, filename="lost.bin")["status"] == "running"
        # let the server notice the closed connection before the save finishes
        time.sleep(2.5)
        hold.release()
    body = wait_status(1, "save", {"completed", "failed"}, filename="lost.bin")
    assert body["status"] == "completed"
    assert body["result"]["filename"] == "lost.bin"
    assert body["result"]["n_written"] == os.path.getsize(slot_path("lost.bin"))
    assert body["result"]["n_saved"] > 0


def test_response_loss_while_queued_still_runs():
    server.start()
    fill_slot(1)
    with Hold("front.bin") as hold:
        t1, out1 = post_async("/slots/1?action=save", {"filename": "front.bin"})
        wait_status(1, "save", {"running"}, filename="front.bin")
        t2, out2 = post_async("/slots/0?action=erase", {"operation_id": "erase-lost"}, timeout=0.5)
        t2.join(10)
        assert "exception" in out2
        time.sleep(2.5)
        assert op_status(0, "erase", operation_id="erase-lost")[0]["status"] == "queued"
        hold.release()
        t1.join(30)
    body = wait_status(0, "erase", {"completed", "failed"}, operation_id="erase-lost")
    assert body["status"] == "completed"
    assert set(body["result"].keys()) == ERASE_KEYS


def test_active_duplicate_rejected_409_without_enqueue():
    server.start()
    fill_slot(1)
    with Hold("dup.bin") as hold:
        t1, out1 = post_async("/slots/1?action=save", {"filename": "dup.bin"})
        wait_status(1, "save", {"running"}, filename="dup.bin")
        res = requests.post(url("/slots/1?action=save"), json={"filename": "dup.bin"}, timeout=5)
        assert res.status_code == 409
        assert res.json()["error"]["type"] == "conflict_error"
        assert res.json()["error"]["operation_status"] == "running"
        # the same filename on another slot is a different identity (slot 3 wraps to slot 1)
        res = requests.post(url("/slots/3?action=save"), json={"filename": "dup.bin"}, timeout=5)
        assert res.status_code == 409

        t2, out2 = post_async("/slots/0?action=restore", {"filename": "dup.bin", "operation_id": "rs-dup"})
        wait_status(0, "restore", {"queued"}, operation_id="rs-dup")
        res = requests.post(url("/slots/0?action=restore"), json={"filename": "dup.bin", "operation_id": "rs-dup"}, timeout=5)
        assert res.status_code == 409
        assert res.json()["error"]["operation_status"] == "queued"
        # erase is a different action, a different identity: accepted and queued
        t3, out3 = post_async("/slots/0?action=erase", {"operation_id": "rs-dup"})
        wait_status(0, "erase", {"queued"}, operation_id="rs-dup")
        hold.release()
        t1.join(30)
        t2.join(30)
        t3.join(30)
    assert out1["status_code"] == 200
    assert out2["status_code"] == 200
    assert out3["status_code"] == 200
    assert wait_status(1, "save", {"completed"}, filename="dup.bin")["result"] == out1["body"]


def test_failed_checkpoint_append_is_failed_then_old_router_repeat_replaces():
    server.start()
    fill_slot(1)
    marker = slot_path("ckfail.bin") + ".fail-checkpoint"
    pathlib.Path(marker).touch()
    try:
        res = server.make_request("POST", "/slots/1?action=save", data={"filename": "ckfail.bin"})
        assert res.status_code == 500
        # the state was written before the checkpoint appendix failed
        assert os.path.getsize(slot_path("ckfail.bin")) > 0
        body, _ = op_status(1, "save", filename="ckfail.bin")
        assert body["status"] == "failed"
        assert body["error"] == "checkpoint_write_failed"
        assert "result" not in body
    finally:
        os.remove(marker)

    # old router: repeats the same filename after a failure; the terminal record is replaced
    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "ckfail.bin"})
    assert res.status_code == 200
    assert set(res.body.keys()) == SAVE_KEYS
    body, _ = op_status(1, "save", filename="ckfail.bin")
    assert body["status"] == "completed"
    assert body["result"] == res.body

    # repeating a completed save replaces it as well
    res2 = server.make_request("POST", "/slots/1?action=save", data={"filename": "ckfail.bin"})
    assert res2.status_code == 200
    body, _ = op_status(1, "save", filename="ckfail.bin")
    assert body["status"] == "completed"
    assert body["result"] == res2.body


def test_outstanding_limit_429_before_enqueue():
    server.start()
    fill_slot(1)
    with Hold("limit.bin") as hold:
        threads = []
        threads.append(post_async("/slots/1?action=save", {"filename": "limit.bin"}))
        wait_status(1, "save", {"running"}, filename="limit.bin")
        # legacy restores are untracked and do not use the quota
        for i in range(5):
            threads.append(post_async("/slots/1?action=restore", {"filename": f"legacy-{i}.bin"}))
        for i in range(127):
            threads.append(post_async("/slots/1?action=restore", {"filename": "nothing.bin", "operation_id": f"lim-{i:03d}"}))
        for i in range(127):
            wait_status(1, "restore", {"queued"}, operation_id=f"lim-{i:03d}")

        res = requests.post(url("/slots/1?action=restore"), json={"filename": "nothing.bin", "operation_id": "lim-over"}, timeout=5)
        assert res.status_code == 429
        assert res.json()["error"]["type"] == "rate_limit_error"
        assert op_status(1, "restore", operation_id="lim-over")[0]["status"] == "unknown"
        res = requests.post(url("/slots/1?action=save"), json={"filename": "limit-over.bin"}, timeout=5)
        assert res.status_code == 429
        assert op_status(1, "save", filename="limit-over.bin")[0]["status"] == "unknown"

        # the limit is per slot
        t, out = post_async("/slots/0?action=erase", {"operation_id": "other-slot"})
        threads.append((t, out))
        wait_status(0, "erase", {"queued"}, operation_id="other-slot")

        hold.release()
        for t, _ in threads:
            t.join(60)

    for i in range(127):
        body = wait_status(1, "restore", {"completed", "failed"}, operation_id=f"lim-{i:03d}")
        assert body["status"] == "failed"
        assert body["error"] == "restore_failed"
    # the quota is free again
    res = requests.post(url("/slots/1?action=erase"), json={"operation_id": "after-limit"}, timeout=10)
    assert res.status_code == 200


def test_terminal_history_bounded_per_slot():
    server.start()
    for i in range(130):
        res = requests.post(url("/slots/1?action=erase"), json={"operation_id": f"hist-{i:03d}"}, timeout=10)
        assert res.status_code == 200
    assert op_status(1, "erase", operation_id="hist-000")[0]["status"] == "unknown"
    assert op_status(1, "erase", operation_id="hist-001")[0]["status"] == "unknown"
    assert op_status(1, "erase", operation_id="hist-002")[0]["status"] == "completed"
    assert op_status(1, "erase", operation_id="hist-129")[0]["status"] == "completed"


def test_no_epoch_and_unknown_after_restart():
    server.start()
    fill_slot(1)
    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "restart.bin"})
    assert res.status_code == 200
    body, _ = op_status(1, "save", filename="restart.bin")
    assert body["status"] == "completed"
    assert set(body.keys()) == {"slot_id", "action", "filename", "status", "result"}

    server.stop()
    server.start()
    body, _ = op_status(1, "save", filename="restart.bin")
    assert body["status"] == "unknown"
    assert set(body.keys()) == {"slot_id", "action", "filename", "status"}


def test_invalid_operation_id_rejected():
    server.start()
    for op in ["", 5, "x" * 256]:
        res = requests.post(url("/slots/0?action=erase"), json={"operation_id": op}, timeout=5)
        assert res.status_code == 400, op
