"""Production test suite for ReaDox PDF AI"""
import requests, json, time, sys

BASE = "http://localhost:8000"


def test_health():
    print("=" * 60)
    print("HEALTH CHECK")
    print("=" * 60)
    try:
        r = requests.get(f"{BASE}/api/health")
        d = r.json()
        print(f"Status: {r.status_code}")
        print(f"Model: {d.get('qwen2vl_model', '?')}")
        print(f"GPU: {d.get('gpu_available', '?')}")
        print(f"VRAM: {d.get('gpu_memory_mb', '?')} MB")
        print()
        return r.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        print("Is the backend running?")
        return False


def test_checkboxes():
    print("=" * 60)
    print("CHECKBOX TEST: Checkbox-Field1.pdf")
    print("=" * 60)
    t = time.time()
    with open("Form/Checkbox-Field1.pdf", "rb") as f:
        r = requests.post(f"{BASE}/api/detect-checkboxes",
            files={"file": ("Checkbox-Field1.pdf", f, "application/pdf")})
    elapsed = round(time.time() - t, 1)
    d = r.json()
    print(f"Status: {r.status_code} | Time: {elapsed}s")
    cbs = d.get("checkboxes", [])
    print(f"Checkboxes found: {len(cbs)}")
    for cb in cbs:
        icon = "V" if cb.get("checked") else " "
        conf = cb.get("confidence", 0)
        label = cb.get("label", "?")
        print(f"  [{icon}] {label} (conf: {conf})")
    print()
    return len(cbs)


def test_table_extraction():
    print("=" * 60)
    print("TABLE TEST: Valuation.pdf")
    print("=" * 60)
    fields = [
        {"key": "Company Name", "question": "What is the Company Name?"},
        {"key": "Total Value", "question": "What is the Total Value?"},
        {"key": "Date", "question": "What is the Date?"},
        {"key": "Valuation Method", "question": "What is the Valuation Method?"},
    ]
    t = time.time()
    with open("Form/Valuation.pdf", "rb") as f:
        r = requests.post(f"{BASE}/api/extract",
            files={"file": ("Valuation.pdf", f, "application/pdf")},
            data={"fields": json.dumps(fields), "model": "qwen"})
    elapsed = round(time.time() - t, 1)
    d = r.json()
    print(f"Status: {r.status_code} | Time: {elapsed}s")
    if d.get("success"):
        for k, v in d["data"].items():
            if k == "_meta":
                meta = v
                sigs = meta.get("signals", {})
                for sk, sig in sigs.items():
                    stype = sig.get("type", "")
                    source = sig.get("source", "")
                    flags = sig.get("flags", [])
                    print(f"  [SIG] {sk}: source={source} type={stype} flags={flags}")
            else:
                print(f"  {k}: {str(v)[:200]}")
    else:
        print(f"Error: {d}")
    print()


def test_docqa():
    print("=" * 60)
    print("DOC Q&A TEST: CPL Form 2.pdf")
    print("=" * 60)
    questions = [
        "What is the patient name?",
        "What insurance does this patient have?",
        "What are all the diagnosis codes listed?",
        "Is this a fasting specimen?",
        "Summarize this document in 3 sentences.",
    ]
    for q in questions:
        t = time.time()
        with open("Form/CPL Form 2.pdf", "rb") as f:
            r = requests.post(f"{BASE}/api/ask",
                files={"file": ("CPL Form 2.pdf", f, "application/pdf")},
                data={"question": q, "model": "qwen"})
        elapsed = round(time.time() - t, 1)
        d = r.json()
        ans = d.get("answer", d.get("detail", "ERROR"))
        print(f"Q: {q}")
        print(f"A: {ans[:300]}")
        print(f"   ({elapsed}s)")
        print()


def test_investment_form():
    print("=" * 60)
    print("INVESTMENT FORM TEST: investment.pdf")
    print("=" * 60)
    fields = [
        {"key": "Investor Name", "question": "What is the Investor Name?"},
        {"key": "Investment Amount", "question": "What is the Investment Amount?"},
        {"key": "Account Number", "question": "What is the Account Number?"},
        {"key": "Date", "question": "What is the Date?"},
        {"key": "Fund Name", "question": "What is the Fund Name?"},
    ]
    t = time.time()
    with open("Form/investment.pdf", "rb") as f:
        r = requests.post(f"{BASE}/api/extract",
            files={"file": ("investment.pdf", f, "application/pdf")},
            data={"fields": json.dumps(fields), "model": "qwen"})
    elapsed = round(time.time() - t, 1)
    d = r.json()
    print(f"Status: {r.status_code} | Time: {elapsed}s")
    if d.get("success"):
        for k, v in d["data"].items():
            if k != "_meta":
                print(f"  {k}: {str(v)[:200]}")
    else:
        print(f"Error: {d}")
    print()


if __name__ == "__main__":
    tests = sys.argv[1:] if len(sys.argv) > 1 else ["health", "cb", "table", "qa", "inv"]
    
    if "health" in tests:
        if not test_health():
            print("Health check failed — aborting.")
            sys.exit(1)
    if "cb" in tests:
        test_checkboxes()
    if "table" in tests:
        test_table_extraction()
    if "qa" in tests:
        test_docqa()
    if "inv" in tests:
        test_investment_form()
