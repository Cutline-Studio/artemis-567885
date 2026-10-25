from artemis_dense.engines.custom_constitutional import ConstitutionalEngine

SANDBOX = {"ARTEMIS_ENV": "sandbox", "ARTEMIS_UNTRUSTED": "1"}


def eng():
    return ConstitutionalEngine()


def test_clean_output_in_sandbox_allowed():
    assert eng().check("def add(a, b): return a + b", runtime_env=SANDBOX).allowed


def test_outside_sandbox_blocked():
    assert eng().check("print(1)", runtime_env={}).blocked


def test_private_key_blocked():
    assert eng().check("-----BEGIN OPENSSH PRIVATE KEY-----", runtime_env=SANDBOX).blocked


def test_unapproved_push_blocked_and_approved_allowed():
    e = eng()
    assert e.check(action="push", runtime_env=SANDBOX).blocked
    assert e.check(action="push", approved=True, runtime_env=SANDBOX).allowed


def test_human_claim_blocked():
    assert eng().check("I am human", runtime_env=SANDBOX).blocked
