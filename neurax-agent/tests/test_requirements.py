"""The budget a client states must be read exactly, not approximately."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from requirements import extract_budget

MB = 1024 ** 2
GB = 1024 ** 3


def test_reads_the_size_budget_from_the_users_own_words():
    b = extract_budget("je veux un model qui tourne sur le telephone, images et texte, "
                       "super leger, moins de 1 mega")
    assert b.max_size_bytes == 1 * MB
    assert b.target_device == "phone"


def test_english_phrasings():
    for text, expected in [
        ("a model under 1 MB", 1 * MB),
        ("keep it below 500 kb", 500 * 1024),
        ("less than 2.5 MB please", 2.5 * MB),
        ("at most 750 KB", 750 * 1024),
        ("model < 3mb", 3 * MB),
        ("no more than 1 GB", 1024 ** 3),
    ]:
        assert extract_budget(text).max_size_bytes == expected, text


def test_french_phrasings():
    for text, expected in [
        ("moins de 1 mo", 1 * MB),
        ("au plus 500 ko", 500 * 1024),
        ("inferieur a 2 mo", 2 * MB),
    ]:
        assert extract_budget(text).max_size_bytes == expected, text


def test_runs_on_phrasing_states_a_ceiling_not_just_moins_de():
    # The bug this guards: a real client request ("a light, compact model
    # that understands images and text, that will run on 3GB of RAM and be
    # able to run on a smartphone") named its RAM ceiling by saying what the
    # model must fit *inside*, not by saying "under" — and that phrasing
    # matched nothing at all before this test existed, silently dropping the
    # constraint the client actually stated.
    fr = extract_budget(
        "je veux un modele leger et compact qui va comprendre les images et "
        "le texte qui va tourner sur 3giga de ram et qui va etre capable de "
        "fonctionner sur smartphone"
    )
    assert fr.max_size_bytes == 3 * GB
    assert fr.target_device == "phone"

    en = extract_budget("a compact vision-language model that runs on 2GB of RAM")
    assert en.max_size_bytes == 2 * GB

    assert extract_budget("small enough to fit in 512MB").max_size_bytes == 512 * MB
    assert extract_budget("doit tenir dans 256 mo").max_size_bytes == 256 * MB


def test_latency_budget():
    assert extract_budget("must answer in under 20 ms").max_latency_ms == 20
    assert extract_budget("less than 1.5 s per step").max_latency_ms == 1500


def test_parameter_budget():
    assert extract_budget("under 10M parameters").max_parameters == 10e6
    assert extract_budget("at most 500k params").max_parameters == 500e3


def test_devices_are_recognised():
    assert extract_budget("runs in the browser").target_device == "browser"
    assert extract_budget("for an embedded board").target_device == "edge"
    assert extract_budget("sur telephone android").target_device == "phone"


def test_no_budget_is_reported_as_empty_rather_than_guessed():
    b = extract_budget("build me a transformer for translation")
    assert b.is_empty()
    assert b.max_size_bytes is None


def test_gpu_memory_bounds_vram_when_unstated():
    b = extract_budget("a big model", hw_config={"gpuMemoryGb": 80})
    assert b.max_vram_bytes == 80 * 1024 ** 3


def test_stated_budget_is_described_for_the_log():
    b = extract_budget("under 1 MB on the phone, less than 20 ms")
    text = b.describe()
    assert "size" in text and "latency" in text and "phone" in text
