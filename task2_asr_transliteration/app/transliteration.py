import logging
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from models.model_config import SOURCE_SCRIPT, TARGET_SCRIPT

logger = logging.getLogger(__name__)

SCRIPT_MAP = {
    "tamil":      sanscript.TAMIL,
    "devanagari": sanscript.DEVANAGARI,
    "telugu":     sanscript.TELUGU,
    "kannada":    sanscript.KANNADA,
    "malayalam":  sanscript.MALAYALAM,
    "latin":      sanscript.ITRANS,
    "itrans":     sanscript.ITRANS,
    "iast":       sanscript.IAST,
    "slp1":       sanscript.SLP1,
}

INDIC_SCRIPTS = {"tamil", "telugu", "kannada", "malayalam", "devanagari"}


BRIDGE_SCHEME = sanscript.SLP1


def get_script(name: str):
    key = name.strip().lower()
    if key not in SCRIPT_MAP:
        raise ValueError(
            f"Unknown script '{name}'. Available: {list(SCRIPT_MAP.keys())}"
        )
    return SCRIPT_MAP[key]


def transliterate_text(
    text: str,
    source_script: str = SOURCE_SCRIPT,
    target_script: str = TARGET_SCRIPT,
) -> str:
    if not text or not text.strip():
        return ""

    src = get_script(source_script)
    tgt = get_script(target_script)

    src_key = source_script.strip().lower()
    tgt_key = target_script.strip().lower()

    try:
        if src_key in INDIC_SCRIPTS and tgt_key in INDIC_SCRIPTS and src_key != tgt_key:
            # Two-step via SLP1 bridge for lossless cross-Indic conversion
            intermediate = transliterate(text, src, BRIDGE_SCHEME)
            result = transliterate(intermediate, BRIDGE_SCHEME, tgt)
        else:
            # Same script, or romanisation target (ITRANS / IAST / SLP1 / latin)
            result = transliterate(text, src, tgt)

        logger.info(
            f"Transliterated ({source_script} → {target_script}): "
            f"{text[:40]} → {result[:40]}"
        )
        return result
    except Exception as e:
        logger.error(f"Transliteration failed: {e}")
        raise


def transliterate_batch(
    texts: list,
    source_script: str = SOURCE_SCRIPT,
    target_script: str = TARGET_SCRIPT,
) -> list:
    return [transliterate_text(t, source_script, target_script) for t in texts]


def list_supported_scripts() -> list:
    return list(SCRIPT_MAP.keys())