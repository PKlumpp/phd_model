from ipapy import UNICODE_TO_IPA
from ipapy.ipachar import IPAChar


SYMBOLS = {
    "r": 1,
    "ʝ": 2,
    "ã": 3,
    "gː": 4,
    "t": 5,
    "n": 6,
    "w": 7,
    "u": 8,
    "l": 9,
    "yː": 10,
    "ʎ": 11,
    "bʲ": 12,
    "ə": 13,
    "ʃʲ": 14,
    "sː": 15,
    "zʲ": 16,
    "kː": 17,
    "y": 18,
    "ɒ": 19,
    "fʲ": 20,
    "ɑ": 21,
    "ʏ": 22,
    "ɣ": 23,
    "s": 24,
    "m": 25,
    "tː": 26,
    "xʲ": 27,
    "vː": 28,
    "ø": 29,
    "h": 30,
    "ɨ": 31,
    "dʲ": 32,
    "dː": 33,
    "bː": 34,
    "ɲː": 35,
    "ɑː": 36,
    "ɪ": 37,
    "ɛ": 38,
    "i": 39,
    "ʔ": 40,
    "g": 41,
    "ʃ": 42,
    "ɜː": 43,
    "mː": 44,
    "øː": 45,
    "fː": 46,
    "p": 47,
    "iː": 48,
    "(...)": 49,
    "v": 50,
    "ʌ": 51,
    "b": 52,
    "k": 53,
    "x": 54,
    "ɲ": 55,
    "ʒ": 56,
    "rː": 57,
    "eː": 58,
    "ç": 59,
    "ŋ": 60,
    "ɔː": 61,
    "œ": 62,
    "ẽ": 63,
    "θ": 64,
    "a": 65,
    "rʲ": 66,
    "vʲ": 67,
    "ʃː": 68,
    "æ": 69,
    "ɶ̃": 70,
    "pː": 71,
    "nː": 72,
    "lʲ": 73,
    "õ": 74,
    "pʲ": 75,
    "ɱ": 76,
    "ð": 77,
    "f": 78,
    "j": 79,
    "o": 80,
    "nʲ": 81,
    "sʲ": 82,
    "lː": 83,
    "e": 84,
    "d": 85,
    "ʊ": 86,
    "gʲ": 87,
    "z": 88,
    "ɛː": 89,
    "tʲ": 90,
    "β": 91,
    "mʲ": 92,
    "uː": 93,
    "ɥ": 94,
    "ʀ": 95,
    "aː": 96,
    "ɐ": 97,
    "ɔ": 98,
    "oː": 99,
    "ʎː": 100,
    "kʲ": 101
}

DESCRIPTORS = {}
for s in SYMBOLS:
        desc = []
        if s == "(...)":
            DESCRIPTORS[s] = "silence"
            continue
        else:
            for sym in s:
                desc.extend(UNICODE_TO_IPA[sym].descriptors)
        if "suprasegmental" in desc:
            desc.remove("suprasegmental")
        if "diacritic" in desc:
            desc.remove("diacritic")
        x = IPAChar(desc)
        DESCRIPTORS[s] = x.canonical_representation


def to_index(symbol: str, elongations=True) -> int:
    if not elongations:
        symbol = symbol.replace('ː', '')
    return SYMBOLS[symbol]


def to_symbol(index: int, elongations=True) -> str:
    idx = list(SYMBOLS.values()).index(index)
    symbol = list(SYMBOLS.keys())[idx] if elongations else list(SYMBOLS.keys())[idx].replace('ː', '')
    return symbol


def get_class_count() -> int:
    return max(SYMBOLS.values())


def symbol_to_descriptor(symbol: str) -> str:
    return DESCRIPTORS[symbol]


def index_to_descriptor(index: int) -> str:
    return DESCRIPTORS[to_symbol(index)]


if __name__ == "__main__":
    print("Running phonetics internal tests")
    for s in SYMBOLS:
        print(f"{s}: {DESCRIPTORS[s]}")
