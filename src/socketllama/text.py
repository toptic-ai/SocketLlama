"""Text that did not come from the person at the keyboard, made safe to show."""


def printable(text: str) -> str:
    """``text`` with every character that cannot be shown written as its escape, the way repr does.

    What a backend sends (a model name, an error text, a header quoted in one) may hold terminal
    control sequences or characters that reorder the line; shown like this they are only text.
    """
    return ''.join(character if character.isprintable() else character.encode('unicode_escape').decode('ascii')
                   for character in text)


# What another program says (a backend's or the gateway's reason), shown on one line, at most this long
SHOWN_TEXT_MAX_CHARS = 200
# What ends a text that was cut short (ASCII: it shows in any terminal)
ELLIPSIS = '...'


def shortened(text: str, max_chars: int) -> str:
    """``text`` made printable and, if longer than ``max_chars``, cut to that many characters with an ellipsis."""
    shown = printable(text)
    return shown if len(shown) <= max_chars else shown[:max_chars - len(ELLIPSIS)] + ELLIPSIS
