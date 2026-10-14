"""Which laptop jobs need the wearer to have named their laptop: those that delete, send, buy or otherwise change
things. It is the job Claude wrote that is checked, so someone else's words, or a misheard sentence, can never do
that on their own.
"""

from __future__ import annotations

import re

# Saying where it is meant for. Speech-to-text sometimes hears "laptop" as "lab top".
MENTIONS_LAPTOP = re.compile(
    r"\b(?:la[pb][ -]?tops?|computers?(?!\s+science)|pcs?|desktops?)\b|\b(?:on|of|to)\s+(?:my|the)\s+(?:screen|monitor)\b",
    re.IGNORECASE,
)

# A job that deletes, sends, buys or otherwise changes things only goes out when they said where it is meant for,
# so someone else's words, or a misheard sentence, can never do that. Opening and looking things up need no such words.
# These are matched against the job Claude wrote, so they look for the action, not for nouns such as "my email".
CHANGES_THINGS = re.compile(
    "|".join(
        [
            r"\b(?:delet|remov|eras)(?:e|es|ing)\b",
            r"\bwip(?:e|es|ing)\b",
            r"\b(?:un)?install(?:s|ing)?\b(?!\s+(?:apps|programs|updates))",
            r"\bdownload\w*\s+and\s+(?:run|install|open|launch)\b",
            r"\b(?:run|execute|launch)\s+(?:the\s+|this\s+|that\s+)?(?:installer|setup|command|script|\S+\.(?:exe|msi|bat|cmd|ps1))\b",
            r"\bsend(?:s|ing)?\b",
            r"\b(?:e-?mail|message|text|whatsapp|dm)(?:s|ed|ing)?\s+(?:him|her|them|it|this|that|back|everyone|(?:\w+\s+){1,2}(?:that|saying|on|to|about|asking|with))\b",
            r"\bwrite\s+(?:back\s+)?to\s+\w+",
            r"\brepl(?:y|ies|ied|ying)\s+(?:to|with|that|saying|all|back)\b",
            r"(?<!\bthe\s)(?<!\ba\s)(?<!\bmy\s)(?<!\bnew\s)(?<!\blast\s)(?<!\blatest\s)(?<!\bher\s)(?<!\bhis\s)\bpost(?:s|ed|ing)?\s+(?:it|this|that|a|an|on|to|the|my|in)\b",
            r"\btweet\w*\b",
            r"\bshar(?:e|es|ed|ing)\s+(?:it|this|that|the|my|a|an|them|these|those)\b",
            r"\bshar(?:e|es|ing)\b[^.;!?\n]*\bwith\b",
            r"\bupload\w*\b",
            r"\bsubmit\w*\b",
            r"\b(?:turn|hand)(?:s|ed|ing)?\s+(?:it\s+|them\s+)?in\b",
            r"(?<!\bbest\s)\b(?:buy|buys|buying)\b",
            r"\bpurchas(?:e|es|ing)\b(?!\s+history)",
            r"\bpa(?:y|ys|id|ying)\s+(?:for\b|the\b|my\b|it\b|this\b|that\b|with\b|them\b|him\b|her\b|back\b|off\b|\$|\d)",
            r"\b(?:place|make|put)\s+(?:in\s+)?an?\s+order\b",
            r"\border(?:s|ed|ing)?\s+(?:it|this|that|a|an|the|some|me|more|another|two|three|\d+)\b",
            r"\btransfer(?:s|red|ring)?\s+(?:\$|\d|money|funds|the|my|it|this|that|all|files?)",
            r"\bcancel\w*\b",
            r"\bunsubscrib\w*\b",
            r"\bmov(?:e|es|ed|ing)\s+(?:the\s+|my\s+|all\s+|every\s+|these\s+|those\s+|this\s+|that\s+)?(?:files?|folders?|documents?|photos?|pictures?|videos?|\S+\.\w{2,4})\b",
            r"\brenam(?:e|es|ed|ing)\b",
            r"\bempty(?:ing)?\s+(?:the\s+|my\s+)?(?:trash|recycle|bin|folder|cart)\b",
            r"\bformat(?:s|ted|ting)?\s+(?:the\s+|my\s+|this\s+)?(?:drive|disk|usb|partition|[a-z]:)",
            r"\boverwrit\w*\b",
            r"\bclear(?:s|ed|ing)?\s+(?:out\s+)?(?:the\s+|my\s+|all\s+)?(?:browser\s+|browsing\s+|search\s+|chat\s+)?(?:history|cache|cookies|data|downloads?|desktop|folder|everything|all|messages|chat)\b",
            r"\b(?:in|into|to)\s+the\s+(?:trash|recycle\s+bin|bin)\b",
            r"\bwithout\s+saving\b",
            r"\bkill\s+(?:the\s+)?(?:process|task|app|program|\w+\.exe)\b|\bterminate\b|\bend\s+task\b|\bforce\s+quit\b",
            r"\b(?:accept|decline|reject)(?:s|ed|ing)?\s+(?:the\s+|this\s+|that\s+|my\s+|an?\s+)?(?:invite|invitation|request|meeting|friend)",
            r"\b(?:shut\s*down|restart|reboot|lock)\s+(?:the\s+|my\s+)?(?:laptop|computer|pc|windows|system)\b",
            r"\bshut(?:s|ting)?\s+(?:it\s+)?down\b",
            r"\b(?:log|sign)(?:s|ged|ging|ning|ed)?\s*(?:out|off)\b",
        ]
    ),
    re.IGNORECASE,
)
# What a change sounds like, for spotting where a job says not to make one.
_CHANGE_STEMS = (
    r"(?:delet|remov|eras|wip|send|sent|buy|purchas|pay|paid|order|post|tweet|shar|submit|upload|install|uninstall|cancel"
    r"|unsubscrib|transfer|chang|mov|renam|empt|format|overwrit|repl|clos|clear|accept|declin|sign|log|text|e-?mail|messag"
    r"|writ|dm|kill|shut|restart|reboot|touch|edit)"
)
# "Don't delete anything", "do not buy it", "without changing anything": a promise not to change things, which the
# check must not read as a change. "Without saving" is left alone, as it throws work away.
NOT_CHANGING = re.compile(
    rf"\b(?:don[’']?t|do\s+not|never|not|no)\b\W+(?:\w+\W+){{0,2}}?{_CHANGE_STEMS}[^.;!?\n]*"
    r"|\bwithout\s+(?:delet|remov|send|buy|chang|mov|post|pay|order|shar|submit|upload|install|touch|clos|edit)\w*[^.;!?\n]*",
    re.IGNORECASE,
)


def changes_things(task: str) -> bool:
    """Whether a laptop job deletes, sends, buys or otherwise changes something, leaving out what it promises not to do."""
    return CHANGES_THINGS.search(NOT_CHANGING.sub(" ", task)) is not None
