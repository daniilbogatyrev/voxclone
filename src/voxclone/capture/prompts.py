from pathlib import Path

_DATA = Path(__file__).parent / "data" / "harvard_sentences.txt"

def _load_harvard() -> list[str]:
    return [ln.strip() for ln in _DATA.read_text(encoding="utf-8").splitlines() if ln.strip()]

EXPRESSIVE: list[str] = [
    "Wait — are you seriously telling me this actually worked?",
    "No way. That is the best news I've heard all week!",
    "Honestly? I'm not sure this is a good idea.",
    "Ugh, why does this always happen to me...",
    "Hold on, let me think about that for a second.",
    "That's incredible — I can't believe we pulled it off!",
    "Well, I suppose we could try it your way.",
    "Stop! Don't touch that, it's still hot.",
    "Hmm, interesting. Tell me more about that.",
    "I'm so proud of you, you have no idea.",
    "Are you kidding me right now? Unbelievable.",
    "Okay, okay, calm down — everything is going to be fine.",
    "What on earth is that supposed to mean?",
    "Yes! Finally! It's about time something went right.",
    "I really, really don't want to do this.",
    "Could you please repeat that? I didn't quite catch it.",
    "Why would anyone do something like that?",
    "Absolutely not — that's completely out of the question.",
    "Oh, you have got to be joking.",
    "Listen carefully, because I will only say this once.",
    "Do you genuinely believe that will work?",
    "That's the most ridiculous thing I've ever heard.",
    "Please, just give me a moment to breathe.",
    "How dare you say that to me!",
    "I can't thank you enough for everything.",
    "Are we there yet? This is taking forever.",
    "Whoa, slow down — you're going way too fast.",
    "It's fine. Really. I'm completely fine.",
    "Did you remember to lock the front door?",
    "This is exactly what I was afraid of.",
    "Congratulations, you absolutely earned this.",
    "I beg your pardon, but that is simply not true.",
    "Let's get out of here before it gets dark.",
    "You did what?! Tell me everything, right now.",
    "Take a deep breath; we'll figure this out together.",
    "Honestly, I'm exhausted and I just want to sleep.",
    "Watch out — there's a step right in front of you!",
    "I knew it. I absolutely knew this would happen.",
    "Could this day possibly get any stranger?",
    "Promise me you'll be careful out there.",
    "That's hilarious — say it again, I dare you.",
    "I'm warning you, this is your last chance.",
    "Wow, the view from up here is breathtaking.",
    "Are you absolutely certain about this decision?",
    "Please don't go; I still have so much to say.",
    "Enough! I don't want to hear another word.",
    "You're telling me this now, of all times?",
    "Thank goodness you're safe — I was terrified.",
    "Let me be perfectly clear about one thing.",
    "Honestly, who could have predicted any of this?",
]

CONVERSATIONAL: list[str] = [
    "So I was thinking we could grab coffee sometime this week.",
    "Yeah, I went to the store earlier and picked up a few things.",
    "It's been a pretty quiet day, nothing much going on, really.",
    "I'm not totally sure yet, but I think the meeting is at three.",
    "We watched a movie last night and honestly it was just okay.",
    "My weekend was good — mostly just relaxing around the house.",
    "Do you want to split the bill or should I just get this one?",
    "I've been meaning to call my parents, I just keep forgetting.",
    "The weather's supposed to turn cold again by the weekend.",
    "Let's plan to leave around eight so we beat the traffic.",
    "I tried that new place downtown and the food was great.",
    "Honestly I just need a slow morning and a good cup of tea.",
    "We should catch up properly soon, it's been way too long.",
    "I'll send you the details later once I sort everything out.",
    "It's not a big deal, we can always reschedule for next week.",
    "I spent most of the afternoon just tidying up the apartment.",
    "Remind me to grab milk on the way home, would you?",
    "I think I left my charger at the office again.",
    "We took the long way home just to enjoy the drive.",
    "Let me know what works for you and we'll figure it out.",
    "I'm pretty flexible this week, so whatever suits you is fine.",
    "Funny enough, I ran into an old friend at the airport.",
]

TECHNICAL: list[str] = [
    "Please transfer 1,250 dollars to account number 4827 by Friday.",
    "The meeting is scheduled for March 3rd at 9:45 in the morning.",
    "Dr. Alvarez will review the report on page 218, section 4.2.",
    "Our flight, BA 2049, departs from gate 37 at 6:15 p.m.",
    "The temperature dropped to minus 12 degrees Celsius overnight.",
    "Version 2.11 fixed roughly 87 percent of the reported bugs.",
    "Call me at 555-0142 or email me at the usual address.",
    "The package weighs 3.6 kilograms and ships within 48 hours.",
    "Mix 250 milliliters of water with 2 tablespoons of the powder.",
    "The CPU runs at 4.2 gigahertz across all 16 cores.",
    "Invoice 90817 is due on the 15th, with a 5 percent discount.",
    "Maxime and Dmitri will arrive from Saint Petersburg on Tuesday.",
    "The GDP grew by 2.3 percent in the fourth quarter of 2025.",
    "Set the oven to 375 degrees and bake for 25 minutes.",
    "My PIN is not 1234, and my zip code is 60611.",
    "The marathon route covers 42.195 kilometers through the city.",
    "Apartment 7B is on the third floor, just past the elevator.",
    "The dataset contains 1.4 million rows and 32 columns.",
    "We need 3 liters of paint to cover 48 square meters.",
    "The contract, clause 12, expires on December 31st, 2026.",
    "NASA launched the probe at 02:30 UTC on January 9th.",
    "Round 19.99 up to 20, then add the 8 percent tax.",
]

PROMPTS: dict[str, list[str]] = {
    "harvard": _load_harvard(),
    "expressive": EXPRESSIVE,
    "conversational": CONVERSATIONAL,
    "technical": TECHNICAL,
}

def get_prompts(category: str) -> list[str]:
    return PROMPTS[category]

def all_prompts() -> list[tuple[str, str]]:
    return [(cat, text) for cat, texts in PROMPTS.items() for text in texts]
