import streamlit as st

# ---- CUSTOM ANSWER KEY (LIST) ----
WORDS = ["skill", "suits", "hired", "sales", "wages", "semen", "groin", "boner", "boobs", "busty", "chest", "buxom", "teens", "horny", "kinky", "juicy", "curvy", "eager", "bimbo", "bosom", "booty", "milky", "giddy", "glitz", "honey", "licks", "lusty", "ogler", "grope", "plump", "sassy", "steam", "tramp", "flesh", "lubes", "clasp", "balls", "gulps", "seeps", "titty", "gowns", "dress", "sight", "sperm", "admin", "calls", "clerk", "email", "filer", "inbox", "notes", "paper", "tasks", "timer", "phone", "scans", "mails", "index", "files", "forms", "fling", "ample", "plush", "fancy", "tease", "glide", "sulky", "vibes", "frisk", "girth", "silky", "swell", "touch", "vixen", "wench", "fluff", "saucy", "skirt", "strut", "pouty", "nudge", "flirt", "nippy", "hands", "greet", "thick", "perky", "fetch", "smirk", "plans", "memos", "charm", "typed", "draft", "proof", "primp", "blush", "chats", "posed", "slink", "swoon", "cling", "gloss", "quirk", "diary", "reply", "quota", "shift", "snack", "break", "stack", "wrote", "faxes", "speak", "focus", "goals", "lunch", "fonts", "smile", "style", "heels", "error", "point", "query", "shake", "laugh", "share", "label", "sharp", "blank", "punch", "pause", "pitch", "route", "clean", "order", "daily", "align", "brief", "check", "trust", "chair", "stamp", "click", "frown", "waste", "ready", "count", "tally", "dates", "month", "times", "rings", "dials", "entry", "spill", "drink", "water", "wipes", "dries", "stain", "sighs", "mouse", "enter", "stare", "loose", "fixed", "talks", "glare", "codes", "quiet", "voice", "rules", "think", "quick", "speed", "write", "shirt", "hello", "beefy", "press", "slide", "stock", "grind", "swipe", "winks", "purse", "brisk", "vivid", "throb", "twirl", "trace", "desks", "xerox", "ogled", "foxes", "preen", "mince", "siren", "tryst", "hussy"];


def is_valid(word, guess, feedback):
    for i in range(5):
        w = word[i]
        g = guess[i]
        f = feedback[i]

        if f == "g" and w != g:
            return False

        if f == "y":
            if g not in word or w == g:
                return False

        if f == "b":
            guess_good = sum(
                1 for j in range(5)
                if guess[j] == g and feedback[j] != "b"
            )
            if word.count(g) > guess_good:
                return False

    return True

def filter_words(words, guess, feedback):
    return [w for w in words if is_valid(w, guess, feedback)]

# ---- STREAMLIT UI ----
st.title("Wordle Solver (Custom Answer Key)")

if "remaining" not in st.session_state:
    st.session_state.remaining = WORDS[:]

st.subheader("Enter Guess and Feedback")

guess = st.text_input("Guess (5 letters)").lower()
feedback = st.text_input("Feedback (g = green, y = yellow, b = black)").lower()

if st.button("Apply"):
    if len(guess) != 5 or len(feedback) != 5:
        st.error("Guess and feedback must be 5 characters long")
    else:
        st.session_state.remaining = filter_words(
            st.session_state.remaining,
            guess,
            feedback
        )

if st.button("Reset"):
    st.session_state.remaining = WORDS[:]

st.subheader("Remaining Possible Words")
st.write(f"Count: {len(st.session_state.remaining)}")
st.write(st.session_state.remaining)
