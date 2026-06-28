"""
GroupDNA - your WhatsApp group chat, decoded.
Single-file version: pass the path to your exported chat .txt file and it
prints the full report.

Usage:
    python groupdna.py hostel_bois.txt
    python groupdna.py /path/to/my_chat.txt

If no path is given on the command line, it falls back to FILE_PATH below.

Built using only Python fundamentals + NumPy. No pandas, no matplotlib,
no regex, no collections.Counter.
"""

import sys
import numpy as np
from datetime import datetime, timedelta

# Used only if you run this with no command-line argument (e.g. inside a
# notebook cell via `%run groupdna.py`, or just double-clicking it).
FILE_PATH = r"C:\unlox\Data Science Data Analytics June Minor Project 1\hostel_bois.txt"


# ============================================================
# FEATURE 1: THE CHAT PARSER
# ============================================================

def is_date_start(line):
    """Checks if the first 8 characters look like a DD/MM/YY date pattern.
    Used to detect multi-line continuation messages: if a line does NOT
    start with a date, it's a continuation of the previous message's text."""
    if len(line) < 8:
        return False
    chunk = line[:8]
    return (chunk[0:2].isdigit() and chunk[2] == '/' and
            chunk[3:5].isdigit() and chunk[5] == '/' and chunk[6:8].isdigit())


def parse_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_lines = f.read().split('\n')

    messages = []          # list of dicts: dt, sender, text, kind
    system_count = 0
    media_count = 0
    deleted_count = 0

    pending = None  # the message dict currently being built (for continuations)

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue  # Edge case (e): skip empty lines silently

        if not is_date_start(line):
            # Edge case (d): multi-line message continuation
            if pending is not None and pending['kind'] != 'system':
                pending['text'] += ' ' + line
            continue

        # a new dated line starts -> flush whatever message we were building
        if pending is not None:
            messages.append(pending)
            pending = None

        if ' - ' not in line:
            continue  # malformed line, skip
        ts_part, rest = line.split(' - ', 1)

        try:
            dt = datetime.strptime(ts_part, '%d/%m/%y, %H:%M')
        except ValueError:
            continue  # malformed timestamp, skip

        if ': ' in rest:
            sender, text = rest.split(': ', 1)
            if text == '<Media omitted>':
                kind = 'media'          # Edge case (b)
                media_count += 1
            elif text == 'This message was deleted':
                kind = 'deleted'        # Edge case (c)
                deleted_count += 1
            else:
                kind = 'normal'
            pending = {'dt': dt, 'sender': sender, 'text': text, 'kind': kind}
        else:
            # Edge case (a): system message - timestamp + dash, no sender colon
            system_count += 1
            pending = {'dt': dt, 'sender': None, 'text': rest, 'kind': 'system'}

    if pending is not None:
        messages.append(pending)

    real_messages = [m for m in messages if m['kind'] != 'system']
    real_messages.sort(key=lambda m: m['dt'])
    participants = sorted(set(m['sender'] for m in real_messages))

    return {
        'messages': real_messages,
        'participants': participants,
        'system_count': system_count,
        'media_count': media_count,
        'deleted_count': deleted_count,
    }


# ============================================================
# FEATURE 4 HELPER: heatmap renderer (used by Feature 4 and Feature 8)
# ============================================================

def render_heatmap(matrix, names):
    print(" ACTIVITY HEATMAP (messages by hour, 0-23, grouped in 3-hr windows)")
    header = "        " + " ".join(f"{h:02d}" for h in range(0, 24, 3))
    print(header)
    for i, name in enumerate(names):
        row = matrix[i]
        row_max = row.max() if row.max() > 0 else 1
        blocks = []
        for h in range(0, 24, 3):
            window_val = row[h:h + 3].sum()
            window_max = row_max * 3
            ratio = window_val / window_max if window_max else 0
            if ratio <= 0.25:
                blocks.append('. ')
            elif ratio <= 0.50:
                blocks.append('\u2591 ')
            elif ratio <= 0.75:
                blocks.append('\u2592 ')
            else:
                blocks.append('\u2588 ')
        print(f"  {name:<8}" + " ".join(blocks))


# ============================================================
# FEATURE 5 HELPER: tokenizer
# ============================================================

STOP_WORDS = {
    'i', 'is', 'the', 'a', 'and', 'or', 'to', 'of', 'in', 'on', 'for',
    'this', 'that', 'it', 'you', 'me', 'my', 'we', 'so', 'but', 'not',
    'do', 'did', 'was', 'were', 'are', 'am', 'be', 'at', 'with', 'as',
    'if', 'just', 'what', 'why', 'how', 'who', 'hai', 'ka', 'ki', 'ko',
}
PUNCTUATION = '.,!?"\'()[]{}:;-_'


def tokenize(text):
    words = []
    for raw_word in text.split():
        w = raw_word.lower().strip(PUNCTUATION)
        if w and w not in STOP_WORDS:
            words.append(w)
    return words


# ============================================================
# FEATURE 7 HELPERS: archetype scoring functions
# ============================================================

CARING_KEYWORDS = ['okay', 'safe', 'eat', 'sleep', 'take care', 'are you',
                    'please', 'reminder', 'drink water', "don't forget"]
COMEDIAN_WORDS = ['lol', 'lmao', 'haha', 'rofl', 'lmfao']
NIGHT_HOURS = {23, 0, 1, 2, 3, 4}


def spammer_score(p, bursts):
    lengths = [length for sender, length in bursts if sender == p]
    return sum(lengths) / len(lengths) if lengths else 0


def group_mom_score(p, messages_by_person):
    score = 0
    for m in messages_by_person[p]:
        if m['kind'] != 'normal':
            continue
        text_lower = m['text'].lower()
        for kw in CARING_KEYWORDS:
            if kw in text_lower:
                score += 1
    return score


def night_owl_score(p, messages_by_person):
    msgs = messages_by_person[p]
    if not msgs:
        return 0
    night = sum(1 for m in msgs if m['dt'].hour in NIGHT_HOURS)
    return night / len(msgs) * 100


def storyteller_score(p, messages_by_person):
    real = [m for m in messages_by_person[p] if m['kind'] == 'normal']
    if not real:
        return 0
    total_words = sum(len(m['text'].split()) for m in real)
    return total_words / len(real)


def drama_queen_score(p, messages_by_person):
    real = [m for m in messages_by_person[p] if m['kind'] == 'normal']
    if not real:
        return 0
    dramatic = 0
    for m in real:
        text = m['text']
        letters_only = ''.join(ch for ch in text if ch.isalpha())
        is_caps = len(letters_only) >= 3 and letters_only.isupper()
        has_double_exclaim = text.count('!') >= 2
        if is_caps or has_double_exclaim:
            dramatic += 1
    return dramatic / len(real) * 100


def ghost_score(p, silent_days_pct):
    return silent_days_pct[p]


def comedian_score(p, messages_by_person):
    real = [m for m in messages_by_person[p] if m['kind'] == 'normal']
    if not real:
        return 0
    funny = sum(1 for m in real if any(cw in m['text'].lower() for cw in COMEDIAN_WORDS))
    return funny / len(real) * 100


def question_master_score(p, messages_by_person):
    real = [m for m in messages_by_person[p] if m['kind'] == 'normal']
    if not real:
        return 0
    questions = sum(1 for m in real if m['text'].strip().endswith('?'))
    return questions / len(real) * 100


def pakka_punctual_score(p, messages_by_person, active_dates, total_days):
    """BONUS / 9th archetype (invented). Rewards consistency + a near-total
    absence of late-night messaging: high active-day ratio AND very low
    night-hour share."""
    msgs = messages_by_person[p]
    if not msgs:
        return 0
    active_ratio = len(active_dates[p]) / total_days
    night_ratio = night_owl_score(p, messages_by_person) / 100
    return max(0, (active_ratio - night_ratio)) * 100


# ============================================================
# MAIN PIPELINE
# ============================================================

def fmt_duration(seconds):
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} minutes"
    return f"{minutes / 60:.1f} hours"


def run_groupdna(file_path):
    # ---------------- Feature 1: Parser ----------------
    parsed = parse_chat(file_path)
    messages = parsed['messages']
    participants = parsed['participants']

    print(f"Successfully parsed {len(messages)} messages from {len(participants)} participants, "
          f"skipped {parsed['system_count']} system messages, {parsed['media_count']} media-omitted, "
          f"{parsed['deleted_count']} deleted messages.\n")

    # ---------------- Feature 2: Group Overview ----------------
    person_counts = {}
    for m in messages:
        person_counts[m['sender']] = person_counts.get(m['sender'], 0) + 1

    total_messages = len(messages)
    first_date = messages[0]['dt'].date()
    last_date = messages[-1]['dt'].date()
    total_days = (last_date - first_date).days + 1

    ranked = sorted(person_counts.items(), key=lambda kv: kv[1], reverse=True)

    # ---------------- Feature 3: Busiest day / hour ----------------
    day_counts = {}
    hour_counts = {h: 0 for h in range(24)}
    for m in messages:
        d_str = m['dt'].strftime('%d/%m/%Y')
        day_counts[d_str] = day_counts.get(d_str, 0) + 1
        hour_counts[m['dt'].hour] += 1

    busiest_day_str, busiest_day_count = max(day_counts.items(), key=lambda kv: kv[1])
    busiest_hour, busiest_hour_count = max(hour_counts.items(), key=lambda kv: kv[1])

    # ---------------- Feature 4: NumPy heatmap ----------------
    person_index = {name: i for i, name in enumerate(participants)}
    heatmap = np.zeros((len(participants), 24), dtype=int)
    for m in messages:
        heatmap[person_index[m['sender']], m['dt'].hour] += 1

    # ---------------- Feature 5: Top words ----------------
    word_freq = {}
    for m in messages:
        if m['kind'] != 'normal':
            continue
        for w in tokenize(m['text']):
            word_freq[w] = word_freq.get(w, 0) + 1
    top_words = sorted(word_freq.items(), key=lambda kv: kv[1], reverse=True)[:10]
    max_word_count = top_words[0][1] if top_words else 1

    # ---------------- Feature 6: Response speed & silent streaks ----------------
    response_gaps = {p: [] for p in participants}
    for i in range(1, len(messages)):
        prev_m, cur_m = messages[i - 1], messages[i]
        if cur_m['sender'] != prev_m['sender']:
            gap_seconds = (cur_m['dt'] - prev_m['dt']).total_seconds()
            response_gaps[cur_m['sender']].append(gap_seconds)

    avg_response = {}
    for p in participants:
        gaps = response_gaps[p]
        avg_response[p] = sum(gaps) / len(gaps) if gaps else float('inf')

    fastest = min(avg_response.items(), key=lambda kv: kv[1])
    slowest = max(avg_response.items(), key=lambda kv: kv[1])

    active_dates = {p: set() for p in participants}
    for m in messages:
        active_dates[m['sender']].add(m['dt'].date())

    all_dates = [first_date + timedelta(days=i) for i in range(total_days)]

    silent_streaks = {}
    for p in participants:
        max_streak = 0
        current_streak = 0
        streak_start = None
        best_start, best_end = None, None
        for d in all_dates:
            if d not in active_dates[p]:
                if current_streak == 0:
                    streak_start = d
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    best_start, best_end = streak_start, d
            else:
                current_streak = 0
        silent_streaks[p] = (max_streak, best_start, best_end)

    # ---------------- Feature 7: Archetype detection ----------------
    messages_by_person = {p: [m for m in messages if m['sender'] == p] for p in participants}

    bursts = []  # list of (sender, length) - consecutive same-sender runs
    i = 0
    while i < len(messages):
        j = i
        while j + 1 < len(messages) and messages[j + 1]['sender'] == messages[i]['sender']:
            j += 1
        bursts.append((messages[i]['sender'], j - i + 1))
        i = j + 1

    silent_days_pct = {}
    for p in participants:
        inactive_days = total_days - len(active_dates[p])
        silent_days_pct[p] = inactive_days / total_days * 100

    PRIMARY_ARCHETYPES = {
        'THE SPAMMER': (lambda p: spammer_score(p, bursts), 3.0,
                        lambda p, s: f"avg {s:.1f} msgs in a row"),
        'THE GROUP MOM': (lambda p: group_mom_score(p, messages_by_person), 50.0,
                          lambda p, s: f"caring keyword score: {int(s)}"),
        'THE NIGHT OWL': (lambda p: night_owl_score(p, messages_by_person), 60.0,
                          lambda p, s: f"{s:.1f}% msgs between 23h-04h"),
        'THE STORYTELLER': (lambda p: storyteller_score(p, messages_by_person), 30.0,
                            lambda p, s: f"avg {s:.1f} words per msg"),
        'THE DRAMA QUEEN': (lambda p: drama_queen_score(p, messages_by_person), 30.0,
                            lambda p, s: f"{s:.1f}% ALL-CAPS / dramatic msgs"),
        'THE GHOST': (lambda p: ghost_score(p, silent_days_pct), 60.0,
                      lambda p, s: f"silent on {round(s/100*total_days)} of {total_days} days"),
        'THE COMEDIAN': (lambda p: comedian_score(p, messages_by_person), 20.0,
                         lambda p, s: f"{s:.1f}% lol/haha-type msgs"),
        'THE QUESTION MASTER': (lambda p: question_master_score(p, messages_by_person), 25.0,
                                 lambda p, s: f"{s:.1f}% msgs end in '?'"),
    }
    BONUS_ARCHETYPE = (
        'THE PAKKA PUNCTUAL ONE',
        lambda p: pakka_punctual_score(p, messages_by_person, active_dates, total_days),
        85.0,
        lambda p, s: f"consistency score {s:.1f}",
    )

    all_scores = {}
    normalized_scores = {}
    for p in participants:
        all_scores[p] = {}
        normalized_scores[p] = {}
        for name, (func, threshold, _) in PRIMARY_ARCHETYPES.items():
            raw = func(p)
            all_scores[p][name] = raw
            normalized_scores[p][name] = raw / threshold

    # Greedy global assignment: highest normalized score across all (person,
    # archetype) pairs wins that archetype first. If two people both pass a
    # threshold, the stronger scorer keeps the label; the other falls through
    # to their next-best available archetype.
    pairs = []
    for p in participants:
        for name in PRIMARY_ARCHETYPES:
            pairs.append((normalized_scores[p][name], p, name))
    pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_archetype = {}
    claimed_archetypes = set()
    for score, p, name in pairs:
        if p in assigned_archetype or name in claimed_archetypes:
            continue
        assigned_archetype[p] = name
        claimed_archetypes.add(name)

    runner_up = {}
    for p in participants:
        ranked_archetypes = sorted(normalized_scores[p].items(), key=lambda kv: kv[1], reverse=True)
        for name, _ in ranked_archetypes:
            if name != assigned_archetype[p]:
                runner_up[p] = name
                break

    bonus_name, bonus_func, bonus_threshold, bonus_describe = BONUS_ARCHETYPE
    bonus_scores = {p: bonus_func(p) for p in participants}
    bonus_qualifiers = [(p, s) for p, s in bonus_scores.items() if s >= bonus_threshold]
    bonus_winner, bonus_score = (max(bonus_qualifiers, key=lambda kv: kv[1])
                                  if bonus_qualifiers else (None, None))

    # ============================================================
    # FEATURE 8: FINAL REPORT
    # ============================================================
    bar_w = 20
    max_count = ranked[0][1]

    print("=" * 60)
    print(' GROUPDNA REPORT')
    print(f" {total_days} days  |  {total_messages:,} messages  |  {len(participants)} members")
    print("=" * 60)
    print(f" Period       : {first_date.strftime('%d %B %Y')} to {last_date.strftime('%d %B %Y')}")
    print(f" Busiest day  : {datetime.strptime(busiest_day_str, '%d/%m/%Y').strftime('%d %B %Y')} ({busiest_day_count} messages)")
    print(f" Busiest hour : {busiest_hour:02d}:00 - {(busiest_hour + 1) % 24:02d}:00")

    print("\n MESSAGES PER PERSON")
    for name, count in ranked:
        pct = count / total_messages * 100
        bar_len = max(1, round(count / max_count * bar_w))
        print(f"   {name:<8}{'\u2588' * bar_len:<{bar_w+1}} {count:>4} ({pct:4.1f}%)")

    print()
    render_heatmap(heatmap, participants)

    print("\n THIS GROUP'S FAVOURITE WORDS")
    for word, count in top_words[:5]:
        bar_len = max(1, round(count / max_word_count * bar_w))
        print(f"   {word:<10}{'\u2588' * bar_len} {count}")

    print("\n RESPONSE PATTERNS")
    print(f"   Fastest replier : {fastest[0]} (avg {fmt_duration(fastest[1])})")
    print(f"   Slowest replier : {slowest[0]} (avg {fmt_duration(slowest[1])})")

    print("\n LONGEST SILENT STREAKS")
    for p, (streak, s, e) in sorted(silent_streaks.items(), key=lambda kv: kv[1][0], reverse=True):
        if streak == 0:
            print(f"   {p:<8}: 0 days (never went silent)")
        else:
            print(f"   {p:<8}: {streak} days ({s.strftime('%d %b')} - {e.strftime('%d %b')})")

    print("\n PERSONALITY ARCHETYPES")
    for p in participants:
        arch = assigned_archetype[p]
        _, threshold, describe = PRIMARY_ARCHETYPES[arch]
        raw = all_scores[p][arch]
        print(f"   {p:<8} -> {arch}  ({describe(p, raw)})  [runner-up: {runner_up[p]}]")

    if bonus_winner:
        print(f"\n BONUS ARCHETYPE (invented)")
        print(f"   {bonus_winner:<8} also qualifies as {bonus_name}  ({bonus_describe(bonus_winner, bonus_score)})")

    print("=" * 60)
    print(" Generated by GroupDNA  |  Built with Python + NumPy")
    print("=" * 60)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else FILE_PATH
    run_groupdna(path)