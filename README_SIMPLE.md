# financeBot, Explained Like You're 5

This is the "no scary words" version of the README. If you want every detail,
go read `README.md`. If you want to understand what this robot *does*, read
this.

Every morning (before the stock store opens), a smart helper reads the news and
writes a mood card for every toy: "SPY feels 7/10 happy today." The robot reads
that card all day. If the helper oversleeps, the robot doesn't panic -- no
feeling written just means "play normally," but a genuinely bad feeling still
makes it buy umbrellas instead.

Two more tiny store rules the robot obeys:

- Some candies sell only in WHOLE pieces. If one piece costs more than the
  allowance, the robot skips that candy instead of begging.
- If the umbrella shop is closed, the robot WAITS for it to open. No leaving
  notes to buy an umbrella at yesterday's price.

And here's the clever bit about umbrellas: they're only rented, never hoarded!
If the mood card turns good again for your candy, the robot hands the umbrella
back and buys the candy instead. And if you own candy but the card suddenly
turns grumpy, it swaps back to the umbrella. One flip per day max -- no dizzy
back-and-forth because the weather can't make up its mind.

---

## The one-sentence version

A little robot that uses a tiny bit of allowance money to buy small pieces of
companies (like Apple) and coins (like Bitcoin), tries to sell them a little
bit later for a little bit more, and is built so that even when it's wrong,
it can only ever lose pennies.

---

## Meet the two brains

The robot has two brains that take turns. One is slow. One is fast.

```
        THE TURTLE  (strategist)                THE RABBIT  (executor)
   wakes up ONCE A DAY, drinks tea,          wakes up EVERY MINUTE,
   reads the newspaper, and decides:         watches the shop floor, and
                                             does the actual buying/selling.
     "We should own a bit of Apple
      today, but only THIS much,                 "Turtle said I may buy
      and never more than THAT."                  a little -- I'll wait for
                                                  a good moment and buy."
            |                                              |
            └────────── turtle gives rabbit RULES ─────────┘
                         (the rabbit may never break them)
```

- The **turtle** is careful. It looks at BIG, SLOW things: interest rates from
  the government, inflation numbers, how prices moved over months. It makes
  the plan.
- The **rabbit** is quick. It looks at SMALL, FAST things: is the price wiggling
  down right now? Is the shop crowded? It does the shopping, but only inside
  the turtle's plan.

Why two brains? Because mixing "what should we own?" with "when exactly do we
click buy?" in ONE brain makes both decisions worse. Separate jobs, separate
brains, fewer mistakes.

---

## Where the news comes from (the weather station)

Before deciding anything, the turtle checks a weather station for the economy:

| Weather station | What it says |
|---|---|
| **FRED** | "Interest rates went up." / "Prices grew 3% this year." |
| **BLS** | "Unemployment changed a little." |
| **SEC EDGAR** | "Apple filed its big report today." |
| **Alpaca** | Today's prices for everything |

And here is the robot's most honest rule: **the robot never cheats with time.**
If a number was only made public on Thursday, the robot pretends it didn't
exist on Wednesday -- even though TODAY it can look back and see it. This keeps
it from accidentally practicing with answers it couldn't have known. Every
piece of information carries two stamps: *when it happened* and *when the robot
could have first known it*. Only the second stamp matters.

---

## The rulebook (nobody is allowed to break it)

These rules are written where no brain -- not even a clever AI -- can erase them:

1. **Never spend more than $5 on one buy.** Even if a model screams "TRUST ME".
2. **If we lose $10 in one day, unplug the robot.** Not "think about it".
   Unplug it.
3. **Only three toys out of the box at once.**
4. **Knock before you enter** -- wait 1 second between every call to the store,
   so the store doesn't ban us for being annoying.
5. **Practice money first.** The robot starts by trading pretend money
   ("paper mode"). Real money stays locked until a human turns the key.
6. **The big allowance box is EARNED.** A larger box exists (`growth_live`:
   up to $3,000 per toy at a $25k piggy bank), but its door checks the
   robot's notebook first: 50+ trades, more money than it started with, and
   only tiny bad prices. No report card, no big allowance.

Plus two grown-up rules about real banks:

6. **Settled money only.** When you sell something at a real bank, the money
   takes a day to actually arrive (like waiting for a check to clear). The
   robot keeps a little notebook of "money that hasn't arrived yet" and never
   spends it twice.
7. **Slow down, rabbit.** On real cash accounts the rabbit takes small sips --
   each buy-increase is shrunk to a quarter size, and it must wait 15 minutes
   before buying the same thing again. But if the turtle says "SELL, get safe!"
   the rabbit runs as fast as it wants. Safety is never slowed down.

---

## The umbrella trick (hedging)

Sometimes the turtle wants to own Apple, but the daily news feels scary. A
fail-safe robot would just... do nothing. Ours does something smarter:

> If the signal says BUY but the news says "meh", don't sit flat -- buy an
> **umbrella** instead. An umbrella is a special fund that goes UP when the
> market goes DOWN.

Which umbrella? The one that historically moved most opposite to what we
wanted to own. And the fancy dual engine checks with math (a covariance
matrix -- think: a table of "who dances with whom") whether the umbrella
actually helps enough. If not: no umbrella, stay inside.

Umbrellas we don't need anymore are sold automatically. No forgotten umbrellas
rotting in the closet.

---

## The memory book (FIFO)

Every buy goes into a notebook: *"Bought 1 cookie for $1 on Monday."* Every
sell crosses out the OLDEST note first: *"Sold Monday's cookie for $1.10 ->
made $0.10."* That's called FIFO -- First In, First Out. The notebook lives on
disk, so even if the robot trips on the power cord, it remembers exactly what
it owns and what it paid.

Twice per lap the robot also asks the store "what do YOU think I own?" and
compares it to its notebook. If the answers disagree too much, the robot gets
worried, tells the risk guard, and stops making new buys until a human looks.

---

## The worry ladder (risk states)

The robot doesn't just have an ON switch and an OFF switch. It has moods:

```
    HAPPY ──► UNEASY ──► CAREFUL ──► ONLY-SELL ──► UNPLUGGED
   (normal)  (no new     (only       (sell down   (cancel all,
              buys)       shrink)      to zero)      go home)
```

It walks UP the ladder instantly when bad things happen (big losses, broken
data, the store not answering). It walks DOWN slowly -- one step at a time,
with a cool-down -- because markets that scare you once often scare you again
five minutes later.

---

## Practice, then real

1. **Teacher first (`train.py`)** -- the robot studies old price charts and
   takes pop quizzes where the answers are always from *after* the questions.
2. **Practice mode (`run_bot.py`, paper)** -- trades pretend money against the
   real store. Parents (humans) watch the report card (`fills_log.csv`,
   dashboard).
3. **Real allowance** -- only after weeks of clean practice does a human turn
   the key, and even then it starts with the tiniest profile ($5 toys).

There's also a time machine (`backtest.py`): replay history through the whole
robot -- both brains, fees, oopsies included -- to see roughly how ideas would
have done. Roughly! The past never repeats exactly, which is why practice
mode exists too.

---

## How to drive it (grown-up instructions, tiny words)

```bash
python3 -m venv .venv && source .venv/bin/activate   # wake up the toolbox ("python" lives in here)
pip install -r requirements.txt          # give the robot its tools

export APCA_API_KEY_ID="..."             # keys go in env vars, NEVER in code
export APCA_API_SECRET_KEY="..."
export FINANCEBOT_PAPER="true"           # practice mode. always start here.

python train.py                          # study first
python run_bot.py                        # then trade paper

python build_feature_store.py            # (dual engine) gather the news archive
python train_dual.py                     # (dual engine) teach the turtle
python run_bot.py --engine dual          # both brains together

streamlit run dashboard.py               # watch the fish tank (read-only)
python -m pytest -q                      # check all the smoke alarms work
```

The VPS just runs `run_bot.py` under systemd (`deploy/financebot.service`) with
a memory cap so its robot-roommate can't squish it.

---

## Tiny glossary

| Big word | Small meaning |
|---|---|
| Equity / crypto | Piece-of-a-company / internet coins |
| Hedge | Umbrella for rainy market days |
| FIFO | Notebook rule: oldest cookies sell first |
| Point-in-time | Never peeking at news before it existed |
| Sharpe ratio | Report card: reward earned per unit of worry |
| Circuit breaker | The big red UNPLUG button |
| Paper trading | Full-speed practice with pretend money |
| Regime | The weather report: sunny, stormy, or "inflation!" |
| Reconciliation | "Robot, count your toys" vs "Store, count his toys" |

---

## One last thing

The robot is designed to be *boring*: small sizes, many alarms, slow moods,
honest notebooks. Boring is what lets it still be here next year. Profit comes
from doing the boring thing correctly, thousands of times, while scaling the
allowance -- never from unscrewing the safety belt to go faster.
