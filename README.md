# Myst (1993), playable -- a formally-verified game model, in generated code

You are playing a proof. Every state machine in here was generated from a Kinner
model that was machine-checked (TLC); the four access mini-puzzles (clock-tower
gears, planetarium, cabin boiler, rocket voltage) are each their OWN verified
model, run by the host as a sub-game.

## Requirements

Python 3.10+. Nothing else -- no install, no internet, no dependencies.

## Play it

    python play_game.py

Pick moves by number. The arc: read Catherine's note, flip all eight marker
switches around the island, enter the count at the imager (it reveals the tower),
read the four tower clues, then at each age's gate attempt the puzzle -- the host
hands you the real mini-puzzle to solve; solve it and the age opens. Carry the red
pages back to Sirrus's book in the library; the freebie page completes it and
reveals the fireplace combination; open the fireplace and carry the two white-page
halves through to Atrus's green book to free him.

## Watch it win itself

    python play_game.py --autowin

Plays the entire game end to end -- every mini-puzzle solved by the host -- to a
freed Atrus, in one continuous run. (And `python play_game.py --demo` runs the
host self-test: each sub-game solves, the attempt->host->resolve->unlock loop, and
the imager count check.)
