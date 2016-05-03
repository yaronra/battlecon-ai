#!/usr/bin/python

# TO DO

# Pulse: when no character has board markers, can use board symmetry
# to trim fork size.  This means every character should have a function
# that says if board is currently empty (or at least symmetrical).

# KNOWN BUGS/PROBLEMS

# Cancel currently ignored by AI (not played or played around).
# Option 1: For each cancel vs. pair, solve 15x8 matrix.
#     Problems:
#         1. Takes a long time.
#         2. Imprecise, because original sims didn't take into account
#            the extra missing pair at end of beat.
#                (can simulate gain, but that's even longer.)
# Option 2: Evaluate in simulation the loss of a pair for 3 beats
#     Problems:
#         1. Always the same evaluation of lost pair.  There's also
#            preferred_range of lost/remaining card, but that's
#            very rough.
#         2. Not clear how to evaluate spent ante, which might be
#            a complete loss, or still useful this beat.

# Cancel vs. Finisher doesn't discard real base played to invoke the
# Finisher.  Can be tricky to fix, since real base played isn't
# determined.

# when a Cancel ends up killing the opponent, points are still deducted
# for spending the special action

# When a clash empties hands, the result is 0.  Actually, cycling should happen.

# Active ante triggers (Heketch's move) are reported twice when
# there's a Cancel

# Discard piles should be ordered.  For example, Kehrolyn's current form
# is the top (=latest) style that isn't a special action.

# Reveal effects happen only once, so they shouldn't change after a
# clash.  Examples: Eligor's Aegis, Aria's Ionic.  This can't be helped
# without a major refactoring.

# "Deal damage" effects are considered attacks, so dodge effects can
# cause them to miss.

# TO CHECK OPENING DISCARDS
# free_for_all (1, <name>, skip=['kehrolyn'], first_beats=True)

from operator import attrgetter
from optparse import OptionParser
import itertools
import math
import numpy
import os.path
import pstats
import random
import re
import solve
import sys
import time

debug_log = []

# MAIN FUNCTIONS

def main():
    parser = OptionParser()
    parser.add_option("-t", "--test",
                      action="store_true", dest="test",
                      default=False,
                      help="run test")
    parser.add_option("-f", "--from_file",
                      dest="from_file",
                      default='',
                      help="run single beat from given file")
    options, unused_args = parser.parse_args()
    if options.test:
        test()
    elif options.from_file:
        play_beat(options.from_file)
    else:
        play()

playable = [
#             'abarene', not a final version
             'adjenna',
             'alexian',
             'alumis',
             'arec',
             'aria',
#             'borneo', weak promo
             'baenvier',
             'byron',
             'cadenza',
             'cesar',
             'claus',
             'clinhyde',
             'danny',
             'demitras',
             'eligor',
             'eustace',
#             'gerard',  too long
             'heketch',
             'hikaru',
             'iri',
             'jager',
#             'juto', weak promo
             'kajia',
             'kallistar',
             'karin',
             'kehrolyn',
             'khadath',
             'larimore',
             'lesandra',
             'lixis',
             'luc',
             'lymn',
             'magdelina',
             'marmelee',
             'mikhail',
#             'oriana', too long
             'ottavia',
             'rexan',
             'rukyuk',
             'runika',
             'sarafina',
             'seth',
             'shekhtur',
             'tanis',
             'tatsumi',
             'vanaah',
             'voco',
             'xenitia',
             'zaamassal']

def test (first=None, bases='alpha'):
    log = []
    random.seed(0)
    found_first = not first
    for i in range (0,len(playable),2):
        n0 = playable[i]
        if not found_first:
            if n0 == first:
                found_first = True
            else:
                continue
        j = i+1 if i+1 < len(playable) else 0
        n1 = playable[j]
        print n0, n1
        start_time = time.time()
        game = Game.from_start (n0, n1, bases, bases,
                                default_discards=True)
        game_log, unused_winner = game.play_game()
        end_time = time.time()
        print "tot time: %d" % (end_time - start_time)
        log.extend(game_log)
    logfilename = "logs/v1.1_test"
    with open (logfilename, 'w') as f:
        for g in log:
            f.write (g+'\n')

def play ():
    names = sorted([k.capitalize() for k in character_dict])
    while True:
        print "Select your character: [1-%d]\n" % (len(names) + 1)
        ans = menu(names + ['Random'], n_cols=3)
        if ans == len(names):
            ans = random.randrange(len(names))
        human = names[ans]
        print "You will be playing", human
        ai_names = [n for n in names if n != human]
        print "\nSelect AI character: [1-%d]\n" % (len(ai_names) + 1)
        ans = menu(ai_names + ['Random'], n_cols=3)
        if ans == len(ai_names):
            ans = random.randrange(len(ai_names))
        ai = ai_names[ans]
        print "AI will be playing", ai
        print "\nWhich set of bases should be used?"
        ans = menu(['Standard bases',
                    'Beta bases',
                    'I use standard, AI uses beta',
                    'I use beta, AI uses standard'])
        ai_bases = 'beta' if ans in (1,2) else 'alpha'
        human_bases = 'beta' if ans in (1,3) else 'alpha'
        print "Default Discards?"
        default_discards = menu (["No", "Yes"])
        game = Game.from_start (ai, human, ai_bases, human_bases,
                                default_discards=default_discards,
                                interactive=True)
        game.select_finishers()
        game_log, unused_winner = game.play_game()
        if not os.path.exists ("logs"):
            os.mkdir ("logs")
        if ai_bases == 'beta':
            ai = ai + '_beta'
        if human_bases == 'beta':
            human = human + '_beta'
        basename = "logs/" + ai + "(AI)_vs_" + human
        name = save_log (basename, game_log)
        print "Log saved at: ", name
        print
        print "\nAnother game?"
        if not menu (["No","Yes"]):
            break

def save_log(basename, log):
    for i in range (1,10000):
        name = basename + "[" + str(i) + "].txt"
        if not os.path.isfile (name):
            with open (name, 'w') as f:
                for g in log:
                    f.write (g+'\n')
            break
    return name

# Play everyone against everyone
def beta_challenge(next_pair=None, last_pair=None, beta0=False, beta1=False):
    if next_pair is None:
        next_pair = (playable[0],playable[0])
    if last_pair is None:
        last_pair = (playable[-1], playable[-1])
    firsts = [name for name in playable if name >= next_pair[0]
                                       and name <= last_pair[0]]
    for first in firsts:
        seconds = playable
        if first == next_pair[0]:
            seconds = [name for name in seconds if name >= next_pair[1]]
        if first == last_pair[0]:
            seconds = [name for name in seconds if name <= last_pair[1]]
        for second in seconds:
            if ((beta0 == beta1 and first < second) or
                (beta0 != beta1 and first != second)):
                duel(first, second, 1, beta0, beta1)

# Play consecutive pairs, so that each character plays one as alpha
# and one as delta.
def alpha_vs_delta_challenge(first=None):
    n = len(playable)
    for i in xrange(n):
        if playable[i] >= first:
            duel(playable[i], playable[(i+1)%n], 1, 'alpha', 'delta')

# play everyone in names against everyone from start onwards, unless in skip
def free_for_all (repeat, names=None, start=None, skip=[],
                  raise_exceptions=True, first_beats=False):
    if isinstance (names, str):
        names = [names]
    if names is None:
        names = playable
    for i in range (len (playable)):
        for j in range (i+1,len(playable)):
            if (playable[i] in names and playable[j] >= start and playable[j] not in skip) or \
               (playable[j] in names and playable[i] >= start and playable[i] not in skip):
                if raise_exceptions:
                    duel (playable[i], playable[j], repeat, first_beats=first_beats)
                else:
                    try:
                        duel (playable[i], playable[j], repeat, first_beats=first_beats)
                    except Exception as e:
                        print "duel: %s vs. %s" %(playable[i],playable[j])
                        print "exception", e

def duel (name0, name1, repeat, bases0='alpha', bases1='alpha',
          first_beats=False):
    victories = [0,0]
    print name0, "vs.", name1
    log = []
    start = time.time()
    for i in range(repeat):
        game = Game.from_start (name0, name1, bases0, bases1,
                                default_discards=False,
                                first_beats=first_beats)
        game_log, winner = game.play_game()
        log.append ("GAME %d\n-------\n" %i)
        log.extend (game_log)
        if winner == None:
            victories[0] += 0.5
            victories[1] += 0.5
        else:
            winner = winner.lower()
        if winner == name0:
            victories[0] += 1
        elif winner == name1:
            victories[1] += 1
        print winner,
        sys.stdout.flush()
    print
    print victories[0], ":", victories[1]
    end = time.time()
    logfilename = "logs/"+name0+"_"+name1+"_log.txt"
    time_string = "total_time: %d" % (end-start)
    log.append(time_string)
    print time_string
    with open (logfilename, 'w') as f:
        for g in log:
            f.write (g+'\n')

# runs one beat from file data
def play_beat (filename='starting states/start.txt'):
    game = Game.from_file (filename)
    print "Simulating..."
    game.simulate_beat()
    print "Solving..."
    game.solve()
    game.print_solution()
    return game

def play_start_beat (name0, name1, bases0='alpha', bases1='alpha'):
    game = Game.from_start(name0, name1, bases0, bases1,
                           default_discards=False)
    print "Simulating..."
    game.simulate_beat()
    print "Solving..."
    game.solve()
    game.print_solution()
    return game


# Profiling

# copy this:
# cProfile.run ("play_beat ('vanaah', 'demitras')", 'profstat')
# profile ('profstat')
def profile (pfile, n=30):
    p = pstats.Stats(pfile)
    p.strip_dirs().sort_stats('cumulative').print_stats(n)

# input functions

# given a list of strings, prints a menu and prompts for a selection
def menu (options, n_cols=1):
    options = [str(o) for o in options]
    col_len = int(math.ceil(len(options) / float(n_cols)))
    max_width = max([len(o) for o in options])
    # displays options with numbers 1..n, in n_cols columns
    for r in xrange(col_len):
        for c in xrange(n_cols):
            i = c * col_len + r
            if i >= len(options):
                break
            option = options[i]
            spaces = ' ' * (max_width + 5 - len(option) - len(str(i+1)))
            print '[%d] %s%s' % (i+1, option, spaces),
        print
    # inputs number in range 1..n
    ans = input_number (len(options)+1, 1)
    # but returns answer in range 0..n-1
    return ans-1

# input a number between k and n-1
def input_number (n, k=0):
    while True:
        s = raw_input('').strip()
        if string_is_int(s) and int(s) in range(k,n):
            return int(s)

def string_is_int (s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# check if some line in lines starts with start, and return it.
def find_start_line (lines, start):
    for line in lines:
        if line.startswith(start):
            return line
    return None

# check if some line in lines starts with start, and return its index
def find_start_index (lines, start):
    for i,line in enumerate(lines):
        if line.startswith(start):
            return i
    return None

def find_start (lines, start):
    return bool(find_start_line(lines, start))

# check if some line in lines ends with end, and return it.
def find_end_line (lines, end):
    for line in lines:
        if line.endswith(end):
            return line
    return None

def find_end (lines, end):
    return bool(find_end_line(lines, end))

# check if given numbers are strongly ordered (either a<b<c, or c<b<a)
def ordered (a,b,c):
    return (a-b)*(c-b) < 0

# set of all positions between a and b, inclusive
def pos_range (a, b):
    if b>a:
        return set(xrange(a,b+1))
    else:
        return set(xrange(b,a+1))

def all_mean_priorities():
    chars = [character(the_game=None, n=0)
             for character in character_dict.itervalues()]
    chars = sorted(chars, key=attrgetter('mean_priority'), reverse=True)
    for c in chars:
        print "%.1f  %s" %(c.mean_priority, c.name)
    print
    print "Mean: %.1f" % (sum(c.mean_priority for c in chars) / len(chars))
    print "Median: %.1f" % chars[len(chars)/2].mean_priority

# Cumulative Gaussian function
def phi(x):
    return 0.5 + 0.5 * math.erf (x/(2**.5))

# GENERAL CLASSES


# main game class, holds all information and methods
class Game:

    # Constants for indicating special events in result matrix.
    # These will never actually given to the solver - they need to
    # be dealt with in pre-processing.
    # (all constants are floats, so that the numpy array stays a float array)
    CLASH_INDICATOR = -1000.0
    CANCEL_0_INDICATOR = -2000.0
    CANCEL_1_INDICATOR = -2001.0
    CANCEL_BOTH_INDICATOR = -2002.0
    CANCEL_INDICATORS = set ([CANCEL_0_INDICATOR, \
                                 CANCEL_1_INDICATOR, \
                                 CANCEL_BOTH_INDICATOR])
    # Very good/bad result constant (used to prevent certain strats
    # from being chosen by AI:
    EXTREME_RESULT = 20.0

    # Constants for priorities of deciding where you are next beat.
    CHOOSE_POSITION_NOW = 2
    CHOOSE_POSITION_BEFORE_ATTACK_PAIRS = 1
    CHOOSE_POSITION_IN_ANTE = 0

    @staticmethod
    def from_file (file_name):
        """Create non-interactive game from text file.
        Text file is beat situation report as written to log by a previous game.
        """
        with open(file_name) as f:
            lines = [line for line in f]
        char1_start = 0
        while lines[char1_start] != "\n":
            char1_start += 1
        char1_start += 1
        char1_end = char1_start
        while lines[char1_end] != "\n":
            char1_end += 1
        board_start = char1_end
        while lines[board_start] == "\n":
            board_start += 1

        name0 = lines[0][:-1].lower()
        name1 = lines[char1_start][:-1].lower()
        bases0, bases1 = ('alpha', 'alpha')
        if name0.endswith(' bases)'):
            bases0 = name0.split()[-2]
            name0 = name0[:-(len(bases0) + 9)]
        if name1.endswith(' bases)'):
            bases1 = name1.split()[-2]
            name1 = name1[:-(len(bases1) + 9)]

        game = Game(name0, name1, bases0, bases1)
        lines0 = lines [2 : char1_start-1]
        lines1 = lines [char1_start+2 : char1_end]
        board = lines [board_start]
        a0 = game.player[0].board_addendum_lines
        a1 = game.player[1].board_addendum_lines
        addendum0 = lines [board_start + 1 :
                           board_start + 1 + a0]
        addendum1 = lines [board_start + 1 + a0 :
                           board_start + 1 + a0 + a1]
        game.player[0].read_my_state(lines0, board, addendum0)
        game.player[1].read_my_state(lines1, board, addendum1)
        game.debugging = True
        game.interactive_mode = True
        game.initialize_simulations()
        game.interactive_mode = False
        game.debugging = False
        return game

    @staticmethod
    def from_start (name0, name1, bases0, bases1,
                    default_discards=True, interactive=False, cheating=0,
                    first_beats=False):
        """Create a game in starting position.
        name0, name1: names of characters
        """
        game = Game(name0, name1, bases0, bases1,
                    interactive, cheating, first_beats=first_beats)
        game.set_starting_setup (default_discards,
                                 use_special_actions = not first_beats)
        game.initialize_simulations()
        return game

    def __init__ (self, name0, name1, bases0, bases1,
                  interactive=False, cheating=0, first_beats=False):

        name0 = name0.lower()
        name1 = name1.lower()
        self.range_weight = 0.3
        # fast clash evaluation - can only be used for one beat checks
        self.clash0 = False
        self.interactive = interactive
        self.cheating = cheating
        # If first_beats==True, we're checking initial discards,
        # so we only play two beats, and start the game with all cards
        # in hand.
        self.first_beats = first_beats
        # Initial beat is 1, even if starting from file.  Might need changing.
        self.current_beat = 1

        self.interactive_mode = False
        self.replay_mode = False
        self.interactive_counter = None
        self.log = []
        self.player = [character_dict[name0](self, 0, bases0),
                       character_dict[name1](self, 1, bases1,
                                             is_user=interactive)]
        for i in range(2):
            self.player[i].opponent = self.player[1-i]
            for card in self.player[i].all_cards():
                card.opponent = self.player[1-i]
        for p in self.player:
            p.set_mean_priority_effects()
        self.debugging = False
        self.reporting = False

    # sets default state for current characters
    def set_starting_setup (self, default_discards, use_special_actions):
        for p in self.player:
            p.set_starting_setup (default_discards, use_special_actions)

    def select_finishers(self):
        for p in self.player:
            p.select_finisher()

    def initialize_simulations (self):
        # save initial game state
        self.initial_state = self.initial_save ()
        # evaluate game situation, as reference point for further evaluation
        self.reset()
        self.initial_evaluation = self.evaluate()

    # Play a game from current situation to conclusion
    def play_game (self):
        full_names = [p.name_with_base_set()
                      for p in self.player]
        log = ["\n" + " vs. ".join(full_names)]
        for p in self.player:
            log.append ("Chosen finisher for %s: " % p.name +
                             ', '.join(o.name for o in p.finishers))
        # Loop over beats:
        while True:
            log.append ("\nBeat %d" % self.current_beat)
            log.append ("-------\n")
            log.extend (self.situation_report ())
            self.initialize_simulations ()
            if self.interactive:
                self.dump(log)
            self.reporting = False
            self.simulate_beat ()
            self.log_unbeatable_strategies(log)
            self.solve()
            self.reporting = True
            log.extend(self.make_pre_attack_decision() + [''])
            if self.interactive:
                self.dump(log)
                if self.cheating > 0:
                    self.dump (self.report_solution())
            else:
                log.extend(self.report_solution())
            final_state, report = self.execute_beat()
            log.extend (report)
            if final_state.winner != None:
                break
            if self.first_beats and self.current_beat == 2:
                break
            self.full_restore (final_state)
            self.prepare_next_beat ()
        winner = final_state.winner
        if winner == 0.5 or winner is None:
            winner = None
        else:
            winner = self.player[winner].name
        self.dump(log)
        return self.log, winner

    def situation_report (self):
        report = []
        for p in self.player:
            report.extend (p.situation_report())
            report.append ("")
        report.extend (self.get_board())
        return report

    def log_unbeatable_strategies (self, log):
        # check if there's a 100% positive strategy, and note it in log:
        # disregard special actions
        if self.interactive:
            return
        n = len(self.pads[0])
        m = len(self.pads[1])
        for i in xrange(n):
            for j in xrange(m):
                results = numpy.array(self.results[i][j])
                s0 = self.strats[0][i]
                s1 = self.strats[1][j]
                regular_0 = [k for k in range(len(s0))
                             if not isinstance (s0[k][0], SpecialAction)]
                regular_1 = [k for k in range(len(s1))
                             if not isinstance (s1[k][0], SpecialAction)]
                results = results[regular_0,:][:,regular_1]
                row_values = results.min(1)
                if row_values.max() > 0:
                    for k in range(len(row_values)):
                        if row_values[k] > 0:
                            if n > 1:
                                log.append("Given %s by %s" % (self.pads[0][i], self.player[0]))
                            if m > 1:
                                log.append("Given %s by %s" % (self.pads[1][j], self.player[1]))
                            log.append ("Unbeatable strategy for %s: %s: %.2f"
                                % (self.player[0].name,
                                   self.player[0].get_strategy_name(
                                                    s0[regular_0[k]]),
                                   row_values[k]))
                    log.append ("")
                col_values = results.max(0)
                if col_values.min() < 0:
                    for k in range(len(col_values)):
                        if col_values[k] < 0:
                            if n > 1:
                                log.append("Given %s by %s" % (self.pads[0][i], self.player[0]))
                            if m > 1:
                                log.append("Given %s by %s" % (self.pads[1][j], self.player[1]))
                            log.append ("Unbeatable strategy for %s: %s: %.2f"
                                % (self.player[1].name,
                                   self.player[1].get_strategy_name(
                                                    s1[regular_1[k]]),
                                   col_values[k]))
                    log.append ("")

    # empty given log into Game.log; print it if game is interactive
    def dump (self, log):
        for line in log:
            self.log.append (line)
            if self.interactive:
                print line
        log[:] = []

    def logfile_name(self):
        return "%s_%s" % (self.player[0].logfile_name(),
                          self.player[1].logfile_name())

    # makes snapshot of game state (pre strategy selection)
    def initial_save (self):
        state = GameState()
        state.player_states = [p.initial_save() for p in self.player]
        state.current_beat = self.current_beat
        return state

    def initial_restore (self, state):
        for i in range(2):
            self.player[i].initial_restore (state.player_states[i])
        self.current_beat = state.current_beat

    # makes full snapshot, mid beat
    def full_save (self, stage):
        state = GameState()
        state.player_states = [p.full_save() for p in self.player]
        state.reports = [s for s in self.reports]
        state.decision_counter = self.decision_counter
        state.winner = self.winner
        state.active = self.active
        state.stop_the_clock = self.stop_the_clock
        state.current_beat = self.current_beat
        state.stage = stage
        return state

    def full_restore (self, state):
        for i in range(2):
            self.player[i].full_restore (state.player_states[i])
        self.reports = [s for s in state.reports]
        self.decision_counter = state.decision_counter
        self.winner = state.winner
        self.active = state.active
        self.stop_the_clock = state.stop_the_clock
        self.current_beat = state.current_beat

    # resets game state to start of beat position (post strategy selection)
    def reset (self):
        for p in self.player:
            p.reset()
        self.reports = []
        self.fork_decisions = []
        self.decision_counter = 0
        self.winner = None
        self.active = None
        self.stop_the_clock = False

    # run simulation for all strategies, creating result table
    def simulate_beat (self):
        # get lists of strategies from both players
        for p in self.player:
            p.strats = p.get_strategies()
        # run simulations, create result table
        self.results = [[self.simulate (s0,s1)
                         for s1 in self.player[1].strats]
                        for s0 in self.player[0].strats]

        self.initial_restore (self.initial_state)

        # remove redundant finisher strategies
        # (that devolve into identical cancels)
        self.remove_redundant_finishers()

        # Usually this does nothing, but some characters might need to
        # fix the result tables and strategies (e.g. Seth, Ottavia).
        for p in self.player:
            p.post_simulation_processing ()

        # Once we're done with special processing, we can discard
        # the full final state of each simulation, and keep just
        # the evaluation.
        self.results = [[float(result[0]) for result in row]
                        for row in self.results]

        # For each player, make list of strategies, separated into
        # sub lists by pre-attack decision (pad).
        self.pads = [[], []]
        self.strats = [[], []]
        for i, p in enumerate(self.player):
            # group player strategies by pre attack decision.
            for pad, pad_strats in itertools.groupby(p.strats,
                                                     lambda s: s[2][2]):
                self.pads[i].append(pad)
                self.strats[i].append(list(pad_strats))
        # split results into subtables according to pads:
        self.results = [[self.get_pad_subresults(self.results,
                                                 [p.strats for p in self.player],
                                                 pad0, pad1)
                         for pad1 in self.pads[1]]
                        for pad0 in self.pads[0]]

    def remove_redundant_finishers (self):
        redundant_finishers = []
        s0 = self.player[0].strats
        for i in xrange(len(self.results)):
            if isinstance (s0[i][1], Finisher):
                for ii in xrange(len(self.results)):
                    if isinstance (s0[ii][1], Cancel) and s0[ii][2] == s0[i][2]:
                        if ([res[0] for res in self.results[i]] ==
                            [res[0] for res in self.results[ii]]):
                            redundant_finishers.append (i)
                            break
        self.results = [self.results[i] for i in xrange(len(self.results)) \
                                        if i not in redundant_finishers]
        self.player[0].strats = [s0[i] for i in xrange(len(s0)) \
                                       if i not in redundant_finishers]

        redundant_finishers = []
        s1 = self.player[1].strats
        for j in xrange(len(self.results[0])):
            if isinstance (s1[j][1], Finisher):
                for jj in xrange(len(self.results[0])):
                    if isinstance (s1[jj][1], Cancel) and \
                       s1[jj][2] == s1[j][2]:
                        if [row[j][0] for row in self.results] == \
                           [row[jj][0] for row in self.results]:
                            redundant_finishers.append (j)
                            break
        self.results = [[r[j] for j in xrange(len(r)) \
                               if j not in redundant_finishers] \
                               for r in self.results]
        self.player[1].strats = [s1[j] for j in xrange(len(s1)) \
                                 if j not in redundant_finishers]


    def get_pad_subresults(self, results, strats, pad0, pad1):
        ii = [i for i, s in enumerate(strats[0]) if s[2][2] == pad0]
        jj = [i for i, s in enumerate(strats[1]) if s[2][2] == pad1]
        return [[results[i][j]
                 for j in jj]
                for i in ii]

    # if state != None, this is a forked simulation
    def simulate(self, s0, s1, state = None):
        # in a forked simulation, restore the given state
        if state != None:
            self.full_restore (state)

        # if this isn't a forked simulation, execute initial setup
        # (up to pulse check)
        else:
            # restore situation to initial pre-strategy state
            self.initial_restore (self.initial_state)

            self.player[0].strat = s0
            self.player[1].strat = s1

            # resets basic beat information to start of beat state
            self.reset()

            if self.reporting:
                self.report ('')
                for p in self.player:
                    self.report (p.name + ": " + p.get_strategy_name(p.strat))
                self.report ('')

            for p in self.player:
                p.pre_attack_decision_effects()

            # Put attack pairs in discard.
            for p in self.player:
                p.style = p.strat[0]
                p.base = p.strat[1]
                if not isinstance (p.style, SpecialAction):
                    # Adding to existing set,
                    # because Vanaah's token might already be there.
                    p.discard[0] |= set ([p.style, p.base])

            for p in self.player:
                p.ante_trigger()

            # Attack pairs are now active.
            for p in self.player:
                p.set_active_cards()

            # Special Actions

            # Finishers devolve into Cancels above 7 life
            # or if failing to meet their specific conditions
            for p in self.player:
                if (isinstance (p.base, Finisher) and
                    p.base.devolves_into_cancel()):
                    p.base = p.cancel

            # Cancel - return an appropriate cancel indicator
            # (depending on who cancelled).
            # This will be solved retroactively.
            # Check that there's no Pulse that trumps the Cancel.
            cancel = [p.base is p.cancel for p in self.player]
            pulse = [p.base is p.pulse for p in self.player]
            # A double Pulse devolves to a double Cancel.
            if all(pulse):
                final_state = self.full_save (None)
                return self.CANCEL_BOTH_INDICATOR, final_state, self.fork_decisions[:]
            if any(cancel) and not any(pulse):
                final_state = self.full_save (None)
                if all(cancel):
                    return self.CANCEL_BOTH_INDICATOR, final_state, self.fork_decisions[:]
                if cancel[0]:
                    return self.CANCEL_0_INDICATOR, final_state, self.fork_decisions[:]
                if cancel[1]:
                    return self.CANCEL_1_INDICATOR, final_state, self.fork_decisions[:]

            # save state before pulse phase (stage 0)
            state = self.full_save (0)

        # catch ForkExceptions and WinExceptions
        try:

            # Stage 0 includes: pulse check, reveal trigger, clash check.
            if state.stage <=0:
                pulsing_players = [p for p in self.player
                                   if p.base is p.pulse]
                # With one pulse - fork to decide new player positions.
                # Not using execute_move, because Pulse negates any blocking
                # or reaction effects (including status effects from last beat).
                if len (pulsing_players) == 1:
                    pulser = pulsing_players[0]
                    opp = pulser.opponent
                    # If opponent used Cancel/Finisher, we have them
                    # retroactively choose which base activated it.
                    # and put the base and the special action in
                    # discard 0, so that they'll cycle.
                    if opp.style is opp.special_action:
                        if opp.base is opp.cancel:
                            options = opp.cancel_generators.copy()
                            if any(f.devolves_into_cancel()
                                   for f in opp.finishers):
                                options |= opp.finisher_generators
                        else:
                            options = opp.finisher_generators.copy()
                        options &= set(opp.bases)
                        options -= (opp.discard[1] | opp.discard[2])
                        options = list(options)
                        prompt = "Which base did you use to play your special action?"
                        ans = self.make_fork(len(options), opp, prompt,
                                             [o.name for o in options])
                        opp.discard[0].add(options[ans])
                        opp.discard[0].add(opp.special_action)

                    # Perform the Pulse.
                    pairs = list(itertools.permutations(xrange(7), 2))
                    prompt = "Choose positions after Pulse:"
                    options = []
                    if pulser.is_user and self.interactive_mode:
                        current_pair = (self.player[0].position,
                                        self.player[1].position)
                        for pair in pairs:
                            (self.player[0].position,
                             self.player[1].position) = pair
                            options.append (self.get_basic_board())
                        (self.player[0].position,
                         self.player[1].position) = current_pair
                    (self.player[0].position,
                     self.player[1].position) = pairs[self.make_fork(
                                    len(pairs), pulser, prompt, options)]
                    if self.reporting:
                        self.report ('%s Pulses:' % pulser)
                        for s in self.get_board():
                            self.report (s)
                    # Skip directly to cycle and evaluation phase.
                    if pulsing_players:
                        self.stop_the_clock = True
                        return self.cycle_and_evaluate()

                for p in self.player:
                    p.reveal_trigger()

                # clash_priority is fraction
                # that represents autowinning/losing clashes
                priority = [p.get_priority() for p in self.player]
                clash_priority = [priority[i] + p.clash_priority()
                                  for i,p in enumerate(self.player)]

                if self.reporting:
                    self.report ("Priorities:  %d | %d" \
                                 %(priority[0], priority[1]))

                if clash_priority[0] > clash_priority[1]:
                    self.active = self.player[0]
                    if self.reporting:
                        self.report (self.active.name + " is active")
                elif clash_priority[1] > clash_priority [0]:
                    self.active = self.player[1]
                    if self.reporting:
                        self.report (self.active.name + " is active")
                # priority tie
                else:
                    # Two clashing finishers turn into cancels.
                    if isinstance (self.player[0].base, Finisher) and \
                       isinstance (self.player[1].base, Finisher):
                        final_state = self.full_save (None)
                        return self.CANCEL_BOTH_INDICATOR, final_state, self.fork_decisions[:]
                    else:
                        if self.reporting:
                            self.report ("Clash!\n")
                        final_state = self.full_save (None)
                        if self.clash0:
                            return 0, final_state, self.fork_decisions[:]
                        else:
                            return self.CLASH_INDICATOR, final_state, self.fork_decisions[:]
                state = self.full_save (1)

            # start triggers
            if state.stage <= 1:
                for p in self.players_in_order():
                    p.start_trigger()
                state = self.full_save (2)

            # player activations
            if state.stage <= 2:
                active = self.active
                # check if attack needs to be re-executed
                while active.attacks_executed < active.max_attacks:
                    if active.is_stunned():
                        break
                    self.activate (active)
                    active.attacks_executed += 1
                state = self.full_save (3)
            if state.stage <= 3:
                reactive = self.active.opponent
                # check if attack needs to be re-executed
                while reactive.attacks_executed < reactive.max_attacks:
                    if reactive.is_stunned():
                        break
                    self.activate (reactive)
                    reactive.attacks_executed += 1
                state = self.full_save (4)

            # end triggers and evaluation
            for p in self.players_in_order():
                p.end_trigger()

            for p in self.player:
                p.unique_ability_end_trigger()

            return self.cycle_and_evaluate()

        # when a fork is raised, rerun the simulation with n_options values
        # appended to self.fork_decisions
        except ForkException as fork:
            # if a fork is created in interactive mode
            if self.interactive_mode:
                # save current state, so that we come back here in next sim
                self.interactive_state = state
                # save the decision counter
                # this is where the replay will go interactive again
                self.interactive_counter = self.decision_counter
                # switch to thinking mode
                self.interactive_mode = False
            # make list of results for all fork options
            results = []
            # locally remember the decision list, and reset it before each branch
            fork_decisions = self.fork_decisions[:]
##            print fork.forking_player, "-", fork.n_options
##            print "fork decisions:", fork_decisions
            for option in range (fork.n_options):
                self.fork_decisions = fork_decisions + [option]
                results.append (self.simulate (self.player[0].strat, \
                                               self.player[1].strat, \
                                               state))
            values = [r[0] for r in results]
            val = (max(values) if fork.forking_player.my_number == 0 \
                   else min(values))
            i = values.index(val)
            return results[i]

        # when a player wins, give them points
        except WinException as win:
            w = win.winner
            self.winner = w
            if self.reporting:
                if w == 0.5:
                    self.report("THE GAME IS TIED!")
                else:
                    self.report (self.player[w].name.upper() + " WINS!")
            if w == 0.5:
                value = 0
            else:
                value = ((-1)**w) * (5 + self.initial_state.player_states[1-w].life)
            final_state = self.full_save(None)
            return value, final_state, self.fork_decisions[:]

    def players_in_order (self):
        if self.active == self.player[0]:
            return [self.player[0], self.player[1]]
        else:
            return [self.player[1], self.player[0]]

    # activation for one player
    def activate (self, p):
        p.before_trigger()
        if p.is_attacking():
            if (p.can_hit() and p.opponent.can_be_hit() and not
                p.opponent.replace_hit()):
                if self.reporting:
                    self.report (p.name + " hits")
                p.hit_trigger()
                p.opponent.take_a_hit_trigger()
                # hits_scored is set after triggers, so that triggers can check
                # if this is first hit this beat
                p.hits_scored += 1
                if p.base.deals_damage:
                    p.deal_damage(p.get_power())
            else:
                if self.reporting:
                    self.report (p.name + " misses")
        p.opponent.after_trigger_for_opponent()
        p.after_trigger()

    def cycle_and_evaluate (self):
        for p in self.player:
            p.cycle ()

        # If 15 beats have been played, raise WinException
        if self.current_beat == 15 and not self.stop_the_clock:
            if self.reporting:
                self.report("Game goes to time")
            for p in self.player:
                if p.wins_on_timeout():
                    raise WinException(p.my_number)
            diff = self.player[0].life - self.player[1].life
            if diff > 0:
                raise WinException (0)
            elif diff < 0:
                raise WinException (1)
            else:
                # A tie.
                raise WinException (0.5)

        evaluation = self.evaluate()
        if self.debugging:
            for p in self.player:
                self.report (p.name+"'s life: " + \
                    str(self.initial_state.player_states[p.my_number].life) + \
                    " -> " + str(p.life))
            self.report ("preferred ranges: %.2f - %.2f    [%d]" \
                        %(self.player[0].preferred_range,\
                          self.player[1].preferred_range,\
                          self.distance()))
            self.report ("range_evaluation: %.2f - %.2f = %.2f" \
                        %(self.player[0].evaluate_range(), \
                          self.player[1].evaluate_range(), \
                          self.player[0].evaluate_range() - \
                          self.player[1].evaluate_range()))
            self.report ("eval: %.2f vs %.2f gives %.2f" \
                        %(evaluation, self.initial_evaluation, \
                          evaluation - self.initial_evaluation))
        final_state = self.full_save (None)
##        if self.reporting:
##            print "decisions returned: ", self.fork_decisions
        return evaluation - self.initial_evaluation, final_state, \
               self.fork_decisions[:]

    def evaluate(self):
        # Some characters (Tanis, Sarafina, Arec with clone)
        # can end the beat in a "superposition" of different positions.
        # We need to evaluate every possibility.
        real_positions = [p.position for p in self.player]
        positions = [p.get_superposed_positions() for p in self.player]
        # This is the order in which they'll collapse the superposition.
        priorities = [p.get_superposition_priority() for p in self.player]
        evaluations = []
        for p0 in positions[0]:
            row = []
            for p1 in positions[1]:
                if p0 != p1:
                    self.player[0].position = p0
                    self.player[1].position = p1
                    for p in self.player:
                        p.set_preferred_range()
                    row.append(self.player[0].evaluate() -
                               self.player[1].evaluate())
            if row:
                evaluations.append(row)
        for i in xrange(2):
            self.player[i].position = real_positions[i]
        # Higher priority chooses first, so evaluated last.
        # In case of tie, player 0 chooses before player 1
        # (it's arbitrary,
        #  but that's how it will happen during the ante phase)
        if priorities[0] >= priorities[1]:
            minima = [min(row) for row in evaluations]
            value = max(minima)
        else:
            evaluations = zip(*evaluations)
            maxima = [max(row) for row in evaluations]
            value = min(maxima)

        return value + (self.player[0].evaluate_superposition() -
                        self.player[1].evaluate_superposition())

    def distance (self):
        return abs(self.player[0].position - self.player[1].position)

    # Number of beats expected until end of game.
    def expected_beats (self):
        # TODO: This takes into account alternate counting of own life
        # (like Byron's), but not alternate counting of opponent's life
        # (like Adjenna's).
        return min(0.5 * min ([p.effective_life() for p in self.player]),
                   15 - self.current_beat)

    # check for a fork
    # n_options = number of branches in fork
    # player = who makes the decision?
    # prompt = string prompting a human player for this decision
    # options = strings enumerating the options for the player
    #   (if options = None, the question is numerical, no enumerated options)
    # choice = when this is not None, this is a "fake" fork, in which
    #       the AI will always pick the given choice.
    #       human player will be prompted normally
    def make_fork (self, n_options, player, prompt, options = None,
                   choice = None):
        # if no options, it's a bug.
        assert n_options > 0, "Fork with 0 options"
        # if just 1 option, no fork needed
        if n_options == 1:
            return 0
##        if self.debugging:
##            print "FORK"
##            print player.name + ": %d options;" %n_options,
##            print "   decisions:", self.fork_decisions, "   counter:", self.decision_counter, "    interactive:", self.interactive_counter
##            print "interactive mode:", self.interactive_mode, "        replay mode:", self.replay_mode

        # If the character who makes the decision is controled by the opponent,
        # let the opponent make the decision
        controlled = player.opponent.controls_opponent()
        if controlled:
            player = player.opponent
            prompt = "Make this decision for opponent:\n" + prompt

        # when a game against a human player is in active progress
        # the player is prompted to make a decision,
        # which is added to the list
        # any further decisions (from a replay) are deleted
        if self.interactive_mode and not self.replay_mode and \
           player.is_user:
            # prompt for decision, and delete any postulated decisions
            print prompt
            if options is None:
                print "[0-%d]"%(n_options-1)
                decision = input_number (n_options)
            else:
                decision = menu (options)
            self.fork_decisions = \
                            self.fork_decisions [:self.decision_counter]
            self.fork_decisions.append (decision)
            self.decision_counter += 1
            return decision
        # in all other situations, check whether the decision was
        # made

        # decision is in list:
        if self.decision_counter < len (self.fork_decisions):
            # in replay mode, switch back to interactive at correct spot
            if self.replay_mode and \
               self.decision_counter == self.interactive_counter:
                self.replay_mode = False
            # return decision from list, increment decision counter
            decision = self.fork_decisions [self.decision_counter]
##            if self.debugging:
##                print "returning fork"
##                print "fd:", self.fork_decisions
##                print "counter:", self.decision_counter
##                print "decision:", decision
            self.decision_counter += 1
            return decision
        # new decision:
        else:
            # If it's a fake fork, return the given choice
            # and add it to fork_decisions.
            # (unless the decision is controlled by opponent - in which
            # case ignore the given a choice and make it normally.)
            if choice != None and not controlled:
                self.fork_decisions.append (choice)
                self.decision_counter += 1
                return choice
            # in a real fork, raise the Exception
            # in interactive mode, saving the current state for next sim
            # and switching to thinking mode are handled by the except block
            else:
                raise ForkException (n_options, player)

    def get_board (self):
        addenda = []
        for p in self.player:
            a = p.get_board_addendum ()
            if a:
                if isinstance (a, str):
                    a = [a]
                addenda += a
        return [''] + [self.get_basic_board()] + addenda + ['']

    def get_basic_board (self):
        board = ['.'] * 7
        for p in self.player:
            if p.position is not None:
                board[p.position] = p.get_board_symbol()
        return ''.join(board)

    # find minmax for results table
    def solve (self):
        self.value = [[None] * len(self.strats[1])
                      for s in self.strats[0]]
        for p in self.player:
            p.mix = [[None] * len(self.strats[1])
                     for s in self.strats[0]]
        self.pre_clash_results = [[[row[:]
                                    for row in pad_col]
                                   for pad_col in pad_row]
                                  for pad_row in self.results]
        for pad0 in xrange(len(self.strats[0])):
            for pad1 in xrange(len(self.strats[1])):
                value, mix0, mix1 = self.solve_per_pad(
                                      self.results[pad0][pad1],
                                      self.pre_clash_results[pad0][pad1],
                                      self.strats[0][pad0],
                                      self.strats[1][pad1])
                self.value[pad0][pad1] = value
                self.player[0].mix[pad0][pad1] = mix0
                self.player[1].mix[pad0][pad1] = mix1

    # Find minmax for one results table (for given set of pads).
    def solve_per_pad(self, results, pre_clash_results, strats0, strats1):
        self.fix_clashes(results, pre_clash_results, strats0, strats1)
        self.fix_cancels(results)
        array_results = numpy.array(results)
        (mix0, value0) = solve.solve_game_matrix (array_results)
        stratmix0 = zip (strats0, list(mix0))
        # No need to calculate strategy mix for human player
        if not self.player[1].is_user:
            (mix1, value1) = solve.solve_game_matrix (-array_results.transpose())
            stratmix1 = zip (strats1, list(mix1))
            assert abs(value0 + value1) < 0.01, ("Error: value0=%f, value1=%f" %
                                                 (value0, value1))
        else:
            stratmix1 = [(s, 0) for s in strats1]
        return value0, stratmix0, stratmix1

    def make_pre_attack_decision(self):
        # Player 0 makes a decision.
        values = [row[0] for row in self.value]
        d0 = max([(val, i) for i, val in enumerate(values)])[1]
        # Player 1 makes a decision.
        if self.interactive:
            d1 = self.player[1].input_pre_attack_decision_index()
        else:
            values = self.value[0]
            d1 = min([(val, i)
                      for i, val in enumerate(values)])[1]
        self.value = self.value[d0][d1]
        self.results = self.results[d0][d1]
        self.pre_clash_results = self.pre_clash_results[d0][d1]
        for p in self.player:
            p.mix = p.mix[d0][d1]
        self.player[0].final_pad = self.pads[0][d0]
        self.player[1].final_pad = self.pads[1][d1]
        return (self.player[0].pre_attack_decision_report(self.pads[0][d0]) +
                self.player[1].pre_attack_decision_report(self.pads[1][d1]))

    def execute_beat (self, post_clash=False):
        self.initial_restore (self.initial_state)
        s0 = self.player[0].choose_strategy(post_clash)
        if self.interactive and self.cheating == 2:
            print self.player[0], "plays", self.player[0].get_strategy_name(s0)
        s1 = self.player[1].choose_strategy(post_clash)
        if self.interactive:
            self.interactive_state = None
            self.replay_mode = False
            while not self.interactive_mode:
                self.interactive_mode = True
                value, final_state, self.fork_decisions = \
                                self.simulate (s0, s1, self.interactive_state)
                self.replay_mode = True
            self.interactive_mode = False
        else:
            value, final_state, unused_forks = self.simulate (s0, s1)
        report = final_state.reports

        ss0 = [m[0] for m in self.player[0].mix]
        ss1 = [m[0] for m in self.player[1].mix]

        # Solve Cancels
        if value in self.CANCEL_INDICATORS:
            # Both players can only cancel into (non special action)
            # strategies with the same ante and no components of original
            # strategy (this last restriction will only affect
            # any player who didn't cancel).
            post_cancel_strats0 = [s for s in ss0 if
                    s[2] == s0[2] and
                    s[0] != s0[0] and
                    s[1] != s0[1] and
                    not isinstance (s[0], SpecialAction)]
            post_cancel_strats1 = [s for s in ss1 if
                    s[2] == s1[2] and
                    s[0] != s1[0] and
                    s[1] != s1[1] and
                    not isinstance (s[0], SpecialAction)]
            # Before re-simulating, we need to update the game state
            # (record the lost special action/s and any discarded
            # attack pairs).
            self.initial_restore (self.initial_state)
            # Anyone who used a special action (for cancel or finisher)
            # loses the action:
            if isinstance(s0[0], SpecialAction):
                self.player[0].special_action_available = False
            if isinstance(s1[0], SpecialAction):
                self.player[1].special_action_available = False
            # In a double cancel, both players put special action
            # in discard.
            if value == self.CANCEL_BOTH_INDICATOR:
                for p in self.player:
                    p.discard[1].add(p.special_action)
            # In a single cancel, opponent puts pair in discard.
            else:
                s01 = (s0,s1)
                opponent = (self.player[0] if self.CANCEL_1_INDICATOR
                            else self.player[1])
                opp_strat = s01[opponent.my_number]
                # Discard style, which may be special action.
                opponent.discard[1].add(opp_strat[0])
                # Discard base.  If using a finisher, decide which
                # base it was.
                if isinstance(opp_strat[0], SpecialAction):
                    self.dump(report)
                    base = opponent.choose_finisher_base_retroactively()
                else:
                    base = opp_strat[1]
                opponent.discard[1].add(base)

            self.initial_state = self.initial_save ()
            # Re-simulate available strategies with updated situation.
            post_cancel_results = [[float(self.simulate (t0,t1)[0])
                                    for t1 in post_cancel_strats1]
                                   for t0 in post_cancel_strats0]
            # solve new table
            array_results = numpy.array(post_cancel_results)
            (mix0, value0) = solve.solve_game_matrix (array_results)
            (mix1, value1) = solve.solve_game_matrix (-array_results.transpose())
            stratmix0 = zip (post_cancel_strats0, list(mix0))
            stratmix1 = zip (post_cancel_strats1, list(mix1))
            assert abs(value0 + value1) < 0.01, ("Error: value0=%f, value1=%f" %
                                                 (value0, value1))
            value = value0
            self.player[0].mix = stratmix0
            self.player[1].mix = stratmix1
            # Both players choose new strategies
            if self.interactive:
                self.dump(report)
                if self.cheating > 0:
                    self.dump (self.report_solution())
            else:
                report.extend (self.report_solution())
            s0 = self.player[0].choose_strategy(limit_antes=True)
            if self.interactive and self.cheating == 2:
                print self.player[0], "plays", self.player[0].get_strategy_name(s0)
            s1 = self.player[1].choose_strategy(limit_antes=True)

            # Simulate beat based on new solutions
            if self.interactive:
                self.interactive_state = None
                self.replay_mode = False
                while not self.interactive_mode:
                    self.interactive_mode = True
                    value, final_state, self.fork_decisions = \
                                  self.simulate (s0, s1, self.interactive_state)
                    self.replay_mode = True
                self.interactive_mode = False
            else:
                value, final_state, unused_forks = self.simulate (s0, s1)
            report.extend (final_state.reports)

        # if we have a real result (possibly after the cancel/s)
        # return it
        if value != self.CLASH_INDICATOR:
            return final_state, report

        # clash - find strategies that can be switched into
        # (same style and ante, different base)
        g0 = [ii for ii in range(len(ss0))
              if  ss0[ii][0] == s0[0]
              and ss0[ii][2] == s0[2]
              and ss0[ii][1] != s0[1]]
        g1 = [jj for jj in range(len(ss1))
              if  ss1[jj][0] == s1[0]
              and ss1[jj][2] == s1[2]
              and ss1[jj][1] != s1[1]]
        # if one player ran out of bases in clash, just cycle
        # (not sure if end-of-beat UAs should happen,
        # but some of them (Voco) require Forks, so I can't
        # have them outside of simulate(), anyway)
        if min (len(g0),len(g1)) == 0:
            for p in self.player:
                p.cycle ()
            state = self.full_save (None)
            report.append ("\nout of bases - cycling")
            return state, report
        # make sub matrix of remaining results
        i = ss0.index(s0)
        j = ss1.index(s1)
        p0 = self.player[0]
        p1 = self.player[1]
        self.results = [[self.pre_clash_results
                               [p0.clash_strat_index(ii,jj,i,j)]
                               [p1.clash_strat_index(jj,ii,j,i)]
                         for jj in g1]
                        for ii in g0]
        self.pre_clash_results = [row[:] for row in self.results]
        # make vectors of remaining strategies
        self.strats[0] = self.player[0].fix_strategies_post_clash(
                                        [ss0[i] for i in g0], s1)
        self.strats[1] = self.player[1].fix_strategies_post_clash(
                                        [ss1[j] for j in g1], s0)
        # solve clash
        value, self.player[0].mix, self.player[1].mix = \
            self.solve_per_pad(self.results, self.pre_clash_results,
                               self.strats[0], self.strats[1])
        # Run this function recursively with post-clash strategies only.
        if self.interactive:
            self.dump(report)
        recursive_state, recursive_report = self.execute_beat(post_clash=True)
        report.extend(recursive_report)
        return recursive_state, report


    # fix clash results
    def fix_clashes (self, results, pre_clash_results, ss0, ss1):
        if self.clash0:
            return
        n = len (results)
        m = len (results[0])
        # if at least one matrix dimension is 1, clashes are final,
        # and approximated with 0
        final_clashes = (min(n,m) == 1)
        # when each clash is resolved, other clashes should still be unresolved
        for i in range(n):
            for j in range(m):
                if pre_clash_results[i][j] == self.CLASH_INDICATOR:
                    if final_clashes:
                        # Cutting a corner here:
                        # In fact, when someone runs out of bases, cycling
                        # happens, which might be better for one player.
                        results[i][j] = 0
                    else:
                        # find indices of strategies that share
                        # style and ante decisions with clash
                        # (but not base, i.e, not exact same strategy)
                        g0 = [ii for ii in range(n)
                              if ss0[ii][0]==ss0[i][0]
                              and ss0[ii][2]==ss0[i][2]
                              and ss0[ii][1]!=ss0[i][1]]
                        g1 = [jj for jj in range(m)
                              if ss1[jj][0]==ss1[j][0]
                              and ss1[jj][2]==ss1[j][2]
                              and ss1[jj][1]!=ss1[j][1]]
                        # make sub matrix of those results
                        p0 = self.player[0]
                        p1 = self.player[1]
                        try:
                            subresults = [[pre_clash_results
                                       [p0.clash_strat_index(ii,jj,i,j)]
                                       [p1.clash_strat_index(jj,ii,j,i)]
                                       for jj in g1]
                                      for ii in g0]
                        except Exception as e:
                            print "Failure"
                            print "n m i j"
                            print n, m, i, j
                            print ss0[i]
                            print ss1[j]
                            print g0
                            print g1
                            print "p0 clash indices"
                            for ii in g0:
                                for jj in g1:
                                    print p0.clash_strat_index(ii,jj,i,j),
                                print
                            print "p1 clash indices"
                            for ii in g0:
                                for jj in g1:
                                    print p1.clash_strat_index(jj,ii,j,i),
                                print
                            raise e
                        # and solve it
                        results[i][j] = self.sub_solve(subresults)

    # Fix cancel results in matrix.
    # Until a better solution is found, just cause AI to ignore the
    # possibility: it doesn't play cancel, or play around cancel.
    def fix_cancels(self, results):
        n = len(results)
        m = len(results[0])
        for i in range(n):
            for j in range(m):
                if results[i][j] in self.CANCEL_INDICATORS:
                    if results[i][j] == self.CANCEL_0_INDICATOR:
                        results[i][j] = -self.EXTREME_RESULT
                    elif results[i][j] == self.CANCEL_1_INDICATOR:
                        results[i][j] = self.EXTREME_RESULT
                    else:
                        results[i][j] = 0

    # solves 4x4 (or smaller) matrix created by clash
    def sub_solve (self, matrix):
        n = len(matrix)
        m = len(matrix[0])
        # trivial matrix: one dimensional
        if min(n,m) == 1:
            # a final clash ends the beat, approximate as 0:
            for i in range(n):
                for j in range(m):
                    if matrix[i][j] == self.CLASH_INDICATOR:
                        matrix[i][j] = 0
            # solve matrix trivially
            if n==1:
                return min(matrix[0])
            else: #m=1
                return max([row[0] for row in matrix])
        # non-trivial matrix
        # fix clashes
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == self.CLASH_INDICATOR:
                    sub_matrix = [[matrix[ii][jj] for jj in range(m) if jj!=j] \
                                                  for ii in range(n) if ii!=i]
                    matrix[i][j] = self.sub_solve(sub_matrix)
        (unused_sol, val) = solve.solve_game_matrix (numpy.array(matrix))
        return val

    # return report of positive strategies
    def report_solution (self):
        report = []
        for p in self.player:
            report.append (p.name + ':')
            for m in p.mix:
                if m[1] > 0.0001:
                    report.append (str(int(100*m[1]+.5))+"% "+ \
                                   p.get_strategy_name(m[0]))
            report.append ("")
        return report

    # print solution with positive strategies,
    # plus any strategies called by name in extra
    # if extra contains a character name, show all for that character
    # transpose if it's p1
    def print_solution (self, extra = []):
        if isinstance (extra, str):
            extra = [extra]
        extra = [e.lower() for e in extra]
        for i, pad0 in enumerate(self.pads[0]):
            for j, pad1 in enumerate(self.pads[1]):
                if len(self.pads[0]) > 1:
                    print "Pre attack decision: %s" % pad0
                if len(self.pads[1]) > 1:
                    print "Pre attack decision: %s" % pad1
                # for each player
                for p in self.player:
                    # keep only positive probs
                    if p.name.lower() in extra:
                        p.filtered_indices = range(len(p.mix[i][j]))
                    else:
                        p.filtered_indices = [k for k in xrange(len(p.mix[i][j])) \
                                              if p.mix[i][j][k][1]>0.0001 \
                                              or p.get_strategy_name(p.mix[i][j][k][0]).lower() in extra]
                    p.filtered_mix = [p.mix[i][j][k] for k in p.filtered_indices]
                    r = random.random()
                    total = 0
                    print "\n", p
                    for m in p.filtered_mix:
                        print str(int(100*m[1]+.5))+"%", \
                              p.get_strategy_name(m[0]),
                        if total+m[1] >= r and total < r:
                            print ' ***',
                        print " "
                        total = total + m[1]
                print '\n' + self.player[0].name + '\'s Value:', self.value[i][j], '\n'
                small_mat = numpy.array ([[self.results[i][j][k][m] \
                                     for m in self.player[1].filtered_indices] \
                                     for k in self.player[0].filtered_indices])
                # if all player 1 strategies displayed, transpose for ease of reading
                if self.player[1].name.lower() in extra and \
                   self.player[0].name.lower() not in extra:
                    small_mat = small_mat.transpose()
                    print "(transposing matrix)"
                print small_mat.round(2)

    # game value if one player uses given strategy, and other player uses
    # calculated mix
    # Assumes no pre-attack decisions (so no Tanis)
    def vs_mix (self, name):
        name = name.lower()
        mix0 = self.player[0].mix[0][0]
        mix1 = self.player[1].mix[0][0]
        ii = [i for i in range(len(mix0))
              if self.player[0].get_strategy_name(mix0[i][0]).lower() == name]
        jj = [j for j in range(len(mix1))
              if self.player[1].get_strategy_name(mix1[j][0]).lower() == name]
        for i in ii:
            value = 0
            for j in range(len(mix1)):
                value += self.results[0][0][i][j] * mix1[j][1]
            print value
        for j in jj:
            value = 0
            for i in range(len(mix0)):
                value += self.results[0][0][i][j] * mix0[i][1]
            print value

    # for each strategy (of each player), print worst possible case
    def worst_case (self):
        array_results = numpy.array(self.results)
        n,m = array_results.shape
        worst = array_results.argmin (1)
        for i in range(n):
            print array_results[i,worst[i]], ':',
            print self.player[0].get_strategy_name(
                self.player[0].strats[i]), '--->',
            print self.player[1].get_strategy_name(
                self.player[1].strats[worst[i]])
        print '##################################################'
        worst = array_results.argmax (0)
        for j in range(m):
            print array_results[worst[j],j], ':',
            print self.player[1].get_strategy_name(
                self.player[1].strats[j]), '--->',
            print self.player[0].get_strategy_name(
                self.player[0].strats[worst[j]])

    # run one simulation by strategy names
    # and print reports
    def debug (self, name0, name1, full_debug = True):
        name0 = name0.lower()
        name1 = name1.lower()
        self.debugging = full_debug
        self.reporting = True
        self.initial_restore (self.initial_state)
        for p in self.player:
            p.set_preferred_range()
        print ("preferred ranges: %.2f - %.2f    [%d]" \
                        %(self.player[0].preferred_range,\
                          self.player[1].preferred_range,\
                          self.distance()))
        print ("range_evaluation: %.2f - %.2f = %.2f" \
                        %(self.player[0].evaluate_range(), \
                          self.player[1].evaluate_range(), \
                          self.player[0].evaluate_range() - \
                          self.player[1].evaluate_range()))
        s0 = [s for s in self.player[0].strats \
              if self.player[0].get_strategy_name(s).lower() == name0]
        s1 = [s for s in self.player[1].strats \
              if self.player[1].get_strategy_name(s).lower() == name1]
        unused_value, state, forks = self.simulate (s0[0], s1[0])
        self.debugging = False
        self.reporting = False
        for s in state.reports:
            print s
        return state, forks

    # add a string to reports
    def report (self, s):
        if self.interactive_mode:
            if not self.replay_mode:
                print s
                self.log.append(s)
        else:
            self.reports.append(s)

    # re-solve matrix assuming that one player antes first
    def first_ante (self, first):
        ss0 = self.player[0].strats
        ss1 = self.player[1].strats
        array_results = numpy.array (self.results)
        # assumes first anteer is player 0
        # if not, reverse everything now, then put it back at the end
        if (first == 1):
            array_results = -array_results.transpose()
            ss0, ss1 = ss1, ss0

        ab0,ab1 = array_results.shape
        b0 = len(set(s[2] for s in ss0))
        b1 = len(set(s[2] for s in ss1))
        if ab0%b0 != 0 or ab1%b1 != 0:
            print "Total strategies not divisible by antes"
            return
        a0 = ab0/b0
        a1 = ab1/b1
        (mix0, value0) = solve.solve_for_player0_with_b0_known \
                         (array_results, a0,b0,a1,b1)
        (mix1, value1) = solve.solve_for_player0_with_b1_known \
                         (-array_results.transpose(), a1,b1,a0,b0)
        # mix1 (2nd ante) has full pair/ante mix for each ante of player 0
        # followed by pre-ante pair mix
        # for each pair 1 x ante 0: print ante 1 mix
        for a1i in range (a1):
            pair_prob = mix1 [b0*a1*b1 + a1i]
            if pair_prob > 0.0001:
                for b0i in range (b0):
                    print self.player[1-first].get_strategy_name \
                                                    (ss1[a1i*b1]),
                    print 'vs.',
                    opposing_ante = \
                        self.player[first].get_ante_name(ss0[b0i][2])
                    print ("No Ante" if opposing_ante == "" else opposing_ante)
                    for b1i in range (b1):
                        prob = mix1[b0i*a1*b1 + a1i*b1 + b1i]
                        if prob > .0001:
                            print "   ", str(int(100*prob/pair_prob+.5))+"%",
                            my_ante = self.player[1-first].get_ante_name \
                                                    (ss1[b1i][2])
                            print ("No Ante" if my_ante == "" else my_ante)
        print

        spread = [0.0 for i in range (ab1)]
        for i in range(a1):
            spread [i*b1] = mix1[b0*a1*b1 + i]
        mix1 = spread

        # if results were transposed, put them back,
        if (first == 1):
            array_results = -array_results.transpose()
            ss0, ss1 = ss1, ss0
            mix0, mix1 = mix1, mix0
            value0, value1 = value1, value0

        stratmix0 = zip (ss0,mix0)
        stratmix1 = zip (ss1,mix1)
        if abs (value0 + value1) > 0.01:
            print "ERROR:"
            print "  value0:", value0
            print "  value1:", value1
            raise Exception()
        self.value = value0
        self.player[0].mix = stratmix0
        self.player[1].mix = stratmix1

        self.print_solution()

    def prepare_next_beat (self):
        if not self.stop_the_clock:
            self.current_beat += 1
        for p in self.player:
            p.prepare_next_beat()


class GameState (object):
    pass

class PlayerState (object):
    pass


class Character (object):

    # Bureaucratic Methods

    def __init__(self, the_game, n, base_set='alpha', is_user=False):
        self.game = the_game
        self.my_number = n
        self.is_user = is_user
        self.starting_life = 20

        # insert spaces before capitals
        self.name = re.sub(r'([a-z](?=[A-Z0-9])|[A-Z0-9](?=[A-Z0-9][a-z]))', r'\1 ',
                           self.__class__.__name__)

        # Placeholders that do nothing.  You have them until the reveal step.
        self.null_style = NullStyle (the_game, self)
        self.null_base = NullBase (the_game, self)

        # All Characters have all bases (alpha, beta, delta).
        # This allows conversion of initial discards from alpha to
        # beta/delta.
        self.strike = Strike (the_game,self)
        self.shot = Shot (the_game,self)
        self.drive = Drive (the_game,self)
        self.burst = Burst (the_game,self)
        self.grasp = Grasp (the_game,self)
        self.dash = Dash (the_game,self)
        # Counter and Wave avoid name-clashes with styles of same name.
        self.counter_base = Counter (the_game,self)
        self.wave_base = Wave (the_game,self)
        self.force = Force (the_game,self)
        self.spike = Spike (the_game,self)
        self.throw = Throw (the_game,self)
        self.parry = Parry (the_game,self)
        self.stagger = Stagger (the_game,self)
        self.leap = Leap (the_game,self)
        self.lunge = Lunge (the_game,self)
        self.sweep = Sweep (the_game,self)
        self.catch = Catch (the_game,self)
        self.block = Block (the_game,self)

        self.base_set = base_set

        # If character doesn't have specific list of bases, give
        # the standard/beta bases:
        try:
            self.bases
        except:
            if base_set == 'alpha':
                self.bases = [self.unique_base,
                              self.strike,
                              self.shot,
                              self.drive,
                              self.burst,
                              self.grasp,
                              self.dash]
            elif base_set == 'beta':
                self.bases = [self.unique_base,
                              self.counter_base,
                              self.wave_base,
                              self.force,
                              self.spike,
                              self.throw,
                              self.parry]
            else:
                self.bases = [self.unique_base,
                              self.stagger,
                              self.leap,
                              self.lunge,
                              self.sweep,
                              self.catch,
                              self.block]

        # Record characters starting cards, in case they lose some
        # during the game.
        self.initial_styles = self.styles[:]
        self.initial_bases = self.bases[:]

        # If specific character's __init__ didn't create lists of
        # status effects or tokens, make an empty ones.
        try:
            self.status_effects
        except:
            self.status_effects = []
        try:
            self.tokens
        except:
            self.tokens = []
        self.pool = []
        self.max_counters = 0
        self.counter_name = None
        self.counters = 0

        self.styles_and_bases_set = set(self.styles) | set(self.bases)
        for i in xrange(5):
            self.styles[i].order = i
        for i in xrange(7):
            self.bases[i].order = i
        # special action card (a style) and bases.
        # (finishers are unique to each character)
        self.special_action = SpecialAction (the_game,self)
        self.pulse = Pulse (the_game,self)
        self.cancel = Cancel (the_game,self)
        # This randomly assigns bases for Juto and Borneo.
        self.pulse_generators = set ([self.bases[4], self.bases[6]])
        self.cancel_generators = set ([self.bases[2], self.bases[5]])
        self.finisher_generators = set ([self.bases[0], self.bases[1],
                                         self.bases[3]])
        # discard[0] is for cards played this beat.
        # they will cycle into discard[1] at end of beat.
        # (discard [0] is empty between beats)
        self.discard = [set() for i in xrange(3)]

        # Create attributes for all card-like objects
        for card in self.all_cards():
            name = card.name.replace('-','_').replace(' ','_').replace("'",'').lower()
            self.__dict__[name] = card

        self.set_mean_priority()

    def select_finisher(self):
        if len(self.finishers) == 1:
            print "Only one Finisher implemented for %s" % self.name
        else:
            if self.is_user:
                print "Select a Finisher for %s:" % self.name
                ans = menu([f.name for f in self.finishers])
            else:
                ans = int(random.random() * len(self.finishers))
            self.finishers = [self.finishers[ans]]
            print "%s selects a Finisher: %s" %(self.name,
                                                self.finishers[0].name)

    # Used by Game to add opponent to all your cards/tokens etc.,
    # and by Character to create specifically named attributes for each
    # object.
    def all_cards (self):
        return self.styles + self.bases + self.tokens + self.finishers + \
               self.status_effects + [self.special_action, self.pulse,
                self.cancel, self.null_style, self.null_base]

    def __str__ (self):
        return self.name

    def logfile_name(self):
        name = self.name
        if self.base_set != 'alpha':
            name += "(%s)" % self.base_set
        if self.opponent.is_user:
            name = name + "(AI)"
        return name

    def name_with_base_set(self):
        name = self.name
        if self.base_set != 'alpha':
            name += " (%s)" % self.base_set
        return name

    # read character state from list of strings (as written in game log)
    # lines - list of strings for this character's situation report
    # board - board string for position
    # addendum - list of strings for this character's board addendum
    # Returns lines specific to the character, for the benefit of inherited
    # methods.
    def read_my_state (self, lines, board, addendum):
        self.life = int(lines[0].split()[-1])
        self.position = board.find(self.get_board_symbol())
        if self.position == -1:
            self.position = None
        cards = self.styles + self.bases
        for d in (1,2):
            self.discard[d] = set(c for c in cards if c.name in lines[d])
        # Figure out where standard portion of report ends,
        # and any character specific data starts.
        next_line_index = 3
        if self.counter_name:
            counters_line = find_start_line(lines, "%s: " % self.counter_name)
            if counters_line:
                self.counters = int(counters_line.split()[1])
            next_line_index += 1
        if (find_start(lines, 'Special action available') or
            find_start(lines, 'special action')):
            self.special_action_available = True
            next_line_index += 1
        else:
            self.special_action_available = False
        removed_line = find_start_line(lines, 'Removed cards:')
        if removed_line:
            self.styles = [s for s in self.styles if s.name not in removed_line]
            self.bases = [b for b in self.bases if b.name not in removed_line]
            next_line_index += 1
        self.active_status_effects = []
        for status in self.status_effects:
            if status.read_my_state(lines) is not None:
                next_line_index += 1
        return lines[next_line_index:]

    # set character state for start of a game
    def set_starting_setup (self, default_discards, use_special_actions):
        self.life = self.starting_life
        self.position = (2 if self.my_number==0 else 4)
        self.special_action_available = use_special_actions

        # Set initial discard piles
        if self.game.first_beats:
            # Have all cards available for initial discard checks
            self.discard[1] = set()
            self.discard[2] = set()
        elif default_discards:
            self.discard[1] = set((self.styles[0], self.bases[6]))
            self.discard[2] = set((self.styles[1], self.bases[5]))
        else:
            # get starting discard from player / ai
            if self.is_user:
                styles = self.styles[:]
                bases = self.bases[:]
                print "Discard a style to discard 1:"
                s1 = styles[menu([s.name for s in styles])]
                styles.remove(s1)
                print "Discard a style to discard 2:"
                s2 = styles[menu([s.name for s in styles])]
                print "Discard a base to discard 1:"
                b1 = bases[menu([b.name for b in bases])]
                bases.remove(b1)
                print "Discard a base to discard 2:"
                b2 = bases[menu([b.name for b in bases])]
            else:
                s1,b1,s2,b2 = self.choose_initial_discards()
                b1 = [b for b in self.bases
                      if b.alpha_name == b1.alpha_name][0]
                b2 = [b for b in self.bases
                      if b.alpha_name == b2.alpha_name][0]
            self.discard[1] = set ((s1,b1))
            self.discard[2] = set ((s2,b2))
        self.active_status_effects = []

    def situation_report (self):
        report = []
        full_name = self.name_with_base_set()
        report.append (full_name)
        report.append ('-' * len (full_name))
        report.append ('Life: %d' %self.life)
        report.append ('Discard 1: ' + self.discard_report (self.discard[1]))
        report.append ('Discard 2: ' + self.discard_report (self.discard[2]))
        if self.counter_name is not None:
            report.append('%s: %d' % (self.counter_name, self.counters))
        initial_cards = set(self.initial_styles + self.initial_bases)
        cards = set(self.styles + self.bases)
        missing_cards = initial_cards - cards
        if missing_cards:
            report.append('Removed cards: ' + self.discard_report (missing_cards))
        if self.special_action_available:
            report.append ('Special action available')
        for status in self.active_status_effects:
            status.situation_report(report)
        return report

    def discard_report (self, pile):
        styles = [card.name for card in pile if isinstance (card, Style)]
        bases = [card.name for card in pile if isinstance (card, Base)]
        other = [card.name for card in pile if not isinstance (card, Style) \
                                           and not isinstance (card, Base)]
        return ' '.join(styles+bases+other)

    # save player state before strategy selection
    def initial_save (self):
        state = PlayerState()
        state.life = self.life
        state.position = self.position
        state.discard = [d.copy() for d in self.discard]
        state.special_action_available = self.special_action_available
        state.pool = self.pool[:]
        state.counters = self.counters
        state.styles = self.styles[:]
        state.bases = self.bases[:]
        return state

    # restore state before setting a new strategy
    def initial_restore (self, state):
        self.life = state.life
        self.position = state.position
        self.discard = [d.copy() for d in state.discard]
        self.special_action_available = state.special_action_available
        self.pool = state.pool[:]
        self.counters = state.counters
        self.styles = state.styles[:]
        self.bases = state.bases[:]

    # reset character to status at start of beat
    def reset (self):
        self.damage_taken = 0
        self.was_stunned = False
        self.hits_scored = 0
        self.moved = False
        self.triggered_dodge = False
        self.attacks_executed = 0
        self.max_attacks = 1
        self.ante = []
        self.active_cards = []
        # replaced in reveal phase
        self.style = self.null_style
        self.base = self.null_base
        # accumulated bonuses from triggers this beat
        self.triggered_power_bonus = 0
        self.triggered_priority_bonus = 0
        # if not None, someone has replaced the power of your pair
        # with this number
        self.alt_pair_power = None
        self.pending_status_effects = []

    # snapshot of game state mid-turn, for forks
    def full_save (self):
        state = self.initial_save()
        state.damage_taken = self.damage_taken
        state.was_stunned = self.was_stunned
        state.hits_scored = self.hits_scored
        state.moved = self.moved
        state.triggered_dodge = self.triggered_dodge
        state.attacks_executed = self.attacks_executed
        state.max_attacks = self.max_attacks
        state.ante = self.ante[:]
        state.active_cards = self.active_cards[:]
        state.style = self.style
        state.base = self.base
        state.triggered_power_bonus = self.triggered_power_bonus
        state.triggered_priority_bonus = self.triggered_priority_bonus
        state.alt_pair_power = self.alt_pair_power
        state.pending_status_effects = self.pending_status_effects[:]
        for status in self.pending_status_effects:
            status.full_save(state)
        return state

    # restoration of mid-turn snapshot (in a fork)
    def full_restore (self, state):
        self.initial_restore (state)
        self.damage_taken = state.damage_taken
        self.was_stunned = state.was_stunned
        self.hits_scored = state.hits_scored
        self.moved = state.moved
        self.triggered_dodge = state.triggered_dodge
        self.attacks_executed = state.attacks_executed
        self.max_attacks = state.max_attacks
        self.ante = state.ante[:]
        self.active_cards = state.active_cards[:]
        self.style = state.style
        self.base = state.base
        self.triggered_power_bonus = state.triggered_power_bonus
        self.triggered_priority_bonus = state.triggered_priority_bonus
        self.alt_pair_power = state.alt_pair_power
        self.pending_status_effects = state.pending_status_effects[:]
        for status in self.pending_status_effects:
            status.full_restore(state)

    # switch "next beat" bonuses to "this beat", etc.
    def prepare_next_beat (self):
        self.active_status_effects = self.pending_status_effects[:]
        for status in self.active_status_effects:
            status.prepare_next_beat()

    def get_pairs (self):
        unavailable_cards = self.discard[1] \
                          | self.discard[2]
        styles = sorted (set(self.styles) - unavailable_cards,
                         key = attrgetter('order'))
        bases = sorted (set(self.bases) - unavailable_cards,
                        key = attrgetter('order'))
        pairs = [(s,b) for s in styles for b in bases]
        bases = set(bases)
        if self.special_action_available:
            if self.pulse_generators & bases:
                pairs.append ((self.special_action, self.pulse))
            if self.cancel_generators & bases:
                pairs.append ((self.special_action, self.cancel))
            if self.finisher_generators & bases:
                for finisher in self.finishers:
                    pairs.append ((self.special_action, finisher))
        return pairs

    # Prompt user for a style and a base.
    # Available cards based on cards in strategy mix
    # (so that they are suitably limited after a clash).
    def input_pair (self):
        styles = sorted(list(set(m[0][0] for m in self.mix)),
                       key=attrgetter('order'))
        bases = sorted(list(set(m[0][1] for m in self.mix)),
                       key=attrgetter('order'))
        # friendly reminder of bases in hand when asking about styles
        if len(styles) > 1:
            print "(Bases in hand: " + \
                  ', '.join(b.name for b in bases
                            if not isinstance (b, SpecialBase)) + ")\n"
        style = (styles[menu(styles)] if len(styles) > 1 else styles[0])
        if isinstance (style, SpecialAction):
            bases = [b for b in bases if isinstance (b, SpecialBase)]
        else:
            bases = [b for b in bases if not isinstance (b, SpecialBase)]
        base = (bases[menu(bases)] if len(bases) > 1 else bases[0])
        return (style, base)

    def get_antes (self):
        return [None]

    def input_ante (self, pair):
        return None

    # Used by characters who give ante decisions to their opponents.
    # (Rexan/Alexian)
    def get_induced_antes (self):
        return [None]

    def input_induced_ante (self):
        return None

    # Used by characters who make a choice public before setting
    # attack pairs (Tanis).
    def get_pre_attack_decisions(self):
        return [None]

    def input_pre_attack_decision_index(self):
        return 0

    # Strategy format:
    # (style, base, (ante, induced_ante, pre_attack_decision))
    # induced_ante is the ante decision you made because of opponent's
    # ability (e.g. vs. Rexan/Alexian).
    # pre_attack_decision is any decision made public before attack pairs
    # are chosen (e.g., Tanis).
    def get_strategies (self):
        return [pair + ((ante, induced_ante, pre_attack_decision),)
                for pre_attack_decision in self.get_pre_attack_decisions()
                for pair in self.get_pairs()
                for ante in self.get_antes()
                for induced_ante in self.opponent.get_induced_antes()]

    def input_strategy (self, limit_antes):
        pair = self.input_pair()
        if limit_antes:
            ante = self.mix[0][0][2]
        # Otherwise, report ai's ante choice
        # and prompt for human ante choice
        else:
            opp = self.opponent
            opp_own_ante_name = opp.get_ante_name(opp.chosen_ante[0])
            opp_induced_ante_name = self.get_induced_ante_name(opp.chosen_ante[1])
            names = [name for name in (opp_own_ante_name,
                                       opp_induced_ante_name)
                     if name]
            opp_ante_name = ' | '.join(names)
            if opp_ante_name != "" :
                print opp.name + "'s ante: " + opp_ante_name
            own_ante = self.input_ante(pair)
            induced_ante = opp.input_induced_ante()
            ante = (own_ante, induced_ante, self.final_pad)
        return pair + (ante,)

    # choose a random strategy according to mix
    # (or prompt human player for a strategy)
    def choose_strategy (self, limit_antes=False):
        if self.is_user:
            strategy = self.input_strategy(limit_antes)
        else:
            # If there's only one option, return it.
            if len(self.mix) == 1:
                return self.mix[0][0]
            r = random.random()
            total = 0
            for m in self.mix:
                total = total + m[1]
                if total >= r :
                    strategy = m[0]
                    break
        # in case I need to report my ante choice to opponent's input_strategy
        self.chosen_ante = strategy[2]
        return strategy

    def get_strategy_name (self, s):
        name = s[0].name + " " + s[1].name
        own_ante_name = self.get_ante_name (s[2][0])
        if own_ante_name:
            name += " (%s)" % own_ante_name
        induced_ante_name = self.opponent.get_induced_ante_name (s[2][1])
        if induced_ante_name:
            name += " | (%s)" % induced_ante_name
        return name

    def get_ante_name (self, ante):
        return ""

    def get_induced_ante_name (self, ante):
        return ""

    def get_board_symbol (self):
        if self.my_number == 0 or self.name[0] != self.opponent.name[0]:
            return self.name[0]
        else:
            return self.name[0].lower()

    board_addendum_lines = 0
    def get_board_addendum (self):
        return None

    def set_active_cards(self):
        self.active_cards = ([self.style, self.base] +
                             self.get_active_tokens() +
                             self.active_status_effects)
    def get_active_tokens(self):
        return ([] if self.opponent.blocks_tokens() else self.ante)

    # Bonus and ability calculation

    def get_priority_pre_penalty(self):
        # This only exists for when opponent is Ottavia.
        return self.get_priority_bonus() + self.triggered_priority_bonus + \
               sum(card.priority+card.get_priority_bonus()
                    for card in self.active_cards)
    def get_priority_bonuses(self):
        return (sum(card.priority for card in self.active_cards
                   if card not in (self.style, self.base)) +
                sum(card.get_priority_bonus()
                    for card in self.active_cards) +
                self.get_priority_bonus() +
                self.triggered_priority_bonus +
                self.opponent.give_priority_penalty())
    def get_printed_priority(self):
        return sum(card.priority for card in
                   set((self.style, self.base)) & set(self.active_cards))
    def get_priority(self):
        if self.swap_priorities() or self.opponent.swap_priorities():
            return self.opponent.get_priority_no_swap()
        else:
            return self.get_priority_no_swap()
    def get_priority_no_swap(self):
        priority = self.get_priority_bonus() + self.triggered_priority_bonus + \
                   sum(card.priority+card.get_priority_bonus()
                        for card in self.active_cards) + \
                   self.opponent.give_priority_penalty()
        if self.opponent.blocks_priority_bonuses():
            priority = min (priority, self.get_printed_priority())
        return priority
    def get_printed_power(self):
        if self.alt_pair_power is None:
            return sum(card.power for card in
                       set((self.style, self.base)) & set(self.active_cards))
        else:
            return self.alt_pair_power
    def get_power_bonuses(self):
        return (sum(card.power for card in self.active_cards
                   if card not in (self.style, self.base)) +
                sum(card.get_power_bonus()
                    for card in self.active_cards) +
                self.get_power_bonus() +
                self.triggered_power_bonus +
                self.opponent.give_power_penalty())
    def get_power (self):
        if self.base.power is None:
            return 0
        if self.alt_pair_power is None:
            # "card.power or 0" safeguards against card.power == None.
            card_power = sum(card.power + card.get_power_bonus()
                             for card in self.active_cards)
        else:
            # When pair power is externally fixed, use it instead
            # of the actual printed power on the two cards.
            card_power = (self.alt_pair_power +
                          sum(card.power for card in self.active_cards
                              if card not in (self.style, self.base)) +
                          sum(card.get_power_bonus()
                              for card in self.active_cards))
        power = (card_power + self.get_power_bonus() +
                 self.triggered_power_bonus +
                 self.opponent.give_power_penalty())
        if self.opponent.blocks_power_bonuses():
            power = min (power, self.get_printed_power())
        # power has a minimum of 0
        return max (0, power)
    def get_minrange (self):
        cards = self.active_cards
        if (self.ignore_style_range(self) or
            self.opponent.ignore_style_range(self)):
            cards = [c for c in cards if c is not self.style]
        minrange = self.get_minrange_bonus() + \
                   sum(card.minrange+card.get_minrange_bonus()
                        for card in cards) + \
                   self.opponent.give_minrange_penalty()
        if self.opponent.blocks_minrange_bonuses():
            printed_minrange = sum(card.minrange for card in
                                   set((self.style, self.base)) & set(cards))
            minrange = min (minrange, printed_minrange)
        return minrange
    def get_maxrange_bonuses(self):
        return (sum(card.maxrange for card in self.active_cards
                   if card not in (self.style, self.base)) +
                sum(card.get_maxrange_bonus()
                    for card in self.active_cards) +
                self.get_maxrange_bonus() +
                self.opponent.give_maxrange_penalty())
    def get_maxrange (self):
        cards = self.active_cards
        if (self.ignore_style_range(self) or
            self.opponent.ignore_style_range(self)):
            cards = [c for c in cards if c is not self.style]
        maxrange = self.get_maxrange_bonus() + \
                   sum(card.maxrange+card.get_maxrange_bonus()
                        for card in cards) + \
                   self.opponent.give_maxrange_penalty()
        if self.opponent.blocks_maxrange_bonuses():
            printed_maxrange = sum(card.maxrange for card in
                                   set((self.style, self.base)) & set(cards))
            maxrange = min (maxrange, printed_maxrange)
        return maxrange
    def get_priority_bonus (self):
        return 0
    def get_power_bonus (self):
        return 0
    def get_minrange_bonus (self):
        return 0
    def get_maxrange_bonus (self):
        return 0
    def give_priority_penalty (self):
        return sum (card.give_priority_penalty() for card in self.active_cards)
    def give_power_penalty (self):
        return sum (card.give_power_penalty() for card in self.active_cards)
    def give_minrange_penalty (self):
        return sum (card.give_minrange_penalty() for card in self.active_cards)
    def give_maxrange_penalty (self):
        return sum (card.give_maxrange_penalty() for card in self.active_cards)

    # Cause my or opponent's style range modifiers to be ignored.
    def ignore_style_range(self, player):
        return False
    def swap_priorities(self):
        return False

    def add_triggered_power_bonus (self, bonus):
        self.triggered_power_bonus += bonus
        if bonus != 0 and self.game.reporting:
            string = str(bonus) if bonus < 0 else "+" + str(bonus)
            self.game.report ("%s gets %s power" % (self.name, string))
    def add_triggered_priority_bonus (self, bonus):
        self.triggered_priority_bonus += bonus
        if bonus != 0 and self.game.reporting:
            string = str(bonus) if bonus < 0 else "+" + str(bonus)
            self.game.report ("%s gets %s priority" % (self.name, string))

    def get_stunguard (self):
        return sum(card.get_stunguard() for card in self.active_cards)
    def get_soak (self):
        return sum(card.get_soak() for card in self.active_cards)

    # Triggers

    def ante_trigger (self):
        pass
    def reveal_trigger (self):
        for card in self.active_cards:
            card.reveal_trigger()
    def start_trigger (self):
        if not self.opponent.blocks_start_triggers():
            self.activate_card_triggers('start_trigger')
    def before_trigger (self):
        if not self.opponent.blocks_before_triggers():
            self.activate_card_triggers('before_trigger')
    def hit_trigger (self):
        if not self.opponent.blocks_hit_triggers():
            self.activate_card_triggers('hit_trigger')
    def take_a_hit_trigger (self):
        self.activate_card_triggers('take_a_hit_trigger')
    def damage_trigger (self, damage):
        # Iri's Knight Form needs stun to be checked before damage
        # triggers.
        # Is the opposite needed for anyone?
        self.opponent.stun (damage)
        if not self.opponent.blocks_damage_triggers():
            self.activate_card_triggers('damage_trigger', [damage])
    def take_damage_trigger (self, damage):
        pass
    def soak_trigger (self, damage_soaked):
        pass
    def after_trigger_for_opponent(self):
        pass
    def after_trigger (self):
        if not self.opponent.blocks_after_triggers():
            self.activate_card_triggers('after_trigger')
    def end_trigger (self):
        if not self.opponent.blocks_end_triggers():
            self.activate_card_triggers('end_trigger')
    # This is for unique abilites that say "end of beat"
    def unique_ability_end_trigger (self):
        pass

    def activate_card_triggers(self, trigger_name, params=[]):
        cards = self.active_cards
        trigger = attrgetter(trigger_name)
        ordered = attrgetter('ordered_' + trigger_name)
        ordered_cards = [c for c in cards if ordered(c)]
        other_cards = [c for c in cards if not ordered(c)]
        for card in other_cards:
            trigger(card)(*params)
        while(ordered_cards):
            if len(ordered_cards) > 1:
                prompt = "Choose trigger to execute next:"
                options = [card.name for card in ordered_cards]
                ans = self.game.make_fork (len(ordered_cards),
                                            self, prompt, options)
            else:
                ans = 0
            abort = trigger(ordered_cards[ans])(*params)
            if abort:
                break
            ordered_cards.pop(ans)

    # special ability calculation
    def is_attacking (self):
        return self.base.is_attack
    def can_hit (self):
        if self.standard_range():
            distance = self.attack_range()
            if self.game.debugging:
                self.game.report ("%s's range: %d-%d attack range: %d"
                                  %(self.name,
                                    self.get_minrange(),
                                    self.get_maxrange(),
                                    distance))
            in_range = distance <= self.get_maxrange() and \
                       distance >= self.get_minrange()
            if self.game.debugging:
                self.game.report("in range: %s" % in_range)
                self.game.report("all cards can hit: %s" % all(card.can_hit() for card in self.active_cards))
        else:
            in_range = False
        return (in_range or self.special_range_hit()) and \
               all(card.can_hit() for card in self.active_cards)
    def can_be_hit (self):
        if self.game.debugging:
            self.game.report("triggered dodge: %s" % self.triggered_dodge)
        return not self.triggered_dodge and \
               all(card.can_be_hit() for card in self.active_cards)
    def set_triggered_dodge(self):
        self.triggered_dodge = True
        if self.game.reporting:
            self.game.report("%s cannot be hit this beat." % self)
    # This does somthing if the player "would be hit".
    def replace_hit(self):
        return False
    def standard_range (self):
        return all (card.standard_range for card in self.active_cards)
    def special_range_hit (self):
        return any(card.special_range_hit() for card in self.active_cards)

    # Utility function that returns positions that are "in range".
    def in_standard_range(self):
        if not self.is_attacking() or not self.standard_range():
            return set()
        minr = max(0, self.get_minrange())
        maxr = self.get_maxrange()
        pos = self.position
        opp = self.opponent.position
        if opp > pos:
            return set(xrange(pos+minr, min(7, pos+maxr+1)))
        else:
            return set(xrange(max(0, pos-maxr), pos-minr+1))

    def reduce_soak (self, soak):
        for card in self.active_cards:
            soak = card.reduce_soak(soak)
        return soak
    def reduce_stunguard (self, stunguard):
        for card in self.active_cards:
            stunguard = card.reduce_stunguard(stunguard)
        return stunguard

    # return blocked destinations, that cannot be moved into or through
    def blocks_opponent_movement (self, initiator, direct):
        blocked = set()
        for card in self.active_cards:
            blocked |= card.blocks_opponent_movement (initiator, direct)
        return blocked
    def blocks_own_movement (self, initiator):
        blocked = set()
        for card in self.active_cards:
            blocked |= card.blocks_own_movement(initiator)
        return blocked

    # Characters that can just transform opponent's move into
    # something else should return a tuple of:
    # (mover, moves, direct, max_move)
    def transform_opponent_moves(self):
        return None

    # Characters that can ignore movement effects they apply
    # to themselves.
    def may_ignore_own_movement(self):
        return False

    # performance hacks:
    # For rare abilities, return False automatically.
    # Characters that have the ability need to override this.

    # block opponent's stats from going above printed style+base
    def blocks_priority_bonuses (self):
        return False
    def blocks_power_bonuses (self):
        return False
    def blocks_minrange_bonuses (self):
        return False
    def blocks_maxrange_bonuses (self):
        return False
    # block effects of anted tokens and cards.
    # block spending of tokens and counters.
    # cannot refer to self.active_cards, because it is used in making it
    def blocks_tokens (self):
        return False
    # react to own or opponent's execute_move()
    # invoked also for failed movement
    def movement_reaction (self, initiator, mover, old_position, direct):
        for card in self.active_cards:
            card.movement_reaction (initiator, mover, old_position, direct)
    def mimics_movement (self):
        return False
    # makes in-beat choices (forks) for opponent
    def controls_opponent (self):
        return False
    def blocks_start_triggers (self):
        return False
    def blocks_before_triggers (self):
        return False
    def blocks_hit_triggers (self):
        return False
    def blocks_damage_triggers (self):
        return False
    def blocks_after_triggers (self):
        return False
    def blocks_end_triggers (self):
        return False

    def has_stun_immunity (self):
        return any (card.has_stun_immunity() for card in self.active_cards)
    # fraction added to priority to break ties in clash
    # used when a card wins/loses ties without clashing
    def clash_priority (self):
        return sum(card.clash_priority for card in self.active_cards)
    # change this when a character can't go below 1 life (or other value)
    def get_minimum_life (self):
        return 0
    def get_damage_cap (self):
        return min (card.get_damage_cap() for card in self.active_cards)

    # execution of game tasks

    # cycle discard piles
    def cycle (self):
        if self.style is self.special_action:
            self.special_action_available = False
        # Cycling happens if you didn't play a special action, or if
        # your non-pulse special action was trumped by a pulse.
        if (self.style is not self.special_action or
            (self.opponent.base is self.opponent.pulse and
             self.base is not self.pulse)):
            # if the special action cycles back, restore it.
            if self.special_action in self.discard[2]:
                self.special_action_available = True
            # Actual cycling.
            self.discard[2] = self.discard[1].copy()
            self.discard[1] = self.discard[0].copy()
        # but remove cards from the virtual discard[0] in any case
        self.discard[0] = set()

    def become_active_player(self):
        self.game.active = self
        if self.game.reporting:
            self.game.report("%s becomes the active player" % self.name)

    # can't go over 20
    def gain_life (self, gain):
        self.life = min (20, self.life + gain)
        if self.game.reporting and gain > 0:
            self.game.report (self.name + " gains %d life (now at %d)" \
                              %(gain, self.life))

    # can't go under 1
    def lose_life (self, loss):
        self.life = max (1, self.life - loss)
        if self.game.reporting and loss > 0:
            self.game.report (self.name + " loses %d life (now at %d)" \
                              %(loss, self.life))

    # deal damage to opponent
    def deal_damage (self, damage):
        self.opponent.take_damage (damage)

    # apply damage from attack or other source
    def take_damage (self, damage):
        soak = self.opponent.reduce_soak(self.get_soak())
        if self.game.reporting and soak:
            self.game.report ('%s has %d soak' % (self, soak))
        damage_soaked = min (soak, damage)
        if damage_soaked:
            self.soak_trigger (damage_soaked)
        remaining_damage = max(self.get_damage_cap() - self.damage_taken, 0)
        final_damage = min (damage - damage_soaked, remaining_damage)

        # life can't go below certain minimum (usually 0)
        self.life = max (self.life - final_damage,
                         self.get_minimum_life())
        if self.game.reporting:
            self.game.report (self.name + \
                              " takes %d damage (now at %d life)" \
                              %(final_damage, self.life))
        if self.life <= 0:
            raise WinException (self.opponent.my_number)
        if final_damage:
            self.opponent.damage_trigger (final_damage)
            self.take_damage_trigger (final_damage)
        # damage_taken updated after triggers, so that triggers can
        # check if this is first damage this beat
        self.damage_taken += final_damage

    # Attempts to stun me with given damage.
    # If damage == None, it's an auto-stun effect
    # Returns True if a stun was scored, False otherwise.
    def stun (self, damage=None):
        if self.has_stun_immunity():
            return False
        if damage == None: #auto stun
            self.was_stunned = True
            if self.game.reporting:
                self.game.report (self.name + " is stunned")
            return True
        else:
            stunguard = self.opponent.reduce_stunguard(self.get_stunguard())
            if damage > stunguard:
                self.was_stunned = True
                if self.game.reporting:
                    self.game.report (self.name + " is stunned")
                return True
            else:
                return False

    # Stun Immunity negates stun retroactively
    def is_stunned(self):
        return self.was_stunned and not self.has_stun_immunity()

    # Returns all of my tokens in my ante to pool.
    # Does not affect induced tokens
    def revoke_ante (self):
        self.pool += [token for token in self.ante if token in self.tokens]
        self.ante = [token for token in self.ante if token not in self.tokens]
        self.set_active_cards()

    # checks if it's possible to spend a token
    # takes either a number of tokens, or a specific token
    def can_spend (self, to_spend):
        if self.opponent.blocks_tokens():
            return False
        # number of tokens
        if isinstance (to_spend, int):
            return len(self.pool) >= to_spend
        # specific token
        else:
            return to_spend in self.pool

    # discard a token if possible, report which token was discarded
    # or False if none
    # if token = None, you choose
    # Doesn't allow discarding induced tokens.
    # That's ok, because:
    #     1. Your own powers only cause you to discard your own tokens.
    #     2. The characters that dole out tokens don't have powers that
    #        cause them to be discarded.
    def discard_token (self, token = None, verb = "discard"):
        if not self.pool:
            return False

        if token == None:
            # if all tokens are the same, discard one:
            if len (set (self.pool)) == 1:
                index = 0
            # otherwise discard lowest value (with fake_fork for human)
            else:
                values = [t.value for t in self.pool]
                index = values.index (min (values))
                prompt = "Select a token to " + verb + ":"
                options = [t.name for t in self.pool]
                index = self.game.make_fork (len(self.pool), self, \
                                                  prompt, options, index)
            discarded = self.pool.pop (index)
            if self.game.reporting:
                self.game.report (self.name + " " + verb + "s " + \
                                  discarded.aname + " token")
            return discarded
        else:
            if token in self.pool:
                self.pool.remove (token)
                if self.game.reporting:
                    self.game.report (self.name + " " + verb + "s " + \
                                      token.aname + " token")
                return token
            else:
                return False

    # like discarding, but you can't do it if the opponent is blocking tokens
    def spend_token (self, token = None, verb = "spend"):
        if self.opponent.blocks_tokens():
            return False
        else:
            return self.discard_token (token, verb)

    # like spending, but it's added to the ante
    def ante_token (self, token = None):
        new_ante = self.spend_token (token, verb = "ante")
        if new_ante:
            self.ante.append (new_ante)
            self.set_active_cards ()

    # recover n tokens, up to your max_tokens
    # if character has multiple kinds of tokens, need to override this
    def recover_tokens (self, n):
        old_pool = len (self.pool)
        self.pool = [self.tokens[0]] * min (self.max_tokens,
                                            len(self.pool) + n)
        gain = len(self.pool) - old_pool
        if self.game.reporting and gain:
            self.game.report ("%s gains %d %s token%s"
                              %(self.name, gain, self.tokens[0].name,
                                ("s" if gain>1 else "")))

    def spend_counters(self, effect_string, max_spend=None,
                       set_amount=False):
        """Creates fork for spending, reports spending, and returns
        amount spent."""
        if (self.counters == 0 or
            self.opponent.blocks_tokens()):
            return 0
        if max_spend is None:
            max_spend = self.counters
        elif max_spend > self.counters:
            if set_amount:
                return 0
            else:
                max_spend = self.counters
        prompt = None
        options = None
        if self.game.interactive_mode and self.is_user:
            if set_amount:
                prompt = "Spend %d %s Counter%s %s?" % (
                    max_spend, self.counter_name,
                    '' if max_spend == 1 else 's',
                    effect_string)
                options = ['No', 'Yes']
            else:
                prompt = "Spend how many %s Counters for %s?" % (
                    self.counter_name, effect_string)
        n_options = 2 if set_amount else max_spend + 1
        spend = self.game.make_fork(n_options, self, prompt, options)
        if spend:
            if set_amount:
                spend = max_spend
            self.counters -= spend
            if self.game.reporting:
                self.game.report("%s spends %d %s Counter%s (now has %d)" %
                    (self.name, spend, self.counter_name,
                     '' if spend == 1 else 's', self.counters))
        return spend

    def discard_counters(self, n):
        old = self.counters
        self.counters = max(0, self.counters - n)
        loss = old - self.counters
        if loss and self.game.reporting:
            self.game.report("%s discards %d %s Counter%s (now has %d)" %
                (self.name, loss, self.counter_name,
                 '' if loss == 1 else 's', self.counters))
        return loss

    def recover_counters(self, n):
        old = self.counters
        self.counters = min(self.counters + n, self.max_counters)
        gain = self.counters - old
        if gain and self.game.reporting:
            self.game.report("%s recovers %d %s Counter%s (now has %d)" %
                (self.name, gain, self.counter_name,
                 '' if gain == 1 else 's', self.counters))
        return gain

    def remove_card_from_game(self, card):
        # Remove card from list of available cards for future beats.
        self.styles = [s for s in self.styles if s is not card]
        self.bases = [b for b in self.bases if b is not card]
        for d in self.discard:
            d.discard(card)
        # If card is current style or base, replace it with null card,
        # so that it doesn't do anything this beat.
        if card is self.style:
            self.style = self.null_style
        if card is self.base:
            self.base = self.null_base
        self.set_active_cards()
        if self.game.reporting:
            self.game.report("%s's %s %s is removed from the game" % (
                self, card, "style" if isinstance(card, Style) else Base))

    # Various convenience movement functions:
    def advance (self, moves, max_move=False, specific_movement_reaction=None):
        self.execute_move (self, moves, max_move=max_move,
                           specific_movement_reaction=specific_movement_reaction)
    def pull (self, moves, max_move=False, specific_movement_reaction=None):
        self.execute_move (self.opponent, moves, max_move=max_move,
                           specific_movement_reaction=specific_movement_reaction)
    def retreat (self, moves, max_move=False, specific_movement_reaction=None):
        moves = [-m for m in moves]
        self.execute_move (self, moves, max_move=max_move,
                           specific_movement_reaction=specific_movement_reaction)
    def push (self, moves, max_move=False, specific_movement_reaction=None):
        moves = [-m for m in moves]
        self.execute_move (self.opponent, moves, max_move=max_move,
                           specific_movement_reaction=specific_movement_reaction)
    def move (self, moves, specific_movement_reaction=None):
        moves = [-m for m in moves if m != 0] + list(moves)
        self.execute_move (self, moves,
                           specific_movement_reaction=specific_movement_reaction)
    def move_opponent (self, moves, specific_movement_reaction=None):
        moves = [-m for m in moves if m != 0] + list(moves)
        self.execute_move (self.opponent, moves,
                           specific_movement_reaction=specific_movement_reaction)
    def move_directly (self, dests, specific_movement_reaction=None):
        self.execute_move (self, dests, direct=True,
                           specific_movement_reaction=specific_movement_reaction)
    def move_opponent_directly (self, dests, specific_movement_reaction=None):
        self.execute_move (self.opponent, dests, direct=True,
                           specific_movement_reaction=specific_movement_reaction)
    def move_to_unoccupied (self, specific_movement_reaction=None):
        self.move_directly (list(set(xrange(7)) - set((self.position, self.opponent.position))),
                            specific_movement_reaction=specific_movement_reaction)
    def move_opponent_to_unoccupied (self, specific_movement_reaction=None):
        self.move_opponent_directly (list(set(xrange(7)) - set((self.position, self.opponent.position))),
                                     specific_movement_reaction=specific_movement_reaction)

    # Handles both moving yourself and moving opponent.
    # Mover is player that is actually being moved.
    # direct: the move skips intervening spaces.
    # Moves are relative for indirect moves, absolute for direct moves.
    # max_move means you move as much as possible.  It's only relevant
    # for indirect moves, which should give the entire range in the
    # moves parameter.
    # specific_movement_reaction is a movement reaction by the initiator for this particular move.
    def execute_move (self, mover, moves, direct=False, max_move=False,
                      specific_movement_reaction=None):
        transform = self.opponent.transform_opponent_moves()
        if transform is not None:
            mover, moves, direct, max_move = transform

        # Some characters can ignore their own movement (adding null move).
        # This doesn't work well with max_move=True, but no characters
        # have both.
        if self is mover and self.may_ignore_own_movement():
            if direct:
                if self.position not in moves:
                    moves.append(self.position)
            else:
                if 0 not in moves:
                    moves.append(0)
            moves = sorted(moves)

        mover.forced_block = False
        mover_pos = mover.position

        self.inner_execute_move (mover, moves, direct, max_move)

        if not direct and mover.opponent.mimics_movement():
            mover.opponent.position += (mover.position - mover_pos)
            mover.opponent.position = min(6, max(0, mover.opponent.position))

        # Note and report movement
        if mover_pos != mover.position:
            mover.moved = True
            if self.game.reporting:
                if mover == self:
                    self.game.report (self.name + " moves:")
                else:
                    self.game.report (self.name + " moves " + mover.name + ":")
                for s in self.game.get_board():
                    self.game.report (s)
        # reactions not dependent on actual movement, some cards react
        # to failed movement.
        if specific_movement_reaction:
            specific_movement_reaction(self, mover, mover_pos, direct)
        self.movement_reaction (self, mover, mover_pos, direct)
        self.opponent.movement_reaction (self, mover, mover_pos, direct)

    # Does heavy lifting of movement.
    def inner_execute_move (self, mover, moves, direct, max_move):
        mover_pos = mover.position
        # obtain set of attempted destinations
        if direct:
            # Moves are given as destinations, just remove opponent's position.
            # Don't remove own position, because some moves say "may".
            dests = (set(moves) - set([mover.opponent.position])) & set(xrange(7))
        else:
            # convert relative moves to destinations
            dests = self.get_destinations(mover, moves)
        mover.dests = dests
        # obtain set of spaces blocked by opponent, and record
        # them for inspection by method that invoked the move.
        mover.mover_block = mover.blocks_own_movement(self)
        mover.mover_block.discard(mover_pos)
        mover.opponent_block = mover.opponent.blocks_opponent_movement(
                                                        self, direct)
        mover.opponent_block.discard(mover_pos)
        blocked = mover.mover_block | mover.opponent_block
        # compute possible destinations after blocking
        if not blocked:
            possible = list(dests)
        else:
            unobstructed = self.get_unobstructed_positions(
                                        mover_pos, blocked, direct)
            possible = list (dests & unobstructed)
        if max_move and possible:
            direction = sum(moves) * (mover.opponent.position -
                                      mover.position)
            if direction > 0:
                possible = [max(possible)]
            else:
                possible = [min(possible)]
        if possible:
            mover_name = ("" if mover == self else mover.name+" ")
            prompt = "Choose position to move " + mover_name + "to:"
            options = []
            possible = sorted(possible)
            if self.is_user and self.game.interactive_mode:
                for p in possible:
                    mover.position = p
                    options.append (self.game.get_basic_board())
                mover.position = mover_pos
            ans = self.game.make_fork (len(possible), self, prompt, options)
            mover.position = possible[ans]


    def get_unobstructed_positions(self, mover_pos, blocked, direct):
        if direct:
            # for direct movement, only directly blocked spaces are obstructed.
            return set(xrange(7)) - blocked
        else:
            # for normal movement, a position is 'unobstructed' if there
            # are no blocked positions between it and the mover
            return set(pos for pos in xrange(7)
                       if len(pos_range(pos,mover_pos) & blocked) == 0)

    # given player moving/being moved, and set of relative moves,
    # return possible destinations, taking into account board size
    # and jumping over opponent
    def get_destinations (self, mover, moves):
        mypos = mover.position
        otherpos = mover.opponent.position
        direction = (otherpos - mypos) / abs (otherpos - mypos)
        destinations = []
        for m in moves:
            dest = mypos + m * direction
            # forward movement takes into account running into opponent
            if m > 0:
                # If mover's opponent mimics movement, stop one short
                if mover.opponent.mimics_movement():
                    if direction == 1 and dest == 6:
                        continue
                    if direction == -1 and dest == 0:
                        continue
                # but normally, jump over opponent
                else:
                    if direction*dest >= direction*otherpos:
                        dest = dest + direction
            if dest >= 0 and dest <= 6:
                destinations.append(dest)
        return set(destinations)

    # range opponent is in when I attack
    # some characters calculate this from some proxy, rather than their own
    # position
    def attack_range (self):
        return abs (self.position - self.opponent.position)

    # opponent uses this to evaluate my life total
    def effective_life (self):
        return self.life

    def expected_soak(self):
        unavailable_cards = self.discard[1] \
                          | self.discard[2]
        styles = set(self.styles) - unavailable_cards
        bases = set(self.bases) - unavailable_cards
        return sum (s.soak for s in styles) / len (styles) + \
               sum (b.soak for b in bases) / len (bases)

    def expected_stunguard(self):
        unavailable_cards = self.discard[1] \
                          | self.discard[2]
        styles = set(self.styles) - unavailable_cards
        bases = set(self.bases) - unavailable_cards
        return sum (s.stunguard for s in styles) / len (styles) + \
               sum (b.stunguard for b in bases) / len (bases)

    # determine the range the character wants to be in this beat:
    def set_preferred_range (self):
        unavailable_cards = self.discard[1] \
                          | self.discard[2]
        styles = set(self.styles) - unavailable_cards
        bases = set(self.bases) - unavailable_cards
        self.preferred_range = \
            sum (s.get_preferred_range() for s in styles) / float(len(styles)) + \
            sum (b.get_preferred_range() for b in bases) / float(len(bases))

    def evaluate_range (self):
        return - self.game.range_weight * \
                      (self.preferred_range - self.game.distance()) ** 2

    # Some characters (Tanis, Arec with clone) can choose their
    # position at the start of next beat.  They should return a list
    # of possible positions.
    def get_superposed_positions(self):
        return [self.position]
    def get_superposition_priority(self):
        return self.game.CHOOSE_POSITION_NOW
    # bonus added to the final composition of superposed evaluations.
    def evaluate_superposition(self):
        return 0

    # opponent's life (negative)
    # special bonuses accumulated,
    # square difference between my preferred range and current range (negative)
    # bonuses related to cards in hand
    # discard rotation (cards in discard 2 are half penalty)
    # having a special action
    def evaluate (self):
        hand = self.styles_and_bases_set - (self.discard[1] | self.discard[2])
        # When playing "first beats", to find best initial discards, we
        # don't want to use card bonuses (the cards that get used due
        # to having bad bonuses are precisely the ones we wanted to discard).
        card_bonus = (0 if self.game.first_beats else
                      sum(card.evaluation_bonus() for card in hand))
        discard_penalty = (0 if self.game.first_beats else
          - sum(card.get_discard_penalty() for card in self.discard[1])
          - sum(card.get_discard_penalty() for card in self.discard[2]) / 2.0)
        if self.special_action_available:
            special_action_bonus = (self.special_action.value +
                                    (max([finisher.evaluate_setup()
                                          for finisher in self.finishers])
                                     if (self.life <= 7) else 0))
        else:
            if self.special_action in self.discard[1]:
                special_action_bonus = self.special_action.value - 2
            elif self.special_action in self.discard[2]:
                special_action_bonus = self.special_action.value - 1
            else:
                special_action_bonus = 0
        status_effect_bonus = sum([status.value for status in
                                   self.pending_status_effects])

        if self.game.debugging:
            self.game.report("%s's evaluation:"%self)
            self.game.report("card bonus: %.2f"% card_bonus)
            self.game.report("discard penalty: %.2f"% discard_penalty)
            self.game.report("special action: %.2f"% special_action_bonus)
            self.game.report("status effects: %.2f"% status_effect_bonus)
        return (- self.opponent.effective_life()
                + self.evaluate_range() + card_bonus
                + discard_penalty + special_action_bonus
                + status_effect_bonus)

    # How many spaces can I retreat?
    def retreat_range(self):
        me = self.position
        opp = self.opponent.position
        return 6-me if me > opp else me

    # Mean priority over all styles/bases/effects.
    mean_priority_bonus = 0
    def set_mean_priority(self):
        self.mean_priority = (self.mean_priority_bonus +
            sum (s.mean_priority for s in self.styles) / float(len(self.styles)) +
            sum (b.mean_priority for b in self.bases) / float(len(self.bases)))

    def set_mean_priority_effects(self):
        # Priority advange is 0 when I'm low, 0.5 when equal,
        # 1 when I'm high.
        e = math.exp(self.mean_priority - self.opponent.mean_priority)
        self.priority_advantage = (e-1) / (e+1)
        # Priority nearness is 0 when I'm low or high, 1 when equal.
        e2 = math.sqrt(e)
        self.priority_nearness = 2 * e2 / (e + 1)

    # Some characters (like Seth) need to fix the result table
    # and strategy lists after simulating all strategies.
    def post_simulation_processing (self):
        return

    # Some characters (like Seth) have information that survives a clash,
    # so for post-clash strategies, they behave as if they picked a different
    # strategy.

    # This changes the indices of post clash strategies in the strategy list
    def clash_strat_index (self, my_new_i, opp_new_i, my_orig_i, opp_orig_i):
        # Most characters just use the strategy they switched to after the clash.
        return my_new_i

    # This fixes the strategies themselves.
    def fix_strategies_post_clash (self, strats, opp_orig):
        return strats

    def pre_attack_decision_effects(self):
        pass
    def pre_attack_decision_report(self, decision):
        return []

    # Does this character auto-win on a timeout?
    def wins_on_timeout(self):
        return False

    # When you play a finisher against a cancel, you discard the base
    # you used.  Figure out which base to discard.
    def choose_finisher_base_retroactively(self):
        options = list((self.finisher_generators & set(self.bases)) -
                       (self.discard[1] | self.discard[2]))
        assert options
        if len(options) == 1:
            return options[0]
        if self.is_user:
            print "Which base did you use to play your finisher?"
            ans = menu([o.name for o in options])
            return options[ans]
        else:
            # Since ante is already set, it might mess up evaluations.
            # (not to mention, it's not fair to evaluate at this point).
            # So AI just uses a set order.
            if self.strike in options:
                return self.strike
            if self.drive in options:
                return self.drive
            return options[0] # unique base

# One card, style or base
class Card (object):
    minrange = 0
    maxrange = 0
    power = 0
    priority = 0
    # These attributes are referenced by Card.get_soak(), but are also
    # used for evaluating expected soak/stunguard after the beat.
    stunguard = 0
    soak = 0

    name_override = None
    # average of minimum and maximum range (including movement)
    preferred_range = 0
    # fraction added to priority to break ties in clash
    # used when a card wins/loses ties without clashing
    clash_priority = 0
    @property
    def mean_priority(self):
        return self.priority

    def __init__(self, the_game, my_player):
        self.game = the_game
        self.me = my_player
        # printed name is class name with spaces before capitals and digits
        self.name = self.name_override or \
                    re.sub(r'([a-z](?=[A-Z0-9])|[A-Z0-9](?=[A-Z0-9][a-z]))', r'\1 ',
                           self.__class__.__name__)
        # name with a/an before it
        self.aname = ("an " if self.name[0] in "AEIOU" else "a ") + self.name

    def __repr__ (self):
        return self.name

    # For effects that care about the printed power.
    # Bases with power N/A should have power = None
    # Bases with power X (implemented with a property) should override
    # printed_power to None
    @property
    def printed_power (self):
        return self.power

    # parameters
    def get_minrange_bonus(self):
        return 0
    def get_maxrange_bonus(self):
        return 0
    def get_priority_bonus(self):
        return 0
    def get_power_bonus(self):
        return 0
    def get_stunguard(self):
        return self.stunguard
    def get_soak(self):
        return self.soak
    def get_preferred_range(self):
        return self.preferred_range

    # Triggers
    # Returning True instead of None causes further triggers of this
    # type (by this player) to be aborted.
    def ante_trigger(self):
        pass
    def reveal_trigger(self):
        pass
    def start_trigger(self):
        pass
    def before_trigger(self):
        pass
    def hit_trigger(self):
        pass
    def take_a_hit_trigger(self):
        pass
    def damage_trigger(self, damage):
        pass
    def after_trigger(self):
        pass
    def end_trigger(self):
        pass
    # Set to True if a card has a trigger of the appropriate type,
    # for which ordering (vs. other triggers) is important.
    ordered_start_trigger = False
    ordered_before_trigger = False
    ordered_hit_trigger = False
    ordered_take_a_hit_trigger = False
    ordered_damage_trigger = False
    ordered_after_trigger = False
    ordered_end_trigger = False

    #interrupting self/opponent
    def can_hit (self):
        return True
    def can_be_hit (self):
        return True
    # change this if attack *only* has special range
    standard_range = True
    # conditions under which special range hits
    def special_range_hit (self):
        return False
    def reduce_soak (self, soak):
        return soak
    def reduce_stunguard (self, stunguard):
        return stunguard
    # return blocked spaces
    # non-direct moves will automatically be prevented from passing through
    # blocked spaces to the spaces beyond
    # (so "direct" is provided for special cases only)
    def blocks_opponent_movement (self, initiator, direct):
        return set()
    def blocks_own_movement (self, initiator):
        return set()

    # note that this is invoked for failed movement as well
    def movement_reaction(self, initiator, mover, old_position, direct):
        pass
    def mimics_movement (self):
        return False
    def has_stun_immunity (self):
        return False
    def give_priority_penalty (self):
        return 0
    def give_power_penalty (self):
        return 0
    def give_minrange_penalty (self):
        return 0
    def give_maxrange_penalty (self):
        return 0
    def get_damage_cap (self):
        return 1000

    # situation dependent bonus (positive or negative) for having this card in
    # hand at end of beat
    def evaluation_bonus (self):
        return 0
    # How bad is it to have this card in discard 1, beyond the average
    # card?
    # Badness is halved in discard 2.
    discard_penalty = 0
    # This is only for special cards that may end up in the discard.
    # Styles and Bases override this.
    def get_discard_penalty (self):
        return self.discard_penalty

class Style (Card):
    def get_discard_penalty (self):
        return self.discard_penalty + 1.5

class Base (Card):
    is_attack = True  #change for bases that don't attack
    deals_damage = True #change for bases that do attack, but can't deal damage
    # Some effects check for using "same base" as opponent.  This
    # comparison equates alpha/beta/delta bases by color.  To enable this,
    # beta/delta bases override this with the corresponding alpha base.
    @property
    def alpha_name(self):
        return self.name
    def get_discard_penalty (self):
        return self.discard_penalty + 0.9

# Placeholders used before the reveal step, so that no card abilities
# are activated, and also when an ability removes cards mid beat.
class NullStyle (Style):
    pass
class NullBase (Base):
    is_attack = False

# only covers token's effects when anted.
# anything else should be covered by the character
class Token (Card):
    value = 0

# Represents a status effect (effect that gives a bonus/penalty
# next turn).
class StatusEffect(Card):
    def read_my_state(self, lines):
        line = find_start_line(lines, self.read_state_prefix)
        if line:
            self.me.active_status_effects.append(self)
        # In case inherited method wants to do more with the line.
        # (Also, so that character's read_my_state() knows to
        # skip a line).
        return line
    def situation_report(self, report):
        if self in self.me.active_status_effects:
            report.append(self.situation_report_line)
    # Only save/restore pending effects.
    # Active effects can't change during the beat.
    # No need to reset pending effects - we remove the object
    # from the pending_status_effects list.
    # No need to rest/save/restore/move the object itself,
    # that is handled by Character code.
    # Only do something if StatusEffect has variable parameters.
    def full_save(self, state):
        pass
    def full_restore(self, state):
        pass
    # Move pending effects to active state.
    def prepare_next_beat(self):
        pass
    def activate(self):
        self.me.pending_status_effects.append(self)
        if self.game.reporting:
            self.game.report(self.activation_line)

class PowerBonusStatusEffect(StatusEffect):
    read_state_prefix = "Power bonus: "
    @property
    def situation_report_line(self):
        return ("Power bonus: %d" %
                self.me.active_power_bonus)
    @property
    def activation_line(self):
        return ("%s will have %d power next beat" %
                (self.me.name, self.me.pending_power_bonus))
    @property
    def value(self):
        return 0.3 * self.me.pending_power_bonus
    def read_my_state(self, lines):
        line = StatusEffect.read_my_state(self, lines)
        if line:
            self.me.active_power_bonus = int(line.split()[2])
        return line
    def full_save(self, state):
        state.pending_power_bonus = self.me.pending_power_bonus
    def full_restore(self, state):
        self.me.pending_power_bonus = state.pending_power_bonus
    def prepare_next_beat(self):
        self.me.active_power_bonus = self.me.pending_power_bonus
    def activate(self, bonus):
        self.me.pending_power_bonus = bonus
        StatusEffect.activate(self)
    def get_power_bonus(self):
        return self.me.active_power_bonus

class PriorityBonusStatusEffect(StatusEffect):
    read_state_prefix = "Priority bonus: "
    @property
    def situation_report_line(self):
        return ("Priority bonus: %d" %
                self.me.active_priority_bonus)
    @property
    def activation_line(self):
        return ("%s will have %d priority next beat" %
                (self.me.name, self.me.pending_priority_bonus))
    @property
    def value(self):
        return 0.3 * self.me.pending_priority_bonus
    def read_my_state(self, lines):
        line = StatusEffect.read_my_state(self, lines)
        if line:
            self.me.active_priority_bonus = int(line.split()[2])
        return line
    def full_save(self, state):
        state.pending_priority_bonus = self.me.pending_priority_bonus
    def full_restore(self, state):
        self.me.pending_priority_bonus = state.pending_priority_bonus
    def prepare_next_beat(self):
        self.me.active_priority_bonus = self.me.pending_priority_bonus
    def activate(self, bonus):
        self.me.pending_priority_bonus = bonus
        StatusEffect.activate(self)
    def get_priority_bonus(self):
        return self.me.active_priority_bonus

class PowerPenaltyStatusEffect(StatusEffect):
    # Penalty represented by negative number
    read_state_prefix = "Opponent has power penalty: -"
    @property
    def situation_report_line(self):
        return ("Opponent has power penalty: %d" %
                self.me.active_power_penalty)
    @property
    def activation_line(self):
        return ("%s will have %d priority next beat" %
                (self.opponent.name, self.me.pending_power_penalty))
    @property
    def value(self):
        return 0.3 * -self.me.pending_power_penalty
    def read_my_state(self, lines):
        line = StatusEffect.read_my_state(self, lines)
        if line:
            self.me.active_power_penalty = int(line.split()[4])
        return line
    def full_save(self, state):
        state.pending_power_penalty = self.me.pending_power_penalty
    def full_restore(self, state):
        self.me.pending_power_penalty = state.pending_power_penalty
    def prepare_next_beat(self):
        self.me.active_power_penalty = self.me.pending_power_penalty
    def activate(self, penalty):
        self.me.pending_power_penalty = penalty
        StatusEffect.activate(self)
    def give_power_penalty(self):
        return self.me.active_power_penalty

class PriorityPenaltyStatusEffect(StatusEffect):
    # Penalty represented by negative number
    read_state_prefix = "Opponent has priority penalty: -"
    @property
    def situation_report_line(self):
        return ("Opponent has priority penalty: %d" %
                self.me.active_priority_penalty)
    @property
    def activation_line(self):
        return ("%s will have %d priority next beat" %
                (self.opponent.name, self.me.pending_priority_penalty))
    @property
    def value(self):
        return 0.3 * -self.me.pending_priority_penalty
    def read_my_state(self, lines):
        line = StatusEffect.read_my_state(self, lines)
        if line:
            self.me.active_priority_penalty = int(line.split()[4])
        return line
    def full_save(self, state):
        state.pending_priority_penalty = self.me.pending_priority_penalty
    def full_restore(self, state):
        self.me.pending_priority_penalty = state.pending_priority_penalty
    def prepare_next_beat(self):
        self.me.active_priority_penalty = self.me.pending_priority_penalty
    def activate(self, penalty):
        self.me.pending_priority_penalty = penalty
        StatusEffect.activate(self)
    def give_priority_penalty(self):
        return self.me.active_priority_penalty

class OpponentPowerPriorityStatusEffect(StatusEffect):
    read_state_prefix = "Opponent has +1 power and +1 priority"
    situation_report_line = "Opponent has +1 power and +1 priority"
    activation_line = "Opponent will have +1 power and +1 priority next beat"
    value = -0.6
    def give_power_penalty(self):
        return 1
    def give_priority_penalty(self):
        return 1

class OpponentImmobilizedStatusEffect(StatusEffect):
    read_state_prefix = "Opponent is immobilized"
    situation_report_line = "Opponent is immobilized"
    activation_line = "Opponent will be immobilized next beat"
    value = 3.5
    def blocks_opponent_movement(self, initiator, direct):
        return set(xrange(7))

class OpponentEliminatedStatusEffect(StatusEffect):
    read_state_prefix = "Opponent will be eliminated at end of beat"
    situation_report_line = "Opponent will be eliminated at end of beat"
    activation_line = "Opponent will be eliminated at end of next beat"
    @property
    def value(self):
        return self.opponent.life + 3
    # Elimination trigger handled by Character.unique_ability_end_trigger()

#For Jager
class RefinedStatusEffect(StatusEffect):
    read_state_prefix = "+1 power & priority to signature moves"
    situation_report_line = "+1 power & priority to signature moves"
    activation_line = "Jager will have +1 power, +1 priority if using a signature move next beat"
    @property
    def value(self):
        return 0.1 * len(self.me.signature_pool)
    def get_power_bonus(self):
        return 1 if self.me.strat[2][0] else 0
    def get_priority_bonus(self):
        return 1 if self.me.strat[2][0] else 0

class HoundingStatusEffect(StatusEffect):
    read_state_prefix = "+3 priority to signature moves"
    situation_report_line = "+3 priority to signature moves"
    activation_line = "Jager will have +3 priority if using a signature move next beat"
    @property
    def value(self):
        return 0.15 * len(self.me.signature_pool)
    def get_priority_bonus(self):
        return 3 if self.me.strat[2][0] else 0

# raised at fork points to throw simulation back up to main simulate() method
class ForkException (Exception):
    def __init__ (self, n_options, player):
        self.n_options = int(n_options)
        self.forking_player = player

# raised when a character wins
class WinException (Exception):
    def __init__ (self, winner):
        self.winner = winner

# raised on bugs to return game object
class DebugException (Exception):
    def __init__ (self, game):
        self.game = game

# CHARACTERS
# Each character is a class.
# A game creates an instance of each player's character

class Abarene(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Tares (the_game, self)
        self.styles = [Lethal       (the_game, self),
                       Intoxicating (the_game, self),
                       Barbed       (the_game, self),
                       Crippling    (the_game, self),
                       Pestilent    (the_game, self)]
        self.finishers = [Flytrap        (the_game, self),
                          HallicrisSnare (the_game, self)]
        self.tokens = [Dizziness (the_game, self), \
                       Fatigue   (the_game, self), \
                       Nausea    (the_game, self), \
                       Lethargy (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.5 # Fatigue

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = []

    def choose_initial_discards (self):
        return (self.lethal, self.burst,
                self.barbed, self.grasp)

    def situation_report (self):
        report = Character.situation_report (self)
        tokens = [t.name for t in self.pool]
        report.append ("Pool: " + ', '.join(tokens))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [t for t in self.tokens if t.name in lines[0]]

    def reset (self):
        self.was_token_recovered = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.was_token_recovered = self.was_token_recovered
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.was_token_recovered = state.was_token_recovered

    def get_antes (self):
        combos = [itertools.combinations(self.pool, n)
                  for n in xrange(len(self.pool) + 1)]
        return sum([list(c) for c in combos], [])

    def input_ante (self, pair):
        if self.pool:
            antes = self.get_antes()
            options = [self.get_ante_name(a) for a in antes]
            options[0] = "None"
            print "Select tokens to ante:"
            return antes[menu(options)]
        else:
            return []

    def ante_trigger (self):
        for token in self.strat[2][0]:
            self.ante_token (token)

    def get_ante_name (self, a):
        return ', '.join(token.name for token in a)

    def recover_tokens (self, choosing_player, from_ante_only=False):
        recoverable = [t for t in self.tokens if t not in self.pool]
        if from_ante_only:
            recoverable = [t for t in recoverable if t in self.ante]
        if recoverable:
            prompt = "Select a token for Abarene to recover:"
            options = [t.name for t in recoverable]
            choice = self.game.make_fork (len(recoverable), choosing_player,
                                          prompt, options)
            recovered = recoverable[choice]
            self.pool += [recovered]
            self.pool = sorted(self.pool, key=attrgetter('name'))
            if self.game.reporting:
                self.game.report ("Abarene recovers her " + recovered.name + " token")
            self.was_token_recovered = True

    def hit_trigger(self):
        self.recover_tokens(self.opponent)
        Character.hit_trigger(self)

    # Lethargy token transforms opponent's moves to "move 1"
    def transform_opponent_moves(self):
        if self.lethargy in self.active_cards:
            return (self.opponent, [-1,1], False, False)
        else:
            return None

    def evaluate(self):
        return Character.evaluate(self) + sum([token.value
                                               for token in self.pool])

class Adjenna (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Gaze (the_game, self)
        self.styles = [Alluring     (the_game, self),
                       Arresting    (the_game, self),
                       Pacifying    (the_game, self),
                       Irresistible (the_game, self),
                       Beckoning    (the_game, self)]
        self.finishers = [BasiliskGaze (the_game, self),
                           Fossilize    (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.petrification = 0

    def choose_initial_discards (self):
        return (self.beckoning, self.burst,
                self.alluring, self.grasp)

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Petrification markers on %s" %(self.petrification,
                                                          self.opponent.name))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.petrification = int(lines[0][0])

    def initial_save (self):
        state = Character.initial_save (self)
        state.petrification = self.petrification
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.petrification = state.petrification

    def reset (self):
        self.petrification_is_blocked = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.petrification_is_blocked = self.petrification_is_blocked
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.petrification_is_blocked = state.petrification_is_blocked

    # When Arresting, pushed back 1 for each damage soaked
    def soak_trigger (self, soaked_damage):
        if self.arresting in self.active_cards:
            # treated as opponent pushing me
            self.opponent.push ([soaked_damage])

    # Adjenna wants to be as close to opponent as possible
    # regardless of her cards?
    def set_preferred_range (self):
        self.preferred_range = 1

    def petrify (self):
        # don't petrify after a successful Gaze
        if self.petrification_is_blocked:
            return
        self.petrification += 1
        if self.game.reporting:
            self.game.report (self.opponent.name +
                              " receives a Petrification Marker")
        if self.petrification > 5:
            if self.game.reporting:
                self.game.report (self.opponent.name +
                                  " has 6 Petrification Markers")
            raise WinException (self.my_number)

    def unique_ability_end_trigger (self):
        if not self.is_stunned() and self.game.distance() == 1:
            self.petrify()

    # 6 tokens kill, so they're like 20 life.
    # But they also have other effects.
    def evaluate (self):
        petri_life = (6 - self.petrification) / 6.0 * 20
        effective_life = min (petri_life, self.opponent.effective_life())
        return Character.evaluate(self) + self.opponent.effective_life() \
               - effective_life + 2 * self.petrification

class Alexian (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Divider (the_game, self)
        self.styles = [Gestalt  (the_game, self), \
                       Regal    (the_game, self), \
                       Stalwart (the_game, self), \
                       Mighty   (the_game, self), \
                       Steeled  (the_game, self)  ]
        self.finishers = [EmpireDivider (the_game, self),
                          HailTheKing (the_game, self)]
        self.induced_tokens = [Chivalry  (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = -0.5 # Chivalry gives priority to opponents.

    def all_cards(self):
        return Character.all_cards(self) + self.induced_tokens

    def choose_initial_discards (self):
        return (self.mighty, self.strike,
                self.regal, self.grasp)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.induced_pool = [self.chivalry]

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%s has %d Chivalry token%s" %
                       (self.opponent, len(self.induced_pool),
                        '' if len(self.induced_pool)==1 else 's'))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.induced_pool = [self.chivalry] * int(lines[0].split()[2])

    def initial_save (self):
        state = Character.initial_save (self)
        state.induced_pool = self.induced_pool[:]
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.induced_pool = state.induced_pool[:]

    def reset (self):
        self.damage_soaked = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.damage_soaked = self.damage_soaked
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.damage_soaked = state.damage_soaked

    def get_induced_antes (self):
        return range (len(self.induced_pool) + 1)

    def input_induced_ante (self):
        n = len(self.induced_pool)
        if n > 0 :
            print "Number of Chivalry tokens [0-%d]: " %n
            return input_number (n+1)
        else:
            return 0

    def ante_trigger (self):
        # opponent antes Chivalry tokens according to their chosen strategy
        for unused in xrange(self.opponent.strat[2][1]):
            self.induced_pool.remove(self.chivalry)
            self.opponent.ante.append(self.chivalry)
            if self.game.reporting:
                self.game.report ("%s antes a Chivalry token" %
                                  self.opponent.name)

    def hit_trigger (self):
        self.give_induced_tokens(1)
        Character.hit_trigger (self)

    def get_induced_ante_name (self, a):
        if a == 0:
            return ""
        if a == 1:
            return "1 Chivalry token"
        return "%d Chivalry tokens" %a

    # Opponent gets n token, up to 3
    def give_induced_tokens (self, n):
        old_pool = len (self.induced_pool)
        self.induced_pool = [self.chivalry] * min (3, len(self.induced_pool) + n)
        gain = len(self.induced_pool) - old_pool
        if self.game.reporting and gain:
            self.game.report ("%s gains %d %s token%s"
                              %(self.opponent.name, gain, self.induced_tokens[0].name,
                                ("s" if gain>1 else "")))

    # record damage soaked (for Steeled and Empire Divider)
    def soak_trigger (self, soaked_damage):
        self.damage_soaked += soaked_damage

    def ignore_style_range(self, player):
        return (player is self.opponent and
                self.mighty in self.active_cards)

    def evaluate (self):
        value = Character.evaluate(self)
        value -= 0.4 * len(self.induced_pool)
        return value

class Alumis(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Dread (the_game, self)
        self.styles = [Tenebrous (the_game, self),
                       Arcane    (the_game, self),
                       Horrific  (the_game, self),
                       Ghastly   (the_game, self),
                       Sinister  (the_game, self)  ]
        self.finishers = [AphoticTerror (the_game, self),
                          UmbralVice    (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.shadow_position = 6 - self.position

    def choose_initial_discards (self):
        return (self.sinister, self.grasp,
                self.tenebrous, self.strike)

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.shadow_position = addendum[0].find('s')

    board_addendum_lines = 1
    def get_board_addendum (self):
        addendum = ['.'] * 7
        addendum [self.shadow_position] = 's'
        return ''.join(addendum)

    def initial_save (self):
        state = Character.initial_save (self)
        state.shadow_position = self.shadow_position
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.shadow_position = state.shadow_position

    def reset(self):
        Character.reset(self)
        self.arcane_soak = 0

    def full_save(self):
        state = Character.full_save(self)
        state.arcane_soak = self.arcane_soak
        return state

    def full_restore(self, state):
        Character.full_restore(self, state)
        self.arcane_soak = state.arcane_soak

    # Shadow follows opponent.
    def movement_reaction(self, initiator, mover, old_position, direct):
        Character.movement_reaction(self, initiator, mover, old_position, direct)
        if mover is self:
            return
        opp = self.opponent.position
        if abs(opp - self.shadow_position) <= 1:
            return
        self.shadow_position = opp+1 if opp < self.shadow_position else opp-1
        if self.game.reporting:
            self.game.report("The Shadow follows %s" % self.opponent)
            for s in self.game.get_board():
                self.game.report (s)

    # Aphotic Terror attacks from shadow.
    def attack_range(self):
        if self.aphotic_terror in self.active_cards:
            return abs(self.shadow_position - self.opponent.position)
        return self.game.distance()

    def get_soak(self):
        return Character.get_soak(self) + self.arcane_soak


class Arec (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Hex (the_game, self)
        self.styles = [Phantom      (the_game, self),
                       Perceptional (the_game, self),
                       Returning    (the_game, self),
                       Mirrored     (the_game, self),
                       Manipulative (the_game, self)  ]
        self.finishers = [UncannyOblivion (the_game, self)]
        self.tokens = [Fear         (the_game, self),
                       Hesitation   (the_game, self),
                       Mercy        (the_game, self),
                       Recklessness (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.1 # Tokens might block priority bonuses.

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = self.tokens[:]
        self.clone_position = None
        self.manipulative_base = None

    #TODO: choose
    def choose_initial_discards (self):
        return (self.phantom, self.strike,
                self.manipulative, self.grasp)

    def situation_report(self):
        report = Character.situation_report (self)
        tokens = [t.name for t in self.pool]
        report.append("Pool: " + ', '.join(tokens))
        if self.manipulative_base:
            report.append("Manipulative forced base: %s" %
                          self.manipulative_base)
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [t for t in self.tokens if t.name in lines[0]]
        line = find_start_line(lines, "Manipulative forced base:")
        if line:
            for base in self.opponent.bases:
                if base.name in line:
                    self.manipulative_base = base
        else:
            self.manipulative_base = None
        self.clone_position = addendum[0].find('c')
        if self.clone_position == -1:
            self.clone_position = None

    board_addendum_lines = 1
    def get_board_addendum (self):
        if self.clone_position is None:
            return ''
        addendum = ['.'] * 7
        addendum [self.clone_position] = 'c'
        return ''.join(addendum)

    def initial_save (self):
        state = Character.initial_save (self)
        state.clone_position = self.clone_position
        state.manipulative_base = self.manipulative_base
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.clone_position = state.clone_position
        self.manipulative_base = state.manipulative_base

    def reset (self):
        self.returning_range_triggered = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.returning_range_triggered = self.returning_range_triggered
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.returning_range_triggered = state.returning_range_triggered

    def get_antes (self):
        clone_antes = ([False] if self.clone_position is None
                       else [False, True])
        token_antes = [None] + self.pool
        return list(itertools.product(clone_antes, token_antes))

    def input_ante (self, pair):
        if self.clone_position is None:
            clone_ante = False
        else:
            print "Move to clone?"
            clone_ante = bool(menu(['No', 'Yes']))
        if self.pool:
            print "Select token to ante:"
            options = [t.name for t in self.pool] + ["None"]
            ans = menu (options)
            token_ante = (self.pool[ans] if ans < len(self.pool)
                          else None)
        else:
            token_ante = None
        return (clone_ante, token_ante)

    def ante_trigger (self):
        if self.strat[2][0][0]:
            assert self.clone_position is not None
            # Setting clone position to None before moving, so that
            # clone doesn't appear in post-move board report.
            pos = self.clone_position
            self.clone_position = None
            self.move_directly([pos])
        if self.strat[2][0][1] is not None:
            self.ante_token (self.strat[2][0][1])
        self.clone_position = None

    def get_ante_name (self, a):
        token_name = a[1].name if a[1] else ""
        clone_name = "move to clone" if a[0] else ""
        names = [token_name, clone_name]
        names = filter(None, names)
        return '; '.join(names)

    def recover_tokens (self):
        recoverable = [t for t in self.tokens if t not in (self.pool+self.ante)]
        if len(recoverable) > 0:
            values = [t.value for t in recoverable]
            prompt = "Select a token to recover:"
            options = [t.name for t in recoverable]
            choice = values.index (max (values))
            # fake fork for AI, just select the best token
            choice = self.game.make_fork (len(recoverable), self, \
                                               prompt, options, choice)
            recovered = recoverable [choice]
            self.pool += [recovered]
            if self.game.reporting:
                self.game.report ("%s recovers his %s token" % (self, recovered.name))

    # Tokens block opponent's triggers.
    def blocks_start_triggers(self):
        return self.fear in self.active_cards
    def blocks_before_triggers(self):
        return self.hesitation in self.active_cards
    def blocks_hit_triggers(self):
        return self.mercy in self.active_cards
    def blocks_damage_triggers(self):
        return self.mercy in self.active_cards
    def blocks_after_triggers(self):
        return self.hesitation in self.active_cards

    # Tokens block own triggers.
    def start_trigger (self):
        if not any(p.blocks_start_triggers() for p in self.game.player):
            self.activate_card_triggers('start_trigger')
    def before_trigger (self):
        if not any(p.blocks_before_triggers() for p in self.game.player):
            self.activate_card_triggers('before_trigger')
    def hit_trigger (self):
        if not any(p.blocks_hit_triggers() for p in self.game.player):
            self.activate_card_triggers('hit_trigger')
    # also, Hex doesn't stun.
    def damage_trigger (self, damage):
        if not any(p.blocks_damage_triggers() for p in self.game.player):
            self.activate_card_triggers('damage_trigger', [damage])
        if self.base != self.unique_base:
            self.opponent.stun (damage)
    def after_trigger (self):
        if not any(p.blocks_after_triggers() for p in self.game.player):
            self.activate_card_triggers('after_trigger')

    # Token blocks opponent's bonuses
    def blocks_power_bonuses(self):
        return self.recklessness in self.active_cards
    def blocks_priority_bonuses(self):
        return self.recklessness in self.active_cards

    # Token blocks own bonuses
    def get_priority_no_swap(self):
        priority = Character.get_priority_no_swap(self)
        if self.blocks_priority_bonuses():
            priority = min (priority, self.get_printed_priority())
        return priority
    def get_power(self):
        power = Character.get_power(self)
        if self.blocks_power_bonuses():
            power = min(power, self.get_printed_power())
        power = max(0, power)
        return power

    def get_minrange(self):
        return 1 if self.returning_range_triggered else Character.get_minrange(self)
    def get_maxrange(self):
        return 6 if self.returning_range_triggered else Character.get_maxrange(self)

    # BUG: doesn't play nice with clashes: will go by final base,
    # rather than initial one.
    def reveal_trigger(self):
        if self.manipulative_base not in (None, self.opponent.base):
            self.opponent.lose_life(2)
        self.manipulative_base = None

    def get_superposed_positions(self):
        if self.clone_position is None:
            return [self.position]
        else:
            # Since Arec only moves in the ante phase, he gives
            # the clone position even if it's on the opponent
            # (the opponent might move before attack pairs are set).
            return [self.position, self.clone_position]
    def get_superposition_priority(self):
        if self.clone_position is None:
            return self.game.CHOOSE_POSITION_NOW
        else:
            return self.game.CHOOSE_POSITION_IN_ANTE
    def evaluate_superposition(self):
        if (self.clone_position is None or
            self.clone_position == self.opponent.position):
            return 0
        # Try for differing ranges, to mix opponent up.
        d0 = self.game.distance()
        d1 = abs(self.clone_position - self.opponent.position)
        return 0.2 + 0.1 * abs(d0-d1)

    def evaluate (self):
        value = (Character.evaluate(self) + sum(t.value for t in self.pool))
        if self.manipulative_base is not None and self.opponent.life > 1:
            # Compare opponent's evalutation with and without chosen base.
            opp_eval = self.opponent.evaluate()
            real_preferred_range = self.opponent.preferred_range
            self.opponent.discard[2].add(self.manipulative_base)
            self.opponent.set_preferred_range()
            new_opp_eval = self.opponent.evaluate()
            self.opponent.discard[2].remove(self.manipulative_base)
            self.opponent.preferred_range = real_preferred_range
            base_value = new_opp_eval - opp_eval
            # Shift range of possible values (say, -10 to 10) to (0 to 2)
            # (2 is maximum value of this ability).
            base_value = base_value / 10.0 + 1
            if self.opponent.life == 2:
                base_value /= 2
            value += base_value
        return value

class Aria (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Reconfiguration (the_game, self)
        self.styles = [Photovoltaic (the_game, self), \
                       Ionic        (the_game, self), \
                       Laser        (the_game, self), \
                       Catalyst     (the_game, self), \
                       Dimensional  (the_game, self)  ]
        self.finishers = [LaserLattice (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.dampening = Dampening()
        self.magnetron = Magnetron()
        self.turret = Turret()
        self.droids = [self.dampening,
                       self.magnetron,
                       self.turret]

    mean_priority_bonus = 0.4 # Magnetron

    def choose_initial_discards (self):
        return (self.laser, self.grasp,
                self.ionic, self.dash)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        # Choose initial droid
        if self.is_user:
            print "Choose initial droid:"
            ans = menu (self.droids)
            self.droids[ans].position = self.position
        else:
            self.magnetron.position = self.position

    board_addendum_lines = 3
    def get_board_addendum (self):
        addendum = []
        for droid in self.droids:
            line = ['.'] * 7
            if droid.position is not None:
                line[droid.position] = droid.initial
            line = ''.join(line)
            addendum.append(line)
        return addendum

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        for i, droid in enumerate(self.droids):
            pos = addendum[i].find(droid.initial)
            if pos > -1:
                droid.position = pos
            else:
                droid.position = None

    def initial_save (self):
        state = Character.initial_save (self)
        state.droid_positions = [droid.position for droid in self.droids]
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        for i,droid in enumerate(self.droids):
            droid.position = state.droid_positions[i]

    def reset (self):
        # where the attack is coming from - either self or a droid
        self.attacker = self
        # which droids have already attacked (in a Laser Lattice)
        for droid in self.droids:
            droid.has_attacked = False
        self.dimensional_no_stun = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.attacker = self.attacker
        state.droids_have_attacked = [droid.has_attacked
                                      for droid in self.droids]
        state.dimensional_no_stun = self.dimensional_no_stun
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.attacker = state.attacker
        for i,droid in enumerate(self.droids):
            droid.has_attacked = state.droids_have_attacked[i]
        self.dimensional_no_stun = state.dimensional_no_stun

    def attack_range (self):
        return abs(self.attacker.position - self.opponent.position)

    def give_power_penalty (self):
        dampening = self.dampening.position
        return (-1
                if dampening is not None and
                   abs (self.opponent.position - dampening) <= 1
                else 0)

    def give_priority_penalty (self):
        penalty = Character.give_priority_penalty(self)
        magnetron = self.magnetron.position
        if (magnetron is not None and
            abs (self.opponent.position - magnetron) <= 1):
            penalty -= 1
        return penalty

    def end_trigger (self):
        Character.end_trigger (self)
        turret = self.turret.position
        if turret is not None and abs (self.opponent.position - turret) <= 1:
            self.opponent.lose_life (1)

    def damage_trigger (self, damage):
        if not self.opponent.blocks_damage_triggers():
            self.activate_card_triggers('damage_trigger', [damage])
        # Dimensional doesn't stun if its trigger prevented it
        if not self.dimensional_no_stun:
            self.opponent.stun (damage)

    # When using Laser, make the choice about origin of attack -
    # just before attacking
    def before_trigger (self):
        Character.before_trigger(self)
        if self.laser in self.active_cards:
            sources = [self] + [d for d in self.droids
                                if d.position is not None]
            ans = self.game.make_fork (len(sources), self,
                                       "Choose source of attack:",
                                       sources)
            self.attacker = sources[ans]

    def evaluate (self):
        value = Character.evaluate(self)
        dampening = self.dampening.position
        magnetron = self.magnetron.position
        turret = self.turret.position
        positions = [droid.position for droid in self.droids
                     if droid.position is not None]

        # Long term value of droid positions
        # 1 life per active droid
        # 0.2 life per position covered by a droid
        covered = set()
        for pos in positions:
            covered |= set((pos-1, pos, pos+1))
        covered &= set(xrange(7))

        value += len(positions)
        value += 0.2 * len (covered)

        # Short term value of droid positions vis-a-vis opponent
        # 0.5 per droid adjacent to opponent
        opp = self.opponent.position
        for pos in positions:
            if abs(pos - opp) <= 1:
                value += 0.5

        # Long term value of droid positions because of styles

        # Dampening is more likely to block attacks from center
        if dampening is not None:
            value += 0.05 * dampening * (6-dampening)
        # Magnetron is more useful when we can pull opponents to edge
        if magnetron is not None:
            value += 0.1 * abs (magnetron - 3)
        # Turret can hit two positions from center
        if turret == 3:
            value += 0.2

        # Short term value vs. style effects depend on which styles are in hand,
        # and are handled by evaluation_bonus() on each style.

        return value

class Baenvier(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Spire (the_game, self)
        self.styles = [Spellforge  (the_game, self), \
                       Curse       (the_game, self), \
                       Runic       (the_game, self), \
                       Crescent    (the_game, self), \
                       Destruction (the_game, self)  ]
        self.finishers = [Spellbreak   (the_game, self)]
        self.tokens = [SpellEater (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = [self.spell_eater] * 3

    def choose_initial_discards (self):
        return (self.destruction, self.strike,
                self.curse, self.grasp)

    # Baenvier's token doesn't have ante effects, so it's not checked
    # for them (for efficiency)
    def get_active_tokens (self):
        tokens = Character.get_active_tokens (self)
        # Induced tokens are still checked normally.
        return [t for t in tokens if t not in self.tokens]

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Spell Eater token%s" %(len(self.pool),
                       '' if len(self.pool)==1 else 's'))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.spell_eater] * int(lines[0][0])

    def get_antes (self):
        if self.pool:
            return [0,1]
        else:
            return [0]

    def input_ante (self, pair):
        if self.pool:
            print ("Ante a Spell Eater Token?")
            return menu (["No", "Yes"])
        else:
            return 0

    def ante_trigger (self):
        if self.strat[2][0]:
            # For now, tokens have no effect (and are returned to hand)
            # if any player used a special action.
            if any(player.style is player.special_action
                   for player in self.game.player):
                return
            self.ante_token(self.spell_eater)
            for player in (self, self.opponent):
                if player.style is player.special_action:
                    # If a player used a special action, we have to
                    # figure out which base activated it.
                    # Since we can't have forks before the reveal
                    # we choose arbitrarily (limiting the opponent's
                    # options unfairly).
                    if player.base is player.cancel:
                        options = player.cancel_generators.copy()
                        if any(f.devolves_into_cancel()
                               for f in player.finishers):
                            options |= player.finisher_generators
                    elif player.base is player.pulse:
                        options = player.pulse_generators.copy()
                    else:
                        options = player.finisher_generators.copy()
                    options &= set(player.bases)
                    options -= (player.discard[1] | player.discard[2])
                    options = sorted(list(options), key=attrgetter('name'))
                    base = options[0] # First option alphabetically.
                    player.discard[0].discard(player.base)
                    player.base = base
                    player.discard[0].add(base)
                player.discard[0].discard(player.style)
                if self.game.reporting:
                    self.game.report("%s is returned to %s's hand" %
                                     (player.style, player))
                player.style = player.null_style

    # Runic bonus transfer.  Bug: can't separate negative and positive
    # bonuses if both are applied simultaneously.
    def blocks_power_bonuses(self):
        return self.runic in self.active_cards and self.opponent.hits_scored
    def take_a_hit_trigger(self):
        if self.runic in self.active_cards:
            self.add_triggered_power_bonus(max(0,
                                    self.opponent.get_power_bonuses()))

    def get_ante_name (self, a):
        return ("Spell Eater" if a==1 else "")

    def ignore_style_range(self, player):
        return (player is self and
                self.spire in self.active_cards)

    def evaluate (self):
        return Character.evaluate(self) + 1.5 * len(self.pool)

class Borneo(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.bases = [ClawBorneo       (the_game,self),
                      DividerBorneo    (the_game,self),
                      HexBorneo        (the_game,self),
                      Knuckle          (the_game,self),
                      MasterPlan       (the_game,self),
                      SceneShiftBorneo (the_game,self),
                      Smoke            (the_game,self)]
        self.styles = [Weaksauce (the_game,self),
                       Clumsy    (the_game,self),
                       Pathetic  (the_game,self),
                       Petulant  (the_game,self),
                       Slippery  (the_game,self)]
        self.finishers = [MamaMazzaroth (the_game, self)]
        self.status_effects = [OpponentPowerPriorityStatusEffect(
                                the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.unstunners = set([self.hex, self.clumsy])

    def choose_initial_discards (self):
        return (self.slippery, self.smoke,
                self.petulant, self.knuckle)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.mazzaroth_active = False

    def situation_report (self):
        report = Character.situation_report (self)
        if self.mazzaroth_active:
            report.append ("Mama Mazzaroth activated")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.mazzaroth_active = find_start(lines, 'Mama Mazzaroth activated')

    def initial_save (self):
        state = Character.initial_save (self)
        state.mazzaroth_active = self.mazzaroth_active
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.mazzaroth_active = state.mazzaroth_active

    def reset (self):
        self.triggers_blocked = False
        self.switched_sides = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.triggers_blocked = self.triggers_blocked
        state.switched_sides = self.switched_sides
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.triggers_blocked = state.triggers_blocked
        self.switched_sides = state.switched_sides

    def damage_trigger (self, damage):
        if not self.opponent.blocks_damage_triggers():
            self.activate_card_triggers('damage_trigger', [damage])
        # Clumsy and Hex don't stun
        if not self.unstunners & set(self.active_cards):
            self.opponent.stun (damage)

    # Hex blocks opponent's triggers.
    def blocks_start_triggers(self):
        return self.triggers_blocked
    def blocks_before_triggers(self):
        return self.triggers_blocked
    def blocks_hit_triggers(self):
        return self.triggers_blocked
    def blocks_damage_triggers(self):
        return self.triggers_blocked
    def blocks_after_triggers(self):
        return self.triggers_blocked
    def blocks_end_triggers(self):
        return self.triggers_blocked

    # Smoke ignores style's range modifier
    def get_maxrange (self):
        if self.base.name == 'Smoke':
            maxrange = self.base.maxrange + \
                       self.opponent.give_maxrange_penalty()
            if self.opponent.blocks_maxrange_bonuses():
                maxrange = min (maxrange, self.base.maxrange)
        else:
            maxrange = Character.get_maxrange(self)
        return maxrange

    def get_minrange (self):
        if self.base.name == 'Smoke':
            minrange = self.base.minrange + \
                       self.opponent.give_minrange_penalty()
            if self.opponent.blocks_minrange_bonuses():
                minrange = min (minrange, self.base.minrange)
        else:
            minrange = Character.get_minrange(self)
        return minrange

    def wins_on_timeout(self):
        return self.mazzaroth_active

    def evaluate(self):
        return Character.evaluate(self) + 6 * self.mazzaroth_active


class Byron (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Smoke (the_game, self)
        self.styles = [Soulless   (the_game, self), \
                       Deathless  (the_game, self), \
                       Heartless  (the_game, self), \
                       Faceless   (the_game, self), \
                       Breathless (the_game, self)  ]
        self.finishers = [SoulTrap (the_game, self),
                          SoulGate (the_game, self)]
        self.status_effects = [PriorityBonusStatusEffect(the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.starting_life = 15
        self.counter_name = "Mask"
        self.max_counters = 5

    def choose_initial_discards (self):
        return (self.soulless, self.strike,
                self.heartless, self.shot)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.counters = 5

    def reset (self):
        self.life_lost = 0
        self.counters_discarded = 0
        self.heartless_ignore_soak = False
        self.heartless_soak = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.life_lost = self.life_lost
        state.counters_discarded = self.counters_discarded
        state.heartless_ignore_soak = self.heartless_ignore_soak
        state.heartless_soak = self.heartless_soak
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.life_lost = state.life_lost
        self.counters_discarded = state.counters_discarded
        self.heartless_ignore_soak = state.heartless_ignore_soak
        self.heartless_soak = state.heartless_soak

    def discard_counters (self, n):
        loss = Character.discard_counters(self, n)
        # recording discards for Soulless
        if loss:
            self.counters_discarded += 1

    def take_damage_trigger (self, damage):
        if not self.damage_taken:
            self.discard_counters(1)

    def lose_life (self, amount):
        Character.lose_life (self, amount)
        self.life_lost += amount

    def ignore_style_range(self, player):
        return (player is self and
                self.smoke in self.active_cards)

    def reduce_soak (self, soak):
        return 0 if self.heartless_ignore_soak else Character.reduce_soak(self, soak)
    def get_soak (self):
        return self.heartless_soak + Character.get_soak(self)

    # restoring life at end, rather than start of beat
    # this lets opponent's evaluation figure that damage is worthless
    # (it's incorrect in beat 15, where the damage might break the tie)

    # Putting in cycle, to make sure Pulse doesn't stop it
    # (Pulse shouldn't stop it,
    #  because it really happens at the start of next beat)
    def cycle (self):
        Character.cycle (self)
        self.life = max (3 * self.counters - self.life_lost, 1)

    # If life is low, just return it
    # But if life is high, the deduction from life loss doesn't matter
    # (you won't die this beat, and it will go away next beat)
    def effective_life (self):
        if self.life >= 8:
            life_loss_effect = 0
        elif self.life <= 3:
            life_loss_effect = 1
        else:
            life_loss_effect = (8 - self.life) / 5.0
        # The final result is scaled from 15 to 20
        return 20.0 / 15 * max (3 * self.counters - self.life_lost * life_loss_effect, 1)

class Cadenza (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Press (the_game, self)
        self.styles = [Battery    (the_game, self), \
                       Clockwork  (the_game, self), \
                       Hydraulic  (the_game, self), \
                       Grapnel    (the_game, self), \
                       Mechanical (the_game, self)  ]
        self.finishers = [RocketPress   (the_game, self),
                           FeedbackField (the_game, self)]
        self.status_effects = [PriorityBonusStatusEffect(the_game, self)]
        self.tokens = [IronBody  (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = [self.iron_body] * 3

    def choose_initial_discards (self):
        return (self.mechanical, self.strike,
                self.grapnel, self.grasp)

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Iron Body token%s" %(len(self.pool),
                       '' if len(self.pool)==1 else 's'))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.iron_body] * int(lines[0][0])

    def reset (self):
        self.token_spent = False
        self.damage_soaked = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.token_spent = self.token_spent
        state.damage_soaked = self.damage_soaked
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.token_spent = state.token_spent
        self.damage_soaked = state.damage_soaked

    def get_antes (self):
        if self.pool:
            return [0,1]
        else:
            return [0]

    def input_ante (self, pair):
        if self.pool:
            print ("Ante an Iron Body Token?")
            return menu (["No", "Yes"])
        else:
            return 0

    def ante_trigger (self):
        if self.strat[2][0]:
            self.ante_token(self.iron_body)

    def get_ante_name (self, a):
        return ("Iron Body" if a==1 else "")

    def get_stunguard (self):
        return (1000000 if self.token_spent else Character.get_stunguard(self))

    def expected_stunguard(self):
        ret = Character.expected_stunguard(self)
        # Assume tokens are mostly used retroactively for stunguard,
        # rather than preemptively for immunity.
        if self.pool:
            ret += 4
        return ret

    # when Cadenza is about to be stunned by damage, he may spend a token
    def stun (self, damage=None):
        if (damage is not None and
            self.can_spend(1) and
            self.iron_body not in self.ante and
            damage > self.get_stunguard()):
            if self.game.make_fork (2, self, "Spend an Iron Body token?",
                                    ["No", "Yes"]):
                self.token_spent = 1
                self.spend_token()
                if self.game.reporting:
                    self.game.report ("Cadenza gains infinite stunguard")
        return Character.stun (self, damage)

    # record damage soaked (for Feedback Field)
    def soak_trigger (self, soaked_damage):
        self.damage_soaked += soaked_damage

    def evaluate (self):
        return Character.evaluate(self) + 2.5 * len(self.pool)

class Cesar (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Suppression (the_game, self)
        self.styles = [Phalanx     (the_game, self),
                       Unstoppable (the_game, self),
                       Fueled      (the_game, self),
                       Bulwark     (the_game, self),
                       Inevitable  (the_game, self)  ]
        self.finishers = [Level4Protocol (the_game, self)]
        self.status_effects = [PowerPenaltyStatusEffect(the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def choose_initial_discards (self):
        return (self.fueled, self.strike,
                self.unstoppable, self.grasp)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.threat_level = 0
        self.defeat_immunity = True

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("Threat level: %d" % self.threat_level)
        if not self.defeat_immunity:
            report.append("Defeat immunity used-up")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.threat_level = int(lines[0][-2])
        self.defeat_immunity = not find_start(lines, "Defeat immunity used")

    def initial_save (self):
        state = Character.initial_save (self)
        state.threat_level = self.threat_level
        state.defeat_immunity = self.defeat_immunity
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.threat_level = state.threat_level
        self.defeat_immunity = state.defeat_immunity

    def reset(self):
        Character.reset(self)
        self.threat_level_at_ante = 0

    def full_save(self):
        state = Character.full_save(self)
        state.threat_level_at_ante = self.threat_level_at_ante
        return state

    def full_restore(self, state):
        Character.full_restore(self, state)
        self.threat_level_at_ante = state.threat_level_at_ante

    def ante_trigger (self):
        self.gain_threat_level()
        self.threat_level_at_ante = self.threat_level

    def gain_threat_level (self):
        self.threat_level += 1
        if self.threat_level > 4:
            self.threat_level = 0
        if self.game.reporting:
            self.game.report("Cesar gains a Threat Level (current level: %d)" %
                             self.threat_level)
        if self.threat_level == 4:
            self.stun()

    def get_power_bonus(self):
        return (self.threat_level_at_ante
                if self.threat_level_at_ante < 4 else 0)

    def get_stunguard(self):
        stunguard = Character.get_stunguard(self)
        if self.threat_level_at_ante <= 2:
            stunguard += self.threat_level_at_ante
        return stunguard

    def expected_stunguard(self):
        ret = Character.expected_stunguard(self)
        expected_level = (self.threat_level + 1) % 5
        stunguard_per_level = [ret, ret+1, ret+2, 0, 0]
        return stunguard_per_level[expected_level]

    def has_stun_immunity(self):
        # You have stun immunity if you got to 3 at ante, but didn't
        # play Fueled.
        return ((self.threat_level_at_ante == 3 and
                 self.threat_level == 3)
                or Character.has_stun_immunity(self))

    def get_soak(self):
        soak = Character.get_soak(self)
        if self.threat_level_at_ante == 4:
            soak += 2
        return soak

    def expected_soak(self):
        ret = Character.expected_soak(self)
        if self.threat_level == 3:
            ret += 2
        return ret

    # Copying all this to implement Cesar's death immunity.
    def take_damage (self, damage):
        soak = self.opponent.reduce_soak(self.get_soak())
        if self.game.reporting and soak:
            self.game.report ('%s has %d soak' % (self, soak))
        damage_soaked = min (soak, damage)
        if damage_soaked:
            self.soak_trigger (damage_soaked)
        remaining_damage = max(self.get_damage_cap() - self.damage_taken, 0)
        final_damage = min (damage - damage_soaked, remaining_damage)

        # life can't go below certain minimum (usually 0)
        self.life = max (self.life - final_damage,
                         self.get_minimum_life())
        if self.game.reporting:
            self.game.report (self.name + \
                              " takes %d damage (now at %d life)" \
                              %(final_damage, self.life))
        if self.life <= 0:
            # This is the different part.
            if self.defeat_immunity:
                self.life = 1
                self.defeat_immunity = False
            else:
                raise WinException (self.opponent.my_number)
        self.opponent.damage_trigger (final_damage)
        self.take_damage_trigger (final_damage)
        # damage_taken updated after triggers, so that triggers can
        # check if this is first damage this beat
        self.damage_taken += final_damage

    def blocks_hit_triggers (self):
        return self.unstoppable in self.active_cards

    def blocks_damage_triggers (self):
        return self.unstoppable in self.active_cards

    # It's best to end the beat with level 4,0,1 and worst to end with 3
    def evaluate (self):
        return (Character.evaluate(self) +
                [1.4, 0.9, -0.35, -2.6, 1.15][self.threat_level] +
                5 * self.defeat_immunity)

class Claus (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Tempest (the_game, self)
        self.styles = [Hurricane (the_game, self),
                       Tailwind  (the_game, self),
                       Blast     (the_game, self),
                       Cyclone   (the_game, self),
                       Downdraft (the_game, self)  ]
        self.finishers = [AutumnsAdvance (the_game, self)]
        self.status_effects = [PriorityBonusStatusEffect(the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def choose_initial_discards (self):
        return (self.hurricane, self.grasp,
                self.cyclone, self.shot)

    # Implement's Claus's movement rule: when you would pass over an
    # opponent, stop short and push opponent instead, causing damage
    # for spaces moved.
    # For now, assuming that :
    # 1. Claus is pulled normally by opponents, may switch sides.
    # 2. If Claus would be blocked from moving, pushing doesn't happen
    #    (even if the UA would result in pushing only).
    # NOTE: not implementing max_move, because Claus doesn't have such
    # movements.
    def inner_execute_move (self, mover, moves, direct, max_move):
        # Moving opponent (Grasp) happens normally.
        if mover is self.opponent:
            return Character.inner_execute_move(self, mover, moves, direct, max_move)
        # From now on, assume:
        # 1. mover is self.
        # 2. movement is not direct (since Claus has no direct moves).
        # 3. there are no mover blocks (claus never blocks own moves).
        initial_pos = self.position
        dests = self.get_destinations(self, moves)
        blocked = self.opponent.blocks_opponent_movement(self, direct)
        blocked.discard(initial_pos)
        self.dests = dests
        self.mover_block = set()
        self.opponent_block = blocked
        if not blocked:
            possible = list (dests)
        else:
            unobstructed = self.get_unobstructed_positions(
                                        initial_pos, blocked, direct)
            possible = list (dests & unobstructed)
        if possible:
            prompt = "Choose position to move to:"
            options = []
            possible = sorted(possible)
            if self.is_user and self.game.interactive_mode:
                for p in possible:
                    self.position = p
                    options.append (self.game.get_basic_board())
                self.position = initial_pos
            ans = self.game.make_fork (len(possible), self, prompt, options)
            dest = possible[ans]
            # If not trying to move over opponent, move normally.
            # Also, against a mimic, move normally (they retreat as you
            # advance, nothing interesting happens).
            if (not ordered(initial_pos, self.opponent.position, dest)
                or self.opponent.mimics_movement()):
                self.position = dest
                return
            # Trying to move over opponent: this the fun part.
            if dest > initial_pos:
                self.position = self.opponent.position - 1
            else:
                self.position = self.opponent.position + 1
            push_distance = abs(dest - self.opponent.position)
            self.push([push_distance], specific_movement_reaction=self.specific_movement_reaction)

    def specific_movement_reaction(self, initiator, mover, old_position, direct):
            self.deal_damage(abs(self.opponent.position -
                                 old_position))

class Clinhyde (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Frenzy (the_game, self)
        self.styles = [Toxic     (the_game, self),
                       Diffusion (the_game, self),
                       Shock     (the_game, self),
                       Gravity   (the_game, self),
                       Phase     (the_game, self)  ]
        self.finishers = [VitalSilverInfusion (the_game, self),
                          RitherwhyteInfusionClinhyde (the_game, self)]
        self.packs = [Crizma   (the_game, self),
                      Ehrlite  (the_game, self),
                      Hylatine (the_game, self) ]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.5 # Ehrlite

    def all_cards (self):
        return Character.all_cards(self) + self.packs

    def choose_initial_discards (self):
        return (self.diffusion, self.unique_base,
                self.phase, self.grasp)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.active_packs = set([])
        self.vital_silver_activated = False
        self.ritherwhyte_activated = False

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("Active Stim Packs: " +
                       ', '.join([p.name for p in self.active_packs]))
        if self.vital_silver_activated:
            report.append ("Vital Silver Infusion activated")
        if self.ritherwhyte_activated:
            report.append ("Ritherwhyte Infusion activated")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.active_packs = set([p for p in self.packs if p.name in lines[0]])
        self.vital_silver_activated = find_start (lines, 'Vital Silver')
        self.ritherwhyte_activated = find_start (lines, 'Ritherwhyte')

    def initial_save (self):
        state = Character.initial_save (self)
        state.active_packs = self.active_packs.copy()
        state.vital_silver_activated = self.vital_silver_activated
        state.ritherwhyte_activated = self.ritherwhyte_activated
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.active_packs = state.active_packs.copy()
        self.vital_silver_activated = state.vital_silver_activated
        self.ritherwhyte_activated = state.ritherwhyte_activated

    def get_antes (self):
        return [p for p in self.packs if p not in self.active_packs] + [None]

    def input_ante (self, pair):
        inactive = [p for p in self.packs if p not in self.active_packs]
        if inactive:
            print "Select a stim pack to activate:"
            options = [p.name for p in inactive] + ["None"]
            ans = menu (options)
            return (inactive[ans] if ans < len(inactive) else None)
        else:
            return None

    def get_ante_name (self, a):
        return ("" if a==None else a.name)

    def set_active_cards (self):
        Character.set_active_cards (self)
        self.active_cards.extend (list(self.active_packs))

    def ante_trigger (self):
        self.ante_activation = self.strat[2][0]
        if self.ante_activation is not None:
            self.ante_activation.activate()
        if len(self.active_packs) > 0:
            self.lose_life (len(self.active_packs))
        if self.vital_silver_activated or self.ritherwhyte_activated:
            self.lose_life (1)

    def get_maxrange_bonus (self):
        return 2 if self.ritherwhyte_activated else 0

    def get_power_bonus (self):
        return 1 if isinstance (self.ante_activation, Crizma) else 0

    def get_priority_bonus (self):
        return 1 if isinstance (self.ante_activation, Ehrlite) else 0

    # shortcut because no other sources of soak
    def get_soak (self):
        return 2 if isinstance (self.ante_activation, Hylatine) else 0

    def expected_soak(self):
        ret = Character.expected_soak(self)
        if self.hylatine not in self.active_packs:
            ret += 1
        return ret

    def expected_stunguard(self):
        ret = Character.expected_stunguard(self)
        if self.hylatine in self.active_packs:
            ret += 2
        else:
            ret += 1
        return ret

    def may_ignore_own_movement(self):
        return self.gravity in self.active_cards

    def set_preferred_range (self):
        Character.set_preferred_range (self)
        if self.ritherwhyte_activated:
            self.preferred_range += 1

    activation_values = [0, -0.5, -1.5, -3]
    def evaluate (self):
        value = Character.evaluate(self)
        value += self.activation_values[len(self.active_packs)]
        if self.ritherwhyte_activated:
            value += 4
        if self.vital_silver_activated:
            value -= 4
        return value


class Clive (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Wrench (the_game, self)
        self.styles = [Upgradeable (the_game, self), \
                       Rocket      (the_game, self), \
                       Burnout     (the_game, self), \
                       Megaton     (the_game, self), \
                       LeapingClive(the_game, self)  ]
        self.finishers = [SystemShock (the_game, self)]
        self.modules = [RocketBoots   (the_game, self), \
                        BarrierChip   (the_game, self), \
                        AtomicReactor (the_game, self), \
                        ForceGloves   (the_game, self), \
                        CoreShielding (the_game, self), \
                        AfterBurner   (the_game, self), \
                        SynapseBoost  (the_game, self), \
                        AutoRepair    (the_game, self), \
                        ExtendingArms (the_game, self)  ]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def all_cards (self):
        return Character.all_cards(self) + self.modules

    def choose_initial_discards (self):
        # Not really checked
        return (self.upgradeable, self.grasp,
                self.rocket, self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.module_stack = self.modules[:]
        self.active_modules = []

    def situation_report (self):
        report = Character.situation_report (self)
        if self.module_stack:
            report.append ("Modules in stack:")
            for module in self.module_stack:
                report.append ("  %s" % module.name)
        else:
            report.append ("No modules in stack")
        if self.active_modules:
            report.append ("Active Modules:")
            for module in self.active_modules:
                report.append ("  %s" % module.name)
        else:
            report.append ("No active modules")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        if lines[0] == "No modules in stack\n":
            self.module_stack = []
            i = 1
        else:
            module_names = []
            i = 1
            while lines[i].startswith('  '):
                module_names.append(lines[i][2:-1])
                i += 1
            self.module_stack = [module for module in self.modules
                                  if module.name in module_names]
        if lines[i] == "No active modules\n":
            self.active_modules = []
        else:
            module_names = []
            i += 1
            while lines[i].startswith('  '):
                module_names.append(lines[i][2:-1])
                i += 1
            self.active_modules = [module for module in self.modules
                                   if module.name in module_names]

    def initial_save (self):
        state = Character.initial_save (self)
        state.module_stack = self.module_stack[:]
        state.active_modules = self.active_modules[:]
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.module_stack = state.module_stack[:]
        self.active_modules = state.active_modules[:]

    def reset (self):
        self.switched_sides = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.switched_sides = self.switched_sides
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.switched_sides = state.switched_sides

    def get_antes (self):
        return [None] + self.module_stack

    def input_ante (self, pair):
        if self.module_stack:
            print "Select a module to activate:"
            options = [m.name for m in self.module_stack] + ["None"]
            ans = menu (options)
            return (self.module_stack[ans]
                    if ans < len(self.module_stack)
                    else None)
        else:
            return None

    def get_ante_name (self, ante):
        return ante.name if ante else ""

    def ante_trigger (self):
        module = self.strat[2][0]
        if module:
            self.activate_module(module)

    def set_active_cards (self):
        Character.set_active_cards(self)
        self.active_cards += self.active_modules

    # Discard all modules when stunned
    def stun (self, damage=None):
        stunned = Character.stun (self, damage)
        if stunned:
            # Oppnent loses life when playing Burnout
            if self.game.distance() == 1 and self.burnout in self.active_cards:
                self.opponent.lose_life(len(self.active_modules))
            while self.active_modules:
                self.discard_module(self.active_modules[0])
        return stunned

    def activate_module (self, module):
        self.module_stack.remove(module)
        self.active_modules.append(module)
        self.set_active_cards()
        if self.game.reporting:
            self.game.report ("Clive activates %s" % module.name)

    def return_module (self, module):
        self.module_stack.append(module)
        if module in self.active_modules:
            self.active_modules.remove(module)
            self.set_active_cards()
        if self.game.reporting:
            self.game.report ("Clive returns %s to his module stack"
                              % module.name)

    def discard_module (self, module):
        self.active_modules.remove(module)
        self.set_active_cards()
        if self.game.reporting:
            self.game.report ("Clive discards %s" % module.name)

    def blocks_hit_triggers (self):
        return self.core_shielding in self.active_cards

    def blocks_damage_triggers (self):
        return self.core_shielding in self.active_cards

    def set_preferred_range (self):
        Character.set_preferred_range(self)
        if self.rocket_boots in self.active_modules:
            self.preferred_range += 0.5
        elif self.rocket_boots in self.module_stack:
            self.preferred_range += 0.1
        if self.extending_arms in self.active_modules:
            self.preferred_range += 0.5
        elif self.extending_arms in self.module_stack:
            self.preferred_range += 0.1

    def expected_stunguard(self):
        ret = Character.expected_stunguard(self)
        if self.barrier_chip in self.active_modules:
            ret += 1
        elif self.barrier_chip in self.module_stack:
            ret += 0.2
        return ret

    # Activating a module is worth 0.1, losing it is worth -1.1
    def evaluate (self):
        return (Character.evaluate(self)
              + 1.1 * len(self.active_modules)
              + 1.0 * len(self.module_stack))


class Danny (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Hellgate (the_game, self)
        self.styles = [Monstrous (the_game, self), \
                       Fallen    (the_game, self), \
                       Shackled  (the_game, self), \
                       Sinners   (the_game, self), \
                       Vicious   (the_game, self)  ]
        self.finishers = [TheDarkRide (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def choose_initial_discards (self):
        return (self.fallen, self.burst,
                self.vicious, self.shot)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.monsters = set()

    board_addendum_lines = 1
    def get_board_addendum (self):
        if not self.monsters:
            return ''
        addendum = ['.'] * 7
        for m in self.monsters:
            addendum [m] = 'm'
        return ''.join(addendum)

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.monsters = set([i for i in xrange(7) if addendum[0][i]=='m'])

    def initial_save (self):
        state = Character.initial_save (self)
        state.monsters = self.monsters.copy()
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.monsters = state.monsters.copy()

    def reset (self):
        self.fallen_ignore_stunguard = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.fallen_ignore_stunguard = self.fallen_ignore_stunguard
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.fallen_ignore_stunguard = state.fallen_ignore_stunguard

    def add_monsters(self, n_monsters, positions):
        positions &= set(xrange(7))
        positions -= self.monsters
        n_monsters = min(n_monsters, len(positions))
        if not n_monsters:
            return
        combos = list(itertools.combinations(positions, n_monsters))
        prompt = ("Add monster:" if n_monsters == 1
                  else "Add monsters:")
        options = []
        if self.is_user and self.game.interactive_mode:
            base_list = [('m' if m in self.monsters else '.'
                          for m in xrange(7))]
            for combo in combos:
                tmp_list = base_list[:]
                for m in combo:
                    tmp_list[m] = 'm'
                options.append (''.join(tmp_list))
        combo = combos[
            self.game.make_fork(len(combos), self, prompt, options)]
        self.monsters |= set(combo)
        if self.game.reporting:
            self.game.report ("Danny adds monsters to the board:")
            for s in self.game.get_board():
                self.game.report (s)

    def remove_monsters(self, n_monsters, positions):
        positions &= self.monsters
        n_monsters = min(n_monsters, len(positions))
        if not n_monsters:
            return
        combos = list(itertools.combinations(positions, n_monsters))
        prompt = ("Remove monster:" if n_monsters == 1
                  else "Remove monsters:")
        options = []
        if self.is_user and self.game.interactive_mode:
            for combo in combos:
                tmp_list = ['.'] * 7
                for x in combo:
                    tmp_list[x] = 'x'
                options.append (''.join(tmp_list))
        combo = combos[
            self.game.make_fork(len(combos), self, prompt, options)]
        self.monsters -= set(combo)
        if self.game.reporting:
            self.game.report ("%s removes monsters:" % self.opponent)
            for s in self.game.get_board():
                self.game.report (s)

    # Put one monster in range of the attack.
    # I assume this happens before other "after activating" effects.
    def after_trigger(self):
        pos_set = self.in_standard_range()
        if pos_set:
            self.add_monsters(n_monsters=1, positions=pos_set)
        Character.after_trigger(self)

    # Remove one monster in opponent's range.
    def after_trigger_for_opponent(self):
        pos_set = self.opponent.in_standard_range()
        if pos_set:
            self.remove_monsters(n_monsters=1, positions=pos_set)

    def reduce_stunguard(self, stunguard):
        return 0 if self.fallen_ignore_stunguard else Character.reduce_stunguard(self, stunguard)

    def special_range_hit (self):
        return (self.opponent.position in self.monsters or
                Character.special_range_hit(self))

    def evaluate_range (self):
        # If the range is closer than my preferred range, the problem
        # is that it's too easy for the opponent to hit.  Monsters
        # don't help
        # But if it's farther, the problem is that it's too hard for
        # me to hit, and they do.

        distance = self.game.distance()
        penalty = - self.game.range_weight * \
                    (self.preferred_range - distance) ** 2
        if distance <= self.preferred_range:
            return penalty
        # Monster on opponent kills 70% of penalty,
        # monsters adjacent kill 20%, monsters at range 2 kill 10%.
        attenuation = 0.7 * (self.opponent.position in self.monsters)
        adjacent = self.get_destinations(self.opponent, [-1,1])
        attenuation += 0.2 * len(adjacent & self.monsters) / len(adjacent)
        two_away = self.get_destinations(self.opponent, [-2,2])
        attenuation += 0.1 * len(two_away & self.monsters) / len(two_away)
        return penalty * (1 - attenuation)

    def evaluate (self):
        # Monsters are worth more earlier in the game.
        return (Character.evaluate(self) +
                0.1 * self.game.expected_beats() * len(self.monsters))

class Demitras (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Deathblow (the_game, self)
        self.styles = [Darkside     (the_game, self), \
                       Bloodletting (the_game, self), \
                       Vapid        (the_game, self), \
                       Illusory     (the_game, self), \
                       Jousting     (the_game, self)  ]
        self.finishers = [SymphonyOfDemise (the_game, self),
                          Accelerando      (the_game, self)]
        self.tokens = [Crescendo  (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.max_tokens = 5

    mean_priority_bonus = 2 # Tokens.

    def choose_initial_discards (self):
        return (self.bloodletting, self.grasp,
                self.illusory, self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 2 * [self.crescendo]

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Crescendo token%s" %(len(self.pool),
                       '' if len(self.pool)==1 else 's'))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.crescendo] * int(lines[0][0])

    def get_antes (self):
        return range (len(self.pool) + 1)

    def input_ante (self, pair):
        n = len(self.pool)
        if n > 0 :
            print "Number of Crescendo tokens [0-%d]: " %n
            return input_number (n+1)
        else:
            return 0

    def ante_trigger (self):
        for unused in range(self.strat[2][0]):
            self.ante_token(self.crescendo)

    def get_ante_name (self, a):
        if a == 0:
            return ""
        if a == 1:
            return "1 token"
        return str(a) + " tokens"

    def get_priority_bonus (self):
        return len(self.pool)

    # add new token before activating style/base triggers
    # to give Deathblow the option to use the new token
    def hit_trigger (self):
        self.recover_tokens (1)
        Character.hit_trigger (self)

    def take_a_hit_trigger (self):
        self.discard_token ()
        Character.take_a_hit_trigger (self)

    def evaluate (self):
        return Character.evaluate (self) + self.token_value (len(self.pool))

    # tokens worth fixed amount when spent for power
    # (1.75 rather than 2 because they might miss)
    # their presence in pool is worth less the more you have
    # [4 3 2 1 0.5]
    # (because you must use them soon, and because of diminishing returns
    #  on priority)
    cumulative_pool_value = [0,4,7,9,10,10.5]

    def token_value (self, tokens):
        spend_value = 1.75 * tokens
        pool_value = self.cumulative_pool_value [tokens]
        return spend_value + pool_value


class Eligor (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Aegis (the_game, self)
        self.styles = [Vengeful     (the_game, self), \
                       CounterStyle (the_game, self), \
                       Martial      (the_game, self), \
                       Chained      (the_game, self), \
                       Retribution  (the_game, self)  ]
        self.finishers = [SweetRevenge   (the_game, self),
                          SheetLightning (the_game, self)]
        self.status_effects = [OpponentImmobilizedStatusEffect(the_game, self)]
        self.tokens = [VengeanceT (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.max_tokens = 5

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 2 * [self.vengeance]

    def choose_initial_discards (self):
        return (self.martial, self.grasp,
                self.chained, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Vengeance token%s" %(len(self.pool),
                       '' if len(self.pool)==1 else 's'))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.vengeance] * int(lines[0][0])

    def get_antes (self):
        return range (len(self.pool) + 1)

    def input_ante (self, pair):
        n = len(self.pool)
        if n > 0 :
            print "Number of Vengeance tokens [0-%d]: " %n
            return input_number (n+1)
        else:
            return 0

    def ante_trigger (self):
        for unused in range(self.strat[2][0]):
            self.ante_token(self.vengeance)

    def get_ante_name (self, a):
        if a == 0:
            return ""
        if a == 1:
            return "1 token"
        return str(a) + " tokens"

    def take_damage_trigger (self, damage):
        self.recover_tokens (damage)

    def has_stun_immunity (self):
        return self.get_active_tokens().count(self.vengeance) == 5

    # Retribution recovers a token for each damage soaked (max 2)
    def soak_trigger (self, soaked_damage):
        if self.retribution in self.active_cards:
            self.recover_tokens (min(soaked_damage, 2))

    def get_minimum_life (self):
        return (1 if self.sweet_revenge in self.active_cards
                else Character.get_minimum_life(self))

    def evaluate (self):
        return Character.evaluate (self) + 0.2 * len(self.pool)

class Eustace(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Cross (the_game, self)
        self.styles = [Elementary (the_game, self), \
                       Ivory      (the_game, self), \
                       Frigid     (the_game, self), \
                       Frostbite  (the_game, self), \
                       Barons     (the_game, self)  ]
        self.finishers = [BoneBreaker (the_game, self),
                          RimeBlitzer (the_game, self)]
        self.status_effects = [PriorityPenaltyStatusEffect(the_game, self)]
        self.tokens = [Frostflow (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.max_tokens = 5

    def choose_initial_discards (self):
        return (self.barons, self.grasp,
                self.elementary, self.drive)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = [self.frostflow] * 3

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Frostflow token%s" %(len(self.pool),
                       '' if len(self.pool)==1 else 's'))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.frostflow] * int(lines[0][0])

    def reset (self):
        self.elementary_range_bonus = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.elementary_range_bonus = self.elementary_range_bonus
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.elementary_range_bonus = state.elementary_range_bonus

    def get_antes (self):
        return range(len(self.pool) + 1)

    def input_ante (self, pair):
        if self.pool:
            print "Number of Frostflow Tokens to ante [0-%d]: " % len(self.pool)
            return input_number(len(self.pool) + 1)
        else:
            return 0

    def ante_trigger (self):
        for unused in range(self.strat[2][0]):
            self.ante_token(self.frostflow)

    def get_ante_name (self, a):
        if a == 0:
            return ""
        if a == 1:
            return "1 token"
        return str(a) + " tokens"

    def get_maxrange(self):
        if self.frostflow in self.active_cards:
            return Character.get_minrange(self)
        else:
            return Character.get_maxrange(self)

    def get_maxrange_bonus(self):
        return self.elementary_range_bonus

    def discard_token(self, token=None, verb="discard"):
        ret = Character.discard_token(self, token, verb)
        if not ret:
            if self.game.reporting:
                self.game.report("Eustace runs out of Frost")
            self.lose_life(3)
            self.opponent.set_triggered_dodge()
        return ret

    # Discard a token on first damage.
    def take_damage_trigger(self, damage):
        Character.take_damage_trigger(self, damage)
        if not self.damage_taken:
            self.discard_token()

    # Lose a counter on first stun.
    def stun (self, damage=None):
        prior_stun = self.is_stunned()
        stunned = Character.stun (self, damage)
        if not prior_stun and stunned:
            self.discard_token()
        return stunned

    def unique_ability_end_trigger(self):
        self.recover_tokens(1)

    # Keep track of switching sides for Cross
    def movement_reaction(self, initiator, mover, old_position, direct):
        Character.movement_reaction(self, initiator, mover, old_position, direct)
        if not direct and mover.opponent.mimics_movement():
            return
        if ordered(old_position, mover.opponent.position, mover.position):
            self.switched_sides = True

    # When anteing, max range is same as minrange.
    def set_preferred_range (self):
        unavailable_cards = self.discard[1] \
                          | self.discard[2]
        styles = set(self.styles) - unavailable_cards
        bases = set(self.bases) - unavailable_cards
        preferred_range = \
            sum (s.get_preferred_range() for s in styles) / float(len(styles)) + \
            sum (b.get_preferred_range() for b in bases) / float(len(bases))
        preferred_range_when_anteing = \
            sum (s.get_preferred_range() - (s.maxrange - s.minrange) / 2 for s in styles) / float(len(styles)) + \
            sum (b.get_preferred_range() - (b.maxrange - b.minrange) / 2  for b in bases) / float(len(bases))
        ante_prob = len(self.pool) * 0.1
        self.preferred_range = (preferred_range * (1 - ante_prob) +
                                preferred_range_when_anteing * ante_prob)

    frost_value = 0.55
    def evaluate(self):
        return (Character.evaluate(self) + self.frost_value * self.counters
            - self.frost_value * (self.counters <= 1))


class Gerard (Character):
    # TODO: Gold is tokens, not counters.
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Larceny (the_game, self)
        self.styles = [Villainous (the_game, self), \
                       Gilded     (the_game, self), \
                       Avaricious (the_game, self), \
                       Hooked     (the_game, self), \
                       Initiation (the_game, self)  ]
        self.finishers = [Windfall   (the_game, self)]
        self.mercenaries = [Brawler     (the_game, self),
                            Archer      (the_game, self),
                            Trebuchet   (the_game, self),
                            HeavyKnight (the_game, self),
                            Gunslinger  (the_game, self),
                            Mage        (the_game, self),
                            Lackey      (the_game, self),
                            Bookie      (the_game, self)]
        self.status_effects = [InitiationStatusEffect(the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.1 # Mage

    def all_cards(self):
        return Character.all_cards(self) + self.mercenaries

    def choose_initial_discards (self):
        return (self.villainous, self.unique_base,
                self.hooked, self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.mercs_in_play = []
        self.removed_mercs = []
        self.gold = 3

    def situation_report (self):
        report = Character.situation_report (self)
        report.append("Gold: %d" % self.gold)
        if self.mercs_in_play:
            report.append ("Mercenaries in play:")
            for merc in self.mercs_in_play:
                report.append ("  %s" % merc.name)
        else:
            report.append ("No mercenaries in play")
        if self.removed_mercs:
            report.append ("Mercenaries removed from the game:")
            for merc in self.removed_mercs:
                report.append ("  %s" % merc.name)
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        lines.append('') # prevents going out of range in while loops.
        self.gold = int(lines[0].split()[1])
        i = 2
        if lines[1] == "No mercenaries in play\n":
            self.mercs_in_play = []
        else:
            merc_names = []
            while lines[i].startswith('  '):
                merc_names.append(lines[i][2:-1])
                i += 1
            self.mercs_in_play = [merc for merc in self.mercenaries
                                  if merc.name in merc_names]
        if lines[i] == "Mercenaries removed from the game\n":
            merc_names = []
            i += 1
            while lines[i].startswith('  '):
                merc_names.append(lines[i][2:-1])
                i += 1
            self.removed_mercs = [merc for merc in self.mercenaries
                                  if merc.name in merc_names]
        else:
            self.removed_mercs = []

    def initial_save (self):
        state = Character.initial_save (self)
        state.gold = self.gold
        state.mercs_in_play = self.mercs_in_play[:]
        state.removed_mercs = self.removed_mercs[:]
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.gold = state.gold
        self.mercs_in_play = state.mercs_in_play[:]
        self.removed_mercs = state.removed_mercs[:]

    def reset(self):
        Character.reset(self)
        self.activated_mercs = []
        self.switched_sides = False

    def full_save(self):
        state = Character.full_save(self)
        state.activated_mercs = self.activated_mercs[:]
        state.switched_sides = self.switched_sides
        return state

    def full_restore(self, state):
        Character.full_restore(self, state)
        self.activated_mercs = state.activated_mercs
        self.switched_sides = state.switched_sides

    def get_antes(self):
        # a "first-beats" game doesn't care about mercenaries,
        # which are more long term.
        if self.game.first_beats:
            return [([],[])]
        antes = []
        # Make list of mercenaries, and cost to ante each.
        # AI never hires Bookie.
        possible_hires = list(set(self.mercenaries) -
                              set(self.mercs_in_play) -
                              set(self.removed_mercs) -
                              set([self.bookie]))
        hiring_gold = self.gold
        if self.initiation_status_effect in self.active_status_effects:
            hiring_gold += 2
        hiring_combos = []
        for n_hires in xrange(len(possible_hires) + 1):
            combos_n = itertools.combinations(possible_hires, n_hires)
            playable = [c for c in combos_n if
                        sum([h.hiring_cost for h in c]) <= hiring_gold]
            hiring_combos += playable
            if not playable:
                break
        for hiring_combo in hiring_combos:
            hiring_combo = list(hiring_combo)
            if hiring_combo:
                remaining_gold = hiring_gold - sum([h.hiring_cost
                                                    for h in hiring_combo])
            else:
                remaining_gold = self.gold
            possible_activations = [merc for merc in self.mercs_in_play +
                                                     hiring_combo
                                    if merc.activation_cost]
            for n_activations in xrange(len(possible_activations) + 1):
                combos_n = itertools.combinations(possible_activations,
                                                  n_activations)
                playable = [list(c) for c in combos_n if
                            sum([a.activation_cost for a in c]) <= remaining_gold]
                antes += [(hiring_combo, activation_combo)
                          for activation_combo in playable]
                if not playable:
                    break
        return antes

    def input_ante(self, pair):
        remaining_gold = self.gold
        initiation = self.initiation_status_effect in self.active_status_effects
        # List of new hirelings, list of activations
        ante = ([], [])
        # Start with hiring:
        while True:
            discount = 2 if initiation and not ante[0] else 0
            possible_hires = [
                merc for merc in self.mercenaries
                if merc not in self.mercs_in_play +
                               self.removed_mercs + ante[0] and
                   merc.hiring_cost - discount <= remaining_gold]
            if not possible_hires:
                break
            print "Choose next mercenary to hire:"
            options = [h.name for h in possible_hires]
            options.append("Done hiring")
            ans = menu(options)
            if ans == len(possible_hires):
                break
            ante[0].append(possible_hires[ans])
            remaining_gold -= (possible_hires[ans].hiring_cost - discount)
        # Proceed with activations:
        while True:
            possible_activations = [
                merc for merc in self.mercs_in_play + ante[0]
                if merc not in ante[1] and merc.activation_cost and
                   merc.activation_cost <= remaining_gold]
            if not possible_activations:
                break
            print "Choose next mercenary to activate:"
            options = [a.name for a in possible_activations]
            options.append("Done activating")
            ans = menu(options)
            if ans == len(possible_activations):
                break
            ante[1].append(possible_activations[ans])
            remaining_gold -= possible_activations[ans].activation_cost
        return ante

    def get_ante_name(self, ante):
        strings = []
        if ante[0]:
            strings.append("Hire " + ', '.join([a.name for a in ante[0]]))
        if ante[1]:
            strings.append("Activate " + ', '.join([a.name for a in ante[1]]))
        return '; '.join(strings)

    def ante_trigger(self):
        hire = self.strat[2][0][0]
        discount = 2 if (self.initiation_status_effect in
                         self.active_status_effects) else 0
        for h in hire:
            self.mercs_in_play.append(h)
            self.gold -= (h.hiring_cost - discount)
            if self.game.reporting:
                self.game.report("Gerard hires a %s for %d gold (%d remaining)" %
                                 (h, h.hiring_cost - discount, self.gold))
            discount = 0
        activate = self.strat[2][0][1]
        for a in activate:
            self.activated_mercs.append(a)
            self.gold -= a.activation_cost
            if self.game.reporting:
                self.game.report("Gerard activates his %s for %d gold (%d remaining)" %
                                 (a, a.activation_cost, self.gold))
        self.set_active_cards()

    def set_active_cards(self):
        Character.set_active_cards(self)
        self.active_cards += [merc for merc in self.mercs_in_play
                              if not merc.activation_cost]
        self.active_cards += self.activated_mercs

    def gain_gold(self, gold):
        previous = self.gold
        self.gold = min(10, self.gold + gold)
        gain = self.gold - previous
        if gain and self.game.reporting:
            self.game.report("Gerard gains %d Gold Token%s (has %d)" %
                             (gain, 's' if gain > 1 else '', self.gold))

    def unique_ability_end_trigger(self):
        self.gain_gold(1)

    def cycle(self):
        Character.cycle(self)
        if self.lackey in self.mercs_in_play:
            self.mercs_in_play.remove(self.lackey)
            self.removed_mercs.append(self.lackey)

    def movement_reaction(self, initiator, mover, old_position, direct):
        Character.movement_reaction(self, initiator, mover, old_position, direct)
        if not direct and mover.opponent.mimics_movement():
            return
        if ordered(old_position, mover.opponent.position, mover.position):
            self.switched_sides = True

    gold_value = 0.5

    def evaluate(self):
        if self.game.debugging:
            self.game.report("gold evaluation: %.2f"% (self.gold*self.gold_value))
            for merc in self.mercs_in_play:
                self.game.report('%s: %.2f'%(merc.name, merc.get_value()))
        return (Character.evaluate(self) + self.gold * self.gold_value +
                sum([merc.get_value() for merc in self.mercs_in_play]))

class Heketch (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Knives (the_game, self)
        self.styles = [Merciless (the_game, self), \
                       Rasping   (the_game, self), \
                       Critical  (the_game, self), \
                       Assassin  (the_game, self), \
                       Psycho    (the_game, self)  ]
        self.finishers = [MillionKnives   (the_game, self),
                           LivingNightmare (the_game, self)]
        self.status_effects = [OpponentImmobilizedStatusEffect(the_game, self)]
        self.tokens = [DarkForce (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.max_tokens = 1

    mean_priority_bonus = 1.2 # Token.

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = [self.dark_force]
        self.living_nightmare_active = False

    def choose_initial_discards (self):
        return (self.rasping, self.burst,
                self.critical, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        if len(self.pool) == 1:
            report.append ("Dark Force in pool")
        if self.living_nightmare_active:
            report.append ("Living Nightmare active")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = ([self.dark_force] if find_start(lines, 'Dark Force')
                     else [])
        self.living_nightmare_active = find_start(lines, "Living Nightmare active")

    def initial_save (self):
        state = Character.initial_save (self)
        state.living_nightmare_active = self.living_nightmare_active
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.living_nightmare_active = state.living_nightmare_active

    def reset (self):
        self.merciless_immobilized = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.merciless_immobilized = self.merciless_immobilized
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.merciless_immobilized = state.merciless_immobilized

    # 0 is no ante
    # -1,1 are ante and going to the left/right of opponent
    # if your destination is impossible, go to the other
    def get_antes (self):
        if self.pool:
            return [0,-1,1]
        else:
            return [0]

    def input_ante (self, pair):
        if self.pool:
            print ("Ante Dark Force token?")
            ans = menu (["No", "Yes"])
            if ans == 0:
                return 0
            else:
                print "Prefer which side of opponent?"
                ans = menu (["Left", "Right"])
                return (-1 if ans==0 else 1)
        else:
            return 0

    def ante_trigger (self):
        ante = self.strat[2][0]
        if ante:
            self.ante_token(self.dark_force)
        if self.dark_force in self.ante:
            # try going to the space indicated by the ante.
            # if it doesn't work, try the other one.
            old_position = self.position
            self.move_directly ((self.opponent.position+ante,))
            if self.position == old_position:
                self.move_directly ((self.opponent.position-ante,))

    def get_ante_name (self, a):
        if a == 0:
            return ""
        if a == -1:
            return "Dark Force - left"
        if a == 1:
            return "Dark Force - right"

    def damage_trigger (self, damage):
        if not self.opponent.blocks_damage_triggers():
            self.activate_card_triggers('damage_trigger', [damage])
        # Knives doesn't stun at range 1
        if not isinstance (self.base, Knives) or self.game.distance() > 1:
            self.opponent.stun (damage)

    def blocks_opponent_movement (self, initiator, direct):
        return (set(xrange(7)) if self.merciless_immobilized
                else Character.blocks_opponent_movement(self, initiator, direct))

    # retrieve token at end of beat
    def unique_ability_end_trigger (self):
        if self.game.distance() >= 3 or self.living_nightmare_active:
            self.recover_tokens(1)

    def evaluate (self):
        ev = Character.evaluate(self)
        if self.pool:
            ev += 0.8 + 0.1 * self.game.distance()
        if self.living_nightmare_active:
            ev += 7
        return ev

class Hepzibah(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Bloodlight (the_game,self)
        self.styles = [Pactbond    (the_game, self),
                       Anathema    (the_game, self),
                       Necrotizing (the_game, self),
                       Darkheart   (the_game, self),
                       Accursed    (the_game, self)  ]
        self.finishers = [Altazziar       (the_game, self)]
        self.pacts = [Almighty    (the_game, self),
                      Corruption  (the_game, self),
                      Endless     (the_game, self),
                      Immortality (the_game, self),
                      InevitableT (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def all_cards(self):
        return Character.all_cards(self) + self.pacts

    # TODO: choose
    def choose_initial_discards (self):
        return (self.pactbond, self.unique_base,
                self.anathema, self.strike)

    def set_active_cards(self):
        Character.set_active_cards(self)
        if not self.opponent.blocks_tokens():
            self.active_cards += self.anted_pacts
            # Double up on pacts if playing the finisher.
            if self.altazziar in self.active_cards:
                self.active_cards += self.anted_pacts

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.active_pactbond = None

    def situation_report (self):
        report = Character.situation_report (self)
        if self.active_pactbond:
            report.append("Pactbond: %s" % self.active_pactbond)
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        line = find_start_line(lines, 'Pactbond:')
        self.active_pactbond = None
        for pact in self.pacts:
            if pact.name in line:
                self.active_pactbond = pact

    def reset (self):
        self.pending_pactbond = None
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.pending_pactbond = self.pending_pactbond
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.pending_pactbond = state.pending_pactbond

    def prepare_next_beat(self):
        Character.prepare_next_beat(self)
        self.active_pactbond = self.pending_pactbond

    def get_antes (self):
        available_life = self.life - 1
        if self.active_pactbond:
            available_life += 1
        max_ante = min(5, available_life)
        combos = [itertools.combinations(self.pacts, n)
                  for n in xrange(max_ante + 1)]
        # Optimization: force ante of active pactbond.
        combos = sum([list(c) for c in combos], [])
        if self.active_pactbond:
            combos = [c for c in combos
                      if self.active_pactbond in c]
        return combos

    def input_ante (self, pair):
        antes = self.get_antes()
        if len(antes) == 1:
            return antes[0]
        options = [self.get_ante_name(a) for a in antes]
        if not options[0]:
            options[0] = "None"
        print "Select Dark Pacts to ante:"
        return antes[menu(options)]

    def ante_trigger (self):
        self.anted_pacts = self.strat[2][0]
        life_loss = len(self.anted_pacts)
        if self.active_pactbond in self.anted_pacts:
            life_loss -= 1
        self.lose_life(life_loss)

    def get_ante_name (self, a):
        return ', '.join(pact.name for pact in a)

    def set_preferred_range(self):
        Character.set_preferred_range(self)
        if self.pending_pactbond is self.inevitable:
            self.preferred_range += 0.5
        elif self.life > 1:
            self.preferred_range += 0.4

    def expected_soak(self):
        ret = Character.expected_soak(self)
        if self.pending_pactbond is self.immortality:
            ret += 2
        elif self.life > 1:
            ret += 1
        return ret

    def evaluate(self):
        # having low life is especially bad, can't ante.
        return (Character.evaluate(self) + 0.5 * min(0, self.life - 5) +
                (self.pending_pactbond is not None))

class Hikaru (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = PalmStrike (the_game,self)
        self.styles = [Trance    (the_game, self), \
                       Focused   (the_game, self), \
                       Advancing (the_game, self), \
                       Sweeping  (the_game, self), \
                       Geomantic (the_game, self)  ]
        self.finishers = [WrathOfElements (the_game, self),
                          FourWinds       (the_game, self)]
        self.tokens = [Earth (the_game, self), \
                       Fire  (the_game, self), \
                       Water (the_game, self), \
                       Wind  (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.5 # Wind

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = self.tokens[:]

    def choose_initial_discards (self):
        return (self.sweeping, self.grasp,
                self.trance, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        tokens = [t.name for t in self.pool]
        report.append ("Pool: " + ', '.join(tokens))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [t for t in self.tokens if t.name in lines[0]]

    def get_antes (self):
        return [None] + self.pool

    def input_ante (self, pair):
        if self.pool:
            print "Select token to ante:"
            options = [t.name for t in self.pool] + ["None"]
            ans = menu (options)
            return (self.pool[ans] if ans < len(self.pool) else None)
        else:
            return None

    def ante_trigger (self):
        if self.strat[2][0] is not None:
            self.ante_token (self.strat[2][0])

    def get_ante_name (self, a):
        return ("" if a==None else a.name)

    def recover_tokens (self):
        recoverable = [t for t in self.tokens if t not in (self.pool+self.ante)]
        if len(recoverable) > 0:
            values = [t.value for t in recoverable]
            prompt = "Select a token to recover:"
            options = [t.name for t in recoverable]
            choice = values.index (max (values))
            # fake fork for AI, just select the best token
            choice = self.game.make_fork (len(recoverable), self, \
                                               prompt, options, choice)
            recovered = recoverable [choice]
            self.pool += [recovered]
            if self.game.reporting:
                self.game.report ("Hikaru recovers his " + recovered.name + " token")

    # Sweeping adds 2 to damage taken
    def take_damage (self, damage):
        if damage:
            if self.sweeping in self.active_cards:
                damage += 2
            Character.take_damage(self, damage)

    # Four Winds blocks Elemental tokens
    def get_active_tokens (self):
        active_tokens = Character.get_active_tokens (self)
        if self.base.name == 'Four Winds':
            # Induced tokens are still active
            return [t for t in active_tokens if t not in self.tokens]
        else:
            return active_tokens

    # +0/1 to range for Water token is 0.5
    # 0.2 because the token is spent, and another token can't be anted
    def set_preferred_range (self):
        Character.set_preferred_range (self)
        self.preferred_range += 0.2 * (self.water in self.pool)

    def expected_soak(self):
        return 1.5 if self.earth in self.pool else 0

    def evaluate (self):
        return Character.evaluate(self) + sum(t.value for t in self.pool)


class Iri(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Starlight (the_game,self)
        self.styles = [Magical  (the_game, self),
                       Comet    (the_game, self),
                       Astral   (the_game, self),
                       Galactic (the_game, self),
                       Aurora   (the_game, self)]
        self.finishers = [LimitExceed            (the_game, self),
                          FinalGalaxySpearAtamos (the_game, self)]
        self.forms = [MagicalForm (the_game, self),
                      KnightForm  (the_game, self),
                      CosmicForm  (the_game, self)]
        self.status_effects = [PowerBonusStatusEffect(the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def all_cards(self):
        return Character.all_cards(self) + self.forms

    def set_active_cards(self):
        Character.set_active_cards(self)
        if self.limit_exceeded:
            self.active_cards += [form for form in self.forms
                                  if form is not self.current_form
                                  or not self.changed_form
                                  or self.galactic in self.active_cards]
        elif not self.changed_form or self.galactic in self.active_cards:
            self.active_cards.append(self.current_form)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.current_form = self.magical_form
        self.limit_exceeded = False

    def choose_initial_discards (self):
        return (self.galactic, self.burst,
                self.aurora, self.grasp)

    def situation_report (self):
        report = Character.situation_report (self)
        if self.limit_exceeded:
            report.append("Limit exceeded")
        else:
            report.append("%s" % self.current_form)
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.limit_exceeded = lines[0].startswith("Limit exceeded")
        if not self.limit_exceeded:
            for form in self.forms:
                if form.name in lines[0]:
                    self.current_form = form

    def initial_save (self):
        state = Character.initial_save (self)
        state.current_form = self.current_form
        state.limit_exceeded = self.limit_exceeded
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.current_form = state.current_form
        self.limit_exceeded = state.limit_exceeded

    def reset (self):
        self.changed_form = False
        self.knight_form_stunguard = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.changed_form = self.changed_form
        state.knight_form_stunguard = self.knight_form_stunguard
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.changed_form = state.changed_form
        self.knight_form_stunguard = state.knight_form_stunguard

    def get_antes (self):
        return self.forms

    def input_ante (self):
        print "Select your form:"
        return self.forms[menu([form.name for form in self.forms])]

    def ante_trigger (self):
        new_form = self.strat[2][0]
        if new_form is not self.current_form:
            self.current_form = new_form
            self.changed_form = True
            new_form.transformation_trigger()

    def get_ante_name (self, a):
        return a.name

    def get_stunguard(self):
        return Character.get_stunguard(self) + self.knight_form_stunguard

    # Take damage trigger for Astral.
    def take_damage_trigger(self, damage):
        if self.astral in self.active_cards:
            self.add_triggered_power_bonus(min(damage, 5))

    def set_preferred_range(self):
        Character.set_preferred_range(self)
        if self.current_form is self.magical_form:
            self.preferred_range += 0.17
        if self.limit_exceeded:
            self.preferred_range += 0.33

    def expected_stunguard(self):
        sg = Character.expected_stunguard(self)
        # You can get +3 by switching / staying at Knight Form.
        sg += 1
        if self.limit_exceeded:
            sg += 2
        return sg

    def evaluate(self):
        value = Character.evaluate(self)
        if self.current_form == self.cosmic_form:
            if self.game.distance() == 1:
                value += 0.2
            else:
                value -= 0.1
        # About 1.4 per beat.
        if self.limit_exceeded:
            value += 4
        return value


class Jager(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Slicer (the_game,self)
        self.styles = [Refined   (the_game, self),
                       Hounding  (the_game, self),
                       Menacing  (the_game, self),
                       BloodMoon (the_game, self),
                       Fanged    (the_game, self)  ]
        self.finishers = [BloodHunt     (the_game, self),
                          SignOfTheWolf (the_game, self)]
        self.signatures = [WolfsFangFugue       (the_game, self),
                           CrescentMoonClaw     (the_game, self),
                           RedMoonRageSignature (the_game, self),
                           MiststepMassacre     (the_game, self)]
        self.status_effects = [RefinedStatusEffect(the_game, self),
                               HoundingStatusEffect(the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def all_cards(self):
        return Character.all_cards(self) + self.signatures

    def set_active_cards(self):
        if self.strat[2][0] and not self.opponent.blocks_tokens():
            self.base = self.strat[2][0]
            self.active_cards = ([self.base] +
                self.active_status_effects + self.get_active_tokens())
        else:
            Character.set_active_cards(self)
            if self.finisher_signature:
                self.active_cards.append(self.finisher_signature)

    def choose_initial_discards (self):
        return (self.refined, self.grasp,
                self.menacing, self.burst)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.signature_pool = [] if self.game.first_beats else self.signatures[:]

    def situation_report (self):
        report = Character.situation_report (self)
        sigs = [s.name for s in self.signature_pool]
        report.append("Signature moves: " + ', '.join(sigs))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.signature_pool = [s for s in self.signatures if s.name in lines[0]]

    def initial_save (self):
        state = Character.initial_save (self)
        state.signature_pool = self.signature_pool[:]
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.signature_pool = state.signature_pool[:]

    def reset (self):
        self.switched_sides = False
        self.finisher_signature = None
        self.fugue_chosen_direction = None
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.switched_sides = self.switched_sides
        state.finisher_signature = self.finisher_signature
        state.fugue_chosen_direction = self.fugue_chosen_direction
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.switched_sides = state.switched_sides
        self.finisher_signature = state.finisher_signature
        self.fugue_chosen_direction = state.fugue_chosen_direction

    def get_antes (self):
        return [None] + self.signature_pool

    def input_ante (self, pair):
        if self.signature_pool and pair[0] is not self.special_action:
            print "Select signature move to play:"
            options = [s.name for s in self.signature_pool] + ["None"]
            ans = menu (options)
            return (self.signature_pool[ans] if ans < len(self.signature_pool) else None)
        else:
            return None

    # Don't allow ante with special action (this might be legal, but
    # problematic to implement because special action base isn't chosen).
    def get_strategies(self):
        strats = Character.get_strategies(self)
        return [s for s in strats
                if s[0] is not self.special_action or s[2][0] is None]

    def ante_trigger (self):
        if self.strat[2][0] is not None:
            self.signature_pool.remove(self.strat[2][0])

    def get_ante_name (self, a):
        return ("" if a==None else a.name)

    # Recording switching sides, for Blood Moon and Blood Hunt.
    def movement_reaction(self, initiator, mover, old_position, direct):
        Character.movement_reaction(self, initiator, mover, old_position, direct)
        if not direct and mover.opponent.mimics_movement():
            return
        if ordered(old_position, mover.opponent.position, mover.position):
            self.switched_sides = True

    def evaluate(self):
        return (Character.evaluate(self) +
                sum(s.get_value() for s in self.signature_pool))


class Juto(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.bases = [Flash         (the_game,self),
                      Lance         (the_game,self),
                      Press         (the_game,self),
                      Scythe        (the_game,self),
                      SpellboltJuto (the_game,self),
                      Staff         (the_game,self),
                      WhirlpoolJuto (the_game,self)]
        self.styles = [Fluffy  (the_game,self),
                       Snuggly (the_game,self),
                       Cuddly  (the_game,self),
                       Cute    (the_game,self),
                       Fuzzy   (the_game,self)]
        self.finishers = [BanHammer (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def choose_initial_discards (self):
        return (self.snuggly, self.flash,
                self.cuddly, self.whirlpool)

    def reset (self):
        self.staff_block_advancing = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.staff_block_advancing = self.staff_block_advancing
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.staff_block_advancing = state.staff_block_advancing

    def damage_trigger (self, damage):
        if not self.opponent.blocks_damage_triggers():
            self.activate_card_triggers('damage_trigger', [damage])
        # Cuddly doesn't stun
        if self.cuddly not in self.active_cards:
            self.opponent.stun (damage)

    def blocks_opponent_movement(self, initiator, direct):
        if direct or initiator is self or not self.staff_block_advancing:
            return set()
        me = self.position
        opp = self.opponent.position
        if me > opp:
            return set(xrange(opp+1, 7))
        else:
            return set(xrange(opp))

class Kajia(Character):

    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Mandibles (the_game,self)
        self.styles = [Stinging  (the_game,self),
                       Swarming  (the_game,self),
                       Burrowing (the_game,self),
                       Parasitic (the_game,self),
                       Biting    (the_game,self)]
        self.finishers = [Wormwood      (the_game, self),
                          ImagoEmergence (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def choose_initial_discards (self):
        return (self.stinging, self.strike,
                self.swarming, self.burst)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        # number of insect counters on each discard pile
        # (current pair, discard 1, discard 2)
        self.insects = [0,0,0]
        self.imago_emergence_active = False

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d insects on %s's discard 1" %
                       (self.insects[1], self.opponent))
        report.append ("%d insects on %s's discard 2" %
                       (self.insects[2], self.opponent))
        if self.imago_emergence_active:
            report.append ("Imago Emergence is active")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        i1 = int(lines[0].split(' ')[0])
        i2 = int(lines[1].split(' ')[0])
        self.insects = [0,i1,i2]
        self.imago_emergence_active = find_start(lines,
                                                "Imago Emergence is active")

    def initial_save (self):
        state = Character.initial_save (self)
        state.insects = self.insects[:]
        state.imago_emergence_active = self.imago_emergence_active
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.insects = state.insects[:]
        self.imago_emergence_active = state.imago_emergence_active

    def reset (self):
        self.stinging_ignores = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.stinging_ignores = self.stinging_ignores
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.stinging_ignores = state.stinging_ignores

    def give_insects (self, n, pile=0):
        if n > 0:
            self.insects[pile] += n
            if self.game.reporting:
                report = "Kajia places %d insect token%s on %s's %s" % \
                         (n,
                          ('s' if n>1 else ''),
                          self.opponent,
                          ('attack pair' if pile==0 else 'discard pile %d' % pile))
                self.game.report (report)
                self.game.report ("(Current insects: %d / %d / %d)" %
                                  tuple(self.insects))

    def infested_piles (self):
        return (self.insects[1] > 0) + (self.insects[2] > 0)
    def total_insects (self):
        return self.insects[1] + self.insects[2]

    # Give counter when opponent moves and switches sides.
    def movement_reaction (self, initiator, mover, old_position, direct):
        Character.movement_reaction(self, initiator, mover, old_position, direct)
        if (mover is self.opponent and
            ordered(self.opponent.position, self.position, old_position)):
            self.give_insects(1)

    def reduce_soak (self, soak):
        return 0 if self.stinging_ignores > 0 else Character.reduce_soak(self, soak)
    def reduce_stunguard (self, stunguard):
        return 0 if self.stinging_ignores > 1 else Character.reduce_stunguard(self, stunguard)

    # Cycle insects with opponents cards
    def cycle (self):
        Character.cycle(self)
        if isinstance(self.opponent.style, SpecialAction):
            # If opponent's cards don't cycle, neither do insects
            # However, new insects are added to discard 1
            self.insects[1] += self.insects[0]
            self.insects[0] = 0
        else:
            self.opponent.lose_life (self.insects[2])
            self.insects[2] = self.insects[1]
            self.insects[1] = self.insects[0]
            self.insects[0] = 0

    def unique_ability_end_trigger (self):
        if self.imago_emergence_active:
            self.give_insects(1)

    # This just evaluates projected life loss.
    # Style tricks are evaluated by style evaluation bonuses.
    def evaluate (self):
        value = Character.evaluate(self)
        insects = self.insects[1]+self.insects[2]
        value += min (insects, self.opponent.life - 1)
        if self.imago_emergence_active:
            value += 3
        return value


class Kallistar (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Spellbolt (the_game,self)
        self.styles = [Flare    (the_game,self), \
                       Caustic  (the_game,self), \
                       Volcanic (the_game,self), \
                       Ignition (the_game,self), \
                       Blazing  (the_game,self)  ]
        self.finishers = [Supernova          (the_game, self),
                           ChainOfDestruction (the_game, self)]
        self.status_effects = [PriorityPenaltyStatusEffect(the_game, self)]
        self.forms = [HumanForm     (the_game, self),
                      ElementalForm (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 1 # Elemental

    def all_cards(self):
        return Character.all_cards(self) + self.forms

    def set_active_cards(self):
        Character.set_active_cards(self)
        self.active_cards.append(self.form)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.form = self.human_form

    def choose_initial_discards (self):
        return (self.flare, self.strike,
                self.caustic, self.drive)

    def situation_report (self):
        report = Character.situation_report (self)
        report.append(self.form.name)
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.form = (self.elemental_form
                     if find_start (lines, 'Elemental')
                     else self.human_form)

    def initial_save (self):
        state = Character.initial_save (self)
        state.form = self.form
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.form = state.form

    def ante_trigger (self):
        Character.ante_trigger(self)
        self.form.ante_trigger()

    # assumes I'll change to elemental next beat,
    # so value of being elemental is value for one beat,
    # plus value of not having to ignite
    def evaluate (self):
        return Character.evaluate (self) + self.form.get_value()


class Karin (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Claw (the_game,self)
        self.styles = [Howling     (the_game,self), \
                       Coordinated (the_game,self), \
                       FullMoon    (the_game,self), \
                       Feral       (the_game,self), \
                       Dual        (the_game,self)  ]
        self.finishers = [RedMoonRage (the_game, self),
                           LunarCross  (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        # give bases different preferred ranges for jager
        # (because jager isn't moved by base)
        self.unique_base.jager_preferred_range = 1.5
        self.strike.jager_preferred_range = 1
        self.shot.jager_preferred_range = 2.5
        self.drive.jager_preferred_range = 1
        self.burst.jager_preferred_range = 2.5
        self.grasp.jager_preferred_range = 1
        self.dash.jager_preferred_range = 2
        self.counter_base.jager_preferred_range = 1
        self.wave_base.jager_preferred_range = 3
        self.force.jager_preferred_range = 1
        self.spike.jager_preferred_range = 2
        self.throw.jager_preferred_range = 1
        self.parry.jager_preferred_range = 2
        self.stagger.jager_preferred_range = 1
        self.leap.jager_preferred_range = 1
        self.lunge.jager_preferred_range = 1.5
        self.sweep.jager_preferred_range = 1.5
        self.catch.jager_preferred_range = 1
        self.block.jager_preferred_range = 2

    def choose_initial_discards (self):
        return (self.dual, self.grasp,
                self.howling, self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        # Jager starts in front of Karin.
        # With new starting position (2 or 4), this always means the center.
        self.jager_position = 3

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.jager_position = addendum[0].find('j')

    def initial_save (self):
        state = Character.initial_save (self)
        state.jager_position = self.jager_position
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.jager_position = state.jager_position

    def reset (self):
        Character.reset (self)
        self.lunar_swap = False

    def full_save (self):
        state = Character.full_save (self)
        state.lunar_swap = self.lunar_swap
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.lunar_swap = state.lunar_swap

    board_addendum_lines = 1
    def get_board_addendum (self):
        addendum = ['.'] * 7
        addendum [self.jager_position] = 'j'
        return ''.join(addendum)

    def end_trigger (self):
        Character.end_trigger (self)
        if not self.opponent.blocks_end_triggers() and not self.is_stunned():
            self.move_jager ([-1,0,1])

    # given moves are relative
    def move_jager (self, moves):
        positions = [pos for pos in xrange(7)
                     if pos - self.jager_position in moves]
        old_pos = self.jager_position
        prompt = "Move Jager:"
        options = []
        if self.is_user and self.game.interactive_mode:
            for pos in positions:
                self.jager_position = pos
                options.append (self.get_board_addendum())
            self.jager_position = old_pos
        self.jager_position = positions [self.game.make_fork \
                            (len(positions), self, prompt, options)]
        if self.game.reporting and self.jager_position != old_pos:
            self.game.report ("Jager moves:")
            for s in self.game.get_board():
                self.game.report (s)

    def attack_range (self):
        attack_source = (self.jager_position
                         if self.howling in self.active_cards else
                         self.position)
        return abs (self.opponent.position - attack_source)

    # different preferred range for karin and jager, based on the styles
    # that let each of them attack
    def set_preferred_range (self):
        unavailable_cards = self.discard[1] \
                          | self.discard[2]
        styles = set(self.styles) - unavailable_cards
        bases = set(self.bases) - unavailable_cards
        karin_styles = [s for s in styles if s.karin_attack]
        jager_styles = [s for s in styles if s.jager_attack]
        self.karin_weight = len (karin_styles)
        self.jager_weight = len (jager_styles)
        base_range = sum (b.get_preferred_range() for b in bases) / len(bases)
        base_range_jager = sum (b.jager_preferred_range for b in bases) / len(bases)
        self.preferred_range = (sum (s.get_preferred_range()
                                      for s in karin_styles) /
                       self.karin_weight) \
                    + base_range
        self.jager_range = (sum (s.get_preferred_range()
                                  for s in jager_styles) /
                       self.jager_weight) \
                    + base_range_jager \
                    if self.jager_weight>0 else 0

    # average of karin and jager's range penalty
    # (weighted by number of styles that let each of them attack)
    def evaluate_range (self):
        karin_penalty = self.karin_weight * \
                        (self.preferred_range - self.game.distance()) ** 2
        jager_penalty = self.jager_weight * \
                        (self.jager_range - (abs(self.jager_position -
                                                 self.opponent.position))) ** 2
        return - self.game.range_weight * (karin_penalty + jager_penalty) / \
                 (self.karin_weight + self.jager_weight)

    def evaluate (self):
        # we want jager as close as possible to opponent (for most style effects),
        # and as far as possible from karin (to give range options)
        return (Character.evaluate(self)
                + 0.1 * abs(self.position-self.jager_position)
                - 0.2 * abs(self.opponent.position-self.jager_position))

class Kehrolyn (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Overload (the_game,self)
        self.styles = [Mutating    (the_game, self), \
                       Bladed      (the_game, self), \
                       Whip        (the_game, self), \
                       Quicksilver (the_game,self), \
                       Exoskeletal (the_game,self)  ]
        self.finishers = [HydraFork (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def set_mean_priority(self):
        self.mean_priority = (self.mean_priority_bonus +
            2 * sum (s.mean_priority for s in self.styles) / float(len(self.styles)) +
            sum (b.mean_priority for b in self.bases) / float(len(self.bases)))

    def choose_initial_discards (self):
        return (self.mutating, self.grasp,
                self.bladed, self.strike)

    def reset (self):
        Character.reset (self)
        self.overloaded_style = None
        self.mutated_style = None

    def full_save (self):
        state = Character.full_save (self)
        state.overloaded_style = self.overloaded_style
        state.mutated_style = self.mutated_style
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.overloaded_style = state.overloaded_style
        self.mutated_style = state.mutated_style

    # current form is active in addition to normal style and base
    def set_active_cards (self):
        Character.set_active_cards (self)
        # find current form
        # TODO: If there are multiple styles in discard 1 (like after
        # a cancel), the latest one should be used.
        # However, discard is currently implemented as a set, so there's
        # no order to it, and we'll get a random (but consistent) card.
        for card in self.discard[1]:
            if isinstance (card, Style) and card is not self.special_action:
                self.current_form = card
                break
        else:
            self.current_form = self.null_style
        self.active_cards.append (self.current_form)
        if self.overloaded_style:
            self.active_cards.append(self.overloaded_style)
        if self.mutating in self.active_cards and self.mutated_style:
            self.active_cards.append(self.mutated_style)

    # if Whip is current form, add Whip bonus (see under Whip)
    # if Mutating is current form, Whip doubling handled by Whip
    def set_preferred_range (self):
        Character.set_preferred_range(self)
        self.preferred_range += (self.whip.preferred_range \
                                 if self.whip in self.discard[1] else 0)

class Khadath (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Snare (the_game,self)
        self.styles = [Hunters    (the_game, self), \
                       Teleport   (the_game, self), \
                       Evacuation (the_game, self), \
                       Blight     (the_game,self), \
                       Lure       (the_game,self)  ]
        self.finishers = [DimensionalExile (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.6 # Trap penalizes opponent.

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.trap_position = None

    def choose_initial_discards (self):
        #(according to AI, Evacuation and Lure)
        return (self.evacuation, self.grasp,
                self.hunters, self.unique_base)

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.trap_position = addendum[0].find('t')
        if self.trap_position == -1:
            self.trap_position = None

    board_addendum_lines = 1
    def get_board_addendum (self):
        if self.trap_position == None:
            return ''
        addendum = ['.'] * 7
        addendum [self.trap_position] = 't'
        return ''.join(addendum)

    def initial_save (self):
        state = Character.initial_save (self)
        state.trap_position = self.trap_position
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.trap_position = state.trap_position

    # Non-direct moves can't enter trap and come out on other side.
    # This doesn't affect me when I'm pulling with Lure.
    def blocks_opponent_movement (self, initiator, direct):
        if (direct or initiator is self or
            self.trap_position == self.position or
            self.trap_position is None):
            return set()
        return set ([pos for pos in xrange(7) \
                     if ordered (self.opponent.position,
                                 self.trap_position,
                                 pos)])

    def reveal_trigger (self):
        Character.reveal_trigger(self)
        if self.trap_position == None:
            return
        trap_distance = abs (self.trap_position - self.opponent.position)
        if trap_distance == 0:
            self.opponent.add_triggered_priority_bonus(-3)
        elif trap_distance == 1:
            self.opponent.add_triggered_priority_bonus(-1)

    def move_trap (self, positions):
        if isinstance (self.base, Snare):
            return
        positions = list (positions)
        prompt = "Move trap:"
        options = []
        if self.is_user and self.game.interactive_mode:
            trap_pos = self.trap_position
            for pos in positions:
                self.trap_position = pos
                options.append (self.get_board_addendum())
            self.trap_position = trap_pos
        assert positions
        self.trap_position = positions [self.game.make_fork
                                    (len(positions), self, prompt, options)]
        if self.game.reporting:
            self.game.report ("Khadath moves trap:")
            for s in self.game.get_board():
                self.game.report (s)

    def evaluate (self):
        trap = 100 if self.trap_position==None else self.trap_position
        opponent_trap_distance = abs (trap - self.opponent.position)
        result = Character.evaluate (self)
        # bonus for priority next beat
        if opponent_trap_distance == 1:
            result += 0.5
        elif opponent_trap_distance == 0:
            result += 1.5
        return result


class Larimore(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Might (the_game,self)
        self.styles = [Dragonbreath (the_game, self),
                       Blistering   (the_game, self),
                       Warmages     (the_game, self),
                       Dire         (the_game, self),
                      Incinerating (the_game, self)  ]
        self.finishers = [Dragonflare (the_game, self),
                          TheJump     (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.counter_name = 'Firepower'
        self.max_counters = 5

    mean_priority_bonus = -0.5

    def choose_initial_discards (self):
        return (self.incinerating, self.might,
                self.blistering, self.burst)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.counters = 2

    def reset (self):
        Character.reset (self)
        self.blistering_range = False
        self.blistering_ignores = False

    def full_save (self):
        state = Character.full_save (self)
        state.blistering_range = self.blistering_range
        state.blistering_ignores = self.blistering_ignores
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.blistering_range = state.blistering_range
        self.blistering_ignores = state.blistering_ignores

    def get_antes(self):
        return range(4)

    def input_ante(self):
        print "Number of Firepower counters to recharge [0-3]: "
        return input_number(4)

    def ante_trigger(self):
        recharge = self.strat[2][0]
        if recharge == 1:
            self.add_triggered_priority_bonus(-1)
        elif recharge == 2:
            self.add_triggered_power_bonus(-1)
            self.add_triggered_priority_bonus(-3)
        elif recharge == 3:
            self.add_triggered_power_bonus(-2)
            self.add_triggered_priority_bonus(-5)
            if self.game.reporting:
                self.game.report("Larimore's ignores movement on his style and base this beat.")

    def unique_ability_end_trigger(self):
        self.recover_counters(self.strat[2][0])

    def get_ante_name (self, a):
        if a == 0:
            return ""
        return "charge: %d" % a

    # Ignore own movement effects if charging 3.
    def execute_move(self, mover, moves, direct=False, max_move=False, specific_movement_reaction=None):
        if self.strat[2][0] == 3:
            return
        Character.execute_move(self, mover, moves, direct, max_move, specific_movement_reaction)

    def get_maxrange_bonus(self):
        return 2 * self.blistering_range

    def reduce_soak(self, soak):
        return (0 if self.blistering_ignores else Character.reduce_soak(self, soak))

    def reduce_stunguard(self, stunguard):
        return (0 if self.blistering_ignores else Character.reduce_stunguard(self, stunguard))

    def evaluate(self):
        return Character.evaluate(self) + 0.7 * self.counters


class Lesandra(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Summons (the_game,self)
        self.styles = [Invocation  (the_game, self),
                       Pactbreaker (the_game, self),
                       Binding     (the_game, self),
                       Guardian    (the_game, self),
                       Window      (the_game, self)  ]
        self.finishers = [InvokeDuststalker (the_game, self),
                          Mazzaroth         (the_game, self)]
        self.status_effects = [OpponentEliminatedStatusEffect(the_game, self)]
        self.familiars = [BorneoFamiliar (the_game, self),
                          Wyvern         (the_game, self),
                          Salamander     (the_game, self),
                          RuneKnight     (the_game, self),
                          RavenKnight    (the_game, self)  ]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.4 # Borneo active and Raven Knight anted.

    def all_cards(self):
        return Character.all_cards(self) + self.familiars

    def choose_initial_discards (self):
        return (self.guardian, self.unique_base,
                self.pactbreaker, self.grasp)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.active_familiar = self.borneo

    def situation_report (self):
        report = Character.situation_report (self)
        report.append("Familiar: %s" % self.active_familiar)
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        line = find_start_line(lines, 'Familiar:')
        self.active_familiar = None
        for familiar in self.familiars:
            if familiar.name in line:
                self.active_familiar = familiar
                break

    def initial_save (self):
        state = Character.initial_save (self)
        state.active_familiar = self.active_familiar
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.active_familiar = state.active_familiar

    def reset (self):
        Character.reset (self)
        self.anted_familiar = None
        self.invocation_soak = 0

    def full_save (self):
        state = Character.full_save (self)
        state.anted_familiar = self.anted_familiar
        state.invocation_soak = self.invocation_soak
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.anted_familiar = state.anted_familiar
        self.invocation_soak = state.invocation_soak

    def get_antes(self):
        if self.active_familiar:
            return [None, self.active_familiar]
        else:
            return [None]

    def input_ante(self, pair):
        if self.active_familiar:
            print "Ante %s?" % self.active_familiar
            if menu(['No', 'Yes']):
                return self.active_familiar
        return None

    def ante_trigger(self):
        ante = self.strat[2][0]
        if ante:
            assert ante is self.active_familiar
            self.anted_familiar = ante
            self.active_familiar = None

    def get_ante_name(self, a):
        return a.name if a else ''

    def set_active_cards(self):
        Character.set_active_cards(self)
        if self.active_familiar:
            self.active_cards.append(self.active_familiar)
        if self.anted_familiar and not self.opponent.blocks_tokens():
            self.active_cards.append(self.anted_familiar)

    def end_trigger(self):
        Character.end_trigger(self)
        unavailable = self.active_familiar or self.anted_familiar
        available_familiars = [f for f in self.familiars
                               if f is not unavailable]
        prompt = "Choose familiar to summon:"
        damage = self.opponent.damage_taken
        if self.game.interactive_mode and self.is_user:
            options = ["%s (cost: %d life)" % (f.name,
                         f.get_cost() if f.get_cost() > damage else 0)
                       for f in available_familiars]
            options.append('None')
        else:
            options = []
        ans = self.game.make_fork(len(available_familiars) + 1,
                                  self, prompt, options)
        if ans < len(available_familiars):
            self.active_familiar = available_familiars[ans]
            if self.game.reporting:
                self.game.report("Lesandra summons her %s" % self.active_familiar)
            cost = self.active_familiar.get_cost()
            if cost > damage:
                self.lose_life(cost)

    def unique_ability_end_trigger(self):
        if self.opponent_eliminated_status_effect in self.active_status_effects:
            raise WinException(self.my_number)

    def expected_soak(self):
        ret = Character.expected_soak(self)
        if self.active_familiar is self.rune_knight:
            ret += 1
        return ret

    def expected_stunguard(self):
        ret = Character.expected_stunguard(self)
        if self.active_familiar is self.rune_knight:
            ret += 1.5
        return ret

    def take_damage_trigger(self, damage):
        if self.invocation in self.active_cards:
            self.add_triggered_power_bonus(-damage)

    def get_soak(self):
        return self.invocation_soak + Character.get_soak(self)

    def evaluate(self):
        ret = Character.evaluate(self)
        if self.active_familiar is not None:
            ret += self.active_familiar.get_value()
        return ret

class Lixis (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Lance (the_game,self)
        self.styles = [Venomous     (the_game, self), \
                       Rooted       (the_game, self), \
                       Naturalizing (the_game, self), \
                       Vine         (the_game, self), \
                       Pruning      (the_game, self)  ]
        self.finishers = [VirulentMiasma (the_game, self)]
        self.status_effects = [PriorityPenaltyStatusEffect(the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.virulent_miasma_active = False

    def choose_initial_discards (self):
        return (self.pruning, self.grasp,
                self.rooted, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        if self.virulent_miasma_active:
            report.append ("Virulent Miasma is active")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.virulent_miasma_active = find_start (lines, 'Virulent Miasma')

    def initial_save (self):
        state = Character.initial_save (self)
        state.virulent_miasma_active = self.virulent_miasma_active
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.virulent_miasma_active = state.virulent_miasma_active

    def hit_trigger (self):
        Character.hit_trigger (self)
        self.discard_opponent_base()

    def discard_opponent_base(self):
        bases = list (set (self.opponent.bases) - self.opponent.discard[0] \
                                                - self.opponent.discard[1] \
                                                - self.opponent.discard[2])
        prompt = "Choose an extra base to discard"
        options = [b.name for b in bases]
        discard = bases [self.game.make_fork (len(bases), self.opponent,
                                              prompt, options)]
        self.opponent.discard[0] |= set ([discard])
        if self.game.reporting:
            self.game.report (self.opponent.name + " discards " + discard.name)

    def give_priority_penalty (self):
        penalty = Character.give_priority_penalty(self)
        # If Virulent Miasma is active and opponent started beat with 1 life,
        # she gets a priority penalty
        if self.virulent_miasma_active and \
           self.game.initial_state.player_states[1-self.my_number].life == 1:
            penalty -= 3
        return penalty

    def may_ignore_own_movement(self):
        return self.rooted in self.active_cards

    def ante_trigger (self):
        if self.virulent_miasma_active:
            self.opponent.lose_life (3)

    # for performance, the following methods just return False by default
    # so Lixis needs to override them

    # block opponent's stats from increasing above style+base
    def blocks_priority_bonuses (self):
        return self.naturalizing in self.active_cards
    def blocks_power_bonuses (self):
        return self.naturalizing in self.active_cards
    def blocks_minrange_bonuses (self):
        return self.naturalizing in self.active_cards
    def blocks_maxrange_bonuses (self):
        return self.naturalizing in self.active_cards
    # cannot refer to self.active_cards, because it is used in making it
    def blocks_tokens (self):
        return self.style is self.naturalizing

    def evaluate(self):
        return Character.evaluate(self) + 6 * self.virulent_miasma_active


class Luc (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Flash (the_game, self)
        self.styles = [Eternal  (the_game, self), \
                       Memento  (the_game, self), \
                       Fusion   (the_game, self), \
                       Feinting (the_game, self), \
                       Chrono   (the_game, self)  ]
        self.finishers = [TemporalRecursion (the_game, self)]
        self.tokens = [Time(the_game, self)]
        # Virtual cards help implement ante effects.
        self.virtual_cards = [Ante1Effect(the_game, self),
                              Ante3Effect(the_game, self),
                              Ante5Effect(the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        # Override the name override...
        self.ante_3_effect = self.virtual_cards[1]
        self.max_tokens = 5

    mean_priority_bonus = 1 # Tokens (1 for priority, 5 to becomea active).

    def all_cards(self):
        return Character.all_cards(self) + self.virtual_cards

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 2 * [self.time]

    def choose_initial_discards (self):
        return (self.eternal, self.flash,
                self.fusion, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Time token%s" % (len(self.pool),
                       '' if len(self.pool)==1 else 's'))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.time] * int(lines[0][0])

    def reset (self):
        self.eternal_soak = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.eternal_soak = self.eternal_soak
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.eternal_soak = state.eternal_soak

    def get_antes (self):
        p = len(self.pool)
        return [a for a in [0,1,3,5] if a <= p]

    def input_ante (self, pair):
        p = len(self.pool)
        if p > 0 :
            possible_antes = [a for a in [0,1,3,5] if a <= p]
            print "Number of Time tokens [" + \
                            ','.join([str(a) for a in possible_antes]) + ']: '
            while True:
                ante = input_number (len(self.pool)+1)
                if ante in possible_antes:
                    break
            return ante
        else:
            return 0

    def get_ante_name (self, a):
        if a == 0:
            return ""
        if a == 1:
            return "1 token"
        return str(a) + " tokens"

    def set_active_cards(self):
        Character.set_active_cards(self)
        n_tokens = self.active_cards.count(self.time)
        self.active_cards = [card for card in self.active_cards
                             if card is not self.time]
        if n_tokens == 1:
            self.active_cards.append(self.ante_1_effect)
        elif n_tokens == 3:
            self.active_cards.append(self.ante_3_effect)
        elif n_tokens == 5:
            self.active_cards.append(self.ante_5_effect)

    def ante_trigger (self):
        for unused in range(self.strat[2][0]):
            self.ante_token(self.time)

    def unique_ability_end_trigger (self):
        self.recover_tokens (1)

    def get_soak (self):
        return self.eternal_soak + Character.get_soak(self)

    # Changed on 28/8/14.  Previous logs use 0.2 per token.
    def evaluate (self):
        return Character.evaluate (self) + 0.15 * len(self.pool)


class Lymn (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Visions (the_game,self)
        self.styles = [Maddening  (the_game, self), \
                       Chimeric   (the_game, self), \
                       Surreal    (the_game, self), \
                       Reverie    (the_game,self), \
                       Fathomless (the_game,self)  ]
        self.finishers = [Megrim  (the_game, self),
                           Conceit (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def choose_initial_discards (self):
        return (self.maddening, self.burst,
                self.chimeric, self.dash)

    def reset (self):
        self.disparity = None
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.disparity = self.disparity
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.disparity = state.disparity

    # Setting Disparity at Start of Beat.
    # It's not needed earlier, and hopefully priorities haven't changed
    # in the interim.
    def start_trigger (self):
        self.disparity = abs (self.get_priority() - self.opponent.get_priority())
        if self.game.reporting:
            self.game.report ("Disparity is %d" % self.disparity)
        Character.start_trigger (self)


class Magdelina (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Blessing (the_game,self)
        self.styles = [Safety        (the_game, self),
                       Priestess     (the_game, self),
                       Spiritual     (the_game,self),
                       Sanctimonious (the_game, self),
                       Excelsius     (the_game,self)]
        self.finishers = [SolarSoul  (the_game, self),
                          Apotheosis (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    # This assumes steady gain (no Blessing or Spiritual) for 15 beats.
    # It might be excessive, but at any point beyond the start of the
    # game we're past the low priority beats, with only high priority
    # to look forward to.
    # TODO: mean priority changes during game.
    mean_priority_bonus = 2.7

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.level = 0
        self.trance = 0

    def choose_initial_discards (self):
        return (self.spiritual, self.shot,
                self.sanctimonious, self.blessing)

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("Level: %d" %self.level)
        report.append ("Trance: %d" %self.trance)
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.level = int(lines[0][-2])
        self.trance = int(lines[1][-2])

    def initial_save (self):
        state = Character.initial_save (self)
        state.level = self.level
        state.trance = self.trance
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.level = state.level
        self.trance = state.trance

    def full_save (self):
        state = Character.full_save (self)
        state.initial_level = self.initial_level
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.initial_level = state.initial_level

    def ante_trigger(self):
        Character.ante_trigger(self)
        # Remember current level, so that later level changes don't
        # change the bonus.
        self.initial_level = self.level
        if self.level != 0 and self.game.reporting:
            self.game.report ("Magdelina gets a +%d bonus to power, priority and stunguard" % self.level)

    def get_level_bonus(self):
        return self.initial_level

    def get_priority_bonus (self):
        return self.get_level_bonus() + Character.get_priority_bonus(self)
    def get_power_bonus (self):
        return self.get_level_bonus() + Character.get_power_bonus(self)
    def get_stunguard (self):
        return self.get_level_bonus() + Character.get_stunguard(self)

    def expected_stunguard(self):
        return Character.expected_stunguard(self) + self.level

    def unique_ability_end_trigger (self):
        # Spiritual only gains a counter if it hit.
        if self.spiritual in self.active_cards and not self.hits_scored:
            return
        if self.trance > self.level:
            self.trance = 0
            if self.game.reporting:
                self.game.report("Magdelina discards all Trance counters")
            self.gain_level()
        else:
            self.gain_trance()

    def gain_trance(self):
        if self.trance < 5:
            self.trance += 1
            if self.game.reporting:
                self.game.report ("Magdelina gains a Trance counter: has %d" % self.trance)

    def gain_level(self):
        if self.level < 5:
            self.level += 1
            if self.game.reporting:
                self.game.report ("Magdelina gains a Level: has %d" % self.level)

    def discard_trance(self, n):
        old = self.trance
        self.trance = max(0, self.trance - n)
        loss = old - self.trance
        if self.game.reporting:
            self.game.report("%s discards %d Trance Counter%s (now has %d)" %
                (self.name, loss, '' if loss == 1 else 's', self.trance))
        return loss

    # only half value to dealing damage
    def evaluate (self):
        # One level (for one beat) is worth 0.3+0.3+0.1 = 0.7 points.
        # However, empirically Magdelina does better at 1.0 points.
        # Each trance token is worth 1.0 points for each level
        # you expect to go up (it hastens the level by one beat).
        return (Character.evaluate(self)
                + self.opponent.effective_life() * 0.7 +
                + 1.0 * (self.level_values[self.level] +
                         self.trance * self.trance_values_per_level[self.level]))

    # importance of going up a level is 15,12,9,6,3
    level_values = [0,6,12,16,18.5,18.5]
    trance_values_per_level = [3, 2, 1, 0.5, 0, 0]

    # TODO: her preferred range should probably be more about not being at the
    # opponent's preferred range (at least at low levels)

class Marmelee (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Meditation (the_game, self)
        self.styles = [Petrifying  (the_game, self), \
                       Magnificent (the_game, self), \
                       Sorceress   (the_game, self), \
                       Barrier     (the_game, self), \
                       Nullifying  (the_game, self)  ]
        self.finishers = [AstralCannon (the_game, self),
                           AstralTrance (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.counter_name = 'Concentration'
        self.max_counters = 5

    def choose_initial_discards (self):
        return (self.petrifying, self.burst,
                self.barrier, self.grasp)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.counters = 2

    def reset (self):
        self.sorceress_range = 0
        self.meditation_soak = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.sorceress_range = self.sorceress_range
        state.meditation_soak = self.meditation_soak
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.sorceress_range = state.sorceress_range
        self.meditation_soak = state.meditation_soak

    def cycle(self):
        Character.cycle(self)
        self.recover_counters (1)

    # lose all counters when stunned
    def stun (self, damage=None):
        stunned = Character.stun (self, damage)
        if stunned:
            if self.game.reporting and self.counters:
                self.game.report ("Marmelee loses all Concentration counters")
            self.counters = 0
        return stunned

    def get_maxrange_bonus (self):
        return 2 if self.sorceress_range else 0

    def get_soak (self):
        return self.meditation_soak + Character.get_soak(self)

    def evaluate (self):
        return Character.evaluate (self) + 0.85 * self.counters

class Mikhail (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Scroll (the_game, self)
        self.styles = [Immutable    (the_game, self), \
                       Transcendent (the_game, self), \
                       Hallowed     (the_game, self), \
                       Apocalyptic  (the_game, self), \
                       Sacred       (the_game, self)  ]
        self.finishers = [MagnusMalleus (the_game, self),
                           TheFourthSeal (the_game, self)]
        self.tokens = [Seal (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.max_tokens = 3

    def set_mean_priority(self):
        self.mean_priority = (self.mean_priority_bonus +
            0.7 * sum (s.mean_priority for s in self.styles) / float(len(self.styles)) +
            sum (b.mean_priority for b in self.bases) / float(len(self.bases)))

    def choose_initial_discards (self):
        return (self.hallowed, self.grasp,
                self.apocalyptic, self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 3 * [self.seal]

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Seal token%s" % (len(self.pool),
                       '' if len(self.pool)==1 else 's'))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.seal] * int(lines[0][0])

    def reset (self):
        self.immutable_soak = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.immutable_soak = self.immutable_soak
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.immutable_soak = state.immutable_soak

    def get_antes (self):
        if self.pool:
            return [0,1]
        else:
            return [0]

    def input_ante (self, pair):
        if self.pool:
            print "Ante Seal token?"
            return menu (["No", "Yes"])
        else:
            return 0

    def ante_trigger (self):
        if self.strat[2][0]:
            self.ante_token(self.seal)

    def get_ante_name (self, a):
        return "Seal" if a else ""

    # Style is inactive when no token anted.
    # This does not affect the SpecialAction "style".
    def set_active_cards (self):
        Character.set_active_cards (self)
        if not (self.seal in self.ante or
                isinstance (self.style, SpecialAction)):
            self.active_cards.remove (self.style)

    # Mikhail's token doesn't have ante effects, so it's not checked for them
    # (for efficiency)
    def get_active_tokens (self):
        tokens = Character.get_active_tokens (self)
        # Induced tokens are still checked normally.
        return [t for t in tokens if t not in self.tokens]

    def unique_ability_end_trigger (self):
        if self.damage_taken:
            self.recover_tokens (1)

    def recover_tokens (self, n):
        if self.apocalyptic not in self.active_cards:
            Character.recover_tokens (self, n)

    def get_soak (self):
        return self.immutable_soak + Character.get_soak(self)

    # 0,4,6,8 didn't go well - he steady-stated at 2 tokens, instead of 1.6,
    # and lost more games.
    token_values = (0,3,4,5)
    def evaluate (self):
        return Character.evaluate(self) + self.token_values[len(self.pool)]


class Oriana (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Meteor (the_game, self)
        self.styles = [Celestial   (the_game, self), \
                       Stellar     (the_game, self), \
                       Unstable    (the_game, self), \
                       Metamagical (the_game, self), \
                       Calamity    (the_game, self)  ]
        self.finishers = [NihilEraser   (the_game, self),
                           GalaxyConduit (the_game, self)]
        self.tokens = [MagicPoint (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.mp = self.magic_point # Convenient abbreviation.
        self.max_tokens = 10

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 6 * [self.mp]

    def choose_initial_discards (self):
        return (self.stellar, self.grasp,
                self.calamity, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Magic Point token%s" % (len(self.pool),
                       '' if len(self.pool)==1 else 's'))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.mp] * int(lines[0].split()[0])

    def get_antes (self):
        return range (len(self.pool) + 1)

    def input_ante (self, pair):
        n = len(self.pool)
        if n > 0 :
            print "Number of Magic Point tokens [0-%d]: " %n
            return input_number (n+1)
        else:
            return 0

    def ante_trigger (self):
        for unused in range(self.strat[2][0]):
            self.ante_token(self.mp)

    def get_ante_name (self, a):
        if a == 0:
            return ""
        if a == 1:
            return "1 token"
        return str(a) + " tokens"

    # oriana's tokens don't have ante effects, so they're not checked for them
    # (for efficiency)
    def get_active_tokens (self):
        tokens = Character.get_active_tokens (self)
        # Induced tokens are still checked normally.
        return [t for t in tokens if t not in self.tokens]

    # convert damage to life loss when using Celestial
    def take_damage (self, damage, for_real=False):
        if self.celestial in self.active_cards:
            self.lose_life (damage)
        else:
            Character.take_damage(self, damage)

    # regain tokens (and maybe don't lose life)  when using Celestial
    def lose_life (self, loss):
        if self.celestial in self.active_cards:
            if self.ante.count(self.mp) < 7:
                Character.lose_life (self, loss)
            self.recover_tokens (loss)
        else:
            Character.lose_life (self, loss)

    def evaluate (self):
        return Character.evaluate (self) + 0.3 * len (self.pool)


class Ottavia (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Shooter  (the_game, self)
        self.styles = [Snapback (the_game, self), \
                       Demolition (the_game, self), \
                       AntiPersonnel (the_game, self), \
                       Cover (the_game, self), \
                       Cybernetic (the_game, self)  ]
        self.finishers = [ExtremePrejudice (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 1.5 # Target lock.

    def choose_initial_discards (self):
        return (self.cybernetic, self.strike,
                self.demolition, self.shot)

    def reset (self):
        self.opponent_priority_pre_penalty = None
        self.target_lock = None
        self.cybernetic_soak = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.opponent_priority_pre_penalty = self.opponent_priority_pre_penalty
        state.target_lock = self.target_lock
        state.cybernetic_soak = self.cybernetic_soak
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.opponent_priority_pre_penalty = state.opponent_priority_pre_penalty
        self.target_lock = state.target_lock
        self.cybernetic_soak = state.cybernetic_soak

    # These aren't really possible antes, but rather possible ante
    # results (correct or incorret guess).
    # The actual strategies will be formed in post-processing.
    def get_antes (self):
        return [False, True]

    def input_ante (self, pair):
        priorities = self.all_opponent_priorities
        print "Guess %s's priority for Target Lock:" % self.opponent
        print "Possible priorities:", priorities
        mn = min(priorities)
        mx = max(priorities)
        while True:
            ans = input_number(mx + 1, mn)
            if ans in priorities:
                return ans

    # Handles both simulation strategies (correct/incorrect guess)
    # and post-processing strategies (actual priority guessed).
    def get_ante_name (self, a):
        # post processing - guess is a number
        if isinstance (a, int):
            return "Target Lock: %d" % a
        # simulation, guess is correct or not
        else:
            assert isinstance(a, bool)
            if a:
                return "Target Acquired"
            else:
                return "Target Missed"

    # Check whether target is locked
    # Handles both boolean guess results and actual priority guesses.
    def give_priority_penalty(self):
        # Make sure we only make the check the first time priority is
        # needed each beat (after the reveal).  Later checks
        # just refer to the previous result.
        if self.target_lock is None:
            self.opponent_priority_pre_penalty = self.opponent.get_priority_pre_penalty()
            guess = self.strat[2][0]
            self.target_lock = (guess is True or
                                guess is self.opponent_priority_pre_penalty)
            if self.game.reporting and self.target_lock:
                self.game.report ("Ottavia acquires target lock")
        return -10 if self.target_lock else 0

    # Ottavia's simulated strategies don't have a real target lock, but
    # rather an assumption about whether the guess was correct or not
    # (which is all that is needed for simulation).
    # After simulation, some post-processing is required to get the real
    # list of strategies and table of results.
    def post_simulation_processing (self):
        # Obtain from results a list of all possible pre-penalty
        # priorities for opponent.
        priorities = [result[1].player_states[self.my_number].opponent_priority_pre_penalty
                      for row in self.game.results for result in row]
        priorities = sorted(list(set(priorities)))
        # When any player Cancels/Pulses, priority is None
        if priorities[0] is None:
            priorities.pop(0)

        fake_strats = self.strats
        opp_strats = self.opponent.strats
        fake_results = self.game.results

        # For each pair of Ottavia's strategies (same except for guess
        # correctness), pick one (and record its index).
        strats_without_guesses = [(s,i) for i,s in enumerate(fake_strats)
                                  if s[2][0] is False]
        # Then, make a new strategy list, with one guess for each
        # possible opponent priority (keeping the index of the original
        # fake strat).
        full_strats = [((s[0][0], s[0][1], (p, s[0][2][1], None)), s[1])
                       for s in strats_without_guesses for p in priorities]

        n = len(full_strats)
        m = len(opp_strats)
        # for each strat combination, take the simulation result
        # corresponding to those pairs, but pick correct/incorrect guess
        # based on whether Ottavia's guess actually corresponds to
        # opponent's priority.
        if self.my_number==0:
            full_results = [[0]*m for unused in xrange(n)]
            for i in xrange(n):
                for j in xrange(m):
                    my_strat = full_strats[i][0]
                    my_fake_index = full_strats[i][1]
                    # Compare my ante to opponent's pre-penalty priority,
                    # found in original results table.
                    is_correct = (my_strat[2][0] ==
                        fake_results[my_fake_index][j][1].player_states[0].opponent_priority_pre_penalty)
                    fake_strat = (my_strat[0], my_strat[1],
                                  (is_correct, my_strat[2][1], None))
                    fake_i = fake_strats.index(fake_strat)
                    full_results[i][j] = fake_results[fake_i][j]
        else:
            full_results = [[0]*n for unused in xrange(m)]
            for i in xrange(m):
                for j in xrange(n):
                    my_strat = full_strats[j][0]
                    my_fake_index = full_strats[j][1]
                    # Compare my ante to opponent's pre-penalty priority,
                    # found in original results table.
                    is_correct = (my_strat[2][0] ==
                        fake_results[i][my_fake_index][1].player_states[1].opponent_priority_pre_penalty)
                    fake_strat = (my_strat[0], my_strat[1],
                                  (is_correct, my_strat[2][1], None))
                    fake_j = fake_strats.index(fake_strat)
                    full_results[i][j] = fake_results[i][fake_j]

        self.game.results = full_results
        self.strats = [s[0] for s in full_strats]

        # Record this, in case we need to ask a human player to choose
        # from the list.
        self.all_opponent_priorities = priorities

    def get_soak(self):
        return self.cybernetic_soak + Character.get_soak(self)

class Rexan (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Malediction  (the_game, self)
        self.styles = [Unyielding   (the_game, self), \
                       Devastating  (the_game, self), \
                       Enervating   (the_game, self), \
                       Vainglorious (the_game, self), \
                       Overlords    (the_game, self)  ]
        self.finishers = [ZeroHour     (the_game, self)]
        self.induced_tokens = [CurseToken (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.5 # Tokens penalize opponent.

    def all_cards(self):
        return Character.all_cards(self) + self.induced_tokens

    def choose_initial_discards (self):
        return (self.unyielding, self.drive,
                self.enervating, self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.induced_pool = []

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%s has %d Curse token%s" %
                       (self.opponent, len(self.induced_pool),
                       '' if len(self.induced_pool)==1 else 's'))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.induced_pool = [self.curse] * int(lines[0].split()[2])

    def initial_save (self):
        state = Character.initial_save (self)
        state.induced_pool = self.induced_pool[:]
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.induced_pool = state.induced_pool[:]

    def reset (self):
        self.malediction_damage_limit = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.malediction_damage_limit = self.malediction_damage_limit
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.malediction_damage_limit = state.malediction_damage_limit

    def get_induced_antes (self):
        return range (len(self.induced_pool) + 1)

    def input_induced_ante (self):
        n = len(self.induced_pool)
        if n > 0 :
            print "Number of Curse tokens [0-%d]: " %n
            return input_number (n+1)
        else:
            return 0

    def ante_trigger (self):
        # opponent antes Curse tokens according to their chosen strategy
        for unused in xrange(self.opponent.strat[2][1]):
            self.induced_pool.remove(self.curse)
            self.opponent.ante.append(self.curse)
            if self.game.reporting:
                self.game.report ("%s antes a Curse token" %
                                  self.opponent.name)

    def get_induced_ante_name (self, a):
        if a == 0:
            return ""
        if a == 1:
            return "1 Curse token"
        return "%d Curse tokens" %a

    def take_a_hit_trigger (self):
        if (not self.opponent.hits_scored and
            self.curse not in self.opponent.ante):
            self.give_induced_tokens(1)

    # Opponent gets n tokens, up to 3
    def give_induced_tokens (self, n):
        old_pool = len (self.induced_pool)
        new_pool = old_pool + n
        self.induced_pool = [self.curse] * min (3, new_pool)
        gain = len(self.induced_pool) - old_pool
        if self.game.reporting and gain:
            self.game.report ("%s gains %d %s token%s"
                              %(self.opponent.name, gain, self.induced_tokens[0].name,
                                ("s" if gain>1 else "")))
        if new_pool > 3:
            self.opponent.lose_life (2 * (new_pool - 3))

    def get_damage_cap (self):
        return 2 if self.malediction_damage_limit else Character.get_damage_cap(self)

    # returning tokens on cycle, so that it happens on a Pulse, too
    def cycle (self):
        if not (self.hits_scored or self.opponent.hits_scored):
            self.give_induced_tokens (self.opponent.ante.count(self.curse))
        Character.cycle (self)

    # There seems to be little punishment for hoarding tokens
    def evaluate (self):
        value = Character.evaluate(self)
        value += 1.0 * len(self.induced_pool)
        return value


class Rukyuk (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Reload  (the_game, self)
        self.styles = [Sniper     (the_game, self), \
                       PointBlank (the_game, self), \
                       Gunner     (the_game, self), \
                       Crossfire  (the_game, self), \
                       Trick      (the_game, self)  ]
        self.finishers = [FullyAutomatic (the_game, self),
                           ForceGrenade   (the_game, self)]
        self.tokens = [APShell        (the_game, self), \
                       ExplosiveShell (the_game, self), \
                       FlashShell     (the_game, self), \
                       ImpactShell    (the_game, self), \
                       LongshotShell  (the_game, self), \
                       SwiftShell     (the_game, self)  ]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.4 # Swift

    def choose_initial_discards (self):
        return (self.crossfire, self.shot,
                self.trick, self.reload)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = [t for t in self.tokens]

    def situation_report (self):
        report = Character.situation_report (self)
        tokens = [t.name for t in self.pool]
        report.append ("Pool: " + ', '.join(tokens))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [t for t in self.tokens if t.name in lines[0]]

    def reset (self):
        self.token_spent = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.token_spent = self.token_spent
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.token_spent = state.token_spent

    def get_antes (self):
        return [None] + self.pool

    def input_ante (self, pair):
        if self.pool:
            print "Select token to ante:"
            options = [t.name for t in self.pool] + ["None"]
            ans = menu (options)
            return (self.pool[ans] if ans < len(self.pool) else None)
        else:
            return None

    def ante_trigger (self):
        if self.strat[2][0] is not None:
            self.ante_token (self.strat[2][0])

    def get_ante_name (self, a):
        return ("" if a == None else a.name)

    # Needs an Ammo token in ante to hit, unless using Force Grenade
    def can_hit (self):
        return Character.can_hit(self) and \
               (len(set(self.ante) & set(self.tokens)) > 0 or
                self.base.name == "Force Grenade")

    def recover_tokens (self):
        self.pool = self.tokens[:]

    # Both finishers block Ammo tokens
    def get_active_tokens (self):
        active_tokens = Character.get_active_tokens (self)
        if self.base in self.finishers:
            # Induced tokens are still active
            return [t for t in active_tokens if t not in self.tokens]
        else:
            return active_tokens

    # the main component is based on the total count
    # specific tokens might have small value bonuses
    def evaluate (self):
        return Character.evaluate(self) + sum(t.value for t in self.pool) \
                                        + self.n_token_values [len(self.pool)]
    n_token_values = [-4, -3, -2, -1.5, -1, -0.5, 0]

class Runika (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Tinker  (the_game, self)
        self.styles = [Channeled   (the_game, self),
                       Maintenance (the_game, self),
                       Explosive   (the_game, self),
                       Impact      (the_game, self),
                       Overcharged (the_game, self)]
        self.finishers = [ArtificeAvarice (the_game, self),
                           UdstadBeam      (the_game, self)]
        self.artifacts = [Autodeflector  (the_game, self),
                          Battlefist     (the_game, self),
                          HoverBoots     (the_game, self),
                          ShieldAmulet   (the_game, self),
                          PhaseGoggles   (the_game, self)     ]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.7 # Hover Boots (with overcharge).

    def all_cards (self):
        return Character.all_cards(self) + self.artifacts

    def choose_initial_discards (self):
        return (self.channeled, self.grasp,
                self.impact, self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.active_artifacts = set (self.artifacts)
        self.deactivated_artifacts = set()
        # Artifacts removed by Overcharged are in neither set.

    def situation_report (self):
        report = Character.situation_report (self)
        if self.active_artifacts:
            report.append ("Active artifacts: ")
            for artifact in self.active_artifacts:
                report.append ("  " + artifact.name)
        else:
            report.append("No active artifacts")
        removed = set(self.artifacts) - self.active_artifacts \
                                      - self.deactivated_artifacts
        if removed:
            report.append ("Removed artifacts: ")
            for artifact in removed:
                report.append ("  " + artifact.name)
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        if lines[0] == 'No active artifacts\n':
            self.active_artifacts = set()
        else:
            names = []
            i = 1
            while i<len(lines) and lines[i].starts_with('  '):
                names.append(lines[i][2:-1])
                i += 1
            self.active_artifacts = set([a for a in self.artifacts
                                         if a.name in names])
        i = find_start_index(lines, 'Removed artifacts:')
        if i is None:
            removed = set()
        else:
            names = []
            i += 1
            while i<len(lines) and lines[i].starts_with('  '):
                names.append(lines[i][2:-1])
                i += 1
            removed = set([a for a in self.artifacts if a.name in names])
        self.deactivated_artifacts = set(self.artifacts) \
                                   - self.active_artriacts \
                                   - removed

    def initial_save (self):
        state = Character.initial_save (self)
        state.active_artifacts = self.active_artifacts.copy()
        state.deactivated_artifacts = self.deactivated_artifacts.copy()
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.active_artifacts = state.active_artifacts.copy()
        self.deactivated_artifacts = state.deactivated_artifacts.copy()

    def reset (self):
        self.overcharged_artifact = None
        self.tinkered_artifact = None
        self.channeled_artifact = None
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.overcharged_artifact = self.overcharged_artifact
        state.tinkered_artifact = self.tinkered_artifact
        state.channeled_artifact = self.channeled_artifact
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.overcharged_artifact = state.overcharged_artifact
        self.tinkered_artifact = state.tinkered_artifact
        self.channeled_artifact = state.channeled_artifact

    def set_active_cards (self):
        Character.set_active_cards(self)
        if self.base.name != 'Udstad Beam':
            self.active_cards += list(self.active_artifacts)

    def activate_artifact (self, artifact):
        if artifact not in self.active_artifacts:
            self.active_artifacts.add(artifact)
            self.deactivated_artifacts.discard(artifact)
            self.set_active_cards ()
            if self.game.reporting:
                self.game.report (artifact.name + " is activated")

    def deactivate_artifact (self, artifact):
        if artifact in self.active_artifacts:
            self.active_artifacts.remove(artifact)
            self.deactivated_artifacts.add(artifact)
            self.set_active_cards ()
            if self.game.reporting:
                self.game.report (artifact.name + " is de-activated")

    def remove_artifact (self, artifact):
        if artifact in self.active_artifacts:
            self.active_artifacts.remove(artifact)
            self.set_active_cards ()
            if self.game.reporting:
                self.game.report (artifact.name + " is removed from the game")

    # Runika chooses an artifact to activate
    def activation_fork (self, prompt='Choose artifact to activate:'):
        deactivated = list(self.deactivated_artifacts)
        if not deactivated:
            if self.game.reporting:
                self.game.report ('No artifacts to activate')
            return None
        options = [artifact.name for artifact in deactivated]
        ans = self.game.make_fork (len(options), self, prompt, options)
        chosen_artifact = deactivated[ans]
        self.activate_artifact(chosen_artifact)
        return chosen_artifact

    # given player chooses 1 artifact to deactivate
    def deactivation_fork (self, player, fake=False):
        # overcharged artifacts cannot be deactivated
        targets = self.active_artifacts - set((self.overcharged_artifact,))
        # tinkered artifacts cannot be deactivated by opponent on hit
        if player is self.opponent:
            targets -= set((self.tinkered_artifact,))
        targets = list(targets)
        if not targets:
            if self.game.reporting:
                self.game.report ('No artifacts to de-activate')
            return None
        prompt = 'Choose an artifact to de-activate:'
        options = [artifact.name for artifact in targets]
        if fake:
            # choose highest/lowest value, depending on choosing player
            values = [artifact.value for artifact in targets]
            if player == self:
                choice = values.index (min (values))
            else:
                choice = values.index (max (values))
        else:
            choice = None
        ans = self.game.make_fork (len(targets), player,
                                   prompt, options, choice)
        chosen_artifact = targets[ans]
        self.deactivate_artifact (chosen_artifact)
        return chosen_artifact

    # opponent deactivates an artifact on first hit of beat
    def take_a_hit_trigger (self):
        if self.opponent.hits_scored or \
           self.base.name == 'Artifice Avarice':
            return
        if self.channeled_artifact in self.active_artifacts:
            self.deactivate_artifact(self.channeled_artifact)
        else:
            self.deactivation_fork (self.opponent)

    # Overcharged Autodeflector blocks life loss
    def lose_life (self, life):
        if self.autodeflector is self.overcharged_artifact and \
           self.autodeflector in self.active_artifacts:
            if self.game.reporting:
                self.game.report ("Runika's Autodeflector blocks life loss")
        else:
            Character.lose_life (self, life)

    # deactivation is 3 times the value of an artifact.
    # removal is 1 more time the value
    def evaluate (self):
        return Character.evaluate (self) + \
            4 * sum(artifact.value for artifact in self.active_artifacts) + \
                sum(artifact.value for artifact in self.deactivated_artifacts)


class Sarafina(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Field (the_game, self)
        self.styles = [Implosion (the_game, self),
                       Locus     (the_game, self),
                       Silver    (the_game, self),
                       Hollow    (the_game, self),
                       Tactical  (the_game, self)  ]
        self.finishers = [RitherwhyteInfusionSarafina (the_game, self),
                          FulminatingVortex   (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.projection = None
        self.ritherwhyte_activated = False

    def choose_initial_discards (self):
        return (self.hollow, self.grasp,
                self.locus, self.field)

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.projection = addendum[0].find('p')
        if self.projection == -1:
            self.projection = None
        self.ritherwhyte_activated = find_start (lines, 'Ritherwhyte')

    board_addendum_lines = 1
    def get_board_addendum (self):
        if self.projection == None:
            return ''
        addendum = ['.'] * 7
        addendum [self.projection] = 'p'
        return ''.join(addendum)

    def situation_report (self):
        report = Character.situation_report (self)
        if self.ritherwhyte_activated:
            report.append ("Ritherwhyte Infusion activated")
        return report

    def initial_save (self):
        state = Character.initial_save (self)
        state.projection = self.projection
        state.ritherwhyte_activated = self.ritherwhyte_activated
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.projection = state.projection
        self.ritherwhyte_activated = state.ritherwhyte_activated

    def reset (self):
        self.switched_sides = False
        self.field_immobilzie = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.switched_sides = self.switched_sides
        state.field_immobilzie = self.field_immobilzie
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.switched_sides = state.switched_sides
        self.field_immobilzie = state.field_immobilzie

    def project(self, n):
        # Put projection n spaces away from Sarafina.
        # If n is None, put projection at edge of arena.
        if n is None:
            positions = set((0,6))
        else:
            positions = set((self.position - n,
                             self.position + n)) & set(xrange(7))
        if self.projection is not None:
            positions.add(self.projection)
        positions.discard(self.position)
        if not positions:
            return
        positions = sorted(list(positions))
        options = []
        if self.is_user and self.game.interactive_mode:
            projection = self.projection
            for pos in positions:
                self.projection = pos
                options.append(self.get_board_addendum())
            self.projection = projection
        self.projection = positions[self.game.make_fork(
            len(positions), self, "Place projection:", options)]
        if self.game.reporting:
            self.game.report ("Sarafina places her projection:")
            for s in self.game.get_board():
                self.game.report (s)

    def get_antes (self):
        return [False] if self.projection is None else [False, True]

    def input_ante (self, pair):
        if self.projection is None:
            return False
        else:
            print "Move to projection?"
            return bool(menu(['No', 'Yes']))

    def ante_trigger (self):
        if self.ritherwhyte_activated:
            self.lose_life (1)
        if self.strat[2][0]:
            assert self.projection is not None
            # Setting projection position to None before moving, so that
            # projection doesn't appear in post-move board report.
            pos = self.projection
            self.projection = None
            self.move_directly([pos])

    def get_ante_name (self, a):
        return "move to projection" if a else ""

    def get_maxrange_bonus (self):
        return 2 if self.ritherwhyte_activated else 0

    def set_preferred_range (self):
        Character.set_preferred_range (self)
        if self.ritherwhyte_activated:
            self.preferred_range += 1

    def blocks_priority_bonuses (self):
        return self.silver in self.active_cards
    def blocks_power_bonuses (self):
        return self.silver in self.active_cards

    # Recording switching sides, for Locus.
    def movement_reaction(self, initiator, mover, old_position, direct):
        Character.movement_reaction(self, initiator, mover, old_position, direct)
        if not direct and mover.opponent.mimics_movement():
            return
        if ordered(old_position, mover.opponent.position, mover.position):
            self.switched_sides = True

    def blocks_opponent_movement(self, initiator, direct):
        return set(xrange(7)) if self.field_immobilzie else Character.blocks_opponent_movement(self, initiator, direct)

    def get_superposed_positions(self):
        if self.projection is None:
            return [self.position]
        else:
            # Since Sarafina only moves in the ante phase, she returns
            # the projection position even if it's on the opponent
            # (the opponent might move before attack pairs are set).
            return [self.position, self.projection]

    def get_superposition_priority(self):
        if self.projection is None:
            return self.game.CHOOSE_POSITION_NOW
        else:
            return self.game.CHOOSE_POSITION_IN_ANTE

    def evaluate_superposition(self):
        if self.opponent.position is None:
            return 0
        if (self.projection is None or
            self.projection in (self.opponent.position,
                                self.position)):
            ret = 0
        else:
            # Try for differing ranges, to mix opponent up.
            d0 = self.game.distance()
            d1 = abs(self.projection - self.opponent.position)
            ret = 0.2 + 0.1 * abs(d0-d1)
        # Locus gets extra priority for switching with opponent.
        if self.locus not in self.discard[1] | self.discard[2]:
            if (self.projection is not None and ordered(self.position,
                                                        self.opponent.position,
                                                        self.projection)):
                ret += 0.2
            else:
                ret -= 0.2
        return ret

    def evaluate (self):
        value = Character.evaluate(self)
        if self.ritherwhyte_activated:
            value += 4
        return value


class Seth(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Omen (the_game, self)
        self.styles = [Fools      (the_game, self),
                       Mimics     (the_game, self),
                       Vanishing  (the_game, self),
                       Wyrding    (the_game, self),
                       Compelling (the_game, self)  ]
        self.finishers = [FortuneBuster (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.6 # Guess base.

    def choose_initial_discards (self):
        return (self.fools, self.grasp,
                self.wyrding, self.strike)

    # these aren't really possible antes, but rather possible ante results
    # (correct or incorret guess)
    # the actual strategies will be formed in post-processing
    def get_antes (self):
        return [False, True]

    def input_ante (self, pair):
        discards = self.opponent.discard[1] | self.opponent.discard[2]
        bases = [b for b in self.opponent.bases if b not in discards]
        print "Name one of %s's bases:" % self.opponent
        options = [b.name for b in bases]
        ans = menu (options)
        return bases[ans]

    # Handles both simulation strategies (correct/incorrect guess)
    # and post-processing strategies (acutal base guessed).
    # After a clash, we go back to correct/incorrect (because correctness
    # has already been decided, and has nothing to do with opponent's new
    # base.
    # TODO: actually, Seth can use the bonuses if the base changes.
    # However, he wins priority ties when he guesses correctly,
    # so a clash is usually not possible.
    def get_ante_name (self, a):
        # post processing - guess is an opponent's base
        if isinstance (a, Base):
            return a.name
        # simulation, guess is correct or not
        else:
            if a:
                return "Correct Guess"
            else:
                return "Incorrect Guess"

    # Check whether Seth's guess was correct.
    # Handles both boolean guess results and actual base guesses.
    # TODO: actually, power and priority can change if the base changes
    # during the beat.  I don't think anyone other than Seth can
    # actually change their base after the clash.  In any case, fixing
    # this would require giving up the "boolean guess" trick.
    def reveal_trigger (self):
        self.correct_guess = (self.strat[2][0] in [True, self.opponent.base])
        if self.correct_guess:
            self.add_triggered_power_bonus (2)
            self.add_triggered_priority_bonus (2)
        Character.reveal_trigger(self)

    def clash_priority (self):
        return 0.1 * self.correct_guess + Character.clash_priority(self)

    # logic to handle Wyrding shenanigans
    def start_trigger (self):
        if (not self.opponent.blocks_start_triggers() and
            self.wyrding in self.active_cards):
            old_base = self.base
            # we might get a new base
            self.style.start_trigger ()
            # if base changed, you can activate old base's start trigger,
            if self.base != old_base:
                # fork for old base is only necessary if it has a start trigger
                if old_base.name in ['Omen','Burst','Spike','Force']:
                    prompt = ("Execute %s's Start of Beat effect before switching?"%
                              old_base.name)
                    options = ["No", "Yes"]
                    if self.game.make_fork (2, self, prompt, options):
                        old_base.start_trigger()
            # in any case, activate trigger for final base
            self.base.start_trigger()
        else:
            Character.start_trigger(self)

    # overrides default method, which I set to pass for performance
    def mimics_movement (self):
        return any (card.mimics_movement() for card in self.active_cards)

    # Seth's simulated strategies don't have a real guess, but rather an
    # assumption about whether the guess was correct or not (which is
    # all that is needed for simulation).
    # After simulation, some post-processing is required get the real
    # list of strategies and table of results.
    def post_simulation_processing (self):
        bases = set(self.opponent.bases) - self.opponent.discard[1] \
                                         - self.opponent.discard[2]
        bases = sorted (list(bases), key = attrgetter('order'))

        fake_strats = self.strats
        opp_strats = self.opponent.strats
        fake_results = self.game.results

        # For each pair of Seth's strategies (same except for guess
        # correctness), pick one.
        strats_without_guesses = [s for s in fake_strats
                                 if s[2][0] is False]
        # Then, make a new strategy list, with 5 actual guess per strategy.
        full_strats = [(s[0], s[1], (b, s[2][1], None))
                       for s in strats_without_guesses for b in bases]

        n = len(full_strats)
        m = len(opp_strats)
        # for each strat combination, take the simulation result corresponding
        # to those pairs, but pick correct/incorrect guess based on whether
        # Seth's guess actually corresponds to the played base
        if self.my_number==0:
            full_results = [[0]*m for unused in xrange(n)]
            for i in xrange(n):
                for j in xrange(m):
                    my_strat = full_strats[i]
                    opp_strat = opp_strats[j]
                    is_correct = (my_strat[2][0] == opp_strat[1])
                    fake_strat = (my_strat[0],
                                  my_strat[1],
                                  (is_correct, my_strat[2][1], None))
                    fake_i = fake_strats.index(fake_strat)
                    full_results[i][j] = fake_results[fake_i][j]
        else:
            full_results = [[0]*n for unused in xrange(m)]
            for i in xrange(m):
                for j in xrange(n):
                    my_strat = full_strats[j]
                    opp_strat = opp_strats[i]
                    is_correct = (my_strat[2][0] == opp_strat[1])
                    fake_strat = (my_strat[0],
                                  my_strat[1],
                                  (is_correct, my_strat[2][1], None))
                    fake_j = fake_strats.index(fake_strat)
                    full_results[i][j] = fake_results[i][fake_j]

        self.game.results = full_results
        self.strats = full_strats


class Shekhtur (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Brand (the_game, self)
        self.styles = [Unleashed   (the_game, self), \
                       Combination (the_game, self), \
                       Reaver      (the_game, self), \
                       Jugular     (the_game, self), \
                       Spiral      (the_game, self)  ]
        # Soul Breaker isn't fully implemented, so only Coffin Nails is listed
        self.finishers = [CoffinNails (the_game, self)]
        self.status_effects = [PowerBonusStatusEffect(the_game, self)]
        self.tokens = [Malice (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        self.max_tokens = 5

    mean_priority_bonus = 1.2 # Tokens

    def choose_initial_discards (self):
        return (self.combination, self.grasp,
                self.reaver, self.shot)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 3 * [self.malice]
        self.hits_scored_last_beat = False
        self.coffin_nails_hit = False

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Malice token%s" % (len(self.pool),
                       '' if len(self.pool)==1 else 's'))
        if self.coffin_nails_hit:
            report.append ("%s has no soak or stunguard (Coffin Nails)" % self.opponent)
        if self.hits_scored_last_beat:
            report.append ("Shekhtur hit her opponent last beat")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.malice] * int(lines[0][0])
        self.hits_scored_last_beat = find_start (lines, 'Shekhtur hit her opponent')
        self.coffin_nails_hit = find_end (lines, '(Coffin Nails)\n')

    def initial_save (self):
        state = Character.initial_save (self)
        state.coffin_nails_hit = self.coffin_nails_hit
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.coffin_nails_hit = state.coffin_nails_hit

    def prepare_next_beat (self):
        Character.prepare_next_beat (self)
        self.hits_scored_last_beat = self.hits_scored

    def get_antes (self):
        n = len(self.pool)
        return range (0, n + 1)

    def input_ante (self, pair):
        n = len(self.pool)
        if n > 0 :
            print "Number of Malice tokens [0-%d]: " %n
            return input_number (n+1)
        else:
            return 0

    def ante_trigger (self):
        for unused in range(self.strat[2][0]):
            self.ante_token(self.malice)

    def get_ante_name (self, a):
        if a == 0:
            return ""
        if a == 1:
            return "1 token"
        return str(a) + " tokens"

    def reduce_soak (self, soak):
        if self.coffin_nails_hit:
            return 0
        else:
            return Character.reduce_soak(self,soak)

    def reduce_stunguard (self, stunguard):
        if self.coffin_nails_hit:
            return 0
        else:
            return Character.reduce_stunguard(self,stunguard)

    def damage_trigger (self, damage):
        self.recover_tokens (damage)
        Character.damage_trigger (self, damage)

    def evaluate (self):
        return (Character.evaluate (self) + 0.3 * len(self.pool) +
                5 * self.coffin_nails_hit)


class Tanis(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = SceneShift  (the_game, self)
        self.styles = [Valiant     (the_game, self), \
                       Climactic  (the_game, self), \
                       Storyteller   (the_game, self), \
                       Playful  (the_game, self), \
                       Distressed (the_game, self)  ]
        self.finishers = [CurtainCall (the_game, self)]
        self.puppets = [Eris     (the_game, self),
                        Loki     (the_game, self),
                        Mephisto (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def all_cards(self):
        return Character.all_cards(self) + self.puppets

    # TODO: choose
    def choose_initial_discards (self):
        return (self.climactic, self.grasp,
                self.playful, self.unique_base)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.possessed_puppet = None
        if self.is_user:
            perms = list(itertools.permutations(self.puppets))
            options = ['...' + ''.join([puppet.initial for puppet in permutation] + '.')
                       for permutation in perms]
            print "Choose initial puppet positions:"
            perm = perms[menu(options)]
            for i, puppet in enumerate(perm):
                puppet.position = i + 3
        else:
            (self.mephisto.position, self.eris.position, self.loki.position) = \
                ((1,2,3) if self.position == 1 else (5,4,3))
        self.position = None

    def situation_report (self):
        report = Character.situation_report (self)
        if self.possessed_puppet:
            report.append ("Possessed puppet: %s" % self.possessed_puppet)
        return report

    board_addendum_lines = 3
    def get_board_addendum (self):
        addendum = []
        for puppet in self.puppets:
            if puppet.position is not None:
                line = ['.'] * 7
                line[puppet.position] = puppet.initial
                line = ''.join(line)
                addendum.append(line)
        return addendum

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.position = None
        for i, puppet in enumerate(self.puppets):
            puppet.position = addendum[i].find(puppet.initial)
        line = find_start_line(lines, 'Possessed puppet:')
        self.possessed_puppet = None
        for puppet in self.puppet:
            if puppet.name in line:
                self.possessed_puppet = puppet
                break

    def initial_save (self):
        state = Character.initial_save (self)
        state.puppet_positions = [puppet.position
                                  for puppet in self.puppets]
        state.possessed_puppet = self.possessed_puppet
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        for i,puppet in enumerate(self.puppets):
            puppet.position = state.puppet_positions[i]
        self.possessed_puppet = state.possessed_puppet

    def reset (self):
        self.switched_sides = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.switched_sides = self.switched_sides
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.switched_sides = state.switched_sides

    def get_pre_attack_decisions(self):
        options = [p for p in self.puppets
                   if p.position != self.opponent.position and
                   p is not self.possessed_puppet]
        return options if options else [self.possessed_puppet]

    def input_pre_attack_decision_index(self):
        options = self.get_pre_attack_decisions()
        if len(options) == 1:
            return 0
        print "Choose puppet to possess this beat:"
        return menu([puppet.name for puppet in options])

    def pre_attack_decision_report(self, decision):
        return ['Tanis possesses %s' % decision]

    # Switch Tanis in.
    def pre_attack_decision_effects(self):
        self.possessed_puppet = self.strat[2][2]
        self.position = self.possessed_puppet.position
        self.possessed_puppet.position = None

    # Switch Tanis out.
    def cycle(self):
        Character.cycle(self)
        self.possessed_puppet.position = self.position
        self.position = None

    def get_superposed_positions(self):
        return list(set(puppet.position for puppet in
                        self.get_pre_attack_decisions()))

    def get_superposition_priority(self):
        return self.game.CHOOSE_POSITION_BEFORE_ATTACK_PAIRS

    # Record switching sides, for SceneShift
    def movement_reaction(self, initiator, mover, old_position, direct):
        Character.movement_reaction(self, initiator, mover, old_position, direct)
        if not direct and mover.opponent.mimics_movement():
            return
        if ordered(old_position, mover.opponent.position, mover.position):
            self.switched_sides = True

    def blocks_priority_bonuses (self):
        return (self.playful in self.active_cards
                and not self.playful.suspend_blocking)
    def blocks_power_bonuses (self):
        return (self.playful in self.active_cards
                and not self.playful.suspend_blocking)
    def blocks_maxrange_bonuses (self):
        return (self.playful in self.active_cards
                and not self.playful.suspend_blocking)

# Add special case juto and style evaluation
class Tatsumi (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Whirlpool  (the_game, self)
        self.styles = [Siren     (the_game, self), \
                       Fearless  (the_game, self), \
                       Riptide   (the_game, self), \
                       Empathic  (the_game, self), \
                       WaveStyle (the_game, self)  ]
        self.finishers = [TsunamisCollide (the_game, self),
                          GuardianXandjinn (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)
        # give bases different preferred ranges for juto
        # (because juto isn't moved by base)
        self.unique_base.juto_preferred_range = 1.5
        self.strike.juto_preferred_range = 1
        self.shot.juto_preferred_range = 2.5
        self.drive.juto_preferred_range = 1
        self.burst.juto_preferred_range = 2.5
        self.grasp.juto_preferred_range = 1
        self.dash.juto_preferred_range = 2
        self.counter_base.juto_preferred_range = 1
        self.wave_base.juto_preferred_range = 3
        self.force.juto_preferred_range = 1
        self.spike.juto_preferred_range = 2
        self.throw.juto_preferred_range = 1
        self.parry.juto_preferred_range = 2
        self.stagger.juto_preferred_range = 1
        self.leap.juto_preferred_range = 1
        self.lunge.juto_preferred_range = 1.5
        self.sweep.juto_preferred_range = 1.5
        self.catch.juto_preferred_range = 1
        self.block.juto_preferred_range = 2

    mean_priority_bonus = 0.2 # Juto

    def choose_initial_discards (self):
        return (self.fearless, self.burst,
                self.riptide, self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.juto_position = self.position
        self.juto_life = 4

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("Juto's life: %d" %self.juto_life)
        return report

    board_addendum_lines = 1
    def get_board_addendum (self):
        if self.juto_position is None:
            return ''
        addendum = ['.'] * 7
        addendum [self.juto_position] = 'j'
        return ''.join(addendum)

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.juto_life = int (lines[0][-2])
        self.juto_position = addendum[0].find('j')
        if self.juto_position == -1:
            self.juto_position = None

    def initial_save (self):
        state = Character.initial_save (self)
        state.juto_position = self.juto_position
        state.juto_life = self.juto_life
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.juto_position = state.juto_position
        self.juto_life = state.juto_life

    def reset (self):
        self.riptide_dodge = False
        self.juto_damage_taken = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.riptide_dodge = self.riptide_dodge
        state.juto_damage_taken = self.juto_damage_taken
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.riptide_dodge = state.riptide_dodge
        self.juto_damage_taken = state.juto_damage_taken

    def zone_0 (self):
        return self.juto_position != None and \
               ordered (self.juto_position,
                        self.position,
                        self.opponent.position)
    def zone_1 (self):
        return self.juto_position == self.position
    def zone_2 (self):
        return self.juto_position != None and \
               ordered (self.position,
                        self.juto_position,
                        self.opponent.position)
    def zone_3 (self):
        # Not using ordered(), because we don't need a strong ordering.
        return self.juto_position != None and \
               (self.juto_position - self.opponent.position) * \
               (self.position - self.opponent.position) <= 0

    def get_maxrange_bonus (self):
        return 2 * self.zone_0()

    def get_power_bonus (self):
        return 1 * self.zone_0()

    def get_priority_bonus (self):
        return 1 * self.zone_3()

    def get_stunguard (self):
        return Character.get_stunguard(self) + 2 * self.zone_2()

    def expected_stunguard(self):
        ret = Character.expected_stunguard(self)
        if self.zone_2():
            ret += 2
        return ret

    def get_soak (self):
        return Character.get_soak(self) + 1 * self.zone_2() + 3 * self.zone_1()

    def expected_soak(self):
        ret = Character.expected_soak(self)
        if self.zone_2():
            ret += 1
        if self.zone_1():
            ret += 3
        return ret

    # Reduce Juto's life when he soaks damage.
    def soak_trigger (self, soaked_damage):
        self.juto_damage_taken += soaked_damage
        self.juto_life -= soaked_damage

    # Copying all this to have Juto disappear when out of life points.
    def take_damage (self, damage):
        soak = self.opponent.reduce_soak(self.get_soak())
        if self.game.reporting and soak:
            self.game.report ('%s has %d soak' % (self, soak))
        damage_soaked = min (soak, damage)
        if damage_soaked:
            self.soak_trigger (damage_soaked)
        remaining_damage = max(self.get_damage_cap() - self.damage_taken, 0)
        final_damage = min (damage - damage_soaked, remaining_damage)

        # life can't go below certain minimum (usually 0)
        self.life = max (self.life - final_damage,
                         self.get_minimum_life())
        if self.game.reporting:
            self.game.report (self.name + \
                              " takes %d damage (now at %d life)" \
                              %(final_damage, self.life))
        if self.life <= 0:
            raise WinException (self.opponent.my_number)
        if final_damage:
            self.opponent.damage_trigger (final_damage)
            self.take_damage_trigger (final_damage)
        # damage_taken updated after triggers, so that triggers can
        # check if this is first damage this beat
        self.damage_taken += final_damage

        # Remove Juto (after stun, so that he has a chance to contribute
        # stunguard).
        if self.juto_life <= 0:
            self.juto_position = None
            self.juto_life = 0

    def move_juto (self, positions):
        if self.juto_position is None:
            return
        positions = list (positions)
        prompt = "Move Juto:"
        options = []
        if self.is_user and self.game.interactive_mode:
            juto_pos = self.juto_position
            for pos in positions:
                self.juto_position = pos
                options.append (self.get_board_addendum())
            self.juto_position = juto_pos
        self.juto_position = positions [self.game.make_fork
                                    (len(positions), self, prompt, options)]
        if self.game.reporting:
            self.game.report ("Juto moves")
            for s in self.game.get_board():
                self.game.report (s)

    # with Fearless, attacks from juto's position
    def attack_range (self):
        if self.fearless in self.active_cards and self.juto_position != None:
            return abs (self.juto_position - self.opponent.position)
        return abs (self.position - self.opponent.position)

    def can_be_hit (self):
        return Character.can_be_hit(self) and (
                        not (self.riptide_dodge and
                             self.opponent.attack_range() >= 3))

    # different preferred range for tatsumi and juto, based on the styles
    # that let each of them attack
    # preferred range is now 4-tuple of preferred ranges, and also of
    # weights (number of styles that let each of them attack)
    def set_preferred_range (self):
        unavailable_cards = self.discard[1] \
                          | self.discard[2]
        styles = set(self.styles) - unavailable_cards
        bases = set(self.bases) - unavailable_cards
        tatsumi_styles = [s for s in styles if s.tatsumi_attack]
        juto_styles = [s for s in styles if s.juto_attack]
        self.tatsumi_weight = len (tatsumi_styles)
        self.juto_weight = len (juto_styles)
        base_range = sum (b.get_preferred_range() for b in bases) / len(bases)
        base_range_juto = sum (b.juto_preferred_range for b in bases) / len(bases)
        self.preferred_range = (sum (s.get_preferred_range()
                                      for s in tatsumi_styles) /
                       self.tatsumi_weight) \
                    + base_range
        self.juto_range = (sum (s.get_preferred_range()
                                  for s in juto_styles) /
                       self.juto_weight) \
                    + base_range_juto \
                    if self.juto_weight>0 else 0

    # average of tatsumi and juto's range penalty
    # (weighted by number of styles that let each of them attack)
    def evaluate_range (self):
        if self.juto_position is None:
            return Character.evaluate_range(self)
        tatsumi_penalty = self.tatsumi_weight * \
                        (self.preferred_range - self.game.distance()) ** 2
        juto_penalty = self.juto_weight * \
                        (self.juto_range - (abs(self.juto_position -
                                                 self.opponent.position))) ** 2
        return - self.game.range_weight * (tatsumi_penalty + juto_penalty) / \
                 (self.tatsumi_weight + self.juto_weight)

    def evaluate (self):
        juto_value = self.juto_life / 4.0
        if juto_value <= 0:
            juto_value = -1
        if self.zone_0():
            juto_value += 1
        return Character.evaluate (self) + juto_value


class Vanaah(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Scythe  (the_game, self)
        self.styles = [Reaping   (the_game, self),
                       Judgment  (the_game, self),
                       Glorious  (the_game, self),
                       Paladin   (the_game, self),
                       Vengeance (the_game, self)  ]
        self.finishers = [DeathWalks     (the_game, self),
                           HandOfDivinity (the_game, self)]
        self.status_effects = [PriorityPenaltyStatusEffect(the_game, self)]
        self.tokens = [DivineRush (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.6 # Token

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = [self.divine_rush]

    def choose_initial_discards (self):
        return (self.vengeance, self.burst,
                self.reaping, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        if self.pool:
            report.append ("Divine Rush token in pool")
        return report

    def read_my_state (self, lines, board, addendum):
        for i in (1,2):
            if 'Divine Rush' in lines[i]:
                token_location = i
                break
        else:
            token_location = 0
        lines = Character.read_my_state (self, lines, board, addendum)
        if token_location == 0:
            self.pool = [self.divine_rush]
        else:
            self.discard[token_location].add(self.divine_rush)

    def get_antes (self):
        return range (len(self.pool) + 1)

    def input_ante (self, pair):
        if self.pool:
            print "Ante Divine Rush token?"
            return menu (["No", "Yes"])
        else:
            return 0

    def ante_trigger (self):
        if self.strat[2][0]:
            self.ante_token(self.divine_rush)

    def get_ante_name (self, a):
        return ("Divine Rush" if a==1 else "")

    # when vanaah antes/discards her token, she puts it
    # into her discard[0]
    def discard_token (self, token = None, verb = "discard"):
        result = Character.discard_token (self, token, verb)
        if result:
            self.discard[0].add(self.divine_rush)
        return result

    def recover_tokens (self):
        self.pool = [self.divine_rush]
        for d in self.discard:
            d.discard(self.divine_rush)

    # token will cycle with cards from 0 to 1 to 2, but needs some help.
    def cycle (self):
        # ... coming back from discard 2...
        if self.divine_rush in self.discard[2] and \
           not isinstance (self.style, SpecialAction):
            self.pool = [self.divine_rush]
        # ... and moving from 0 to 1 with finisher/pulse
        # (which don't cycle).
        if (self.style is self.special_action and
            self.divine_rush in self.discard[0]):
            self.discard[1].add(self.divine_rush)
        Character.cycle (self)


class Voco (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Shred  (the_game, self)
        self.styles = [Monster     (the_game, self), \
                       Metal       (the_game, self), \
                       Hellraising (the_game, self), \
                       Abyssal     (the_game, self), \
                       Thunderous  (the_game, self)  ]
        self.finishers = [ZMosh   (the_game, self),
                           TheWave (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.zombies = set()

    def choose_initial_discards (self):
        return (self.thunderous, self.unique_base,
                self.monster, self.burst)

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.zombies = set([i for i in xrange(7) if addendum[0][i]=='z'])

    def initial_save (self):
        state = Character.initial_save (self)
        state.zombies = self.zombies.copy()
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.zombies = state.zombies.copy()

    def reset(self):
        Character.reset(self)
        self.zombie_soak = 0

    def full_save(self):
        state = Character.full_save(self)
        state.zombie_soak = self.zombie_soak
        return state

    def full_restore(self, state):
        Character.full_restore(self, state)
        self.zombie_soak = state.zombie_soak

    board_addendum_lines = 1
    def get_board_addendum (self):
        if not self.zombies:
            return ''
        addendum = ['.'] * 7
        for z in self.zombies:
            addendum [z] = 'z'
        return ''.join(addendum)

    def add_zombies (self, to_add):
        n = len (self.zombies)
        self.zombies |= to_add
        if self.game.reporting and len(self.zombies) != n:
            for s in self.game.get_board():
                self.game.report (s)

    # remove n zombies from given list
    # if n > len(list), remove all in list
    # if n < len(list), fork to decide which ones to remove
    def remove_zombies (self, removable_positions, n_removed):
        if n_removed == 0:
            return
        n = len(self.zombies)
        removable_zombies = removable_positions & self.zombies
        if n_removed >= len(removable_zombies):
            # remove all relevant zombies
            self.zombies -= removable_positions
        else:
            # choose some zombies to remove
            prompt = "Choose zombies to remove from board"
            options = []
            combos = list (itertools.combinations (removable_positions,
                                                   n_removed))
            if self.is_user and self.game.interactive_mode:
                for combo in combos:
                    zs = ['z' if i in combo else '.' for i in xrange(7)]
                    options.append (''.join(zs))
            fork_result = self.game.make_fork (len(combos), self,
                                               prompt, options)
            self.zombies -= set(combos[fork_result])
            if self.game.reporting and len(self.zombies) != n:
                for s in self.game.get_board():
                    self.game.report (s)

    # the positions in which zombies can soak damage
    # current version - attacker's space and adjacent spaces
    def soak_positions (self):
        return set ((self.opponent.position-1,self.opponent.position+1)) \
               & set(xrange(7))

    def get_soak (self):
        return self.zombie_soak

    def expected_soak(self):
        return Character.expected_soak(self) + len (self.zombies & self.soak_positions())

    def take_a_hit_trigger(self):
        Character.take_a_hit_trigger(self)
        self.zombie_soak += len(self.soak_positions() & self.zombies)
        if self.z_mosh not in self.active_cards:
            self.remove_zombies(self.soak_positions(), 2)

    def unique_ability_end_trigger (self):
        opp = self.opponent.position
        if opp in self.zombies:
            if self.game.make_fork (2, self, "Remove zombie to cause 1 life loss?",
                                    ["No", "Yes"]):
                self.opponent.lose_life (1)
                self.remove_zombies(set([opp]), 1)

    # zombies in the soak range or on opponent are .5
    # zombies anywhere else are .25
    def evaluate (self):
        op = self.opponent.position
        better = self.soak_positions() | set([op])
        value =  Character.evaluate(self) + 0.25 * len(self.zombies) \
                                         + 0.25 * len(self.zombies & better)
        return value


class Xenitia(Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = Bloodweapon  (the_game, self)
        self.styles = [Draining (the_game, self), \
                       Guilty   (the_game, self), \
                       Razor    (the_game, self), \
                       Rending  (the_game, self), \
                       Purging  (the_game, self)  ]
        self.finishers = [Exsanguinate (the_game, self),
                          Phlebotomy   (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.blood_awakened = False

    def choose_initial_discards (self):
        return (self.rending, self.burst,
                self.razor, self.grasp)

    def reset (self):
        self.razor_ignore_stunguard = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.razor_ignore_stunguard = self.razor_ignore_stunguard
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.razor_ignore_stunguard = state.razor_ignore_stunguard

    def max_draw(self):
        return min(5, self.life -1)

    def get_antes (self):
        return range(1 + self.max_draw())

    def input_ante (self, pair):
        md = self.max_draw()
        if md == 0:
            return 0
        print "Draw how much life? [0-%d]" % md
        return input_number(md + 1)

    def ante_trigger (self):
        draw = self.strat[2][0]
        if self.game.reporting:
            self.game.report("Xenitia's draw is %d" % draw)
        self.lose_life(draw)

    def get_ante_name (self, a):
        return "%d life" % a

    def damage_trigger(self, damage):
        Character.damage_trigger(self, damage)
        if self.strat[2][0] == 0:
            self.gain_life(1)

    def unique_ability_end_trigger(self):
        if not self.is_stunned():
            self.gain_life(self.strat[2][0])

    # Copying to allow Guilty to block damage.
    def take_damage (self, damage):
        soak = self.opponent.reduce_soak(self.get_soak())
        if self.game.reporting and soak:
            self.game.report ('%s has %d soak' % (self, soak))
        damage_soaked = min (soak, damage)
        if damage_soaked:
            self.soak_trigger (damage_soaked)
        remaining_damage = max(self.get_damage_cap() - self.damage_taken, 0)
        final_damage = min (damage - damage_soaked, remaining_damage)

        # Xenitia's addition
        if (self.guilty in self.active_cards and
            self.strat[2][0] == final_damage):
            final_damage = 0

        # life can't go below certain minimum (usually 0)
        self.life = max (self.life - final_damage,
                         self.get_minimum_life())
        if self.game.reporting:
            self.game.report (self.name + \
                              " takes %d damage (now at %d life)" \
                              %(final_damage, self.life))
        if self.life <= 0:
            raise WinException (self.opponent.my_number)
        if final_damage:
            self.opponent.damage_trigger (final_damage)
            self.take_damage_trigger (final_damage)
        # damage_taken updated after triggers, so that triggers can
        # check if this is first damage this beat
        self.damage_taken += final_damage

    def reduce_stunguard(self, stunguard):
        return 0 if self.razor_ignore_stunguard else Character.reduce_stunguard(self, stunguard)

    def evaluate(self):
        return Character.evaluate(self) + 0.2 * self.max_draw()


class Zaamassal (Character):
    def __init__ (self, the_game, n, base_set='alpha', is_user=False):
        self.unique_base = ParadigmShift  (the_game, self)
        self.styles = [Malicious (the_game, self), \
                       Sinuous   (the_game, self), \
                       Urgent    (the_game, self), \
                       Sturdy    (the_game, self), \
                       Warped    (the_game, self)  ]
        self.paradigms = [Pain       (the_game, self), \
                          Fluidity   (the_game, self), \
                          Haste      (the_game, self), \
                          Resilience (the_game, self), \
                          Distortion (the_game, self)  ]
        self.finishers = [OpenTheGate  (the_game, self),
                          PlaneDivider (the_game, self)]
        Character.__init__ (self, the_game, n, base_set, is_user)

    mean_priority_bonus = 0.1 # Haste

    def all_cards (self):
        return Character.all_cards (self) + self.paradigms

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.active_paradigms = set()

    def choose_initial_discards (self):
        return (self.malicious, self.grasp,
                self.sinuous, self.burst)

    def situation_report (self):
        report = Character.situation_report (self)
        paradigms = [p.name for p in self.active_paradigms]
        paradigm_rep = ', '.join(paradigms) if paradigms else "None"
        report.append ("Active paradigm: " + paradigm_rep)
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.active_paradigms = set([p for p in self.paradigms
                                     if p.name in lines[0]])

    def initial_save (self):
        state = Character.initial_save (self)
        state.active_paradigms = self.active_paradigms.copy()
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.active_paradigms = state.active_paradigms.copy()

    def reset (self):
        self.resilience_soak = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.resilience_soak = self.resilience_soak
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.resilience_soak = state.resilience_soak

    def set_active_paradigms (self, paradigms):
        self.active_paradigms = set(paradigms)
        if self.game.reporting:
            if not paradigms:
                self.game.report("Zaamassal loses all paradigms")
            elif len(paradigms) == 1:
                self.game.report ("Zaamassal assumes the Paradigm of " + paradigms[0].name)
            else:
                self.game.report ("Zaamassal assumes the following paradigms:")
                for paradigm in paradigms:
                    self.game.report ("   " + paradigm.name)
        self.set_active_cards ()

    def set_active_cards (self):
        Character.set_active_cards (self)
        self.active_cards.extend (list(self.active_paradigms))

    def get_soak (self):
        return self.resilience_soak + Character.get_soak (self)

    def expected_soak(self):
        ret = Character.expected_soak(self)
        if self.resilience in self.active_paradigms:
            ret += 2
        return ret

    # lose all paradigms when stunned
    def stun (self, damage=None):
        stunned = Character.stun (self, damage)
        if stunned:
            self.set_active_paradigms ([])
        return stunned

    def may_ignore_own_movement(self):
        return self.sturdy in self.active_cards

    def set_preferred_range (self):
        Character.set_preferred_range (self)
        if self.fluidity in self.active_paradigms:
            self.preferred_range += 0.5
        if self.distortion in self.active_paradigms:
            self.preferred_range = 3.5
            # distortion also reduces opponent's preferred range
            # paradigm shift can make a mess

    def evaluate (self):
        return Character.evaluate(self) + \
               sum (p.evaluate() for p in self.active_paradigms)

# STYLES AND BASES
# Each style/base is a "constant" object, referenced by players

# Alpha Bases

class Strike (Base):
    minrange = 1
    maxrange = 1
    power = 4
    priority = 3
    stunguard = 5
    preferred_range = 1

class Shot (Base):
    minrange = 1
    maxrange = 4
    power = 3
    priority = 2
    stunguard = 2
    preferred_range = 2.5

class Drive (Base):
    minrange = 1
    maxrange = 1
    power = 3
    priority = 4
    preferred_range = 2 # drive can hit at ranges 1-3
    def before_trigger(self):
        self.me.advance ([1,2])
    ordered_before_trigger = True

class Burst (Base):
    minrange = 2
    maxrange = 3
    power = 3
    priority = 1
    # extra effective range in corner
    preferred_range = 1.5 # explicit so that characters can override it
    def get_preferred_range (self):
        return 2.5 if self.me.position in [0,6] else self.preferred_range
    def start_trigger(self):
        self.me.retreat ([1,2])
    ordered_start_trigger = True
    def evaluation_bonus(self):
        r = self.me.retreat_range()
        if r == 0:
            return -0.3
        if r == 1:
            return 0
        return 0.3

class Grasp (Base):
    minrange = 1
    maxrange = 1
    power = 2
    priority = 5
    preferred_range = 1
    def hit_trigger(self):
        self.me.move_opponent([1])
    ordered_hit_trigger = True
    def evaluation_bonus(self):
        if self.opponent.position in [0,6]:
            return -0.15
        return 0.05

class Dash (Base):
    power = None
    priority = 9
    preferred_range = 2 # dash can switch sides at range 1-3
    is_attack = False
    def after_trigger(self):
        self.me.move([1,2,3], specific_movement_reaction=self.specific_movement_reaction)
    ordered_after_trigger = True
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        if (ordered(old_position, self.opponent.position, self.me.position) and
            not self.opponent.mimics_movement()):
            self.me.set_triggered_dodge()
    # Dash is usually a strong out, that's worth keeping in hand
    discard_penalty = 0.5
    def evaluation_bonus(self):
        if self.opponent.position in [0,6]:
            return -0.3
        return 0.1


# Beta Bases

class Counter (Base):
    alpha_name = 'Strike'
    minrange = 1
    maxrange = 1
    power = 4
    priority = 1
    soak = 1
    stunguard = 4
    preferred_range = 2 # counter can hit at ranges 1-3
    def before_trigger(self):
        self.me.advance ([1,2])
    ordered_before_trigger = True

class Wave (Base):
    alpha_name = 'Shot'
    minrange = 2
    maxrange = 4
    power = 3
    priority = 4
    # extra range in corner
    preferred_range = 2
    def get_preferred_range (self):
        return 3 if self.me.position in [0,6] else self.preferred_range
    def before_trigger(self):
        self.me.retreat ([1,2])
    ordered_before_trigger = True
    def evaluation_bonus(self):
        r = self.me.retreat_range()
        if r == 0:
            return -0.3
        if r == 1:
            return 0
        return 0.3

class Force (Base):
    alpha_name = 'Drive'
    minrange = 1
    maxrange = 1
    power = 3
    priority = 2
    stunguard = 2
    preferred_range = 2 # force can hit at ranges 1-3
    def start_trigger(self):
        self.me.advance ([1,2])
    ordered_start_trigger = True
    def reduce_stunguard (self, stunguard):
        return 0
    def evaluation_bonus(self):
        return 0.03 * self.opponent.expected_stunguard() - 0.03

class Spike (Base):
    alpha_name = 'Burst'
    minrange = 2
    maxrange = 2
    power = 3
    priority = 3
    # range restriction in corner
    preferred_range = 2
    corner_preferred_range = 2.5
    def get_preferred_range (self):
        return (self.corner_preferred_range if self.me.position in [0,6]
                else self.preferred_range)
    def start_trigger(self):
        self.me.move([0, 1])
    ordered_start_trigger = True
    def hit_trigger(self):
        self.me.set_triggered_dodge()
    def evaluation_bonus(self):
        if self.me.position in [0,6]:
            return -0.1
        return 0.2

class Throw (Base):
    alpha_name = 'Grasp'
    minrange = 1
    maxrange = 1
    power = 2
    priority = 5
    preferred_range = 1
    def hit_trigger(self):
        self.me.execute_move(self.opponent, [-1,2])
    ordered_hit_trigger = True

class Parry (Base):
    alpha_name = 'Dash'
    power = None
    priority = 3
    preferred_range = 1.5 # arbitrary average
    is_attack = False
    def end_trigger(self):
        self.me.move([1,2,3])
    ordered_end_trigger = True
    # Opponent has to match priority parity to hit.
    def can_be_hit(self):
        return (1 + self.me.get_priority()
                  + self.opponent.get_priority()) % 2


# Delta Bases

class Stagger (Base):
    alpha_name = 'Strike'
    minrange = 1
    maxrange = 1
    power = 4
    priority = 3
    stunguard = 4
    preferred_range = 1
    def hit_trigger(self):
        self.me.push([1,2])
    # Cheating a little, by checking position when stunning, rather
    # than after the move
    def reduce_stunguard(self, stunguard):
        return 0 if self.opponent.position in (0,6) else stunguard
    ordered_hit_trigger = True
    def evaluation_bonus(self):
        return 0.05 if self.opponent.retreat_range() <= 2 else -0.15

class Leap (Base):
    alpha_name = 'Shot'
    minrange = 1
    maxrange = 1
    power = 3
    priority = 2
    preferred_range = 3
    def can_be_hit(self):
        return self.opponent.attack_range() not in (3,4)
    def before_trigger(self):
        self.me.advance([2,3])
    ordered_before_trigger = True
    def evaluation_bonus(self):
        d = self.game.distance()
        val = -0.15
        if d in (3,4):
            val += 0.5 # For evasion.
        if d == 2 and self.opponent.position in (0,6):
            val -= 0.05 # Hits at range 2, unless can't advance
        if d == 1 and self.opponent.position in (0,1,5,6):
            val += 0.05 # Misses at range 1, unless can't advance.
        return val

class Lunge (Base):
    alpha_name = 'Drive'
    minrange = 1
    maxrange = 2
    power = 3
    priority = 4
    stunguard = 2
    preferred_range = 2
    def before_trigger(self):
        self.me.move([1])
    ordered_before_trigger = True
    def after_trigger(self):
        self.me.move([1])
    ordered_after_trigger = True

class Sweep (Base):
    alpha_name = 'Burst'
    minrange = 1
    maxrange = 2
    power = 4
    priority = 1
    preferred_range = 1.5
    def can_be_hit(self):
        return self.opponent.attack_range() != 1
    def after_trigger(self):
        self.me.move([0,1])
    ordered_after_trigger = True
    def evaluation_bonus(self):
        return 0.5 if self.game.distance() == 1 else -0.25

class Catch (Base):
    alpha_name = 'Grasp'
    minrange = 1
    maxrange = 1
    power = 2
    priority = 5
    preferred_range = 1
    @property
    def power (self):
        return (0 if self.opponent.base.printed_power is None else
                self.opponent.base.printed_power)
    printed_power = None
    def hit_trigger(self):
        self.me.push([1])
    ordered_hit_trigger = True
    def evaluation_bonus(self):
        if self.opponent.position in [0,6]:
            return -0.15
        return 0.05

class Block (Base):
    alpha_name = 'Dash'
    power = None
    preferred_range = 2 # something average
    is_attack = False
    soak = 4
    stunguard = 2
    def end_trigger(self):
        self.me.move([1,2,3])
    ordered_end_trigger = True
#     # Block is usually a strong out, that's worth keeping in hand
#     discard_penalty = 0.5


# Special Action style and bases

class SpecialAction (Style):
    name_override = 'Special'
    order = 10 # presented after regular styles in lists
    # value of keeping this in hand
    value = 5.0
    def get_discard_penalty(self):
        return 0

# Cancel, Pulse and Finisher inherit from this
class SpecialBase (Base):
    pass

# classes for specific finishers inherit from this
class Finisher (SpecialBase):
    order = 10
    # finishers win priority, even vs. characters that win priority
    clash_priority = 0.3
    # returns True if finisher devolves into a cancel
    # normally, that happens above 7 life
    def devolves_into_cancel (self):
        return self.me.life > 7
    # bonus evaluation points for having finisher set up
    # (does not check life<=7 and availability of special action
    def evaluate_setup (self):
        return 0

class Cancel (SpecialBase):
    order = 20

class Pulse (SpecialBase):
    order = 30

#Abarene

class Flytrap(Finisher):
    minrange = 1
    maxrange = 2
    power = 7
    priority = 2
    def reduce_soak(self, soak):
        return 0
    def reduce_stunguard(self, stunguard):
        return 0
    def reveal_trigger(self):
        if self.me.pool:
            if self.game.reporting:
                self.game.report("Abarene discards all Poison tokens")
            self.opponent.add_triggered_priority_bonus(-2 * len(self.me.pool))
            self.me.pool = []
    def evaluate_setup(self):
        return 0.2 * len(self.me.pool) * (self.game.distance() in [1,2])


class HallicrisSnare(Finisher):
    minrange = 4
    maxrange = 7
    power = 1
    priority = 6
    def hit_trigger(self):
        self.me.pull(range(6), specific_movement_reaction=self.specific_movement_reaction)
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        # Get +1 power per pull
        spaces_pulled = abs(self.opponent.position - old_position)
        # If opponent switched sides, actual advance is one less then distance moved
        if ordered (old_position, self.me.position, self.opponent.position):
            spaces_pulled -= 1
        self.me.add_triggered_power_bonus(spaces_pulled)

    def evaluate_setup(self):
        return 0.5 * (self.game.distance() in [4,5,6])

class Tares(Base):
    minrange = 1
    maxrange = 1
    power = 2
    priority = 2
    stunguard = 4
    preferred_range = 3
    def get_maxrange_bonus(self):
        return 4 - len(self.me.pool)

class Lethal(Style):
    power = -1
    def get_power_bonus(self):
        return len(self.me.pool)
    def before_trigger(self):
        self.me.move_directly([0, 6])
    ordered_before_trigger = True
    def end_trigger(self):
        if not self.me.was_token_recovered:
            self.me.recover_tokens(self.opponent)
    def evaluation_bonus(self):
        return 0.05 * len(self.me.pool) - 0.1

class Intoxicating(Style):
    power = 1
    priority = 1
    def hit_trigger(self):
        self.me.recover_tokens(self.me, from_ante_only = True)
    def evaluation_bonus(self):
        if not self.me.pool:
            return -0.1
        else:
            return 0.03

class Barbed(Style):
    minrange = 1
    maxrange = 1
    power = -1
    priority = 1
    preferred_range = 1
    def movement_reaction (self, initiator, mover, old_pos, direct):
        if mover is self.me:
            return
        opp = self.opponent.position
        if opp == old_pos:
            return
        if direct:
            if self.game.distance() == 1:
                self.opponent.lose_life(2)
        else:
            passed = pos_range(opp, old_pos)
            passed.remove(old_pos)
            me = self.me.position
            if me + 1 in passed:
                self.opponent.lose_life(2)
            if me - 1 in passed:
                self.opponent.lose_life(2)

    def hit_trigger(self):
        self.me.pull([1])
    ordered_hit_trigger = True
    def evaluation_bonus(self):
        return 0.1 if (self.game.distance() == 2 and
                       self.opponent.life > 3) else -0.3

class Crippling(Style):
    priority = -2
    preferred_range = 1
    stunguard = 3
    # Block retreats.
    def blocks_opponent_movement (self, initiator, direct):
        if direct or initiator is self.me:
            return set([])
        opp = self.opponent.position
        if opp < self.me.position:
            return set(xrange(opp))
        else:
            return set(xrange(opp+1, 7))
    def before_trigger(self):
        self.me.advance([1,2])
    ordered_before_trigger = True
    def evaluation_bonus(self):
        return -0.2 if self.opponent.position in (0,6) else 0.1

class Pestilent(Style):
    maxrange = 1
    power = -1
    priority = 1
    def after_trigger(self):
        self.me.advance(range(4))
    ordered_after_trigger = True
    # Block switching sides
    def blocks_opponent_movement(self, initiator, direct):
        opp = self.opponent.position
        me = self.me.position
        return set(xrange(me)) if me < opp else set(xrange(me+1, 7))
    def blocks_own_movement(self, initiator):
        if self.opponent.mimics_movement():
            return set()
        opp = self.opponent.position
        me = self.me.position
        return set(xrange(opp)) if opp < me else set(xrange(opp+1, 7))

class Dizziness(Token):
    def give_power_penalty(self):
        return -2
    value = 0.7

class Fatigue(Token):
    def give_priority_penalty(self):
        return -3
    value = 0.9

class Nausea(Token):
    def reduce_soak(self, soak):
        return 0
    def reduce_stunguard(self, stunguard):
        return 0
    @property
    def value(self):
        # If it's not useful, that's a reason to keep it (so you
        # can get others).
        return min(0.6, 0.1 * self.opponent.expected_stunguard() +
                        0.35 * self.opponent.expected_soak())

class Lethargy(Token):
    # move transform handled by Abarene.transform_opponent_movement()
    value = 1.0

#Adjenna

class BasiliskGaze (Finisher):
    minrange = 2
    maxrange = 3
    power = None
    priority = 3
    deals_damage = False
    def reveal_trigger (self):
        self.opponent.add_triggered_priority_bonus(-self.me.petrification)
    def hit_trigger (self):
        raise WinException (self.me.my_number)
    # auto-win on hit, but easy to evade
    def evaluate_setup (self):
        return 2 if self.game.distance() in (2,3) else 0

class Fossilize (Finisher):
    minrange = 1
    maxrange = 5
    soak = 3
    stunguard = 1
    def take_a_hit_trigger (self):
        self.me.petrify()
    def hit_trigger (self):
        self.me.add_triggered_power_bonus (2 * self.me.petrification)
    def evaluate_setup (self):
        return 0.5 * self.me.petrification if self.game.distance() < 6 else 0

class Gaze (Base):
    minrange = 2
    maxrange = 4
    power = 1
    priority = 4
    preferred_range = 3
    def reduce_stunguard (self, stunguard):
        return 0
    def hit_trigger (self):
        self.me.petrify()
        self.me.petrification_is_blocked = True
    # preventing further petrification handled by Adjenna.petrify()
    def evaluation_bonus(self):
        return 0.03 * self.opponent.expected_stunguard() - 0.03

class Alluring (Style):
    power = -1
    priority = -1
    def hit_trigger (self):
        self.me.add_triggered_power_bonus (self.me.petrification)
    def can_be_hit (self):
        return self.opponent.attack_range() != 1
    def evaluation_bonus (self):
        return 0.5 if self.game.distance() == 1 else -0.5

class Arresting (Style):
    power = -1
    priority = -1
    def before_trigger (self):
        self.me.advance ((1,2))
    ordered_before_trigger = True
    @property
    def soak(self):
        return self.me.petrification
    # back push handled by Adjenna.soak_trigger()

class Pacifying (Style):
    power = -1
    def after_trigger (self):
        self.me.move ((1,2))
    ordered_after_trigger = True
    def movement_reaction (self, initiator, mover, old_pos, direct):
        if mover is self.opponent and mover.position != old_pos:
            self.opponent.lose_life (self.me.petrification)

class Irresistible (Style):
    stunguard = 3
    def before_trigger (self):
        if self.opponent.hits_scored:
            self.me.pull ((3,2,1,0))
    @property
    def ordered_before_trigger(self):
        return self.opponent.hits_scored

class Beckoning (Style):
    minrange = 2
    maxrange = 5
    power = -2
    priority = -1
    preferred_range = 3.5
    @property
    def stunguard (self):
        return 5 - self.me.petrification
    def hit_trigger (self):
        self.me.pull ((0,1,2))
    ordered_hit_trigger = True


#Alexian

class HailTheKing (Finisher):
    name_override = "Hail the King"
    minrange = 1
    maxrange = 1
    soak = 6
    def before_trigger(self):
        self.me.advance([0,1,2,3,4,5], specific_movement_reaction=self.specific_movement_reaction)
    ordered_before_trigger = True
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        # Get +1 power per advance.
        spaces_advanced = abs(self.me.position-old_position)
        # If I switched sides, actual advance is one less then distance moved
        if ordered (old_position, self.opponent.position, self.me.position):
            spaces_advanced -= 1
        self.me.add_triggered_power_bonus (spaces_advanced)
    # recording soaked damage handled by Alexian.soak_trigger()
    def get_power_bonus (self):
        return self.me.damage_soaked
    def evaluate_setup (self):
        return 0.25 * (self.game.distance()-1)

class EmpireDivider (Finisher):
    minrange = 1
    maxrange = 2
    power = 7
    priority = 4
    soak = 4
    def has_stun_immunity (self):
        return True
    def hit_trigger (self):
        self.me.give_induced_tokens(3)
    def evaluate_setup (self):
        return 1 if self.game.distance() <= 2 else 0

class Divider (Base):
    minrange = 1
    maxrange = 1
    power = 4
    priority = 2
    soak = 2
    preferred_range = 1
    def has_stun_immunity(self):
        return True
    def end_trigger (self):
        self.me.give_induced_tokens(1)

class Gestalt (Style):
    power = 2
    priority = -1
    stunguard = 3
    preferred_range = 0.5
    def blocks_own_movement (self, initiator):
        return set() if initiator is self.me else set(xrange(7))
    def start_trigger (self):
        self.me.advance ([0,1])
    ordered_start_trigger = True

class Regal (Style):
    power = 1
    priority = 1
    stunguard = 3
    mean_priority = 1.5 # cancels tokens
    # Opponents at range 1 can't retreat
    def blocks_opponent_movement (self, initiator, direct):
        if direct or initiator is self.me or self.game.distance() > 1:
            return set()
        if self.me.position < self.opponent.position:
            return set(xrange(self.opponent.position + 1, 7))
        else:
            return set(xrange(self.opponent.position))
    # Better if opponent has tokens
    def evaluation_bonus (self):
        return 0.1 * (len(self.me.induced_pool) - 1) + \
               (0.5 if self.game.distance() == 1 else -0.5)
    # disabling tokens handled by tokens themselves

class Stalwart (Style):
    power = 1
    priority = -3
    stunguard = 5
    preferred_range = 2 # range is 0-3, but more is more power
    def before_trigger(self):
        # advance up to 3, but don't switch sides
        self.me.advance (range(min(self.game.distance(), 4)), specific_movement_reaction=self.specific_movement_reaction)
    ordered_before_trigger = True
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        spaces_advanced = abs(self.me.position-old_position)
        self.me.add_triggered_power_bonus (spaces_advanced)

class Mighty (Style):
    priority = 1
    soak = 1
    stunguard = 2
    # Ignoring opponent's style range handled by Alexian.ignore_style_range()

class Steeled (Style):
    power = 2
    priority = -1
    preferred_range = 2 # can usually advance 3, and higher range gives more soak
    @property
    def soak (self):
        return self.game.distance() - 1
    # Advance up to 1 space for each damage soaked.
    # Recording soak handled by Alexian.soak_trigger()
    def before_trigger (self):
        self.me.advance(range(self.me.damage_soaked+1))
    ordered_before_trigger = True

class Chivalry (Token):
    def get_power_bonus(self):
        return 1 if self.me.regal not in self.me.active_cards else 0
    def get_priority_bonus(self):
        return 1 if self.me.regal not in self.me.active_cards else 0

#Alumis

class AphoticTerror(Finisher):
    minrange = 1
    maxrange = 1
    power = 7
    priority = 4
    # attack range calculated by Alumis.attack_range()
    def evaluate_setup(self):
        return (1 if abs(self.opponent.position -
                         self.me.shadow_position) == 1 else 0)

class UmbralVice(Finisher):
    minrange = 1
    maxrange = 5
    power = 7
    priority = 1
    def blocks_opponent_movement(self, initiator, direct):
        # Can't move so that shadow moves.
        # I think this is the same as blocking positions not adjacent
        # to shadow.
        # TODO: if shadow "got away" earlier (Pulse), this implementation
        # might block an indirect movement getting back to it.
        s = self.me.shadow_position
        return set(xrange(7)) - set((s-1, s, s+1))

class Dread(Base):
    standard_range = False
    power = 3
    priority = 4
    preferred_range = 2 # arbitrary average
    def special_range_hit(self):
        return ordered(self.me.position,
                       self.opponent.position,
                       self.me.shadow_position)
    def before_trigger(self):
        self.me.pull([1])
    ordered_before_trigger = 1

class Horrific(Style):
    maxrange = 1
    power = 1
    priority = -1
    preferred_range = 0.5
    stunguard = 2
    def get_preferred_range(self):
        m = self.me.position
        o = self.opponent.position
        s = self.me.shadow_position
        if ordered(s,o,m):
            return 1
        if o == s:
            return 1.5
        return 2.5
    def reduce_soak(self, soak):
        return 0
    def reduce_stunguard(self, stunguard):
        return 0
    def start_trigger(self):
        # Pull up to 1 space towards shadow.
        m = self.me.position
        o = self.opponent.position
        s = self.me.shadow_position
        if ordered(s,o,m):
            self.me.push([0,1])
        elif o != s:
            self.me.push([0,1])
    ordered_start_trigger = True
    def evaluation_bonus(self):
        ret = (0.1 * self.opponent.expected_soak() +
               0.03 * self.opponent.expected_stunguard() - 0.08)
        m = self.me.position
        o = self.opponent.position
        s = self.me.shadow_position
        # pushing at start of beat messes with their attack.
        return ret + 0.1 if ordered(s,o,m) else ret - 0.2

class Ghastly(Style):
    priority = 3
    def reveal_trigger(self):
        if self.me.position == self.me.shadow_position:
            self.me.add_triggered_power_bonus(2)
    def evaluation_bonus(self):
        return 0.15 if self.me.shadow_position == self.me.position else -0.05

class Arcane(Style):
    maxrange = 1
    priority = -2
    preferred_range = 1
    stunguard = 2
    def start_trigger(self):
        if self.me.position == self.me.shadow_position:
            self.me.arcane_soak = 2
            if self.game.reporting:
                self.game.report("Alumis gains soak 2")
        self.me.move([1])
        if self.me.position == self.me.shadow_position:
            self.me.arcane_soak = 2
            if self.game.reporting:
                self.game.report("Alumis gains soak 2")
    ordered_start_trigger = True
    @ property # for eval
    def soak(self):
        d = abs(self.me.position - self.me.shadow_position)
        if d == 0:
            return 1
        if d == 1 and self.opponent.position != self.me.shadow_position:
            return 1
        if d == 2 and self.game.distance() == 1:
            return 1
        return 0
    def get_soak(self):
        return 0 # Character adds it
    def evaluation_bonus(self):
        return 0.1 if self.soak else -0.05

class Sinister(Style):
    minrange = 1
    maxrange = 3
    def get_preferred_range(self):
        return 1 if self.me.retreat_range() >= 2 else 2
    def start_trigger(self):
        self.me.retreat([2])
    ordered_start_trigger = True
    def hit_trigger(self):
        # May move to shadow marker
        self.me.move_directly([self.me.position, self.me.shadow_position])
    ordered_hit_trigger = True
    def evaluation_bonus(self):
        return (-0.05 if self.opponent.position ==
                         self.me.shadow_position else 0.02)

class Tenebrous(Style):
    maxrange = 2
    preferred_range = 1
    def blocks_opponent_movement(self, initiator, direct):
        me = self.me.position
        opp = self.opponent.position
        shadow = self.me.shadow_position
        # Block indirect movement across shadow.
        # (this doesn't work right for opponents with minimum movement
        #  beyond shadow - it stops all movement instead of stopping
        #  at shadow).
        if direct or shadow in (me, opp):
            return set()
        if shadow > self.opponent.position:
            return set(xrange(shadow+1, 7))
        else:
            return set(xrange(shadow))
    def movement_reaction(self, initiator, mover, old_position, direct):
        if mover is self.me:
            return
        shadow = self.me.shadow_position
        if old_position != shadow and self.opponent.position == shadow:
            self.opponent.lose_life(2)

    def evaluation_bonus(self):
        return (-0.1 if self.me.shadow_position in
                    (self.me.position, self.opponent.position) else 0.05)

#Arec

class UncannyOblivion(Finisher):
    minrange = 2
    maxrange = 2
    power = 2
    priority = 8
    def hit_trigger(self):
        # TODO: actually, against an opposing finisher, remove
        # the base used to invoke it.
        if self.opponent.style is not self.opponent.special_action:
            self.opponent.remove_card_from_game(self.opponent.style)
            self.opponent.remove_card_from_game(self.opponent.base)
    def evaluate_setup(self):
        return 1.0 if self.game.distance() == 2 else 0

# Not Implemented
class DominatePerson(Finisher):
    pass

class Hex(Base):
    standard_range = False
    power = 3
    priority = 3
    preferred_range = 3
    def special_range_hit(self):
        return True
    def hit_trigger(self):
        self.me.move_opponent([0,1])
        self.me.recover_tokens()
    ordered_hit_trigger = True
    # Not stunning handled by Arec.damage_trigger()
    def evaluation_bonus(self):
        return 0.05 if len(self.me.pool) < 4 else -0.1

class Phantom(Style):
    maxrange = 2
    power = -1
    priority = 1
    preferred_range = 1
    def reduce_soak(self, soak):
        return 0
    def end_trigger(self):
        self.me.recover_tokens()
    def evaluation_bonus(self):
        return ((0.05 if len(self.me.pool) < 4 else -0.1) +
                0.1 * self.opponent.expected_soak())

class Perceptional(Style):
    minrange = 1
    maxrange = 2
    priority = 1
    preferred_range = 1.5
    def start_trigger(self):
        # fork to decide which token is anted (or none)
        if self.me.can_spend(1):
            n_options = len(self.me.pool) + 1
            prompt = "Select a token to ante with Perceptional:"
            options = [t.name for t in self.me.pool] + ["None"]
            token_number = self.game.make_fork(n_options, self.me,
                                               prompt, options)
            if token_number < len (self.me.pool):
                self.me.ante_token(self.me.pool[token_number])
        # If Fear spent, block further start triggers by Arec.
        if self.me.fear in self.me.active_cards:
            return True
    # Order only matters if you can spend a Fear token to block
    # your own Burst.
    @property
    def ordered_start_trigger(self):
        return self.me.can_spend(self.me.fear)
    def hit_trigger(self):
        self.me.pull([0,1])
    ordered_hit_trigger = True
    def evaluation_bonus(self):
        return 0.1 if len(self.me.pool) > 0 else -0.2

class Returning(Style):
    maxrange = 1
    priority = 1
    preferred_range = 0.5
    clash_priority = 0.1
    mean_priority = 1.5
    def start_trigger (self):
        # Bases from different standard sets equate by color.
        if self.me.base.alpha_name == self.opponent.base.alpha_name:
            self.opponent.stun()
            self.me.returning_range_triggered = True

class Mirrored(Style):
    minrange = 1
    maxrange = 2
    power = 1
    preferred_range = 1.5
    def can_be_hit(self):
        return self.opponent.attack_range() != 2
    def end_trigger(self):
        # Fork to decide clone location.
        positions = [i for i in xrange(7) if i not in (self.me.position,
                                                       self.opponent.position)]
        prompt = "Place clone:"
        options = []
        if self.me.is_user and self.game.interactive_mode:
            for pos in positions:
                self.me.clone_position = pos
                options.append (self.me.get_board_addendum())
        self.me.clone_position = positions [
            self.game.make_fork(len(positions), self.me, prompt, options)]
        if self.game.reporting:
            self.game.report ("Arec places clone:")
            for s in self.game.get_board():
                self.game.report (s)
    ordered_end_trigger = True
    def evaluation_bonus(self):
        return 0.4 if self.game.distance() == 2 else -0.2

class Manipulative(Style):
    power = 1
    def hit_trigger(self):
        available_bases = list(set(self.opponent.bases)
                               - self.opponent.discard[0]
                               - self.opponent.discard[1]
                               - self.opponent.discard[2])
        ans = self.game.make_fork(len(available_bases), self.me,
                "Choose base in %s's hand:" % self.opponent,
                [base.name for base in available_bases])
        self.me.manipulative_base = available_bases[ans]
        if self.game.reporting:
            self.game.report("Arec chooses %s's base: %s" %
                             (self.opponent, self.me.manipulative_base))
    def damage_trigger(self, damage):
        self.me.recover_tokens()
    def evaluation_bonus(self):
        return 0.05 if len(self.me.pool) < 4 else -0.1

# blocks start triggers
class Fear(Token):
    value = 0.5

# blocks before/after triggers
class Hesitation(Token):
    value = 0.8

# blocks hit/damage triggers
class Mercy(Token):
    value = 0.4

# blocks power/priority bonuses
class Recklessness(Token):
    value = 0.6


#Aria

# Not implemented
class SynchroMerge (Finisher):
    minrange = 2
    maxrange = 3
    power = 3
    priority = 6
    def start_trigger (self):
        raise NotImplementedError

class LaserLattice (Finisher):
    minrange = 1
    maxrnage = 2
    power = 2
    priority = 6
    def hit_trigger (self):
        self.me.move_opponent([1])
    ordered_hit_trigger = True
    def after_trigger (self):
        droids = [droid for droid in self.me.droids
                  if droid.position is not None
                  and not droid.has_attacked]
        if not droids:
            return
        ans = self.game.make_fork(len(droids), self.me,
                                  "Choose droid that attacks next:",
                                  droids)
        droid = droids[ans]
        self.me.attacker = droid
        droid.has_attacked = True
        if self.game.reporting:
            self.game.report("Additional attack by %s droid" % droid)
        self.game.activate(self.me)
    ordered_after_trigger = True
    def evaluate_setup (self):
        return 0.5 if self.game.distance() <= 2 else 0

class Reconfiguration (Base):
    standard_range = False
    power = 2
    priority = 3
    preferred_range = 2 # average, so that it doesn't change things too much
    def special_range_hit (self):
        return self.opponent.position in [droid.position
                                          for droid in self.me.droids]
    def before_trigger (self):
        self.me.move_to_unoccupied()
        droids = [d for d in self.me.droids if d.position is None]
        if droids:
            ans = self.game.make_fork(len(droids), self.me,
                                      "Choose droid to be added to board:",
                                      droids)
            droid = droids[ans]
            occupied = (self.me.position, self.opponent.position)
            positions = [pos for pos in xrange(7) if pos not in occupied]
            options = []
            if self.me.is_user and self.game.interactive_mode:
                for pos in positions:
                    line = ['.'] * 7
                    line[pos] = droid.initial
                    options.append(''.join(line))
            ans = self.game.make_fork(len(positions), self.me,
                                      "Choose a space for your %s droid"%droid,
                                      options)
            droid.position = positions[ans]
            if self.game.reporting:
                self.game.report ("Aria adds her %s droid:" % droid)
                for s in self.game.get_board():
                    self.game.report (s)
    @property
    def ordered_before_trigger(self):
        return any([d.position is None for d in self.me.droids])
    def evaluation_bonus (self):
        return (0.25 if self.opponent.position in [droid.position
                                                   for droid in self.me.droids]
                else -0.25)


class Photovoltaic (Style):
    maxrange = 2
    power = -1
    priority = 1
    preferred_range = 1
    def hit_trigger (self):
        droids = [d for d in self.me.droids if d.position is not None]
        for droid in droids:
            if self.game.make_fork (2, self.me,
                                    "Remove %s droid from the board?" % droid,
                                    ["No", "Yes"]):
                droid.position = None
                if self.game.reporting:
                    self.game.report ("Aria removes her %s droid:" % droid)
                    for s in self.game.get_board():
                        self.game.report (s)
                self.me.add_triggered_power_bonus(1)
    def after_trigger (self):
        droids = [d for d in self.me.droids if d.position is None]
        if droids:
            ans = self.game.make_fork(len(droids), self.me,
                                      "Choose droid to add to Aria's space:",
                                      droids)
            droid = droids[ans]
            droid.position = self.me.position
            if self.game.reporting:
                self.game.report ("Aria adds her %s droid:" % droid)
                for s in self.game.get_board():
                    self.game.report (s)
    @property
    def ordered_after_trigger(self):
        return any([d.position is None for d in self.me.droids])

class Ionic (Style):
    maxrange = 1
    priority = 1
    # maximum range is 2 if I can pull opponent towards me
    def get_preferred_range (self):
        if self.me.magnetron.position is None:
            return 0.5
        if self.opponent.position == self.me.magnetron.position or \
           ordered (self.me.position,
                    self.opponent.position,
                    self.me.magnetron.position):
            return 0.5
        else:
            return 1
    # Opponent's base provides 3 priority instead of printed value.
    def give_priority_penalty(self):
        # This implementation is hacky, but should work as long as
        # opponents don't reference their own printed priority
        return 3 - self.opponent.base.priority
    # Pull opponent up to 1 space towards Magnetron.
    def start_trigger (self):
        # Can't pull if no Magnetron, or if Magnetron on opponent.
        if self.me.magnetron.position in [None, self.opponent.position]:
            return
        if ordered (self.me.position,
                    self.opponent.position,
                    self.me.magnetron.position):
            self.me.push([1,0])
        else:
            self.me.pull([0,1])
    @property
    def ordered_start_trigger(self):
        return self.me.magnetron.position is not None
    def evaluation_bonus (self):
        return (0.1 if self.me.magnetron.position not in (None,
                                                          self.opponent.position)
                else -0.1)

class Laser (Style):
    def end_trigger (self):
        if self.me.turret.position is not None and \
           abs (self.opponent.position - self.me.turret.position) == 3:
            self.opponent.lose_life(3)
    @property
    def ordered_end_trigger(self):
        return self.me.turret.position is not None
    def evaluation_bonus (self):
        value = 0
        her = self.opponent.position
        turret = self.me.turret.position
        discard = self.me.discard
        reconfig = self.me.unique_base not in (discard[1]|discard[2])
        # add value if opponent adjacent to droid - that allows many bases
        adjacent = set((her - 1, her + 1))
        droids = set([droid.position for droid in self.me.droids])
        value += (0.25 if adjacent & droids else -0.25)
        # add value if turret in range 3 of opponent
        value += (0.25 if turret is not None and abs(her - turret) == 3
                  else -0.05)
        # add value if turret outside and you have Reconfiguration
        value += (0.1 if turret is None and reconfig else -0.1)
        return value
    # Choice of attack origin handled by Aria.before_trigger()

class Catalyst (Style):
    minrange = 1
    maxrange = 2
    power = -1
    priority = 1
    preferred_range = 1.5
    mean_priority = 2
    def can_be_hit (self):
        return self.me.dampening.position is None or \
               not ordered (self.me.position,
                            self.me.dampening.position,
                            self.opponent.position)
    def reveal_trigger (self):
        if self.me.position == self.me.magnetron.position:
            self.me.add_triggered_priority_bonus (4)
    def evaluation_bonus (self):
        value = 0
        me = self.me.position
        her = self.opponent.position
        dampening = self.me.dampening.position
        discard = self.me.discard
        reconfig = self.me.unique_base not in (discard[1]|discard[2])
        # add value if dampening is between you and opponent
        value += (0.5 if dampening is not None and
                         ordered (me, dampening, her)
                  else -0.2)
        # add value if you're on magnetron
        value += (0.4 if me == self.me.magnetron.position else -0.1)
        # add value if dampening outside and you have Reconfiguration
        value += (0.1 if dampening is None and reconfig else -0.1)
        return value

class Dimensional (Style):
    minrange = 2
    maxrange = 2
    power = -1
    priority = 4
    preferred_range = 2
    def reveal_trigger (self):
        if self.game.distance() == 3:
            self.me.add_triggered_power_bonus(2)
        else:
            self.me.dimensional_no_stun = True
    def end_trigger (self):
        self.me.move_to_unoccupied()
    ordered_end_trigger = True
    def evaluation_bonus (self):
        return 0.5 if self.game.distance() == 3 else -0.2

class Droid (object):
    def __init__ (self):
        self.name = self.__class__.__name__
        self.initial = self.name[0].lower()
        self.position = None
    def __str__ (self):
        return self.name

class Dampening (Droid):
    pass

class Magnetron (Droid):
    pass

class Turret (Droid):
    pass

#Baenvier

class Spellbreak(Finisher):
    minrange = 1
    maxrange = 2
    power = 7
    def has_stun_immunity(self):
        return True
    def blocks_own_movement(self, initiator):
        return set(xrange(7)) if initiator is self.opponent else set()
    def before_trigger(self):
        self.me.advance([1])
    ordered_before_trigger = True
    def hit_trigger(self):
        self.me.gain_life(4)
    def evaluate_setup(self):
        return 0.2 * self.me.life if self.game.distance() <= 3 else 0

# Not Implemented.
# Need to figure out how to shut down opponent's style without removing it
# (i.e., without changing opp.style to opp.null_style).
class Felfire(Finisher):
    minrange = 1
    maxrange = 1
    power = 7
    priority = 2
    def before_trigger(self):
        self.me.advance(range(4))
    def after_trigger(self):
        self.me.move_opponent([1])

class Spire(Base):
    minrange = 3
    maxrange = 5
    power = 4
    priority = 4
    preferred_range = 3.5 #subtracting average style preferred range (0.5)
    def hit_trigger(self):
        opp = self.opponent.position
        self.me.move_directly([opp-1, opp+1])
    # Ignoring style range handled by Baenvier.ignore_style_range

class Runic(Style):
    maxrange = 1
    power = 1
    priority = -1
    preferred_range = 0.5
    stunguard = 2
    # Transfering bonuses handled by Baenvier.blocks_power_bonuses()
    # and Baenvier.take_a_hit_trigger()

class Curse(Style):
    maxrange = 1
    power = 2
    priority = 1
    preferred_range = 0.5
    def after_trigger(self):
        self.opponent.add_triggered_power_bonus(4)

class Spellforge(Style):
    priority = 2
    def hit_trigger(self):
        self.opponent.discard[0].discard(self.opponent.style)
        if self.game.reporting:
            self.game.report("%s is returned to %s's hand" %
                             (self.opponent.style, self.opponent))
        self.opponent.style = self.opponent.null_style
        self.opponent.set_active_cards()
    ordered_hit_trigger = True

class Crescent(Style):
    minrange = 1
    maxrange = 2
    priority = 1
    preferred_range = 1.5
    stunguard = 3
    def hit_trigger(self):
        self.me.move_opponent([2])

class Destruction(Style):
    power = 1
    priority = 1
    def reduce_soak(self, soak):
        return 0
    def reduce_stunguard(self, stunguard):
        return 0
    def evaluation_bonus(self):
        return (0.1 * self.opponent.expected_soak() +
                0.03 * self.opponent.expected_stunguard()) - 0.1

class SpellEater(Token):
    pass

#Borneo

class MamaMazzaroth(Finisher):
    is_attack = False
    priority = 5
    def start_trigger(self):
        # Assuming it's 3 beats including current one.
        self.game.current_beat = 13
        self.me.mazzaroth_active = True
    def after_trigger(self):
        self.me.move([1,2,3])
    ordered_after_trigger = True

class ClawBorneo (Base):
    name_override = 'Claw'
    minrange = 1
    maxrange = 2
    power = 3
    priority = 4
    preferred_range = 2.5
    def before_trigger (self):
        self.me.advance ((1,2))
    ordered_before_trigger = True

class DividerBorneo (Base):
    name_override = 'Divider'
    minrange = 1
    maxrange = 1
    power = 4
    priority = 2
    soak = 2
    preferred_range = 1
    mean_priority = 1
    def has_stun_immunity(self):
        return True
    def end_trigger (self):
        self.me.opponent_power_priority_status_effect.activate()

class HexBorneo(Base):
    name_override = 'Hex'
    standard_range = False
    power = 3
    priority = 3
    preferred_range = 3
    def special_range_hit(self):
        return True
    def hit_trigger(self):
        self.me.move_opponent([0,1])
        self.me.triggers_blocked = True
    ordered_hit_trigger = True
    # Not stunning handled by Borneo.damage_trigger()

class Knuckle(Base):
    minrange = 1
    maxrange = 1
    power = 2
    priority = 5
    preferred_range = 1
    def hit_trigger(self):
        self.me.move([1])
    ordered_hit_trigger = True
    def reduce_soak(self, soak):
        return 0

class MasterPlan(Base):
    standard_range = False
    power = 4
    prefered_range = 3.5
    def special_range_hit(self):
        return self.opponent.moved
    def has_stun_immunity(self):
        return True

class SceneShiftBorneo(Base):
    name_override = 'Scene Shift'
    standard_range = False
    power = 3
    priority = 4
    preferred_range = 2.5
    def special_range_hit(self):
        return self.me.switched_sides
    def before_trigger(self):
        self.me.move(xrange(6), specific_movement_reaction=self.specific_movement_reaction)
    ordered_before_trigger = True
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        if (ordered(old_position, self.opponent.position, self.me.position) and
            not self.opponent.mimics_movement()):
            self.me.switched_sides = True
    def evaluation_bonus(self):
        return -0.4 if self.opponent.position in [0,6] else 0.2

class Weaksauce(Style):
    power = -2
    priority = 1
    def after_trigger(self):
        self.me.retreat([0,1,2])
    ordered_after_trigger = True
    def evaluation_bonus(self):
        ret = self.me.retreat_range()
        if ret == 0:
            return -0.25
        if ret == 1:
            return -0.05
        return 0.15

class Clumsy(Style):
    maxrange = 1
    power = 1
    preferred_range = 0.5
    # Not stunning handled by Borneo.damage_trigger()

class Pathetic(Style):
    maxrange = 2
    power = -1
    preferred_range = 1
    def reduce_stunguard(self, stunguard):
        return 0

class Petulant(Style):
    priority = -2
    stunguard = 3
    def blocks_own_movement(self, initiator):
        return set() if initiator is self.me else set(xrange(7))

class Slippery(Style):
    power = -1
    preferred_range = 1
    def before_trigger(self):
        self.me.advance([1,2])
    ordered_before_trigger = True

#Byron

class SoulTrap (Finisher):
    minrange = 1
    maxrange = 3
    power = 7
    priority = 2
    def has_stun_immunity (self):
        return True
    def hit_trigger (self):
        self.me.recover_counters(1)
    def evaluate_setup (self):
        return 0.1 * min(7, self.opponent.life) if self.game.distance() in (1,2,3) else 0

class SoulGate (Finisher):
    minrange = 3
    maxrnage = 4
    power = 25
    def can_be_hit (self):
        return self.opponent.moved
    def evaluate_setup (self):
        return 0.1 * self.opponent.life if self.game.distance() in (3,4) else 0

class Smoke (Base):
    minrange = 1
    maxrange = 5
    power = 2
    priority = 3
    preferred_range = 2.5 # ignores style range
    def hit_trigger (self):
        self.me.move_opponent((0,1))
    ordered_hit_trigger = True
    # Ignoring style range handled by Byron.ignore_style_range

class Soulless (Style):
    power = 1
    priority = -4
    def has_stun_immunity (self):
        return True
    @property
    def soak (self):
        return 5 - self.me.counters
    def reveal_trigger (self):
        if self.game.distance() == 1:
            self.opponent.lose_life (2)
            self.me.add_triggered_power_bonus (2)
    def end_trigger (self):
        if self.me.counters_discarded == 0:
            self.me.discard_counters(1)
    def evaluation_bonus (self):
        soak_bonus  = 0.2 * (self.soak - 2)
        range_bonus = 0.3 if self.game.distance() == 1 else -0.6
        return soak_bonus + range_bonus

class Deathless (Style):
    minrange = 1
    maxrange = 3
    power = 2
    priority = -1
    preferred_range = 2
    def damage_trigger (self, damage):
        if self.me.attack_range() == 4:
            self.me.move_opponent_to_unoccupied()
    ordered_damage_trigger = True
    def get_damage_cap (self):
        return 1

class Heartless (Style):
    minrange = 1
    maxrange = 2
    power = 1
    priority = 1
    preferred_range = 1.5
    def hit_trigger (self):
        if self.me.attack_range() == 3:
            self.me.heartless_ignore_soak = True
            self.opponent.lose_life (self.opponent.get_soak())
    ordered_hit_trigger = True
    def damage_trigger (self, damage):
        self.me.heartless_soak = damage
    # for evaluation
    soak = 2
    def get_soak (self):
        return 0
    def evaluation_bonus(self):
        return (0.2 * self.opponent.expected_soak()
                if self.game.distance() == 3 else 0)

class Faceless (Style):
    maxrange = 1
    power = 1
    preferred_range = 0.5
    mean_priority = 1
    def hit_trigger (self):
        self.me.set_triggered_dodge()
    def damage_trigger (self, damage):
        if self.me.attack_range() == 2:
            self.me.priority_bonus_status_effect.activate(damage)
    ordered_damage_trigger = True

class Breathless (Style):
    minrange = 3
    maxrange = 4
    preferred_range = 3.5
    def start_trigger (self):
        # may move to other side of opponent
        opp = self.opponent.position
        me = self.me.position
        dests = range(opp) if me > opp else range(opp+1, 7)
        dests.append(me)
        self.me.move_directly(dests)
    ordered_start_trigger = True
    def hit_trigger (self):
        if self.game.distance() == 5:
            self.opponent.add_triggered_power_bonus(-3)
            self.opponent.lose_life(2)
    ordered_hit_trigger = True
    def evaluation_bonus (self):
        if self.game.distance() == 5:
            return 0.5
        opp = self.opponent.position
        me = self.me.position
        spaces_on_other_side = opp if me > opp else 6-opp
        if spaces_on_other_side < 3:
            return -0.25
        if spaces_on_other_side >= 5:
            return 0.75
        return 0.25

#Cadenza

class RocketPress (Finisher):
    minrange = 1
    maxrange = 1
    power = 8
    soak = 3
    def has_stun_immunity (self):
        return True
    def before_trigger (self):
        self.me.advance ((2,3,4,5))
    ordered_before_trigger = True
    def evaluate_setup (self):
        return 2 if self.game.distance() > 1 or self.opponent.position in (0,6)\
               else 0

class FeedbackField (Finisher):
    minrange = 1
    maxrange = 2
    power = 1
    soak = 5
    def hit_trigger (self):
        self.me.add_triggered_power_bonus(self.me.damage_soaked)

class Press (Base):
    minrange = 1
    maxrange = 2
    power = 1
    stunguard = 6
    preferred_range = 1.5
    def get_power_bonus (self):
        return self.me.damage_taken

class Battery (Style):
    power = 1
    priority = -1
    mean_priority = 3
    def end_trigger (self):
        self.me.priority_bonus_status_effect.activate(4)

class Clockwork (Style):
    power = 3
    priority = -3
    soak = 3
    # Clockwork is probably the strongest style.
    discard_penalty = 0.5

class Hydraulic (Style):
    power = 2
    priority = -1
    soak = 1
    preferred_range = 0.5 # adds 1 to max range, but usually not to minrange
    def before_trigger (self):
        self.me.advance ((1,))
    ordered_before_trigger = True

class Grapnel (Style):
    minrange = 2
    maxrange = 4
    preferred_range = 3
    def hit_trigger (self):
        self.me.pull ((0,1,2,3))
    ordered_hit_trigger = True
    # discard bonus to encourage use of weak style
    discard_penalty = -0.5

class Mechanical (Style):
    power = 2
    priority = -2
    def end_trigger (self):
        self.me.advance ((0,1,2,3))
    ordered_end_trigger = True
    # discard bonus to encourage use of weak style
    discard_penalty = -0.5

# only ante effect.  spending effect handled by character
class IronBody (Token):
    def has_stun_immunity (self):
        return True

#Cesar

class Level4Protocol (Finisher):
    minrange = 1
    maxrange = 1
    power = 7
    priority = 11
    soak = 4
    def reveal_trigger (self):
        if self.me.was_stunned and self.game.reporting:
            self.game.report ("Cesar is no longer stunned.")
        self.me.was_stunned = False
    def evaluate_setup (self):
        return 1 if self.game.distance() == 1 else 0

# Not implemented - not sure how to evaluate it
class EndlessVigil (Finisher):
    is_attack = False
    power = None
    priority = 8
    soak = 4
    def end_trigger (self):
        raise NotImplementedError()

class Suppression (Base):
    minrange = 1
    maxrange = 1
    power = 2
    priority = 2
    preferred_range = 1
    # Opponents cannot move past you.
    def blocks_opponent_movement(self, initiator, direct):
        if direct:
            return set()
        else:
            return set([self.me.position])
    def can_be_hit (self):
        return self.opponent.attack_range() < 3
    def end_trigger (self):
        self.me.move((0,1,2,3))
    ordered_end_trigger = True
    def evaluation_bonus (self):
        return 0.5 if self.game.distance() <= 2 else -0.5

class Phalanx (Style):
    maxrange = 1
    priority = -2
    stunguard = 1
    preferred_range = 2
    def before_trigger (self):
        if self.opponent.hits_scored:
            self.me.advance (range(self.me.damage_taken+1))
    @property
    def ordered_before_trigger(self):
        return self.opponent.hits_scored

class Unstoppable (Style):
    priority = -2
    # I assume this requires a move to actually happen.
    # (FAQ disagrees).
    def movement_reaction(self, initiator, mover, old_pos, direct):
        if initiator is self.opponent and mover.position != old_pos:
            self.me.advance((1,2))
    # blocking hit/damage triggers handled by Cesar

class Fueled (Style):
    power = 1
    priority = -1
    preferred_range = 1
    def start_trigger(self):
        # Losing immunity handled by Cesar.has_stun_immunity().
        # Stunning handled by Cesar.gain_threat_level().
        self.me.gain_threat_level()
    def before_trigger(self):
        self.me.advance([1,2])
    ordered_before_trigger = True

class Bulwark (Style):
    priority = -2
    stunguard = 3
    def blocks_own_movement (self, initiator):
        return set() if initiator is self.me else set(xrange(7))
    def after_trigger (self):
        self.me.move ([0,1,2,3])
    ordered_after_trigger = True

class Inevitable (Style):
    stunguard = 3
    def hit_trigger (self):
        self.opponent.stun()
        self.me.power_penalty_status_effect.activate(-3)

#Claus

class AutumnsAdvance(Finisher):
    name_override = "Autumn's Advance"
    minrange = 1
    maxrange = 2
    power = 3
    priority = 6
    def before_trigger(self):
        self.me.advance(range(7))
    def hit_trigger(self):
        if self.opponent.position in [0,6]:
            self.me.add_triggered_power_bonus(3)
    def evaluate_setup(self):
        return 0.5 if self.opponent.position in [0,1,5,6] else 0

class Tempest(Base):
    minrange = 1
    maxrange = 3
    power = 1
    priority = 4
    preferred_range = 2.5
    def start_trigger(self):
        self.me.advance(range(4))
    ordered_start_trigger = True
    def evaluation_bonus(self):
        push = min(4 - self.game.distance(),
                   self.opponent.retreat_range())
        return -0.15 + 0.15 * max(push, 0)

class Hurricane(Style):
    power = 3
    mean_priority = -4
    def end_trigger(self):
        self.me.priority_bonus_status_effect.activate(-4)

class Tailwind(Style):
    maxrange = 1
    power = 1
    preferred_range = 1.5
    def start_trigger(self):
        self.me.advance([1,2])
    ordered_start_trigger = True
    def evaluation_bonus(self):
        push = min(3 - self.game.distance(),
                   self.opponent.retreat_range())
        return -0.2 + 0.25 * max(push, 0)

class Blast(Style):
    maxrange = 1
    priority = -3
    stunguard = 2
    preferred_range = 1
    def start_trigger(self):
        self.me.advance([1])
    ordered_start_trigger = True
    def evaluation_bonus(self):
        push = min(2 - self.game.distance(),
                   self.opponent.retreat_range())
        return -0.1 + 0.25 * max(push, 0)

class Cyclone(Style):
    power = 1
    priority = 1
    def end_trigger(self):
        if self.opponent.position in [0,6] and self.game.distance() == 1:
            self.opponent.lose_life(3)
    ordered_end_trigger = True

class Downdraft(Style):
    priority = -1
    def can_be_hit(self):
        return self.opponent.attack_range() < 3
    def evaluation_bonus(self):
        # Difference isn't very high, because can easily generate
        # the distance with Dash/Drive
        return 0.2 if self.game.distance() >= 3 else -0.2


#Clinhyde

class VitalSilverInfusion (Finisher):
    is_attack = False
    power = None
    priority = 4
    def after_trigger (self):
        self.me.gain_life (10)
        self.me.vital_silver_activated = True

class RitherwhyteInfusionClinhyde(Finisher):
    name_override = "Ritherwhyte Infusion"
    is_attack = False
    power = None
    priority = 4
    def after_trigger (self):
        self.me.ritherwhyte_activated = True

class Frenzy (Base):
    minrange = 1
    maxrange = 1
    priority = 3
    preferred_range = 1.5
    @property
    def power (self):
        return len(self.me.active_packs) + 1
    printed_power = None
    def start_trigger (self):
        self.me.advance ([1])
    ordered_start_trigger = True
    def after_trigger (self):
        active = self.me.active_packs
        if active:
            combos = sum((list(itertools.combinations (active, n))
                          for n in range(len(active)+1)),[])
            prompt = "Choose Stim Packs to deactivate:"
            options = [', '.join([pack.name for pack in combo])
                       for combo in combos]
            options[0] = 'None'
            ans = self.game.make_fork (len(combos), self.me, prompt, options)
            for pack in combos[ans]:
                pack.deactivate()

class Toxic (Style):
    priority = 1
    mean_priority = 1.5 # Double Ehrlite
    def get_preferred_range (self):
        # we might activate a stim before start of beat.
        projected_stims = min (3, len(self.me.active_packs)+0.5)
        return (projected_stims + 1) / 2.0
    # Doesn't directly give stunguard, but might add to hylatine
    def get_stunguard(self):
        return 0
    @property
    def stunguard(self):
        return 2 if self.me.hylatine in self.me.active_packs else 1
    def start_trigger (self):
        for unused in xrange(len(self.me.active_packs) + 1):
            self.me.advance([1])
    ordered_start_trigger = True
    def evaluation_bonus (self):
        # if we have 2 stims at end of beat, we can easily go to 3 next beat
        n = min (2, len(self.me.active_packs))
        return 0.2 * (n-1)
    # doubling of stims handled by stims themselves

class Shock (Style):
    power = -1
    priority = 1
    def get_preferred_range(self):
        return 3 if self.me.retreat_range() <= 2 else 0
    def before_trigger (self):
        self.me.move ([0,3,4,5])
    ordered_before_trigger = True
    def damage_trigger (self, damage):
        if self.game.distance() == 1:
            self.opponent.stun()
    ordered_damage_trigger = True

class Diffusion (Style):
    maxrange = 1
    power = -1
    priority = 1
    preferred_range = 0.5
    def hit_trigger (self):
        self.me.push ([2,1,0])
        if self.opponent.position in (0,6):
            self.opponent.lose_life (2)
    ordered_hit_trigger = True
    def damage_trigger (self, damage):
        active = list(self.me.active_packs)
        if active:
            prompt = "Choose a Stim Pack to deactivate:"
            options = [p.name for p in active]
            ans = self.game.make_fork (len(active), self.me, prompt, options)
            active[ans].deactivate()
    def end_trigger (self):
        # advance until adjacent to opponent
        self.me.advance ([self.game.distance()-1])
    ordered_end_trigger = True

class Gravity (Style):
    minrange = 1
    maxrange = 3
    preferred_range = 2
    def hit_trigger (self):
        self.me.move_opponent_to_unoccupied()
    ordered_hit_trigger = True
    def blocks_own_movement (self, initiator):
        # blocking self initiated movement handled by Clinhyde.execute_move()
        if initiator is self.me:
            return set()
        if self.game.make_fork(2, self.me,
            "Block %s's attempt to move you?" % self.opponent,
            ["No", "Yes"]):
            return set(xrange(7)) #block
        else:
            return set() #don't block

class Phase (Style):
    maxrange = 2
    power = 1
    priority = -3
    preferred_range = 1
    def can_be_hit (self):
        return not self.opponent.moved
    def can_hit (self):
        return self.opponent.moved

class StimPack (Card):
    def activate (self):
        if self not in self.me.active_packs:
            self.me.active_packs.add (self)
            self.me.set_active_cards ()
            if self.game.reporting:
                self.game.report ("Clinhyde activates " + self.name)
    def deactivate (self):
        if self in self.me.active_packs:
            self.me.active_packs.discard (self)
            self.me.set_active_cards ()
            if self.game.reporting:
                self.game.report ("Clinhyde de-activates " + self.name)

# These are the "while active" powers.  Clinhyde handles the activations.
class Crizma (StimPack):
    def get_power_bonus (self):
        return 2 if self.me.toxic in self.me.active_cards else 1
class Ehrlite (StimPack):
    def get_priority_bonus (self):
        return 2 if self.me.toxic in self.me.active_cards else 1
class Hylatine (StimPack):
    def get_stunguard (self):
        return 4 if self.me.toxic in self.me.active_cards else 2

#Clive

# Not implemented, fork is too large.
class SystemReset (Finisher):
    is_attack = False
    power = None
    priority = 5
    soak = 4
    def after_trigger (self):
        raise NotImplementedError()

class SystemShock (Finisher):
    minrange = 1
    maxrange = 4
    power = 9
    priority = 7
    # Only activates if all modules are discarded
    def devolves_into_cancel (self):
        return (Finisher.devolves_into_cancel(self) or
                self.me.module_stack or self.me.active_modules)
    def evaluate_setup (self):
        return 2 if self.game.distance()<= 4 and not (self.me.module_stack or
                                                      self.me.active_modules) \
               else 0

class Wrench (Base):
    minrange = 1
    maxrange = 2
    power = 1
    priority = 4
    preferred_range = 1.5
    def get_power_bonus(self):
        return min (4, len(self.me.active_modules))
    def after_trigger(self):
        # TODO: allow human players to make any choice,
        # and fake fork it for AI to allow all or nothing.
        if self.game.make_fork (2, self.me,
            "Return all modules to your module stack?", ["No", "Yes"]):
            while self.me.active_modules:
                self.me.return_module(self.me.active_modules[0])

class Upgradeable (Style):
    maxrange = 1
    power = -2
    priority = 2
    stunguard = 2
    preferred_range = 0.5
    def get_priority_bonuse (self):
        return -min(4, len(self.me.active_modules))
    def get_power_bonuse (self):
        return min(4, len(self.me.active_modules))
    def before_trigger (self):
        self.me.move ([1])
    ordered_before_trigger = True

class Rocket (Style):
    minrange = 2
    maxrange = 3
    power = 1
    priority = 1
    preferred_range = 2 # actually depends on corner
    def start_trigger (self):
        self.me.retreat ((0,1))
    ordered_start_trigger = True
    def hit_trigger (self):
        self.me.move_opponent ((1,2))
    ordered_hit_trigger = True

class Burnout (Style):
    minrange = 1
    maxrange = 2
    power = -1
    preferred_range = 1.5
    def hit_trigger (self):
        if self.me.active_modules:
            ans = self.game.make_fork (len(self.me.active_modules), self.me,
                            "Choose a module to discard for +2 Power:",
                            [module.name for module in self.me.active_modules])
            self.me.discard_module (self.me.active_modules[ans])
            self.me.add_triggered_power_bonus(2)
    # life loss on stun handled by Clive.stun()

class Megaton (Style):
    power = 1
    priority = -1
    stunguard = 2
    def hit_trigger (self):
        self.me.retreat(range(6), max_move=True)
        self.me.push ([1])
    ordered_hit_trigger = True

class LeapingClive (Style):
    name_override = 'Leaping'
    priority = 1
    # Adds 1 to minrange, 3 to maxrange, and 3 is better.
    preferred_range = 2.25
    def can_hit (self):
        return not self.me.switched_sides
    def before_trigger (self):
        old_pos = self.me.position
        self.me.advance ((1,2,3))
        spaces_advanced = abs(self.me.position-old_pos)
        # If I switched sides, actual advance is one less then distance moved
        if ordered (old_pos, self.opponent.position, self.me.position):
            spaces_advanced -= 1
        if spaces_advanced == 3:
            self.me.add_triggered_power_bonus (2)
    ordered_before_trigger = True
    def movement_reaction(self, initiator, mover, old_position, direct):
        if mover is self.me and ordered(mover.position,
                                        mover.opponent.position,
                                        old_position):
            self.me.switched_sides = True

class Module (Card):
    pass

class RocketBoots (Module):
    def before_trigger (self):
        self.me.advance ([1])
    ordered_before_trigger = True

class BarrierChip (Module):
    stunguard = 1

class AtomicReactor (Module):
    power = 1

class ForceGloves (Module):
    def hit_trigger (self):
        self.me.push ([1])
    ordered_hit_trigger = True

class CoreShielding (Module):
    pass
    # Blocking hit and damage trigger handled by Clive

# TODO: can skip retreating if returning with Wrench
class AfterBurner(Module):
    def after_trigger(self):
        self.me.retreat([1])
    ordered_after_trigger = True

class SynapseBoost (Module):
    priority = 1

class AutoRepair (Module):
    def end_trigger (self):
        self.me.gain_life(1)
        self.me.return_module(self)

class ExtendingArms (Module):
    def before_trigger(self):
        self.me.pull([1])
    ordered_before_trigger = True


#Danny
class TheDarkRide(Finisher):
    standard_range = False
    power = 2
    priority = 6
    def hit_trigger(self):
        self.me.monsters.remove(self.opponent.position)
        self.me.move_opponent([1])
        self.me.max_attacks += 1
    def evaluate_setup(self):
        # 0.5 for each time the attack hits beyond the first.
        if self.opponent.position not in self.me.monsters:
            return 0
        pos = self.opponent.position - 1
        count_a = 0
        while pos in self.me.monsters:
            count_a += 1
            pos -= 1
        pos = self.opponent.position + 1
        count_b = 0
        while pos in self.me.monsters:
            count_b += 1
            pos += 1
        return 0.5 * max(count_a, count_b)

class Hellgate(Base):
    is_attack = False
    power = None
    soak = 3
    preferred_range = 2
    def after_trigger(self):
        self.me.move_to_unoccupied()
    ordered_after_trigger = True
    def end_trigger(self):
        self.me.add_monsters(3, set(xrange(7)))
    def evaluation_bonus(self):
        monsters_to_add = max(3, 7 - len(self.me.monsters))
        return 0.1 * (monsters_to_add - 2)

class Monstrous(Style):
    power = 1
    priority = -1
    stunguard = 2
    def end_trigger(self):
        if self.me.hits_scored:
            self.me.add_monsters(1, set(xrange(7)))

class Fallen(Style):
    maxrange = 1
    power = -1
    preferred_range = 0.5
    def hit_trigger(self):
        if self.opponent.position in self.me.monsters:
            self.me.fallen_ignore_stunguard = True
            self.me.add_triggered_power_bonus(2)
    ordered_hit_trigger = True
    def evaluation_bonus(self):
        return 0.3 + (0.03 * self.opponent.expected_stunguard() - 0.03
                if self.opponent.position in self.me.monsters else -0.33)

class Shackled(Style):
    maxrange = 1
    priority = -1
    def blocks_opponent_movement(self, initiator, direct):
        return self.me.monsters
    def evaluation_bonus(self):
        adjacent = self.me.get_destinations(self.opponent, [-1,1])
        two_away = self.me.get_destinations(self.opponent, [-2,2])
        return (0.3 * len(adjacent & self.me.monsters) +
                0.1 * len(two_away & self.me.monsters) - 0.3)

class Sinners(Style):
    name_override = "Sinner's"
    minrange = 1
    maxrange = 3
    power = -1
    preferred_range = 2
    def after_trigger(self):
        # Move up to 3 monsters to adjacent spaces.
        # Method: for every possible placement of existing monsters,
        # check if it's a valid move.
        if not self.me.monsters:
            return
        monsters = sorted(list(self.me.monsters))
        good_combos = []
        for combo in itertools.combinations(xrange(7), len(self.me.monsters)):
            shifts = [abs(m-c) for m,c in zip(monsters, combo)]
            if max(shifts) <= 1 and sum(shifts) <= 3:
                good_combos.append(combo)
        prompt = "Move monsters:"
        options = []
        if self.me.is_user and self.game.interactive_mode:
            for combo in good_combos:
                tmp_list = ['m' if m in combo else '.'
                            for m in xrange(7)]
                options.append(''.join(tmp_list))
        combo = good_combos[
            self.game.make_fork(len(good_combos), self.me, prompt, options)]
        self.me.monsters = set(combo)
        if self.game.reporting:
            self.game.report ("Danny moves his monsters:")
            for s in self.game.get_board():
                self.game.report (s)
    def evaluation_bonus(self):
        monsters = max(2, len(self.me.monsters))
        return 0.03 * (monsters - 0.5)

class Vicious(Style):
    power = -1
    priority = 1
    mean_priority = 3
    def get_priority_bonus(self):
        me = self.me.position
        opp = self.opponent.position
        return len([m for m in self.me.monsters
                    if (me - opp) * (m - opp) > 0])
    def hit_trigger(self):
        me = self.me.position
        opp = self.opponent.position
        self.me.add_triggered_power_bonus(
            len([m for m in self.me.monsters
                 if (me - opp) * (m - opp) < 0]))
    ordered_hit_trigger = True
    def evaluation_bonus(self):
        return 0.3 * (len(self.me.monsters) - 3)

#Demitras

class SymphonyOfDemise (Finisher):
    name_override = 'Symphony of Demise'
    minrange = 1
    maxrange = 1
    priority = 9
    def before_trigger (self):
        self.me.advance ((0,1,2,3,4))
    ordered_before_trigger = True
    def hit_trigger (self):
        self.me.pool = 5 * [self.me.crescendo]
        if self.game.reporting:
            self.game.report ("Demitras now has 5 Crescendo Tokens")
    def evaluate_setup (self):
        return 2 if self.game.distance()<6 and len(self.me.pool) < 3 else 0

class Accelerando (Finisher):
    minrange = 1
    maxrange = 2
    power = 2
    priority = 4
    def reduce_stunguard (self, stunguard):
        return 0
    def before_trigger (self):
        self.me.advance(range(6), max_move=True)
    ordered_before_trigger = True
    def hit_trigger (self):
        if self.me.can_spend (1):
            spend = self.game.make_fork(
                len(self.me.pool)+1, self.me,
                "Choose number of Crescendo tokens to spend:")
            for unused in range(spend):
                self.me.spend_token()
            self.me.add_triggered_power_bonus(2*spend)
    def evaluate_setup (self):
        # when you can advance to range 2, and have the tokens
        if self.game.distance() > 1 or self.opponent.position in [2,3,4] or \
           (self.opponent.position==1 and self.me.position==0) or \
           (self.opponent.position==5 and self.me.position==6):
            return 0.5 * len(self.me.pool)
        else:
            return 0

class Deathblow (Base):
    minrange = 1
    maxrange = 1
    priority = 8
    def hit_trigger (self):
        if self.me.can_spend(1):
            spend = self.game.make_fork(
                len(self.me.pool)+1, self.me,
                "Choose number of Crescendo tokens to spend:")
            for unused in range (spend):
                self.me.spend_token()
            self.me.add_triggered_power_bonus(2*spend)
    def after_trigger(self):
        if self.me.hits_scored:
            self.me.recover_tokens (1)

class Darkside (Style):
    power = -2
    priority = 1
    def can_be_hit (self):
        return self.opponent.attack_range() < 4
    def hit_trigger (self):
        self.me.retreat (range(6))
    ordered_hit_trigger = True
    def evaluation_bonuse (self):
        # Dark side is good when I can retreat to range 4
        if self.me.position > self.opponent.position:
            return 0.5 if self.opponent.position <= 2 else -0.5
        else:
            return 0.5 if self.opponent.position >= 4 else -0.5

class Bloodletting (Style):
    power = -2
    priority = 3
    def hit_trigger (self):
        self.me.gain_life (self.me.ante.count(self.me.crescendo))
    def reduce_soak (self, soak):
        return 0
    def evaluation_bonus (self):
        return 0.1 * (len(self.me.pool) - 3) + 0.1 * self.opponent.expected_soak()

class Vapid (Style):
    maxrange = 1
    power = -1
    preferred_range = 0.5
    def hit_trigger (self):
        if self.opponent.get_priority() <= 3 :
            self.opponent.stun()

class Illusory (Style):
    power = -1
    priority = 1
    def reveal_trigger (self):
        self.me.revoke_ante()
    def can_be_hit (self):
        return self.me.get_priority() >= self.opponent.get_priority() \
               or len(self.me.pool) > 2
    def evaluation_bonus (self):
        n = min (3, len(self.me.pool))
        return (1-n) * 0.2


class Jousting (Style):
    power = -2
    priority = 1
    preferred_range = 2.5 # adds 0 to minrange and 5 to maxrange
    def start_trigger (self):
        # advance until adjacent to opponent
        self.me.advance ([self.game.distance()-1])
    ordered_start_trigger = True
    def hit_trigger (self):
        # advance as far as possible
        if self.me.position > self.opponent.position:
            move = self.me.position - 1
        else:
            move = 5 - self.me.position
        self.me.advance ((move,))
    ordered_hit_trigger = True

# Only token's ante bonus is listed here
# The permanent priority bonus is handled by the character
class Crescendo (Token):
    power = 2

#Eligor

class SweetRevenge (Finisher):
    minrange = 1
    maxrange = 2
    stunguard = 3
    @property
    def power (self):
        return 0 if self.opponent.base.printed_power is None else \
               3 * (self.opponent.get_printed_power())
    printed_power = None
    # minimum life of 1 handled by Eligor.get_minimum_life()
    def evaluate_setup (self):
        return 2 if self.game.distance() <= 2 else 0

class SheetLightning (Finisher):
    minrange = 3
    maxrange = 6
    power = 4
    priority = 6
    def hit_trigger (self):
        self.me.advance ([self.game.distance()-1])
        self.me.opponent_immobilized_status_effect.activate()
    ordered_hit_trigger = True
    def evaluate_setup (self):
        return 1 if self.me.attack_range() >= 3 else 0

class Aegis (Base):
    minrange = 1
    maxrange = 1
    preferred_range = 1
    @property
    def power (self):
        return 0 if self.opponent.base.printed_power is None else \
               (self.opponent.get_printed_power())
    printed_power = None
    def get_soak (self):
        return self.me.ante.count(self.me.vengeance)
    # for evaluation
    @property
    def soak(self):
        return len(self.me.pool) * 0.6

class Vengeful (Style):
    power = 1
    stunguard = 3
    preferred_range = 0.5
    def before_trigger (self):
        self.me.advance([1])
    def hit_trigger (self):
        self.me.recover_tokens(2)
    def blocks_own_movement (self, initiator):
        return set() if initiator is self.me else set(xrange(7))

class CounterStyle (Style):
    name_override = 'Counter'
    power = 1
    priority = -1
    def start_trigger (self):
        # Bases from different standard sets equate by color.
        if self.me.base.alpha_name == self.opponent.base.alpha_name:
            self.opponent.stun()
    def before_trigger (self):
        if self.me.damage_taken > 0:
            self.me.advance (range(self.me.damage_taken+1))
    @property
    def ordered_before_trigger(self):
        return self.me.damage_taken

class Martial (Style):
    maxrange = 1
    power = 1
    priority = -1
    preferred_range = 0.5
    def before_trigger (self):
        if self.me.damage_taken > 0:
            self.me.add_triggered_power_bonus(2)
        if len(self.me.pool) == 5:
            self.me.add_triggered_power_bonus(2)

class Chained (Style):
    priority = -1
    preferred_range = 0.5
    def before_trigger (self):
        if self.me.can_spend (1):
            max_pull = (self.opponent.position-1 \
                        if self.opponent.position > self.me.position \
                        else 5-self.opponent.position)
            max_pull = min (max_pull, len(self.me.pool))
            pull = self.game.make_fork (max_pull + 1, self.me, \
                      "Choose number of Vengeance tokens to spend for pulling:")
            for unused in range (pull):
                self.me.spend_token()
            if pull > 0:
                self.me.pull((pull,))
    @property
    def ordered_before_trigger(self):
        return self.me.can_spend(1)

class Retribution (Style):
    priority = -1
    preferred_range = 2
    soak = 2
    # if I was hit, may jump adjacent to opponent
    def before_trigger (self):
        if self.opponent.hits_scored:
            self.me.move_directly ((self.me.position,
                                    self.opponent.position-1,
                                    self.opponent.position+1))
    @property
    def ordered_before_trigger(self):
        return self.opponent.hits_scored
    # regaining tokens for soak handled by Eligor.soak_trigger()

class VengeanceT (Token):
    name_override = 'Vengeance'
    stunguard = 2

#Eustace

class BoneBreaker(Finisher):
    minrange = 1
    maxrange = 1
    power = 7
    priority = 5
    def start_trigger(self):
        self.me.advance(range(6))
    ordered_start_trigger = True
    def hit_trigger(self):
        self.me.push(range(6))
    ordered_hit_trigger = True

class RimeBlitzer(Finisher):
    minrange = 1
    maxrange = 1
    power = 8
    priority = 3
    soak = 2
    def before_trigger(self):
        self.me.advance([1,2])
    ordered_before_trigger = True
    def evaluate_setup(self):
        return 0.5 if self.game.distance() <= 3 else 0

class Cross(Base):
    minrange = 1
    maxrange = 1
    power = 4
    priority = 4
    preferred_range = 1.5
    def before_trigger(self):
        self.me.advance([1,2])
    ordered_before_trigger = True
    def after_trigger(self):
        self.me.move([1])
    ordered_after_trigger = True
    def evaluation_bonus(self):
        return -0.2 if self.opponent.position in (0,6) else 0.1

class Frigid(Style):
    def reveal_trigger(self):
        self.me.add_triggered_power_bonus(self.me.strat[2][0])
    def end_trigger(self):
        self.me.advance([self.game.distance() - 1])
    ordered_end_trigger = True
    def evaluation_bonus(self):
        return 0.05 * (self.me.counters - 2)

class Ivory(Style):
    power = 1
    preferred_range = 0.5
    stunguard = 2
    def before_trigger(self):
        self.me.advance([1])
    ordered_before_trigger = True

class Elementary(Style):
    power = 1
    priority = -1
    def start_trigger(self):
        self.me.retreat([0,1], specific_movement_reaction=self.specific_movement_reaction)
    ordered_start_trigger = True
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        if self.me.position != old_position:
            self.me.elementary_range_bonus += 1

class Frostbite(Style):
    priority = -3
    def has_stun_immunity(self):
        return True

class Barons(Style):
    name_override = "Baron's"
    power = 2
    priority = -2
    stunguard = 1
    def end_trigger(self):
        if self.me.hits_scored:
            self.me.recover_tokens(1)

class Frostflow(Token):
    soak = 1

#Gerard

class Windfall(Finisher):
    minrange = 1
    maxrange = 1
    priority = 5
    def before_trigger(self):
        self.me.advance([1,2])
    ordered_before_trigger = True
    def hit_trigger(self):
        self.me.add_triggered_power_bonus(self.me.gold)
    def evaluate_setup(self):
        return 0.2 * max(0, self.me.gold - 4) if self.game.distance() <= 3 else 0

# Not implemented, possible big fork
class UltraBeatdown(Finisher):
    minrange = 1
    maxrange = 3
    power = 2
    priority = 4
    stunguard = 4
    def hit_trigger(self):
        pass

class Larceny(Base):
    minrange = 1
    maxrange = 1
    power = 4
    priority = 3
    preferred_range = 1.5
    def can_hit(self):
        return self.me.switched_sides
    def before_trigger(self):
        self.me.pull([1])
        self.me.advance([1])
    def damage_trigger(self, damage):
        self.me.gain_gold(3)

class Villainous(Style):
    power = 1
    def hit_trigger(self):
        self.me.push([1,2])
    ordered_hit_trigger = True
    def end_trigger(self):
        # Not using specific_movement_reaction, because opponent has a chance to escape with reaction.
        # Or does he?  Rules unclear.
        self.me.advance([1,2,3])
        if self.opponent.position in [0,6] and self.game.distance() == 1:
            self.me.gain_gold(2)
    def evaluation_bonus(self):
        value = self.me.gold_value * 0.3
        return (value * 0.25 if self.opponent.retreat_range() <= 2 and
                             self.me.retreat_range() >= 2
                else -value * 0.75)

class Gilded(Style):
    maxrange = 1
    priority = -1
    preferred_range = 0.5
    def get_soak(self):
        return 2 if self.me.gold > len(self.me.mercs_in_play) else 0
    def get_stunguard(self):
        return 0 if self.me.gold > len(self.me.mercs_in_play) else 4
    # For evaluation: (might lose gold / add mercs in ante).
    @property
    def soak(self):
        return 1 if self.me.gold > len(self.me.mercs_in_play) else 0
    @property
    def stunguard(self):
        return 2 if self.me.gold > len(self.me.mercs_in_play) else 4
    def before_trigger(self):
        self.me.pull([0,1])

class Avaricious(Style):
    preferred_range = 0.5
    mean_priority_bonus = 0.5
    def get_power_bonus(self):
        return 1 if self.me.gold > len(self.me.mercs_in_play) else 0
    def get_priority_bonus(self):
        return 0 if self.me.gold > len(self.me.mercs_in_play) else 1
    def start_trigger(self):
        self.me.advance([1])
    ordered_start_trigger = True
    def damage_trigger(self, damage):
        self.me.gain_gold(2)

class Hooked(Style):
    minrange = 2
    maxrange = 3
    power = 1
    priority = -1
    def get_preferred_range(self):
        return 2.5 - self.me.retreat_range()
    # TODO: this doesn't retreat at all if full retreat is blocked.
    def start_trigger(self):
        self.me.retreat([self.me.retreat_range()])
    ordered_start_trigger = True
    def damage_trigger(self, damage):
        self.me.pull(range(damage+1))
    ordered_damage_trigger = True

class Initiation(Style):
    priority = 1
    preferred_range = 0.5
    def before_trigger(self):
        self.me.advance([0,1])
    ordered_before_trigger = True
    def hit_trigger(self):
        self.me.initiation_status_effect.activate()

class InitiationStatusEffect(StatusEffect):
    read_state_prefix = "First hireling costs 2 gold less"
    situation_report_line = "First hireling costs 2 gold less"
    activation_line = "First hireling next beat will cost 2 gold less"
    @property
    def value(self):
        return 1.5 * self.me.gold_value
    # Reducing cost handled by Gerard methods: get_antes, input_ante,
    #                                          ante_trigger

class Mercenary(Card):
    activation_cost = 0
    def get_value(self):
        return 0

class Bookie(Mercenary):
    hiring_cost = 3
    def end_trigger(self):
        if self.me.hits_scored and not self.me.was_stunned:
            self.me.gain_gold(1)

class Brawler(Mercenary):
    hiring_cost = 3
    def after_trigger(self):
        if self.game.distance() in [1,2]:
            self.opponent.lose_life(1)
    # We only want to ask about order if there's movement after
    # activation.
    @property
    def ordered_after_trigger(self):
        return self.me.style.ordered_after_trigger or self.me.base.ordered_after_trigger
    def get_value(self):
        pr = self.opponent.preferred_range
        next_beats_value = 0.2 * (6 - self.game.distance())
        return 0.6 * (6-pr) + next_beats_value

class Archer(Mercenary):
    hiring_cost = 3
    def after_trigger(self):
        if self.game.distance() in [3,4]:
            self.opponent.lose_life(1)
    # We only want to ask about order if there's movement after
    # activation.
    @property
    def ordered_after_trigger(self):
        return self.me.style.ordered_after_trigger or self.me.base.ordered_after_trigger
    def get_value(self):
        next_beats_value = 0.5 * (2.5 - abs(3.5 - self.game.distance()))
        return 2.0 + next_beats_value

class Trebuchet(Mercenary):
    hiring_cost = 3
    def after_trigger(self):
        if self.game.distance() in [5,6]:
            self.opponent.lose_life(1)
    # We only want to ask about order if there's movement after
    # activation.
    @property
    def ordered_after_trigger(self):
        return self.me.style.ordered_after_trigger or self.me.base.ordered_after_trigger
    def get_value(self):
        pr = self.opponent.preferred_range
        next_beats_value = 0.2 * (self.game.distance() - 1)
        return 0.6 * (pr-1) + next_beats_value

class Lackey(Mercenary):
    hiring_cost = 7
    soak = 5
    # End of beat removal handled by Gerard.cycle()

class Mage(Mercenary):
    hiring_cost = 2
    activation_cost = 3
    power = 1
    priority = 1
    def get_value(self):
        return self.me.gold_value

class Gunslinger(Mercenary):
    hiring_cost = 2
    activation_cost = 3
    def reduce_stunguard(self, stunguard):
        return 0
    def get_value(self):
        return self.me.gold_value

class HeavyKnight(Mercenary):
    hiring_cost = 2
    activation_cost = 3
    def blocks_opponent_movement(self, initiator, direct):
        return set() if direct else set([self.me.position])
    def get_value(self):
        return self.me.gold_value

#Heketch

class MillionKnives (Finisher):
    minrange = 1
    maxrange = 4
    power = 3
    priority = 7
    def hit_trigger (self):
        old_pos = self.me.position
        self.me.advance ((1,))
        if self.me.position != old_pos and self.game.distance() > 1:
            self.me.max_attacks += 1
    ordered_hit_trigger = True
    def evaluate_setup (self):
        dist = self.game.distance()
        return 0.5*(dist-1) if dist <= 4 else 0

class LivingNightmare (Finisher):
    minrange = 1
    maxrange = 1
    power = 3
    priority = 2
    def hit_trigger (self):
        self.opponent.stun()
        self.me.living_nightmare_active = True
    def evaluate_setup (self):
        # only real way of hitting is with +3 priority from token
        return 1 if len(self.me.pool) == 1 else 0

class Knives (Base):
    minrange = 1
    maxrange = 2
    power = 4
    priority = 5
    clash_priority = 0.1
    preferred_range = 1.75
    mean_priority = 5.5
    # blocking your own stun handled by Heketch.damage_trigger()
    discard_penalty = 0.5

class Merciless (Style):
    maxrange = 1
    power = -1
    preferred_range = 0.5
    def movement_reaction (self, initiator, mover, old_pos, direct):
        if (mover == self.opponent and not direct and
            ordered(mover.position, self.me.position, old_pos)):
            self.opponent.lose_life (2)
            self.me.merciless_immobilized = True
            if self.game.reporting:
                self.game.report (self.opponent.name + " cannot move again this beat")
    def after_trigger (self):
        if len (self.me.pool) == 1:
            self.me.set_triggered_dodge()
    # reaction to moving opponent handled by Heketch.execute_move()

class Rasping (Style):
    maxrange = 1
    power = -1
    priority = 1
    preferred_range = 0.5
    def hit_trigger (self):
        if self.game.distance() == 1 and self.me.can_spend(1):
            if self.game.make_fork (2, self.me, \
                                    "Spend Dark Force Token for +3 power?",
                                    ["No", "Yes"]):
                self.me.spend_token ()
                self.me.add_triggered_power_bonus(3)
    @property
    def ordered_hit_trigger(self):
        return self.me.can_spend(1)
    def damage_trigger (self, damage):
        self.me.gain_life ((damage+1)/2)

class Critical (Style):
    power = -1
    priority = 1
    def hit_trigger (self):
        if self.game.distance() == 1 and self.me.can_spend(1):
            if self.game.make_fork (2, self.me, \
                                    "Spend Dark Force Token for +3 power?",
                                    ["No", "Yes"]):
                self.me.spend_token ()
                self.me.add_triggered_power_bonus(3)
    @property
    def ordered_hit_trigger(self):
        return self.me.can_spend(1)
    def reduce_stunguard (self, stunguard):
        return 0
    def evaluation_bonus(self):
        return 0.03 * self.opponent.expected_stunguard() - 0.03

class Assassin (Style):
    def hit_trigger (self):
        self.me.retreat (range(6))
    ordered_hit_trigger = True
    def damage_trigger (self, damage):
        if self.game.distance() == 1 and self.me.can_spend(1):
            # fake fork, assume I immobilize if possible
            if self.game.make_fork (2, self.me,
                "Spend Dark Force Token to immobilze opponent next beat?",
                                    ["No", "Yes"], 1):
                self.me.spend_token ()
                self.me.opponent_immobilized_status_effect.activate()
    @property
    def ordered_hit_trigger(self):
        return self.me.can_spend(1)

class Psycho (Style):
    priority = 1
    preferred_range = 2.5
    def start_trigger (self):
        # advance until adjacent to opponent
        self.me.advance ([self.game.distance()-1])
    ordered_start_trigger = True
    def end_trigger (self):
        if self.game.distance() == 1 and self.me.can_spend(1):
            if self.game.make_fork (2, self.me, \
                "Spend Dark Force Token to activate again?", ["No", "Yes"]):
                self.me.spend_token ()
                if self.game.reporting:
                    self.game.report ("Heketch reactivates attack")
                self.game.activate (self.me)
    @property
    def ordered_end_trigger(self):
        return self.me.can_spend(1)

class DarkForce (Token):
    priority = 3

#Hepzibah

class Altazziar(Finisher):
    minrange = 1
    maxrange = 1
    power = 6
    priority = 2
    preferred_range = 1
    def start_trigger(self):
        self.me.lose_life(self.me.life - 1)
    # Token doubling handled by Hepzibah.set_active_cards()
    def evaluate_setup(self):
        return 0 if self.game.distance() > 3 else 0.3 * min(4, self.me.life)

class Bloodlight(Base):
    minrange = 1
    maxrange = 3
    power = 2
    priority = 3
    preferred_range = 2
    def damage_trigger(self, damage):
        self.me.gain_life(min(damage, len(self.me.anted_pacts)))

class Pactbond(Style):
    power = -1
    priority = -1
    def reveal_trigger(self):
        self.me.gain_life(min(2, len(self.me.anted_pacts)))
    def end_trigger(self):
        prompt = "Choose a free Pact for next beat:"
        options = [pact.name for pact in self.me.pacts]
        # Fake fork: AI always chooses Immortality.
        ans = self.game.make_fork(5, self.me, prompt, options, choice=3)
        self.me.pending_pactbond = self.me.pacts[ans]
        if self.game.reporting:
            self.game.report("Hepzibah Chooses %s as her free Pact" %
                             self.me.pending_pactbond)

class Anathema(Style):
    power = -1
    priority = -1
    def get_power_bonus(self):
        return min(3, len(self.me.anted_pacts))
    def get_priority_bonus(self):
        return min(3, len(self.me.anted_pacts))

class Necrotizing(Style):
    maxrange = 2
    power = -1
    preferred_range = 1
    def hit_trigger(self):
        max_spend = min(3, self.me.life - 1)
        prompt = "How much life to spend for Power?"
        spend = self.game.make_fork(max_spend + 1, self.me, prompt)
        self.me.lose_life(spend)
        self.me.add_triggered_power_bonus(spend)
        # TODO: make opponents life worth 1.01, so that
        # Hepzibah plays Necrotizing aggressively.

class Darkheart(Style):
    priority = -1
    def hit_trigger(self):
        self.me.gain_life(2)
        self.opponent.discard_token()
    def evaluation_bonus(self):
        return 0.1 if self.opponent.pool else -0.1

class Accursed(Style):
    maxrange = 1
    power = -1
    preferred_range = 0.5
    def has_stun_immunity (self):
        return len(self.me.ante) >= 3

class Almighty(Card):
    power = 2

class Corruption(Card):
    def reduce_stunguard(self, stunguard):
        return 0

class Endless(Card):
    priority = 2

class Immortality(Card):
    soak = 2

class InevitableT(Card):
    name_override = 'Inevitable'
    maxrange = 1

#Hikaru

class WrathOfElements (Finisher):
    name_override = 'Wrath of Elements'
    minrange = 1
    maxrange = 1
    power = 7
    priority = 6
    def reveal_trigger (self):
        for t in self.me.pool[:]:
            self.me.ante_token (t)
    def evaluate_setup (self):
        dist = self.game.distance()
        return 1 + 0.5 * len(self.me.pool) \
               if dist == 1 or (dist==2 and self.me.water in self.me.pool) \
               else 0

class FourWinds (Finisher):
    minrange = 1
    maxrange = 1
    power = 2
    priority = 5
    # Disabling tokens handled by Hikaru.get_active_tokens()
    def before_trigger (self):
        self.me.advance ((0,1))
    ordered_before_trigger = True
    # Attack again each time you successfully regain a token
    def hit_trigger (self):
        len_pool = len(self.me.pool)
        self.me.recover_tokens()
        if len(self.me.pool) > len_pool:
            self.me.max_attacks += 1
    # need at least 3 recoverable tokens to be worth it, four is better
    def evaluate_setup (self):
        return self.game.distance() <= 2 and \
               max (0, 2 - len(self.me.pool)
                         - len(set(self.me.ante) & set(self.me.tokens)))

class PalmStrike (Base):
    minrange = 1
    maxrange = 1
    power = 2
    priority = 5
    preferred_range = 0.5 # advance 1 makes effective range 1-2
    def start_trigger (self):
        self.me.advance ([1])
    ordered_start_trigger = True
    def damage_trigger (self, damage):
        self.me.recover_tokens ()

class Trance (Style):
    maxrange = 1
    preferred_range = 0.5
    def reveal_trigger (self):
        # Only revokes own tokens, not induced tokens.
        # (Cards says otherwise, ask Brad)
        self.me.revoke_ante()
    def end_trigger (self):
        self.me.recover_tokens ()

class Focused (Style):
    priority = 1
    stunguard = 2
    def hit_trigger (self):
        self.me.recover_tokens ()

class Advancing (Style):
    power = 1
    priority = 1
    preferred_range = 0.5 # advance only adds to maxrange
    def start_trigger (self):
        self.me.advance ((1,), specific_movement_reaction=self.specific_movement_reaction)
    ordered_start_trigger = True
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        if ordered(old_position, self.opponent.position, self.me.position):
            self.me.add_triggered_power_bonus(1)

class Sweeping (Style):
    power = -1
    priority = 3
    # Increasing incoming damage handled by Hikaru.take_damage()
    discard_penalty = -0.5

class Geomantic (Style):
    power = 1
    def start_trigger (self):
        if self.me.can_spend(1):
            # fork to decide which token is anted (or none)
            n_options = len(self.me.pool) + 1
            prompt = "Select a token to ante with Geomantic:"
            options = [t.name for t in self.me.pool] + ["None"]
            token_number = self.game.make_fork (n_options, self.me, prompt, options)
            if token_number < len (self.me.pool):
                self.me.ante_token (self.me.pool[token_number])

class Earth (Token):
    soak = 3
    value = 0.9

class Fire (Token):
    power = 3
    value = 0.8

class Water (Token):
    minrange = -1
    maxrange = 1
    value = 0.6

class Wind (Token):
    priority = 2
    value = 0.7

#Iri

class LimitExceed(Finisher):
    minrange = 1
    maxrange = 1
    power = 1
    priority = 7
    # Hit: The opponent is stunned, and you activate all of your Forms.
    # They all remain active for the rest of the game, but you can choose
    # a transformation effect as an ante.
    def hit_trigger(self):
        self.opponent.stun()
        self.me.limit_exceeded = True
        self.me.set_active_cards()
    def evaluate_setup(self):
        return 0.5 if self.game.distance() == 1 else 0

class FinalGalaxySpearAtamos(Finisher):
    minrange = 3
    maxrange = 6
    power = 7
    priority = 2
    def get_minrange_bonus(self):
        return -1 if self.me.current_form is self.me.magical_form else 0
    def get_priority_bonus(self):
        return 2 if self.me.current_form is self.me.magical_form else 0
    def get_stunguard(self):
        return 4 if self.me.current_form is self.me.knight_form else 0
    def can_be_hit(self):
        return (self.opponent.attack_range() < 4 or
                self.me.current_form is not self.me.cosmic_form)
    def evaluate_setup(self):
        return (1 if self.game.distance() >= 3 else 0)

class Starlight(Base):
    minrange = 1
    maxrange = 2
    power = 2
    priority = 4
    preferred_range = 1.5
    def get_power_bonus(self):
        ret = 0
        if self.me.current_form is self.me.cosmic_form or self.me.limit_exceeded:
            ret += 1
        if self.me.current_form is self.me.knight_form or self.me.limit_exceeded:
            ret += 2
        return ret
    def get_priority_bonus(self):
        ret = 0
        if self.me.current_form is self.me.magical_form or self.me.limit_exceeded:
            ret += 1
        if self.me.current_form is self.me.knight_form or self.me.limit_exceeded:
            ret -= 2
        return ret
    mean_priority = 3.7 # for eval
    def get_stunguard(self):
        return 2 if self.me.current_form is self.me.knight_form or self.me.limit_exceeded else 0
    stunguard = 0.7 # for eval

class Astral(Style):
    maxrange = 1
    power = -2
    priority = -1
    stunguard = 3
    preferred_range = 0.5
    # When you are take damage this beat, gain power equal to the damage taken (max 5).
    # Handled in Iri.take_damage()
    def end_trigger(self):
        self.me.move([0,1])

class Comet(Style):
    power = 1
    def get_preferred_range(self):
        # If I can move 3 forward, max range is up 3, and so is min range
        # (because switching causes a miss).
        return 3 if self.me.retreat_range() <= 2 else 0
    def before_trigger(self):
        self.me.advance([3], specific_movement_reaction=self.specific_movement_reaction)
    ordered_before_trigger = True
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        if ordered(old_position, self.opponent.position, self.me.position):
            self.opponent.set_triggered_dodge()
    def hit_trigger(self):
        self.me.pull([1,2])
    ordered_hit_trigger = True

class Magical(Style):
    maxrange = 1
    priority = 1
    preferred_range = 0.5
    def after_trigger(self):
        self.me.retreat([1])
    ordered_after_trigger = True

class Galactic(Style):
    def reveal_trigger(self):
        # If limit exceeded, trigger all non-current transforms.
        # Trigger current form transform if it wasn't already triggered.
        for form in self.me.forms:
            if ((self.me.limit_exceeded and not form is self.me.current_form) or
                (form is self.me.current_form and not self.me.changed_form)):
                form.transformation_trigger()

class Aurora(Style):
    minrange = 1
    maxrange = 2
    power = 1
    preferred_range = 1.5
    def hit_trigger(self):
        # Opponent discards extra base.
        bases = list(set(self.opponent.bases) - self.opponent.discard[0]
                                              - self.opponent.discard[1]
                                              - self.opponent.discard[2])
        prompt = "Choose an extra base to discard"
        options = [b.name for b in bases]
        discard = bases[self.game.make_fork(len(bases), self.opponent,
                                            prompt, options)]
        self.opponent.discard[0] |= set ([discard])
        if self.game.reporting:
            self.game.report (self.opponent.name + " discards " + discard.name)

class Form(Card):
    pass

class MagicalForm(Form):
    maxrange = 1
    def hit_trigger(self):
        if self.game.distance() == self.me.get_maxrange():
            self.me.add_triggered_power_bonus(1)
    ordered_hit_trigger = True
    def transformation_trigger(self):
        self.me.add_triggered_priority_bonus(1)
        self.me.add_triggered_power_bonus(-1)

class KnightForm(Form):
    def get_stunguard(self):
        return 3 if (self.me.get_minrange() <=
                     self.game.distance <=
                     self.me.get_maxrange()) else 0
    def transformation_trigger(self):
        self.me.add_triggered_priority_bonus(-2)
        self.me.knight_form_stunguard += 3
        if self.game.reporting:
            self.game.report("Iri gets +3 stunguard")

class CosmicForm(Form):
    def reveal_trigger(self):
        if self.game.distance() == 1:
            self.opponent.add_triggered_priority_bonus(-3)
    def transformation_trigger(self):
        self.me.add_triggered_power_bonus(2)
        self.me.lose_life(1)

#Jager

class BloodHunt(Finisher):
    standard_range = False
    power = 7
    priority = 4
    def special_range_hit(self):
        return self.me.switched_sides
    def start_trigger(self):
        self.me.advance(range(5))
    def evaluate_setup(self):
        return (0 if self.opponent.position in (0,6)
                or self.game.distance() >= 5 else 0.5)

class SignOfTheWolf(Finisher):
    name_override = 'Sign of the Wolf'
    priority = 8
    @property
    def is_attack(self):
        return (self.me.finisher_signature and
                self.me.finisher_signature.is_attack)
    @property
    def deals_damage(self):
        return (self.me.finisher_signature and
                self.me.finisher_signature.deals_damage)
    def after_trigger(self):
        if self.me.finisher_signature:
            return
        ans = self.game.make_fork(
            1 + len(self.me.signatures), self.me,
            "Choose a Signature Move to execute:",
            [s.name for s in self.me.signatures] + ["None"])
        if ans == len(self.me.signatures):
            return
        sig = self.me.signatures[ans]
        self.me.signature_pool = [s for s in self.me.signature_pool if s is not sig]
        if self.game.reporting:
            self.game.report("Jager executes %s" % sig)
        self.me.max_attacks += 1
        self.me.finisher_signature = sig
        self.me.set_active_cards()
    def evaluate_setup(self):
        return 0.25 * len(self.me.signature_pool)

class Slicer(Base):
    minrange = 1
    maxrange = 2
    power = 3
    priority = 4
    preferred_range = 1.5
    def hit_trigger(self):
        self.me.move([1], specific_movement_reaction=self.specific_movement_reaction)
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        # +1 power if switched sides.
        if (ordered(old_position, self.opponent.position, self.me.position) and
            not self.opponent.mimics_movement()):
            self.me.add_triggered_power_bonus(1)
    ordered_hit_trigger = True
    def after_trigger(self):
        self.me.move([1])
    ordered_after_trigger = True

class Refined(Style):
    mean_priority = 1.5
    priority = 1
    def hit_trigger(self):
        self.me.move_opponent([0,1,2])
    ordered_hit_trigger = True
    def after_trigger(self):
        self.me.move([0,1])
    ordered_after_trigger = True
    def end_trigger(self):
        self.me.refined_status_effect.activate()

class Hounding(Style):
    mean_priority = 2
    priority = 1
    def after_trigger(self):
        self.me.move([1,2])
    ordered_after_trigger = True
    def hit_trigger(self):
        self.me.hounding_status_effect.activate()

class Menacing(Style):
    mean_priority = 0
    priority = -2
    def has_stun_immunity (self):
        return True
    def reduce_soak(self, soak):
        return 0
    def reduce_stunguard(self, stunguard):
        return 0
    # If the opponent reveals a base that matches one of your discard piles,
    # become the active player.
    def start_trigger(self):
        base_list = []
        for card in self.me.discard[1] | self.me.discard[2]:
            if hasattr(card,"alpha_name"):
                base_list.append(card.alpha_name)
        if self.opponent.base.alpha_name in base_list:
            self.me.become_active_player()
    def evaluation_bonus(self):
        opp_names = [b.alpha_name for b in self.opponent.bases
                     if b not in self.opponent.discard[1]|self.opponent.discard[2]]
        my_names = [b.alpha_name for b in self.me.discard[1]|self.me.discard[2]
                    if hasattr(b, "alpha_name")]
        match = len(set(opp_names) & set(my_names))
        return (0.1 * self.opponent.expected_soak() +
                0.03 * self.opponent.expected_stunguard() +
                0.15 * match - 0.25)

class BloodMoon(Style):
    power = 1
    priority = -2
    stunguard = 2
    soak = 2
    def can_be_hit (self):
        return self.opponent.attack_range() < 4
    def can_hit(self):
        return not self.me.switched_sides
    def evaluation_bonus(self):
        d = self.game.distance()
        return 1 if d >= 4 else 0.5 * (d-3)

class Fanged(Style):
    minrange = 1
    maxrange = 2
    priority = 1
    def get_preferred_range(self):
        # Usually adds 2-4 to effective range.
        # When next to opponent, can advance 2 to avoid minimum range.
        return 2 if (self.game.distance() == 1 and
                     self.opponent.retreat_range() > 1) else 3
    def before_trigger(self):
        self.me.advance ([1,2], specific_movement_reaction=self.specific_movement_reaction)
    ordered_before_trigger = True
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        if ordered(old_position, self.opponent.position, self.me.position):
            self.me.add_triggered_power_bonus(1)

class Signature(Base):
    # More than regular bonuses, less than Finishers.
    clash_priority = 0.2
    def get_value(self):
        return 0.25 if self.game.distance() in self.good_ranges else 0

class WolfsFangFugue(Signature):
    name_override = "Wolf's Fang Fugue"
    minrange = 1
    maxrange = 2
    power = 3
    priority = 4
    good_ranges = set((1,2))
    def hit_trigger(self):
        if self.me.hits_scored == 0:
            self.me.move([1])
            self.me.max_attacks += 1

class CrescentMoonClaw(Signature):
    minrange = 2
    maxrange = 4
    power = 4
    priority = 6
    good_ranges = set((2,3,4))
    def hit_trigger(self):
        self.me.advance([1,2])

class RedMoonRageSignature(Signature):
    name_override = 'Red Moon Rage'
    minrange = 1
    maxrange = 1
    power = 8
    priority = 3
    stunguard = 5
    good_ranges = set((1,))

class MiststepMassacre(Signature):
    minrange = 1
    maxrange = 1
    power = 5
    priority = 1
    good_ranges = set((5,6))
    def can_be_hit(self):
        return self.opponent.attack_range() not in self.good_ranges
    def before_trigger(self):
        opp = self.opponent.position
        self.me.move_directly([opp-1, opp+1])

#Juto

class BanHammer(Finisher):
    minrange = 1
    maxrange = 1
    power = 5
    priority = 5
    def hit_trigger(self):
        for player in [self.opponent, self.me]:
            opp = player.opponent
            prompt = "Select a style to remove:"
            options = [s.name for s in opp.styles]
            ans = self.game.make_fork(len(options), player, prompt, options)
            opp.remove_card_from_game(opp.styles[ans])
            prompt = "Select a base to remove:"
            options = [b.name for b in opp.bases]
            ans = self.game.make_fork(len(options), player, prompt, options)
            opp.remove_card_from_game(opp.bases[ans])
    def evaluate_setup(self):
        return 1 if self.game.distance() == 1 else 0

class SpellboltJuto(Base):
    name_override = "Spellbolt"
    minrange = 2
    maxrange = 6
    power = 2
    priority = 3
    preferred_range = 4
    def hit_trigger (self):
        prompt = "Choose a Spellbolt effect:"
        options = ["Pull %s" % self.opponent,
                   "Give %s -2 power" % self.opponent]
        if self.game.make_fork(2, self.me, prompt, options):
            self.opponent.add_triggered_power_bonus(-2)
        else:
            self.me.pull ([1,2])
    ordered_hit_trigger = True

class Staff(Base):
    minrange = 1
    maxrange = 2
    power = 3
    priority = 4
    stunguard = 4
    prefered_range = 1.5
    def hit_trigger(self):
        self.me.push([1])
        self.me.staff_block_advancing = True
    ordered_hit_trigger = True

class WhirlpoolJuto (Base):
    name_override = "Whirlpool"
    minrange = 1
    maxrange = 2
    power = 3
    priority = 3
    preferred_range = 2.5
    def start_trigger (self):
        self.me.pull([1])
    ordered_start_trigger = True
    def after_trigger (self):
        self.me.move ([0,1,2,3,4])
    ordered_after_trigger = True

class Fluffy(Style):
    power = -1
    priority = 1
    def after_trigger(self):
        self.me.move([0,1,2,3])
    ordered_after_trigger = True

class Snuggly(Style):
    maxrange = 1
    power = -1
    priority = -1
    preferred_range = 0.5
    def hit_trigger(self):
        self.me.gain_life(2)

class Cuddly(Style):
    power = 1
    priority = 1
    # Not stunning handled by Juto.damage_trigger()

class Cute(Style):
    maxrange = 1
    priority = -2
    preferred_range = 0.5
    def start_trigger(self):
        self.me.retreat([0,1])
    ordered_start_trigger = True

class Fuzzy(Style):
    power = 1
    priority = -2
    stunguard = 5
    def evaluation_bonus (self):
        return -0.3 if self.me.position in (0,6) else 0.15

#Kajia

class Wormwood (Finisher):
    minrange = 1
    maxrange = 1
    power = 3
    priority = 5
    def start_trigger (self):
        insects = self.me.total_insects()
        self.opponent.lose_life(insects)
        self.me.pull([insects])
    ordered_start_trigger = True
    def hit_trigger (self):
        self.me.add_triggered_power_bonus (self.me.total_insects())
    ordered_hit_trigger = True
    def evaluate_setup (self):
        insects = self.me.total_insects()
        return (0.25 * max(0, insects - 1)
                if self.game.distance() <= insects + 1
                else 0)

class ImagoEmergence (Finisher):
    minrange = 3
    maxrange = 6
    power = 1
    priority = 7
    def hit_trigger (self):
        self.opponent.stun()
        self.me.give_insects(1, pile=1)
        self.me.imago_emergence_active = True
    ordered_hit_trigger = True
    def evaluate_setup (self):
        return 0.5 if self.game.distance() >= 3 else 0

class Mandibles (Base):
    minrange = 1
    maxrange = 2
    power = 3
    priority = 3
    preferred_range = 2
    def before_trigger (self):
        self.me.advance ([1])
    ordered_before_trigger = True
    def hit_trigger (self):
        self.me.pull ([1])
    ordered_hit_trigger = True
    def damage_trigger (self, damage):
        if self.me.insects[1] and self.me.insects[2]:
            self.opponent.stun()
    ordered_damage_trigger = True
    def after_trigger (self):
        self.me.advance ([1])
    ordered_after_trigger = True
    def evaluation_bonus (self):
        stun_potential = self.me.insects[1] and self.me.insects[2]
        insect_potential = 1 if self.game.distance() <= 2 else 0
        insect_value = 0.3 * (insect_potential - 0.5)
        return (0.25 if stun_potential else -0.25) + \
               insect_value

class Burrowing (Style):
    priority = -1
    prefered_range = 0.5
    mean_priority = 0
    def reveal_trigger (self):
        piles = self.me.infested_piles()
        self.me.add_triggered_power_bonus (piles)
        self.me.add_triggered_priority_bonus (piles)
    def before_trigger (self):
        self.me.pull ((0,1))
    ordered_before_trigger = True
    def evaluation_bonus (self):
        return 0.3 * (self.me.infested_piles() - 1)

class Swarming (Style):
    maxrange = 1
    priority = -1
    preferred_range = 0.5
    def can_be_hit(self):
        attack_range = self.opponent.attack_range()
        return attack_range > self.me.infested_piles() or attack_range < 1
    def end_trigger(self):
        if self.game.distance() == 1:
            self.me.give_insects(1)
    def evaluation_bonus (self):
        return 0.5 if self.game.distance() <= self.me.infested_piles() else -0.5

class Parasitic (Style):
    @property
    def maxrange (self):
        return self.me.total_insects()
    priority = -1
    def get_preferred_range (self):
        return 0.5 * (self.me.total_insects())
    def start_trigger (self):
        self.me.pull ([3])
    ordered_start_trigger = True
    def evaluation_bonus (self):
        value = 0.25 * (self.me.total_insects() - 2)
        me = self.me.position
        opp = self.opponent.position
        pull_works = (opp > 3) if me < opp else (opp < 3)
        if pull_works:
            max_insect_range = 3
            value += (0.5 if self.game.distance() <= max_insect_range else 0.25)
        else:
            value -= 0.25
        return value

class Stinging (Style):
    priority = 1
    def reveal_trigger (self):
        self.me.stinging_ignores = self.me.infested_piles()
    def hit_trigger (self):
        self.me.give_insects(1)
    def evaluation_bonus (self):
        piles = self.me.infested_piles()
        value = -0.05
        if piles > 0:
            value += 0.1 * self.opponent.expected_soak()
        if piles > 1:
            value += 0.03 * self.opponent.expected_stunguard()
        return value

class Biting (Style):
    maxrange = 2
    power = -1
    priority = -1
    stunguard = 3
    preferred_range = 1
    def reveal_trigger (self):
        self.opponent.add_triggered_power_bonus (-self.me.infested_piles())
    def damage_trigger (self, damage):
        self.me.pull ((0,1,2))
    ordered_damage_trigger = True
    def evaluation_bonus (self):
        insect_potential = (self.game.distance() <= 2 and
                            not self.me.position in (0,6))
        insect_value = 0.3 * (insect_potential - 0.5)
        return insect_value + \
               0.15 * (self.me.infested_piles() - 1)

#Kallistar

class Supernova (Finisher):
    minrange = 1
    maxrange = 2
    power = 8
    priority = 5
    # devolves to cancel if I'm human
    def devolves_into_cancel (self):
        return Finisher.devolves_into_cancel(self) or self.me.form.is_human
    # if you got here, you lose
    def end_trigger (self):
        raise WinException (self.opponent.my_number)
    def evaluate_setup (self):
        return 1 if self.me.form.is_elemental and self.game.distance() <=2 and \
               self.opponent.life <= 8 else 0

class ChainOfDestruction (Finisher):
    name_override = 'Chain of Destruction'
    minrange = 4
    maxrange = 6
    power = 4
    priority = 5
    # devolves to cancel if I'm elemental
    def devolves_into_cancel (self):
        return Finisher.devolves_into_cancel(self) or self.me.form.is_elemental
    def hit_trigger (self):
        if self.me.life >= 4:
            if self.game.make_fork (2, self.me,
                                    "Spend 3 life to repeat this attack?",
                                    ("No", "Yes")):
                self.me.lose_life (3)
                self.me.max_attacks += 1
    def evaluate_setup (self):
        if self.game.distance() >= 4 and self.me.form.is_human:
            return (2.5 if self.me.life >= 7 else
                    (1.5 if self.me.life >= 4 else 0.5))
        else:
            return 0

class Spellbolt (Base):
    minrange = 2
    maxrange = 6
    power = 2
    priority = 3
    preferred_range = 4
    def hit_trigger (self):
        if self.me.form.is_elemental:
            self.me.pull ((0,1,2))
        else:
            self.opponent.add_triggered_power_bonus(-2)
    @property
    def ordered_hit_trigger(self):
        return self.me.form.is_elemental

class Flare (Style):
    power = 3
    mean_priority = 1
    def reveal_trigger (self):
        if self.me.form.is_human:
            self.me.lose_life (3)
            self.me.add_triggered_priority_bonus(3)
    def end_trigger (self):
        self.me.form = self.me.human_form
        self.me.set_active_cards()

class Caustic (Style):
    power = 1
    priority = -1
    def hit_trigger (self):
        if self.me.form.is_elemental:
            self.opponent.stun()
    @property
    def soak (self):
        return 2 * (self.me.form.is_human)

class Volcanic (Style):
    minrange = 2
    maxrange = 4
    preferred_range = 3
    mean_priority = 1
    def hit_trigger (self):
        if self.me.form.is_elemental:
            self.me.priority_penalty_status_effect.activate(-2)
    def end_trigger (self):
        if self.me.form.is_human:
            self.me.move_to_unoccupied()
    @property
    def ordered_end_trigger(self):
        return self.me.form.is_human

class Ignition (Style):
    power = 1
    priority = -1
    def reveal_trigger (self):
        if self.me.form.is_elemental:
            self.me.lose_life (3)
            self.me.add_triggered_power_bonus(3)
    def end_trigger (self):
        self.me.form = self.me.elemental_form
    # if Kallistar is human and Ignition is in discard, that delays ignition
    @property
    def discard_penalty (self):
        return 0 if self.me.form.is_elemental else 2 * self.me.elemental_form.get_value()

class Blazing (Style):
    priority = 1
    def get_maxrange_bonus (self):
        return 1 * self.me.form.is_elemental
    def get_preferred_range (self):
        return 0.5 * self.me.form.is_elemental
    def after_trigger(self):
        if self.me.form.is_human:
            self.me.move ((1,2))
    @property
    def ordered_after_trigger(self):
        return self.me.form.is_human

class HumanForm(Card):
    is_human = True
    is_elemental = False
    soak = 1
    def get_value(self):
        return 0

class ElementalForm(Card):
    is_elemental = True
    is_human = False
    power = 2
    priority = 2
    def ante_trigger(self):
        self.me.lose_life(1)
    def get_value(self):
        return 1 + (self.me.life <= 2)

#Karin

class RedMoonRage (Finisher):
    power = 10
    priority = 12
    standard_range = False
    def special_range_hit (self):
        return abs (self.me.position - self.me.jager_position) == 2 and \
               self.me.position + self.me.jager_position == \
                       2 * self.opponent.position
    def evaluate_setup (self):
        return 3 if abs (self.me.position - self.me.jager_position) == 2 and \
           self.me.position + self.me.jager_position == 2 * self.opponent.position \
           else 0

class LunarCross (Finisher):
    power = 6
    priority = 5
    standard_range = False
    # Swap places with Jager if possible.
    def before_trigger (self):
        if self.me.jager_position not in \
                                       [self.me.position, self.opponent.position]:
            old_pos = self.me.position
            self.me.move_directly([self.me.jager_position])
            if self.me.position != old_pos:
                self.me.jager_position = old_pos
                if ordered(self.me.position,
                           self.opponent.position,
                           old_pos):
                    self.me.lunar_swap = True
    ordered_before_trigger = True
    def special_range_hit (self):
        return self.me.lunar_swap
    def evaluate_setup (self):
        return 1 if ordered(self.me.jager_position,
                            self.opponent.position,
                            self.me.position) else 0

class Claw (Base):
    minrange = 1
    maxrange = 2
    power = 2
    priority = 4
    preferred_range = 2.5
    def before_trigger (self):
        self.me.advance ((1,2))
    ordered_before_trigger = True
    def hit_trigger (self):
        if self.me.jager_position == self.opponent.position:
            self.me.add_triggered_power_bonus (2)
        self.me.move_jager ((-1,1))
    ordered_hit_trigger = True
    def evaluation_bonus (self):
        return (0.15 if self.opponent.position == self.me.jager_position
                else -0.05)

class Howling (Style):
    minrange = -1
    priority = 1
    karin_attack = False
    jager_attack = True
    preferred_range = -0.5
    def has_stun_immunity (self):
        return True
    def hit_trigger (self):
        if self.me.jager_position == self.opponent.position:
            self.me.add_triggered_power_bonus(2)
    ordered_hit_trigger = True
    def end_trigger (self):
        self.me.move_jager ((-1,-0,1))
    def evaluation_bonus (self):
        return (0.2 if self.opponent.position == self.me.jager_position
                else -0.1)

class Coordinated (Style):
    maxrange = 1
    preferred_range = 0.5
    karin_attack = True
    jager_attack = False
    # block opponent from moving into Jager's space
    # if Jager is on Karin, don't block it, so that opponent can jump over Karin
    def blocks_opponent_movement (self, initiator, direct):
        if self.me.position == self.me.jager_position:
            return set()
        else:
            return set([self.me.jager_position])
    # deals damage if opponent attempted to enter Jager's position
    def movement_reaction (self, initiator, mover, old_position, direct):
        # For an opponent to run afoul of Jager, the following must be
        # true:
        # Opponent was supposed to move.
        if mover is self.me:
            return
        # She didn't.
        if mover.position != old_position:
            return
        # It wasn't by choice.
        if old_position in mover.dests:
            return
        # Moving to Jager was a possible choice.
        if self.me.jager_position not in mover.dests:
            return
        # Moving to Jager wasn't obstructed by opponent's ability.
        unobstructed = self.me.get_unobstructed_positions(
                               old_position, mover.mover_block, direct)
        if self.me.jager_position not in unobstructed:
            return
        if self.game.reporting:
            self.game.report("%s is blocked by Jager"
                             % self.opponent.name)
        self.me.deal_damage(2)
    def end_trigger (self):
        self.me.move_jager ((-1,0,1))
    def evaluation_bonus (self):
        opp = self.opponent.position
        me = self.me.position
        jager = self.me.jager_position
        dist = abs(jager-opp)
        # Blocking is irrelevant when jager behind me, on me, on opponent,
        # or very far from opponent
        if ordered(opp, me, jager) or jager == me or jager == opp or dist > 2:
            return -0.5
        # It's best when blocking forward movement
        elif ordered (opp, jager, me):
            return 0.5
        # and less so when blocking backward movement
        else:
            return 0.25

class FullMoon (Style):
    maxrange = 1
    priority = 1
    preferred_range = 0.5
    karin_attack = True
    jager_attack = False
    def get_soak(self):
        return 2 if ordered(self.me.position,
                            self.me.jager_position,
                            self.opponent.position) else 0
    # For next turn, can also get it with swapping
    @property
    def soak (self):
        return 2 if ((self.me.position - self.me.opponent.position) *
                      (self.me.jager_position - self.me.opponent.position) > 0
                      and self.me.position != self.me.jager_position) else 0
    def get_power_bonus (self):
        return 2 if ordered(self.me.position,
                            self.opponent.position,
                            self.me.jager_position) else 0
    def start_trigger (self):
        if self.me.jager_position not in [self.me.position, \
                                          self.opponent.position]:
            if self.game.make_fork (2, self.me, "Swap places with Jager?",
                                    ["No", "Yes"]):
                old_pos = self.me.position
                self.me.move_directly ([self.me.jager_position])
                if self.me.position == self.me.jager_position:
                    self.me.jager_position = old_pos
    ordered_start_trigger = True
    def evaluation_bonus (self):
        me = self.me.position
        opp = self.opponent.position
        jager = self.me.jager_position
        # With swapping, I can get either power or soak,
        # as long as Jager not on opponent.
        if jager in (opp,me):
            return -0.2
        else:
            return 0.1

class Feral (Style):
    priority = 1
    preferred_range = 1
    karin_attack = True
    jager_attack = False
    def start_trigger (self):
        self.me.advance ([1,2])
    ordered_start_trigger = True
    def hit_trigger (self):
        self.me.retreat ([2])
        if self.opponent.position == self.me.jager_position:
            self.opponent.stun()
    ordered_hit_trigger = True
    def end_trigger (self):
        self.me.move ([0,1])
    ordered_end_trigger = True
    def evaluation_bonus (self):
        return (0.3 if self.opponent.position == self.me.jager_position
                else -0.1)

class Dual (Style):
    karin_attack = True
    jager_attack = False
    def before_trigger (self):
        jager_on_opponent = (self.me.jager_position == self.opponent.position)
        # jager uses relative movement
        relative_moves = [pos - self.me.jager_position for pos in xrange(7)
                          if pos not in (self.me.position,
                                         self.opponent.position,
                                         self.me.jager_position)]
        self.me.move_jager (relative_moves)
        if jager_on_opponent:
            self.me.move_opponent_directly ([self.me.jager_position])
    ordered_before_trigger = True
    def evaluation_bonus (self):
        return (0.45 if self.opponent.position == self.me.jager_position
                else -0.15)

#Kehrolyn

class HydraFork (Finisher):
    minrange = 1
    maxrange = 3
    power = 6
    def has_stun_immunity (self):
        return True
    def after_trigger (self):
        self.me.gain_life (5)
    def evaluate_setup (self):
        return 1 if self.game.distance() <= 3 else 0

# NOT IMPLEMENTED
class TheAugustStrain (Finisher):
    minrange = 1
    maxrange = 2
    power = 4
    priority = 5
    soak = 2
    stunguard = 2
    # On Hit: choose a style and set it aside.  It is applied from now on.

class Overload (Base):
    minrange = 1
    maxrange = 1
    power = 3
    priority = 3
    preferred_range = 1
    # automatically loses clashes
    clash_priority = -0.1
    mean_priority = 2.5
    def start_trigger (self):
        # list 2 available styles to play
        available_styles = set (self.me.styles) - set([self.me.style]) \
                         - self.me.discard[1] - self.me.discard[2]
        available_styles = sorted (list(available_styles), \
                                   key=attrgetter('order'))
        # choose one, and add it to active styles
        prompt = "Select a style to overload:"
        options = [s.name for s in available_styles]
        choice = self.game.make_fork (len(available_styles), self.me, prompt,
                                      options)
        self.me.overloaded_style = available_styles[choice]
        if self.game.reporting:
            self.game.report ("Kehrolyn overloads %s" % self.me.overloaded_style)
        self.me.set_active_cards()
        # In case Mutating was overloaded.
        self.me.overloaded_style.reveal_trigger()

class Mutating (Style):
    # There's errata to have this work on reveal rather start of beat.
    def reveal_trigger (self):
        if self == self.me.current_form:
            self.me.mutated_style = self.me.style
        if self in (self.me.style, self.me.overloaded_style):
            self.me.mutated_style = self.me.current_form
        self.me.set_active_cards()
    def end_trigger(self):
        self.me.lose_life(1)
    # doubles as Whip if Whip is current form
    def get_preferred_range (self):
        return (self.me.whip.preferred_range if self.me.whip in self.me.discard[1] else 0)
    # Doesn't grant soak, but can be expected to when evaluating
    def get_soak(self):
        return 0
    @property
    def soak(self):
        return 2 if self.me.exoskeletal in self.me.discard[1] else 0
    discard_penalty = -0.5
    # Duplication handled on character level.

class Bladed (Style):
    power = 2
    stunguard = 2

class Whip (Style):
    maxrange = 1
    # adds 1 range (for 0.5 bonus), but more powerful at range 1
    preferred_range = 0.3
    # doubled if Mutating is current form
    def get_preferred_range (self):
        return (2 if self.me.mutating in self.me.discard[1] else 1) \
               * self.preferred_range
    def hit_trigger (self):
        if self.game.distance() == 1:
            self.opponent.stun()
        else:
            self.me.pull ([1])
    ordered_hit_trigger = True

class Quicksilver (Style):
    priority = 2
    def end_trigger (self):
        self.me.move ((0,1))
    ordered_end_trigger = True

class Exoskeletal (Style):
    # soak is 2, but can be expected to give 4 if mutating is current
    # form
    def get_soak(self):
        return 2
    @property
    def soak(self):
        return 4 if self.me.mutating in self.me.discard[1] else 2
    # may block pull/push  - fork to decide whether to block
    def blocks_own_movement (self, initiator):
        if initiator is self.opponent and self.game.make_fork(2, self.me,
            "Block %s's attempt to move you?" % self.opponent,
            ["No", "Yes"]):
            return set(xrange(7)) #block
        else:
            return set() #don't block

#Khadath

class DimensionalExile (Finisher):
    power = 25
    priority  = 0
    standard_range = False
    def special_range_hit (self):
        return self.opponent.position == self.me.trap_position
    def has_stun_immunity (self):
        return True
    def evaluate_setup (self):
        return 1 if self.opponent.position == self.me.trap_position else 0

# NOT IMPLEMENTED
class PlanarCrossing (Finisher):
    minrange = 1
    maxrange = 2
    power = 4
    priority = 5
    # On hit: move all characters and markers to any legal position

class Snare (Base):
    power = 3
    priority = 1
    standard_range = False
    # just something average - the evaluation bonus really does the trick
    preferred_range = 2
    def has_stun_immunity (self):
        return True
    def special_range_hit (self):
        return (self.me.trap_position is not None and
                abs(self.me.trap_position - self.opponent.position) <= 1)
    # immobility of trap handled Khadath.move_trap()
    def evaluation_bonus(self):
        if self.me.trap_position is not None and \
           abs(self.me.trap_position - self.opponent.position) <= 1:
            return 0.5
        else:
            return -0.5

class Hunters (Style):
    name_override = "Hunter's"
    mean_priority = 1.2
    def reveal_trigger (self):
        if self.me.trap_position is not None and \
           abs(self.me.trap_position - self.opponent.position) <= 1:
            self.me.add_triggered_power_bonus(2)
            self.me.add_triggered_priority_bonus(2)
    def evaluation_bonus(self):
        if self.me.trap_position is not None and \
           abs(self.me.trap_position - self.opponent.position) <= 1:
            return 0.5
        else:
            return -0.5

class Teleport (Style):
    power = 1
    priority = -3
    def can_be_hit (self):
        trap = self.me.trap_position
        return trap is None or not ordered(self.me.position,
                                           trap,
                                           self.opponent.position)
    def end_trigger (self):
        self.me.move_to_unoccupied()
        # move trap anywhere (except current location, I guess)
        positions = set(xrange(7))
        if self.me.trap_position is not None:
            positions.remove(self.me.trap_position)
        self.me.move_trap (positions)
    ordered_end_trigger = True
    def evaluation_bonus(self):
        # Teleport is better when trap is between me and opponent
        if self.me.trap_position is not None and \
           ordered (self.me.position,
                    self.me.trap_position,
                    self.opponent.position):
            return 0.25
        else:
            return -0.25

class Evacuation (Style):
    maxrange = 1
    # in corner, no retreat, so extra range is effective
    def get_preferred_range (self):
        return 0.5 if self.me.position in [0,6] else 0
    def start_trigger (self):
        self.me.move_trap ([self.me.position])
        self.me.retreat ([1])
    ordered_start_trigger = True
    def can_be_hit (self):
        return self.opponent.position != self.me.trap_position
    def evaluation_bonus (self):
        return -0.5 if self.me.position in (0,6) else 0.25

class Blight (Style):
    maxrange = 2
    preferred_range = 1
    def start_trigger (self):
        pos_set = self.me.in_standard_range()
        pos_set.discard(self.opponent.position)
        if pos_set:
            self.me.move_trap (pos_set)
    ordered_start_trigger = True

class Lure (Style):
    maxrange = 5
    power = -1
    priority = -1
    preferred_range = 2.5
    def hit_trigger (self):
        self.me.pull (range(self.game.distance()))
    ordered_hit_trigger = True

#Larimore
class Dragonflare(Finisher):
    minrange = 2
    maxrange = 6
    power = 20
    priority = 6
    def before_trigger(self):
        spend = self.me.spend_counters('to attack', 5, True)
        if not spend:
            self.opponent.set_triggered_dodge()
    def evaluate_setup(self):
        return (1 if self.game.distance() in (3,4,5) and
                     self.me.counters >= 2 else 0)

class TheJump(Finisher):
    minrange = 1
    maxrange = 1
    power = 8
    priority = 2
    def can_be_hit(self):
        return self.opponent.moved
    def before_trigger(self):
        p = self.opponent.position
        self.me.move_directly([p-1, p+1])
    ordered_before_trigger = True

class Might(Base):
    minrange = 1
    maxrange = 3
    power = 1
    priority = 3
    preferred_range = 2
    stunguard = 3
    def hit_trigger(self):
        spend = self.me.spend_counters('for +1 power and 1 push each')
        if spend:
            self.me.add_triggered_power_bonus(spend)
            self.me.push(range(spend+1), max_move=True)

class Warmages(Style):
    name_override = "Warmage's"
    minrange = 2
    maxrange = 4
    priority = 1
    preferred_range = 2.5 # 2-4, but can be made 0-4 at a cost
    def start_trigger(self):
        if self.game.distance() == 1:
            max_ret = self.opponent.retreat_range()
            spend = self.me.spend_counters('to push opponent', max_ret)
            if spend:
                self.me.push([spend])
    @property
    def ordered_start_trigger(self):
        return self.me.counters > 0
    def hit_trigger(self):
        max_pull = self.game.distance() + self.me.retreat_range()
        spend = self.me.spend_counters('to pull opponent', max_pull)
        if spend:
            self.me.pull([spend])
    @property
    def ordered_hit_trigger(self):
        return self.me.counters > 0

class Blistering(Style):
    power = 2
    priority = -1
    def get_preferred_range(self):
        return 0.5 if self.me.counters >= 2 else 0
    def start_trigger (self):
        if self.me.spend_counters('to become active player', 5, True):
            self.me.become_active_player()
            self.game.report("Larimore becomes Active Player")
        if self.me.spend_counters('for +0-2 range', 2, True):
            self.me.blistering_range = True
            if self.game.reporting:
                self.game.report("Larimore gains +0~2 range")

class Dragonbreath(Style):
    minrange = 1
    maxrange = 3
    preferred_range = 2
    stunguard = 3 # for eval
    def get_stunguard(self):
        # 4, minus 1 per counter recharged.
        return 4 - self.me.strat[2][0]
    def hit_trigger(self):
        if (self.me.attack_range() == 3 and
            self.me.spend_counters('for +2 power', 2, True)):
            self.me.add_triggered_power_bonus(2)
    def end_trigger(self):
        self.me.move ([1,2])

class Dire(Style):
    power = 1
    priority = 1
    def has_stun_immunity(self):
        return self.me.get_priority() < 0

class Incinerating(Style):
    maxrange = 1
    power = 1
    preferred_range = 0.5
    def hit_trigger(self):
        if self.me.spend_counters('to ignore soak and stunguard', 3, True):
            self.me.blistering_ignores = True
            if self.game.reporting:
                self.game.report("Larimore ignores soak and stunguard")
    def damage_trigger(self, damage):
        max_spend = min(2, self.opponent.retreat_range())
        spend = self.me.spend_counters('to push opponent', max_spend)
        if spend:
            self.me.push([spend])
    @property
    def ordered_damage_trigger(self):
        return self.me.counters > 0

#Lesandra

class InvokeDuststalker(Finisher):
    minrange = 1
    maxrange = 3
    power = 3
    priority = 5
    def start_trigger(self):
        loss = self.me.life - 1
        self.me.lose_life(loss)
        self.me.add_triggered_power_bonus(loss)
    def damage_trigger(self, damage):
        self.me.gain_life(damage)
    def evaluate_setup(self):
        return 0.2 * self.me.life if self.game.distance() <= 3 else 0

class Mazzaroth(Finisher):
    is_attack = False
    priority = 1
    def start_trigger(self):
        self.me.lose_life(self.me.life -1)
    def after_trigger(self):
        self.me.opponent_eliminated_status_effect.activate()

class Summons(Base):
    minrange = 1
    maxrange = 1
    power = 2
    priority = 2
    stunguard = 3
    preferred_range = 1
    def start_trigger(self):
        me = self.me.position
        self.me.move_opponent_directly([me-1, me+1])
    ordered_start_trigger = True
    def evaluation_bonus(self):
        return 0.1 * self.game.distance() - 0.3

class Invocation(Style):
    minrange = 1
    maxrange = 3
    power = 2
    priority = -2
    preferred_range = 2
    def take_a_hit_trigger(self):
        if (self.me.active_familiar and
            self.game.make_fork(2, self.me, "Banish %s for Soak 2?" %
                                self.me.active_familiar, ["No", "Yes"])):
            if self.game.reporting:
                self.game.report("Lesandra banishes her %s for Soak 2" %
                                 self.me.active_familiar)
            self.me.active_familiar = None
            self.me.invocation_soak = 2
    @property # for eval
    def soak(self):
        return 1 if self.me.active_familiar else 0
    def get_soak(self):
        return 0
    # Taking damage trigger handled by Lesandra.take_damage_trigger()
    def evaluation_bonus(self):
        return 0.1 if self.me.active_familiar else -0.2

class Pactbreaker(Style):
    maxrange = 1
    power = 1
    priority = 1
    preferred_range = 0.5
    def can_hit(self):
        return self.me.anted_familiar is not None
    def evaluation_bonus(self):
        return 0.3 if self.me.active_familiar else -0.6

class Binding(Style):
    maxrange = 2
    priority = -2
    preferred_range = 1
    stunguard = 2
    def movement_reaction(self, initiator, mover, old_position, direct):
        # Requires actual movement (FAQ disagrees).
        if initiator is self.opponent and mover.position != old_position:
            self.opponent.lose_life(1)
            self.opponent.add_triggered_power_bonus(-1)

class Guardian(Style):
    maxrange = 1
    power = 1
    priority = -1
    preferred_range = 0.5
    def get_stunguard(self):
        return 2 if self.me.active_familiar else 0
    # for evaluation
    @property
    def stunguard(self):
        return 1 if self.me.active_familiar else 0
    # Cost reduction handled by Familiar.get_cost()

class Window(Style):
    @property
    def maxrange(self):
        return 3 if self.me.active_familiar else 0
    priority = 1
    def get_preferred_range(self):
        # 1.5 if we have a familar, but we might ante it.
        return 1 if self.me.active_familiar else 0
    def hit_trigger(self):
        if self.game.reporting and self.me.active_familiar:
            self.game.report("Lesandra banishes her %s" % self.me.active_familiar)
        self.me.active_familiar = None
        self.me.pull([1,2])
    ordered_hit_trigger = True
    def evaluation_bonus(self):
        #Range is nice, but then there's forced banishment.
        return 0.1 if self.me.active_familiar else -0.2

class Familiar(Card):
    def get_cost(self):
        cost = self.cost
        if self.me.guardian in self.me.active_cards:
            cost -= self.me.damage_taken
        if self.me.borneo is self.me.anted_familiar:
            cost -= 1
        return max(cost, 0)

# When evaluating familiars, assume one beat of activation,
# then an ante.

class BorneoFamiliar(Familiar):
    name_override = 'Borneo'
    cost = 1
    @property
    def clash_priority(self):
        return 0.1 if self is self.me.active_familiar else 0
    # Ante ability handled by Familar.get_cost()
    def get_value(self):
        return 0.85 if self.me.life > 1 else 0.15

class Wyvern(Familiar):
    cost = 2
    def can_be_hit(self):
        return not (self is self.me.active_familiar and
                    self.game.distance() == 4)
    def before_trigger(self):
        if self is self.me.anted_familiar:
            self.me.move_to_unoccupied()
    @property
    def ordered_before_trigger(self):
        return self is self.me.anted_familiar
    def get_value(self):
        return 3 if self.game.distance() == 4 else 1.5

class Salamander(Familiar):
    cost = 3
    def get_power_bonus(self):
        return 2 if self is self.me.anted_familiar else 1
    def reduce_soak(self, soak):
        return 0
    def reduce_stunguard(self, stunguard):
        return 0 if self is self.me.anted_familiar else stunguard
    def get_value(self):
        expected_damage = 3 + 2 * self.opponent.expected_soak()
        return 0.3 * expected_damage + 0.1 * self.opponent.expected_stunguard()

class RuneKnight(Familiar):
    cost = 4
    def get_stunguard(self):
        return 3 if self is self.me.active_familiar else 0
    def get_soak(self):
        return 2 if self is self.me.anted_familiar else 0
    def get_value(self):
        return 1.8

class RavenKnight(Familiar):
    cost = 5
    def after_trigger(self):
        if self is self.me.active_familiar:
            self.opponent.lose_life(1)
    def get_priority_bonus(self):
        return 4 if self is self.me.anted_familiar else 0
    def get_value(self):
        life = self.opponent.life
        if life == 1:
            loss = 0
        elif life <= 5:
            loss = 0.5
        else:
            loss = 1
        return 1.2 + loss * 0.8

#Lixis

class VirulentMiasma (Finisher):
    minrange = 1
    maxrange = 3
    power = 4
    priority = 5
    def hit_trigger (self):
        self.me.virulent_miasma_active = True
    def evaluate_setup (self):
        return 1.5 if self.game.distance() <= 3 else 0

# NOT IMPLEMENTED
class LifeVirus (Finisher):
    minrange = 3
    maxrnage = 6
    power = 4
    priority = 3
    soak = 3
    stunguard = 2
    # On hit, name a style and base. Each time opp. reveals one, they lose 3 life

class Lance (Base):
    minrange = 2
    maxrange = 2
    power = 3
    priority = 5
    preferred_range = 2
    # block movement into spaces adjacent to me
    def blocks_opponent_movement (self, initiator, direct):
        pos = self.me.position
        return set ([pos+1, pos-1]) & set(xrange(7))

class Venomous (Style):
    power = 1
    stunguard = 2
    mean_priority = 1
    def before_trigger (self):
        self.me.advance ([0,1])
    ordered_before_trigger = True
    def hit_trigger (self):
        self.me.priority_penalty_status_effect.activate(-2)

class Rooted (Style):
    minrange = -1
    power = 1
    priority = -2
    soak = 2
    # preferred_range is 0, reducing minrange isn't usually effective
    # may block pull/push  - fork to decide whether to block
    def blocks_own_movement (self, initiator):
        # optional negation of own movement implemented by always adding 0
        # to Lixis' move options (handled by Lixis.execute_move())
        if initiator is self.opponent and self.game.make_fork(2, self.me,
            "Block %s's attempt to move you?" % self.opponent,
            ["No", "Yes"]):
            return set(xrange(7)) #block
        else:
            return set() #don't block

class Naturalizing (Style):
    maxrange = 1
    power = -1
    priority = 1
    preferred_range = 0.5
    # blocking bonuses and tokens handled by Lixis.blocks_xxx() methods.

class Vine (Style):
    maxrange = 2
    priority = -2
    stunguard = 3
    preferred_range = 1
    def hit_trigger (self):
        self.me.pull ([0,1,2])
    ordered_hit_trigger = True

class Pruning (Style):
    maxrange = 1
    power = -1
    priority = -2
    mean_priority = 1
    preferred_range = 0.5
    def reveal_trigger (self):
        bases = self.count_bases()
        self.me.add_triggered_power_bonus(bases)
        self.me.add_triggered_priority_bonus(bases)
    def hit_trigger(self):
        self.me.discard_opponent_base()
    def count_bases (self):
        her = self.opponent
        return len (set(her.bases) & (her.discard[1] | her.discard[2]))
    # generally weak but not when there are lots of bases in opponent's discard
    def evaluation_bonus (self):
        return 0.3 * (self.count_bases() - 3)

#Luc

class TemporalRecursion (Finisher):
    minrange = 1
    maxrange = 1
    power = 3
    priority = 5
    def before_trigger (self):
        self.me.advance ((0,1))
    ordered_before_trigger = True
    # for efficiency, immediately decide how many forks to make
    # this may be incorrect if opponent has forks during your activation
    def after_trigger (self):
        if self.me.max_attacks == 1 and self.me.can_spend (1):
            extra_attacks = self.game.make_fork (len(self.me.pool) + 1, \
                                                 self.me, \
                            "Number of Time tokens to spend for extra attacks?")
            for unused in range (extra_attacks):
                self.me.spend_token()
            self.me.max_attacks += extra_attacks
            if self.game.reporting and extra_attacks > 0:
                self.game.report ("Luc attacks %d more times" %extra_attacks)

# NOT IMPLEMENTED
class StassisCharge (Finisher):
    minrange = 1
    maxrange = 1
    power = 4
    priority = 3
    stunguard = 4
    # Each time opponent would activate a trigger,
    # negate it and gain a token instead.

class Flash (Base):
    minrange = 1
    maxrange = 1
    power = 1
    priority = 6
    preferred_range = 1.5 # effective range 1-2
    def start_trigger(self):
        self.me.advance ([1])
    ordered_start_trigger = True
#     def damage_trigger(self, damage):
#         # Can't check if stunned yet, but will be stunned unless
#         # immune.
#         if not self.opponent.has_stun_immunity():
#             self.me.recover_tokens(1)
    def reduce_stunguard(self, stunguard):
        return 0
    def evaluation_bonus(self):
        return 0.03 * self.opponent.expected_stunguard() - 0.03

class Eternal (Style):
    priority = -4
    def take_a_hit_trigger (self):
        if self.me.can_spend (1):
            self.me.eternal_soak = \
                self.game.make_fork (1+len(self.me.pool), self.me,
                                     "Spend how many tokens (+1 Soak each)?")
            for unused in range (self.me.eternal_soak):
                self.me.spend_token()
            if self.me.eternal_soak and self.game.reporting:
                self.game.report ("Luc gains +%d soak (%d total)"
                                  %(self.me.eternal_soak,
                                    self.me.eternal_soak+1))
    # for evaluation
    @property
    def soak(self):
        return 1 + 0.6 * len(self.me.pool)
    def get_soak (self):
        return 1

class Memento (Style):
    priority = 1
    def after_trigger (self):
        if self.me.max_attacks == 1 and self.me.can_spend (2):
            if self.game.make_fork (2, self.me, \
                                    "Spend 2 Time tokens for an extra attack?",
                                    ["No", "Yes"]):
                for unused in xrange(2):
                    self.me.spend_token ()
                self.me.max_attacks += 1
                if self.game.reporting:
                    self.game.report ("Luc attacks again")

class Fusion (Style):
    priority = 1
    def damage_trigger (self, damage):
        # Check if destination for full push is beyond board.
        blow_out = (self.me.get_destinations(self.opponent, (-damage,))
                    == set())
        opp = self.opponent.position
        push_positions = (pos_range(0, opp-1)
                          if opp < self.me.position
                          else pos_range(opp+1, 7))
        self.me.push(range(damage+1), max_move=True)
        # if destination beyond board, and opponent didn't block any
        # spaces behind her, life loss occurs.
        if blow_out and not (self.opponent.mover_block & push_positions):
            self.opponent.lose_life(2)
    ordered_damage_trigger = True

class Feinting (Style):
    minrange = 1
    maxrange = 1
    priority = -2
    # in corner, no retreat, so extra range is effective
    def get_preferred_range (self):
        return 1 if self.me.position in [0,6] else 0
    def start_trigger (self):
        self.me.retreat ([1])
    ordered_start_trigger = True
    def end_trigger (self):
        self.me.advance ((1,2))
    ordered_end_trigger = True
    def evaluation_bonus (self):
        return -0.3 if self.me.position in (0,6) else 0.15

class Chrono (Style):
    priority = 1
    # can go far, but at a cost
    preferred_range = 0.3
    def start_trigger (self):
        if self.me.can_spend (1):
            # Can't go past opponent.
            max_advance = abs(self.me.position -
                              self.opponent.position) - 1
            max_advance = min (max_advance, len (self.me.pool))
            advance = self.game.make_fork (max_advance + 1, self.me, \
                                "How many Time tokens to spend for advancing?")
            for unused in range(advance):
                self.me.spend_token()
            if advance > 0:
                self.me.advance ([advance])
    @property
    def ordered_start_trigger(self):
        return self.me.can_spend(1)

# I can't give each token an ante effect, as the effect depends on the total
# number anted
class Time (Token):
    pass

# Virtual Cards that help implement Luc's ante effects.
class Ante1Effect(Card):
    priority = 1

class Ante3Effect(Card):
    name_override = "Ante Effect" # For order fork options
    def start_trigger(self):
        self.me.advance([0,1,2])
    ordered_start_trigger = True

class Ante5Effect(Card):
    def start_trigger(self):
        self.me.become_active_player()

#Lymn

class Megrim (Finisher):
    is_attack = False
    power = None
    def can_be_hit(self):
        return self.me.disparity < 3
    def end_trigger(self):
        self.opponent.lose_life (self.me.disparity)

class Conceit (Finisher):
    is_attack = False
    power = None
    priority = 7
    def can_be_hit(self):
        return self.me.disparity < 3
    def end_trigger(self):
        self.opponent.lose_life (self.me.disparity)

class Visions (Base):
    minrange = 1
    maxrange = 1
    @property
    def power (self):
        return min (5, self.me.disparity)
    printed_power = None
    priority = 3
    preferred_range = 1.5
    def before_trigger (self):
        self.me.advance(range(self.me.disparity + 1))
    ordered_before_trigger = True
    def hit_trigger (self):
        self.me.move_opponent([1,2])
    ordered_hit_trigger = True

class Maddening (Style):
    maxrange = 1
    printed_power = None
    preferred_range = 0.5
    def start_trigger (self):
        self.me.alt_pair_power = self.me.disparity
        self.opponent.alt_pair_power = self.me.disparity
    def get_soak (self):
        if self.me.disparity >= 5:
            return 5
        if self.me.disparity >= 3:
            return 2
        return 0
    # for evaluation
    soak = 1

class Chimeric (Style):
    power = -1
    priority = 2
    preferred_range = 0.5
    def get_power_bonus (self):
        disparity = self.me.disparity
        bonus = 0
        if disparity >= 2:
            bonus += 1
        if disparity >= 4:
            bonus += 2
        if disparity >= 6:
            bonus += 3
        return bonus
    def before_trigger (self):
        self.me.advance([1])
    ordered_before_trigger = True

class Surreal (Style):
    minrange = 1
    @property
    def maxrange (self):
        return self.me.disparity
    priority = 1
    preferred_range = 1.5
    def after_trigger (self):
        pos_set = self.me.in_standard_range()
        if pos_set:
            self.me.move_directly(sorted(list(pos_set)))
    @property
    def ordered_after_trigger(self):
        return self.me.is_attacking() and self.me.standard_range()

class Reverie (Style):
    minrange = 1
    maxrange = 3
    power = 3
    priority = -1
    preferred_range = 2
    clash_priority = 0.1
    mean_priority = -0.5
    def start_trigger (self):
        disparity = self.me.disparity
        if disparity >= 3:
            self.me.lose_life (2)
        if disparity >= 5:
            self.me.stun()

class Fathomless (Style):
    maxrange = 1
    power = -1
    priority = -2
    mean_priority = -1
    preferred_range = 0.5
    def get_stunguard(self):
        return 2 * (self.me.disparity >= 3)
    # for evaluation
    stunguard = 1
    def get_power_bonus(self):
        if self.me.disparity >= 6:
            return 4
        if self.me.disparity >= 3:
            return 2
        return 0
    def has_stun_immunity (self):
        return (self.me.disparity >= 6)
    def start_trigger (self):
        if self.me.disparity >= 8:
            self.me.become_active_player()
            self.me.move_to_unoccupied()
    ordered_start_trigger = True
    def evaluation_bonus(self):
        # Good against Dash
        if self.opponent.base_set != 'alpha':
            return 0
        return (-0.5 if self.opponent.dash in (self.opponent.discard[1] |
                                               self.opponent.discard[2])
                else 0.5)


#Magdelina

class SolarSoul (Finisher):
    minrange = 1
    maxrange = 2
    power = 2
    priority = 4
    soak = 2
    def hit_trigger (self):
        spend = self.game.make_fork (1+self.me.trance, self.me,
                                 "Spend how many Trance counters (+1 Power each)?")
        if spend:
            if self.game.reporting:
                self.game.report("Magdelinga spends %d Trance counters" % spend)
            self.me.trance -= spend
            self.me.add_triggered_power_bonus(spend)
    def evaluate_setup (self):
        return (1 + 0.25 * self.me.trance
                if self.game.distance() <= 2 else 0)

class Apotheosis(Finisher):
    minrange = 1
    maxrange = 1
    power = 1
    priority = 4
    def hit_trigger(self):
        self.opponent.stun()
        self.me.gain_level()
    def evaluate_setup(self):
        if self.game.distance() > 1:
            return 0
        if self.me.level == 5:
            return 0.25
        return 1

class Blessing (Base):
    minrange = 1
    maxrange = 2
    power = None
    priority = 3
    stunguard = 3
    preferred_range = 1.5
    deals_damage = False
    def hit_trigger (self):
        self.me.gain_life (3)
        self.opponent.gain_life (3)
        self.me.gain_trance()

class Safety (Style):
    power = -2
    priority = -1
    def get_damage_cap (self):
        return self.me.level + 3
    def end_trigger (self):
        if self.opponent.hits_scored:
            self.me.move ((0,1,2,3))
    @property
    def ordered_end_trigger(self):
        return self.opponent.hits_scored
    def evaluation_bonus(self):
        return 0.15 * (0.5 ** self.me.level - 0.5)

class Priestess (Style):
    power = -2
    priority = -1
    def hit_trigger (self):
        self.opponent.add_triggered_power_bonus (-2)
    def after_trigger (self):
        self.me.gain_life (1)

class Spiritual (Style):
    power = 1
    priority = 1
    # Don't gain a Trance counter unless you scored a hit.
    # Handled by Magdelina.unique_ability_end_trigger()

class Sanctimonious (Style):
    priority = -2
    def get_maxrange_bonus (self):
        return self.me.trance
    def get_preferred_range (self):
        return self.me.trance / 2.0
    def evaluation_bonus (self):
        return 0.3 * (self.me.trance - 1)

class Excelsius (Style):
    maxrange = 1
    power = -2
    priority = -1
    preferred_range = 1 # adds 2 to effective maxrange
    def before_trigger(self):
        self.me.move ([1])
    ordered_before_trigger = True
    def hit_trigger (self):
        if self.me.level >= 2:
            self.me.push ((2,1,0))
    @property
    def ordered_hit_trigger(self):
        return self.me.level >= 2

#Marmelee

class AstralCannon (Finisher):
    minrange = 2
    maxrange = 4
    priority = 4
    def has_stun_immunity (self):
        return True
    def start_trigger (self):
        n = self.me.counters
        self.me.discard_counters(n)
        self.me.add_triggered_power_bonus(2*n)
    def evaluate_setup (self):
        return 1 if self.me.counters >= 3 and self.game.distance() in (2,3,4) \
               else 0

class AstralTrance (Finisher):
    is_attack = False
    power = None
    soak = 5
    # recover all tokens
    def after_trigger (self):
        self.me.recover_counters (5)

class Meditation (Base):
    minrange = 1
    maxrange = 1
    power = 2
    priority = 3
    preferred_range = 1
    def start_trigger (self):
        self.meditation_soak = self.me.spend_counters('for 1 soak each')
        if self.game.reporting and self.me.meditation_soak:
            self.game.report ("Marmelee gains Soak %d"
                              %self.me.meditation_soak)
    # for evaluation
    @property
    def soak(self):
        return 0.6 * self.me.counters
    def get_soak (self):
        return 0
    def end_trigger (self):
        self.me.recover_counters (1)
    # When pool is empty, Meditation/Sorceress order matters.
    @property
    def ordered_end_trigger(self):
        return self.me.counters == 0

class Petrifying (Style):
    power = 1
    priority = -1
    mean_priority = 0.5
    def start_trigger (self):
        if self.me.spend_counters('to become active player', 3, True):
            self.me.become_active_player()
    def hit_trigger (self):
        if self.me.spend_counters('to stun opponent', 2, True):
            self.opponent.stun()
    def evaluation_bonus (self):
        value = 0.2 if self.me.counters >=3 else -0.3
        if self.me.counters == 2 or self.me.counters == 5:
            value += 0.1
        return value

class Magnificent (Style):
    minrange = 1
    maxrange = 2
    power = -1
    preferred_range = 1.5
    def hit_trigger (self):
        spend = self.me.spend_counters('for +1 power each')
        self.me.add_triggered_power_bonus(spend)
    def after_trigger (self):
        if self.me.spend_counters('to move anywhere', 2, True):
            self.me.move_to_unoccupied ()
    @property
    def ordered_after_trigger(self):
        return self.me.counters >= 2
    def evaluation_bonus(self):
        return 0.1 * (self.me.counters - 2)

class Sorceress (Style):
    priority = -1
    preferred_range = 0.5 # 0-2 is 1, but it costs a token
    def before_trigger (self):
        if self.me.is_attacking():
            self.me.sorceress_range = \
                self.me.spend_counters('for +0-2 range', 1, True)
            if self.me.sorceress_range and self.game.reporting:
                self.game.report ("Marmelee gains +0~2 range")
    def end_trigger (self):
        if self.me.spend_counters('to move 1 space', 1, True):
            self.me.move ([1])
    # With counters, order matters vs. Parry.
    # Without, it matters vs. Meditation.
    ordered_end_trigger = True
    def evaluation_bonus(self):
        n = min(2, self.me.counters)
        return 0.1 * (n-1)

class Barrier (Style):
    maxrange = 1
    power = -1
    preferred_range = 0.5
    def start_trigger (self):
        if self.me.spend_counters('to dodge all attacks', 4, True):
            self.me.set_triggered_dodge()
    def before_trigger (self):
        if self.game.distance() == 1:
            max_push = self.opponent.retreat_range()
            spend = self.me.spend_counters('to push opponent', max_push)
            if spend:
                self.me.push([spend])
    @property
    def ordered_before_trigger(self):
        return self.me.counters >= 1
    def evaluation_bonus (self):
        return 0.3 if self.me.counters >=4 else -0.1

class Nullifying (Style):
    minrange = 1
    maxrange = 1
    priority = 1
    def get_preferred_range (self):
        return 1 if self.me.position in (0,6) else 0
    def start_trigger (self):
        self.me.retreat ([1])
    ordered_start_trigger = True
    def hit_trigger (self):
        spend = self.me.spend_counters('for -1 power to opponent each')
        self.opponent.add_triggered_power_bonus(-spend)
    def evaluation_bonus(self):
        return ((-0.3 if self.me.position in (0,6) else 0.15) +
                0.05 * (self.me.counters - 2))

#Mikhail

class MagnusMalleus (Finisher):
    minrange = 2
    maxrange = 4
    power = 2
    priority = 5
    def hit_trigger (self):
        if self.me.can_spend (1):
            spend = self.game.make_fork (1+len(self.me.pool), self.me,
                                     "Spend how many tokens (+3 Power each)?")
            for unused in range(spend):
                self.me.spend_token()
            self.me.add_triggered_power_bonus(3 * spend)
    def evaluate_setup (self):
        return 0.5 * len(self.me.pool) if self.me.attack_range() in [2,3,4] \
               else 0

class TheFourthSeal (Finisher):
    minrange = 1
    maxrange = 2
    power = 7
    priority = 6
    def can_hit (self):
        return not self.me.pool
    def evaluate_setup (self):
        return 1 if self.me.attack_range() <= 2 and not self.me.pool else 0

class Scroll (Base):
    minrange = 1
    maxrange = 2
    power = 2
    priority = 2
    preferred_range = 1.5
    soak = 1
    stunguard = 3
    def end_trigger (self):
        if self.me.damage_taken == 0:
            self.me.recover_tokens(1)

class Immutable (Style):
    maxrange = 1
    power = 2
    priority = -3
    preferred_range = 0.25 # halved because not necessarily active
    def take_a_hit_trigger (self):
        if not self.opponent.hits_scored and self.me.discard_token():
            self.me.immutable_soak = 3
    # for evaluation, halved because not necessarily active
    @property
    def soak(self):
        return 1.5 if self.me.pool else 0
    def get_soak (self):
        return 0

class Transcendent (Style):
    maxrange = 2
    power = -1
    priority = -1
    preferred_range = 0.5 # halved because not necessarily active
    stunguard = 5
    def before_trigger (self):
        self.me.add_triggered_power_bonus(self.me.damage_taken)

class Hallowed (Style):
    maxrange = 1
    power = 3
    priority = 1
    preferred_range = 0.5 # halved because not necessarily active
    def start_trigger (self):
        self.me.pull ([1])
    ordered_start_trigger = True
    def damage_trigger (self, damage):
        self.me.push (range(damage+1))
    ordered_damage_trigger = True

class Apocalyptic (Style):
    minrange = 2
    maxrange = 4
    power = 3
    priority = -3
    preferred_range = 1.5 # halved because not necessarily active
    soak = 2
    def has_stun_immunity (self):
        return True
    # prevention of token recovery handled by Mikhail.recover_tokens()

class Sacred (Style):
    maxrange = 1
    power = 1
    preferred_range = 0.25 # halved because not necessarily active
    def can_be_hit (self):
        return self.opponent.attack_range() < 3
    def evaluation_bonus(self):
        return 0.5 if self.game.distance() >= 3 else -0.5

class Seal (Token):
    pass

#Oriana

class NihilEraser (Finisher):
    minrange = 1
    maxrange = 6
    priority = 4
    def get_power_bonus (self):
        return 25 if self.me.ante.count(self.me.mp) >= 10 else 0
    def evaluate_setup (self):
        return 1 if self.me.ante.count(self.me.mp) == 10 else 0

class GalaxyConduit (Finisher):
    is_attack = False
    power = None
    priority = 4
    def after_trigger (self):
        ante = self.me.ante.count(self.me.mp)
        self.me.gain_life (ante)
        self.me.recover_tokens (2 * ante)
    def evaluate_setup (self):
        pool = len (self.me.pool)
        # 1 point per life, 0.5 point per MP recovered
        value = pool + 0.5 * min (pool, 10-pool)
        return 0.1 * value

class Meteor (Base):
    minrange = 2
    maxrange = 6
    power = 1
    preferred_range = 4
    def get_power_bonus (self):
        return (1 + self.me.ante.count(self.me.mp)) / 2 # round up
    def get_stunguard (self):
        return self.me.ante.count(self.me.mp)
    # for evaluation
    @property
    def stunguard(self):
        return min(5, len(self.me.pool))
    def end_trigger (self):
        self.me.recover_tokens(2)

class Celestial (Style):
    power = -1
    priority = -4
    # converting damage to life loss handled by Oriana.take_damage()
    # regaining tokens/life on life loss handled by Oriana.lose_life()
    def after_trigger (self):
        if self.me.life > 1:
            life_loss = self.game.make_fork (min(self.me.life,4), self.me,
                                        "How much life would you like to lose?")
            self.me.lose_life (life_loss)

class Stellar (Style):
    minrange = 2
    maxrange = 4
    preferred_range = 3 # without the mp; can be anything with the mp
    def before_trigger (self):
        if self.me.ante.count(self.me.mp) >= 5:
            self.me.move_to_unoccupied()
    @property
    def ordered_before_trigger(self):
        return self.me.ante.count(self.me.mp) >= 5
    def hit_trigger (self):
        if self.me.ante.count(self.me.mp) >= 2:
            self.me.move_opponent_to_unoccupied()
    @property
    def ordered_hit_trigger(self):
        return self.me.ante.count(self.me.mp) >= 2

class Unstable (Style):
    maxrange = 1
    power = 1
    priority = 1
    preferred_range = 0.5
    def hit_trigger (self):
        n_effects = self.me.ante.count(self.me.mp) / 2
        if n_effects:
            effects = self.effects_names.keys()
            combos = list(itertools.combinations (effects, n_effects))
            options = []
            if self.opponent.is_user and self.game.interactive_mode:
                options = [', '.join([self.effects_names[e] for e in combo])
                           for combo in combos]
            result = self.game.make_fork (len(combos), self.opponent,
                                     "Choose effect/s for Oriana to activate:",
                                     options)
            for effect in combos[result]:
                effect(self)
    @property
    def ordered_hit_trigger(self):
        return self.me.ante.count(self.me.mp) >= 2
    def choose_regain (self):
        self.me.recover_tokens(5)
    def choose_move (self):
        self.me.move_opponent_to_unoccupied()
        self.opponent.stun()
    def choose_lose_2 (self):
        self.opponent.lose_life (2)
    def choose_lose_3 (self):
        self.opponent.lose_life (3)
    def choose_discard (self):
        bases = list (set (self.opponent.bases) - self.opponent.discard[0] \
                                                - self.opponent.discard[1] \
                                                - self.opponent.discard[2])
        combos = list (itertools.combinations (bases, 2))
        prompt = "Choose two extra bases to discard"
        options = [', '.join([base.name for base in combo]) for combo in combos]
        chosen_combo = combos [self.game.make_fork (len(combos), self.me,
                                                    prompt, options)]
        self.opponent.discard[0] |= set (chosen_combo)
        if self.game.reporting:
            for base in chosen_combo:
                self.game.report (self.opponent.name + " discards " + base.name)

    effects_names = {choose_regain : "Oriana regains 5 MP",
                     choose_move : "Oriana moves you to any space and stuns you",
                     choose_lose_2 : "You lose 2 life",
                     choose_lose_3 : "You lose 3 life",
                     choose_discard : "You discard two additional bases this beat"}

class Metamagical (Style):
    maxrange = 1
    power = -2
    priority = 3
    preferred_range = 0.5
    mean_priority = 0.5
    def get_prefered_range (self):
        # each token adds 1 to max, but we don't have to ante them.
        return self.preferred_range + 0.2 * min (5, len(self.me.pool))
    def get_maxrange_bonus (self):
        return min (5, self.me.ante.count(self.me.mp))
    def get_power_bonus (self):
        return min (5, self.me.ante.count(self.me.mp))
    def get_priority_bonus (self):
        return -min (5, self.me.ante.count(self.me.mp))
    def hit_trigger (self):
        self.me.recover_tokens (3)

class Calamity (Style):
    power = 1
    def get_stunguard (self):
        return min (self.me.ante.count(self.me.mp), 6)
    # for evaluation
    @property
    def stunguard(self):
        return min(5, len(self.me.pool))
    def hit_trigger (self):
        mp = self.me.ante.count(self.me.mp)
        if mp >= 2:
            if self.opponent.pool and self.game.make_fork (2, self.opponent,
                                'Discard a token to prevent loss of 2 life?',
                                ['No', 'Yes']):
                self.opponent.discard_token()
            else:
                self.opponent.lose_life (2)
            self.me.recover_tokens (3)
        if mp >= 5:
            self.opponent.stun()
            self.me.recover_tokens (3)

class MagicPoint (Token):
    pass


#Ottavia

class ExtremePrejudice(Finisher):
    standard_range = False
    power = 10
    priority = 9
    def special_range_hit(self):
        return self.me.target_lock

# Not implemented.
class DoubleBarrel(Finisher):
    is_attack = False
    priority = 4

class Shooter(Base):
    minrange = 1
    maxrange = 4
    power = 4
    priority = 2
    preferred_range = 2.5
    def reduce_stunguard(self, stunguard):
        return 0
    def evaluation_bonus(self):
        return 0.03 * self.opponent.expected_stunguard() - 0.03

class Snapback(Style):
    minrange = 1
    maxrange = 3
    preferred_range = 2
    def blocks_opponent_movement(self, initiator, direct):
        if self.opponent.get_priority() < self.me.get_priority():
            return set(xrange(7))
        else:
            return set()
    def after_trigger(self):
        self.me.retreat([1,2])
    ordered_after_trigger = True

class Demolition(Style):
    priority = 1
    def hit_trigger(self):
        if self.opponent.get_priority() <= 1:
            self.opponent.stun()
    def after_trigger(self):
        self.me.advance(range(5))
    ordered_after_trigger = True

class Cover(Style):
    minrange = 1
    maxrange = 2
    power = 1
    priority = -3
    soak = 1
    stunguard = 3
    preferred_range = 1.5
    def after_trigger(self):
        self.me.set_triggered_dodge()

class AntiPersonnel(Style):
    name_override = 'Anti-Personnel'
    minrange = 2
    maxrange = 3
    power = 3
    priority = -1
    preferred_range = 2.5
    def take_a_hit_trigger(self):
        self.me.stun()
    def end_trigger(self):
        pos_set = self.me.in_standard_range()
        if pos_set:
            self.me.move_directly(sorted(list(pos_set)))
    ordered_end_trigger = True

class Cybernetic(Style):
    def damage_trigger(self, damage):
        self.me.move_opponent(range(4))
    def after_trigger(self):
        self.me.cybernetic_soak = 2
    # for evaluation
    soak = 1
    def get_soak(self):
        return 0

#Rexan

class ZeroHour (Finisher):
    is_attack = False
    power = None
    def start_trigger(self):
        self.me.deal_damage(3 * len(self.me.induced_pool))
    def end_trigger (self):
        if len(self.me.induced_pool) == 3:
            raise WinException (self.me.my_number)
    def evaluate_setup (self):
        values = [0, 0.25, .75, 1]
        return values[len(self.me.induced_pool)]

# Restricting all stats to printed *base* power is not implemented
class BlackEclipse (Finisher):
    minrange = 1
    maxrange = 1
    power = 3
    priority = 7
    def hit_trigger (self):
        raise NotImplementedError

class Malediction (Base):
    minrange = 1
    maxrange = 6
    power = 2
    priority = 4
    preferred_range = 3
    def hit_trigger (self):
        self.me.malediction_damage_limit = True
        self.me.give_induced_tokens(1)
    # To give the token before Enervating counts it
    ordered_hit_trigger = True
    def end_trigger (self):
        max_pull = len(self.me.induced_pool)
        if max_pull:
            self.me.pull (range(max_pull+1))
    @property
    def ordered_end_trigger(self):
        return self.me.induced_pool

class Unyielding (Style):
    maxrange = 1
    power = -1
    preferred_range = 0.5
    def special_range_hit(self):
        return self.me.curse in self.opponent.ante
    def can_be_hit (self):
        return len(self.me.induced_pool) < 3
    def evaluation_bonus (self):
        return 0.5 if len(self.me.induced_pool) == 3 else -0.1

class Devastating (Style):
    power = 2
    priority = -1
    # advance until adjacent to opponent
    def start_trigger (self):
        self.me.advance([self.game.distance() - 1])
    ordered_start_trigger = True
    def get_damage_cap (self):
        return 4

class Enervating (Style):
    priority = -2
    stunguard = 3
    preferred_range = 0.5
    def give_power_penalty (self):
        return -len(self.me.induced_pool)
    def hit_trigger (self):
        self.me.add_triggered_power_bonus(len(self.me.induced_pool))
    # To count after Maledicition
    ordered_hit_trigger = True

class Vainglorious (Style):
    power = 1
    preferred_range = 0.5
    def give_power_penalty (self):
        return -2 if self.game.distance() == 1 else 0
    def before_trigger (self):
        self.me.pull ((0,1))
    ordered_before_trigger = True
    def evaluation_bonus (self):
        dist = self.game.distance()
        if dist == 1:
            return 0.25
        else:
            return -0.25

class Overlords (Style):
    name_override = "Overlord's"
    power = 1
    mean_priority = 0.5
    preferred_range = 0.75 # can be anything, but long ranges lose too much power
    def reveal_trigger (self):
        self.opponent.add_triggered_priority_bonus(-len(self.me.induced_pool))
    def before_trigger (self):
        if self.me.base.is_attack and self.me.base.deals_damage:
            power = self.me.get_power()
            self.me.pull(range(power+1), specific_movement_reaction=self.specific_movement_reaction)
    @property
    def ordered_before_trigger(self):
        return self.me.base.is_attack and self.me.base.deals_damage
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        # Get -1 power per pull
        spaces_pulled = abs(self.opponent.position - old_position)
        # If opponent switched sides, actual advance is one less then distance moved
        if ordered (old_position, self.me.position, self.opponent.position):
            spaces_pulled -= 1
        self.me.add_triggered_power_bonus(-spaces_pulled)


class CurseToken (Token):
    name_override = 'Curse'
    power = -1
    priority = -1

#Rukyuk

class FullyAutomatic (Finisher):
    minrange = 3
    maxrange = 6
    power = 2
    priority = 6
    # Blocking ammo effect handled by Rukyuk.get_active_tokens()

    # for efficiency, immediately decide how many tokens to discard
    # this may be incorrect if opponent has fork
    def hit_trigger (self):
        if self.me.max_attacks == 1 and self.me.can_spend(1):
            extra_attacks = self.game.make_fork (len(self.me.pool) + 1, \
                                                 self.me, \
                            "Number of tokens to spend for extra attacks?")
            for unused in xrange(extra_attacks):
                self.me.spend_token()
            self.me.max_attacks += extra_attacks
            if self.game.reporting and extra_attacks > 0:
                self.game.report ("Rukyuk attacks %d more times" %extra_attacks)
    def evaluate_setup (self):
        return 2 if len(self.me.pool) > 2 and self.game.distance() >= 3 else 0

class ForceGrenade (Finisher):
    minrange = 1
    maxrange = 2
    power = 4
    priority = 4
    # Blocking ammo effect handled by Rukyuk.get_active_tokens()
    # Not needing a token to hit handled by Rukyuk.can_hit()
    def hit_trigger (self):
        self.me.push (range(6))
    ordered_hit_trigger = True
    def after_trigger (self):
        self.me.retreat (range(6))
    ordered_after_trigger = True
    def evaluate_setup (self):
        return 1 if self.game.distance() <= 2 else 0

class Reload (Base):
    priority = 4
    power = None
    is_attack = False
    preferred_range = 1 # low, becuase when you reload it matters less
    def after_trigger (self):
        self.me.move_to_unoccupied()
    ordered_after_trigger = True
    def end_trigger (self):
        self.me.recover_tokens()

class Sniper (Style):
    minrange = 3
    maxrange = 5
    power = 1
    priority = 2
    preferred_range = 4
    def after_trigger (self):
        self.me.move ((1,2,3))
    ordered_after_trigger = True

class PointBlank (Style):
    maxrange = 1
    stunguard = 2
    preferred_range = 0.5
    def damage_trigger (self, damage):
        self.me.push ((2,1,0))
    ordered_damage_trigger = True

class Gunner (Style):
    minrange = 2
    maxrange = 4
    preferred_range = 3
    def get_minrange_bonus (self):
        return -self.me.token_spent
    def get_maxrange_bonus (self):
        return self.me.token_spent
    # fork to decide whether to discard a token for better range
    def before_trigger (self):
        if self.me.can_spend (1):
            if self.game.make_fork (2, self.me, \
                                    "Discard a token for -1/+1 range?",
                                    ["No", "Yes"]):
                self.me.token_spent = 1
                self.me.spend_token()
                if self.game.reporting:
                    self.game.report ("Rukyuk gains -1/+1 range")
    def after_trigger (self):
        self.me.move ((1,2))
    ordered_after_trigger = True

class Crossfire (Style):
    minrange = 2
    maxrange = 3
    priority = -2
    soak = 2
    preferred_range = 2.5
    def hit_trigger (self):
        if self.me.can_spend (1):
            if self.game.make_fork (2, self.me, \
                                    "Discard a token for +2 power?",
                                    ["No", "Yes"]):
                self.me.spend_token()
                self.me.add_triggered_power_bonus(2)

class Trick (Style):
    minrange = 1
    maxrange = 2
    priority = -3
    preferred_range = 1.5
    def has_stun_immunity (self):
        return True

class APShell (Token):
    name_override = 'AP'
    @property
    def value(self):
        return 0.2 * self.opponent.expected_soak()
    def reduce_soak (self, soak):
        return 0

class ExplosiveShell (Token):
    name_override = 'Explosive'
    value = 0.4
    power = 2

class FlashShell (Token):
    name_override = 'Flash'
    @property
    def value(self):
        return 0.06 * self.opponent.expected_stunguard()
    def reduce_stunguard (self, stunguard):
        return 0

class ImpactShell (Token):
    name_override = 'Impact'
    value = 0.2
    def hit_trigger (self):
        self.me.push ([2])
    ordered_hit_trigger = True

class LongshotShell (Token):
    name_override = 'Longshot'
    value = 0.5
    minrange = -1
    maxrange = 1

class SwiftShell (Token):
    name_override = 'Swift'
    value = 0.6
    priority = 2

#Runika

class ArtificeAvarice (Finisher):
    minrange = 1
    maxrange = 1
    power = 3
    priority = 3
    def start_trigger (self):
        for artifact in self.me.artifacts:
            self.me.activate_artifact(artifact)
        active = list(self.me.active_artifacts)
        prompt = 'Choose artifact to overcharge:'
        options = [artifact.name for artifact in active]
        ans = self.game.make_fork (len(active), self.me, prompt, options)
        self.me.overcharged_artifact = active [ans]
        if self.game.reporting:
            self.game.report ('Runika overcharges her ' +
                              self.me.overcharged_artifact.name)
        # Take care of Hover Boots' Overcharged effect.
        # (Cutting a corner - player should have the choice of using the
        # trigger before the boots are overcharged, thus negating it)
        if self.me.overcharged_artifact is self.me.hover_boots:
            self.me.become_active_player()
    def evaluate_setup (self):
        repair_value = len(self.me.deactivated_artifacts)
        return repair_value if self.game.distance() <= 2 else repair_value / 2.0
    # protecting artifacts handled by Runika.take_a_hit_trigger()

class UdstadBeam (Finisher):
    minrange = 4
    maxrange = 5
    power = 7
    priority = 3
    def has_stun_immunity(self):
        return True
    def start_trigger (self):
        self.me.retreat ([2,1,0])
    ordered_start_trigger = True
    def evaluate_setup (self):
        me = self.me.position
        opp = self.opponent.position
        if me > opp:
            pos_after_retreat = min (6, me + 2)
        else:
            pos_after_retreat = max (0, me - 2)
        return 1 if abs(pos_after_retreat - opp) >= 4 else 0
    # disabling artifacts handled by Runika.set_active_cards()

class Tinker (Base):
    minrange = 1
    maxrange = 1
    power = 3
    priority = 3
    stunguard = 3
    def after_trigger (self):
        self.me.tinkered_artifact = self.me.activation_fork()
    # Blocking deactivation handled by Runika.take_a_hit_trigger()

class Channeled (Style):
    power = 1
    priority = 1
    def start_trigger (self):
        self.me.channeled_artifact = self.me.activation_fork ()
    def end_trigger (self):
        # At end of beat, the only thing that matters is artifact value,
        # so might as well make it a fake fork.
        self.me.deactivation_fork (self.me, fake=True)
    # Forced deactivation choice handled by Runika.take_a_hit_trigger()

class Maintenance (Style):
    power = -1
    priority = -1
    @property
    def soak (self):
        return len(self.me.deactivated_artifacts)
    def after_trigger (self):
        self.me.retreat ([2,1])
    ordered_after_trigger = True

class Explosive (Style):
    power = -1
    preferred_range = 0.5
    def start_trigger (self):
        self.me.pull([0,1])
    ordered_start_trigger = True
    def hit_trigger (self):
        if self.me.active_artifacts:
            if self.game.make_fork (2, self.me,
                    "Deactivate artifact to gain +2 power?",
                               ["No", "Yes"]):
                self.me.deactivation_fork (self.me)
                self.me.add_triggered_power_bonus (2)

class Impact (Style):
    power = -1
    preferred_range = 0.5
    def before_trigger (self):
        self.me.advance ([1])
    ordered_before_trigger = True
    def hit_trigger (self):
        self.me.push ([2,1])
    ordered_hit_trigger = True

class Overcharged (Style):
    # Phase Goggles can be overcharged to increase max range by 1
    # (so mean range is +0.5, and less because I might use something else)
    def get_preferred_range (self):
        if self.me.phase_goggles in self.me.active_artifacts:
            return 0.5 / len(self.me.active_artifacts)
        else:
            return 0
    def start_trigger (self):
        active = list(self.me.active_artifacts)
        if not active:
            if self.game.reporting:
                self.game.report ('No active artifacts to overcharge')
            return
        prompt = 'Choose artifact to overcharge:'
        options = [artifact.name for artifact in active]
        ans = self.game.make_fork (len(active), self.me, prompt, options)
        self.me.overcharged_artifact = active [ans]
        if self.game.reporting:
            self.game.report ('Runika overcharges her ' +
                              self.me.overcharged_artifact.name)
        # Take care of Hover Boots' Overcharged effect.
        # (Cutting a corner - player should have the choice of using the
        # trigger before the boots are overcharged, thus negating it)
        if self.me.overcharged_artifact is self.me.hover_boots:
            self.me.become_active_player()
    def end_trigger (self):
        self.me.remove_artifact (self.me.overcharged_artifact)
    # deactivation immunity handled by Runika.deactivation_fork()

# values are per beat of being active
class Artifact (Card):
    pass

class Autodeflector (Artifact):
    value = 0.8
    def get_soak (self):
        if self is self.me.overcharged_artifact:
            return 4
        return 2
    # life loss immunity handled by Runika.lose_life()

class PhaseGoggles (Artifact):
    value = 0.75
    def get_maxrange_bonus (self):
        if self is self.me.overcharged_artifact:
            return 2
        return 1
    def reduce_stunguard(self, stunguard):
        return 0 if self is self.me.overcharged_artifact else stunguard

class HoverBoots (Artifact):
    value = 0.7
    priority = 2
    # Overcharged trigger is handled by Overcharged.start_trigger().
    # This makes sure it activates, even if HoverBoots just became active
    # (via Tinker)

class ShieldAmulet (Artifact):
    value = 0.65
    def get_stunguard (self):
        if self is self.me.overcharged_artifact:
            return 0
        return 3
    def has_stun_immunity (self):
        return self is self.me.overcharged_artifact
    def blocks_own_movement (self, initiator):
        return (set(xrange(7))
                if self is self.me.overcharged_artifact and
                    initiator is self.opponent
                else set())

class Battlefist (Artifact):
    value = 0.6
    def get_power_bonus (self):
        if self is self.me.overcharged_artifact:
            return 3
        return 1
    def reduce_soak(self, soak):
        return 0 if self is self.me.overcharged_artifact else soak

#Sarafina

class RitherwhyteInfusionSarafina(Finisher):
    name_override = "Ritherwhyte Infusion"
    minrange = 1
    maxrange = 3
    power = 3
    priority = 4
    def after_trigger (self):
        self.me.ritherwhyte_activated = True

class FulminatingVortex(Finisher):
    minrange = 3
    maxrange = 3
    power = 8
    priority = 8
    def evaluate_setup(self):
        return 1 if self.game.distance() == 3 else 0

class Field(Base):
    standard_range = False
    power = 3
    priority = 4
    preferred_range = 3
    def special_range_hit(self):
        return (self.me.projection is not None and
                abs(self.opponent.position - self.me.projection) == 1)
    def hit_trigger(self):
        self.me.field_immobilzie = True
    ordered_hit_trigger = True
    def evaluation_bonus(self):
        p = self.me.projection
        o = self.opponent.position
        return -0.2 if (p is None or abs(o - p) != 1) else 0.4

class SarafinaStyle(Style):
    def end_trigger(self):
        self.me.project(self.projection_range)
    ordered_end_trigger = True

class Silver(SarafinaStyle):
    priority = 1
    preferred_range = 1
    projection_range = 2
    def before_trigger(self):
        self.me.advance([1,2])
    ordered_before_trigger = True
    # Blocking power and priority bonuses handled by Sarafina.

class Locus(SarafinaStyle):
    priority = -1
    mean_priority = 0
    projection_range = 3
    def reveal_trigger(self):
        if self.me.switched_sides:
            self.me.add_triggered_priority_bonus(4)
    def evaluation_bonus (self):
        return (0.3 if self.me.projection is not None
                and ordered(self.me.position, self.opponent.position, self.me.projection)
                else -0.1)

class Implosion(SarafinaStyle):
    power = -1
    priority = -1
    projection_range = 4
    def can_be_hit(self):
        return self.opponent.attack_range() != 1
    def hit_trigger(self):
        self.me.pull([1,2])
    ordered_hit_trigger = True
    def evaluation_bonus(self):
        return 0.3 if self.game.distance() == 1 else -0.7

class Hollow(SarafinaStyle):
    minrange = 1
    maxrange = 2
    priority = -2
    projection_range = 1
    preferred_range = 1.5
    def hit_trigger(self):
        if self.game.distance() == 3:
            self.me.add_triggered_power_bonus(2)
    def evaluation_bonus(self):
        return 0.05 if self.game.distance() == 3 else -0.15

class Tactical(SarafinaStyle):
    power = 1
    priority = 1
    def end_trigger(self):
        if not self.me.is_stunned():
            self.me.project(None)

#Seth

# Not Implemented
class ReadingFate (Finisher):
    minrange = 1
    maxrange = 3
    priority = 6
    def hit_trigger (self):
        self.opponent.stun()
        self.reading_fate_active = True
    def evaluate_setup (self):
        return 2 if self.game.distanc() <= 3 else 0

class FortuneBuster (Finisher):
    deals_damage = False
    minrange = 1
    maxrange = 6
    power = None
    priority = 13
    # Wins priority, even against other finishers (that get 0.3)
    clash_priority = 0.4
    def hit_trigger (self):
        if isinstance(self.opponent.base, Finisher):
            raise WinException (self.me.my_number)
    def evaluate_setup (self):
        return (0.5 if self.opponent.special_action_available and
                     self.opponent.life <= 7
                else 0)

class Omen (Base):
    minrange = 1
    maxrange = 1
    power = 3
    priority = 1
    preferred_range = 1
    def start_trigger (self):
        if self.me.correct_guess:
            self.opponent.stun()

class Fools (Style):
    name_override = "Fool's"
    power = -1
    priority = -4
    def give_minrange_penalty (self):
        return -1
    def give_maxrange_penalty (self):
        return -1
    def evaluation_bonus(self):
        return 0.9 - 0.3 * self.opponent.preferred_range

class Mimics (Style):
    name_override = "Mimic's"
    power = 1
    def mimics_movement (self):
        return True

class Vanishing (Style):
    minrange = 1
    maxrange = 1
    # in corner, no retreat, so extra range is mandatory
    def get_preferred_range (self):
        return 1 if self.me.position in [0,6] else 0.5
    def start_trigger (self):
        self.me.retreat ((1,0))
    ordered_start_trigger = True
    def can_be_hit (self):
        return self.opponent.attack_range() < 4
    def evaluation_bonus (self):
        cornered = self.me.position in (0,6)
        dodge_range = 4 if cornered else 3
        return ((0.25 if self.game.distance() >= dodge_range else -0.25) +
                (-0.3 if cornered else 0.15))

class Wyrding (Style):
    priority = 1
    # some of the logic is on Seth.start_trigger()
    def start_trigger (self):
        if self.me.correct_guess:
            available_bases = set(self.me.bases) \
                            - self.me.discard[0] \
                            - self.me.discard[1] \
                            - self.me.discard[2]
            available_bases = sorted (list(available_bases), \
                                      key = attrgetter('order'))
            prompt = "Choose new base:"
            options = [b.name for b in available_bases] + ['Keep my base']
            ans = self.game.make_fork (len(available_bases)+1, self.me,
                                       prompt, options)
            new_base = available_bases[ans] if ans<len(available_bases) else None
            if new_base:
                self.me.discard[0].add(new_base)
                self.me.base = new_base
                self.me.set_active_cards()
                if self.game.reporting:
                    self.game.report ("Seth selects " + new_base.name + " as his new base")

class Compelling (Style):
    preferred_range = 0.5
    def before_trigger (self):
        self.me.move_opponent([1])
    ordered_before_trigger = True
    def after_trigger (self):
        self.me.move_opponent([1])
    ordered_after_trigger = True

#Shekhtur

# NOT IMPLEMENTED
class SoulBreaker (Finisher):
    minrange = 1
    maxrange = 1
    power = 3
    priority = 3
    def damage_trigger (self, damage):
        self.opponent.stun()
        # MISSING: opponent can't ante, except for mandatory.

class CoffinNails (Finisher):
    minrange = 1
    maxrange = 1
    power = 3
    priority = 3
    def damage_trigger (self, damage):
        self.opponent.stun()
        self.me.coffin_nails_hit = True

class Brand (Base):
    minrange = 1
    maxrange = 2
    power = 3
    priority = 2
    preferred_range = 1.5
    def reduce_stunguard (self, stunguard):
        return 0 if self.me.get_priority() >= 6 else stunguard
    def after_trigger (self):
        if self.me.hits_scored and self.me.can_spend(2):
            max_leech = min(4,len(self.me.pool)) / 2
            leech = self.game.make_fork (max_leech+1, self.me, \
                      "How much life to leech (spend 2 tokens per life point)?")
            for unused in xrange(2*leech):
                self.me.spend_token()
            self.opponent.lose_life (leech)
            self.me.gain_life (leech)
    def evaluation_bonus(self):
        return 0.02 * self.opponent.expected_stunguard() - 0.02

class Unleashed (Style):
    maxrange = 1
    power = -1
    preferred_range = 0.5
    def after_trigger (self):
        self.me.retreat ((2,1))
    ordered_after_trigger = True
    def end_trigger (self):
        self.me.recover_tokens(2)
        self.me.power_bonus_status_effect.activate(1)

class Combination (Style):
    power = 2
    def reduce_soak (self, soak):
        return 0 if self.me.get_priority() >= 7 else soak
    def can_hit (self):
        return self.me.attack_range() < 3
    def hit_trigger (self):
        if self.me.hits_scored_last_beat:
            self.me.add_triggered_power_bonus(2)
    def evaluation_bonus (self):
        return (0.4 if self.game.distance() <=2 else -0.4) + \
               (0.2 if self.me.hits_scored else -0.2) + \
               0.1 * self.opponent.expected_soak()

class Reaver (Style):
    maxrange = 1
    preferred_range = 0.5
    def damage_trigger (self, damage):
        self.me.push([damage])
    ordered_damage_trigger = True
    def end_trigger (self):
        self.me.advance ((1,2))
    ordered_end_trigger = True

class Jugular (Style):
    power = 1
    priority = 2
    def hit_trigger (self):
        self.me.move_opponent ([1])
    ordered_hit_trigger = True
    def end_trigger (self):
        while len(self.me.pool) > 3:
            self.me.discard_token()
        if len(self.me.pool) < 3:
            self.me.recover_tokens (3-len(self.me.pool))

class Spiral (Style):
    priority = -1
    preferred_range = 1 # range 0-3, but prefers shorter ones
    def before_trigger (self):
        self.me.advance ([0,1,2,3], specific_movement_reaction=self.specific_movement_reaction)
    ordered_before_trigger = True
    def specific_movement_reaction(self, initiator, mover, old_position, direct):
        # Get -1 power per advance.
        spaces_advanced = abs(self.me.position-old_position)
        # If I switched sides, actual advance is one less then distance moved
        if ordered (old_position, self.opponent.position, self.me.position):
            spaces_advanced -= 1
        self.me.add_triggered_power_bonus (spaces_advanced)


class Malice (Token):
    priority = 1

#Tanis

# Not Implemented
class EmpathyStrings(Finisher):
    pass

class CurtainCall(Finisher):
    minrange = 1
    maxrange = 2
    power = 2
    priority = 5
    def before_trigger(self):
        for puppet in self.me.puppets:
            if (puppet.position is not None and
                abs(puppet.position - self.me.position) == 1):
                self.me.add_triggered_power_bonus(4)
    def evaluate_setup(self):
        if self.game.distance() > 2:
            return 0
        value = 0
        for puppet in self.me.puppets:
            if abs(puppet.position - self.me.position) == 1:
                value += 0.5
        return value

class SceneShift(Base):
    standard_range = False
    power = 2
    priority = 4
    preferred_range = 2.5
    def special_range_hit(self):
        return self.me.switched_sides
    def before_trigger(self):
        self.me.advance(range(5))
    ordered_before_trigger = True
    # Move each puppet up to 3 spaces.
    # Possessed puppet isn't on board, so it isn't moved.
    def hit_trigger(self):
        for puppet in self.me.puppets:
            if puppet is not self.me.possessed_puppet:
                old_pos = puppet.position
                positions = range(max(0, old_pos - 3),
                                  min(6, old_pos + 3) + 1)
                prompt = "Choose new position for %s:" % puppet
                options = []
                if self.me.is_user and self.game.interactive_mode:
                    for pos in positions:
                        opt = ['.'] * 7
                        opt[pos] = puppet.initial
                        options.append(''.join(opt))
                pos = positions[
                    self.game.make_fork(len(positions), self.me,
                                        prompt, options)]
                puppet.position = pos
                if (old_pos == self.opponent.position and
                    pos != old_pos):
                    self.me.add_triggered_power_bonus(1)
        if self.game.reporting:
            self.game.report("Tanis moves her puppets:")
            for line in self.game.get_board():
                self.game.report(line)
    def evaluation_bonus(self):
        opp = self.opponent.position
        if opp in [0,6]:
            return -0.5
        value = 0
        for puppet in self.me.puppets:
            if puppet.position == opp:
                value += 0.1
        return value

class Valiant(Style):
    minrange = 1
    maxrange = 1
    priority = -1
    preferred_range = 1
    @property
    def soak(self):
        if (self.me.loki.position is not None and
            ordered(self.me.position,
                    self.me.loki.position,
                    self.opponent.position)):
            return 3
        else:
            return 0
    def blocks_own_movement(self, initiator):
        return self.blocked_set()
    def blocks_opponent_movement(self, initiator, direct):
        return self.blocked_set()
    def blocked_set(self):
        loki = self.me.loki.position
        if loki is None:
            return set()
        else:
            return set([loki])
    def evaluation_bonus(self):
        return 0.4 if ordered(self.me.position,
                              self.me.loki.position,
                              self.opponent.position) else -0.1

class Climactic(Style):
    priority = 1
    def reduce_soak(self, soak):
        return 0
    def before_trigger(self):
        mephisto = self.me.mephisto.position
        if mephisto not in (self.me.position, self.opponent.position, None):
            old_pos = self.me.position
            self.me.move_directly([mephisto])
            if self.me.position == mephisto:
                self.me.mephisto.position = old_pos
                if self.game.reporting:
                    self.game.report("Mephisto moves:")
                    for line in self.game.get_board():
                        self.game.report(line)
    @property
    def ordered_before_trigger(self):
        return self.me.mephisto.position not in [self.me.position,
                                                 self.opponent.position,
                                                 None]
    def evaluation_bonus(self):
        me = self.me.position
        opp = self.opponent.position
        mephisto = self.me.mephisto.position
        if mephisto == opp:
            return -0.1
        my_range = abs(me - opp)
        mephisto_range = abs(mephisto - opp)
        diff = abs(my_range - mephisto_range)
        return 0.1 * diff - 0.1 + 0.1 * self.opponent.expected_soak()

class Storyteller(Style):
    power = 1
    priority = 1
    def get_preferred_range(self):
        # extra range if I possess Eris:
        if (self.me.position == self.me.eris.position and
            self.me.possessed_puppet is not self.me.eris):
            return 1.5
        else:
            return 0
    def reduce_stunguard(self, stunguard):
        return 0 if self.me.possessed_puppet is self.me.mephisto else stunguard
    def can_be_hit(self):
        return not (self.me.possessed_puppet is self.me.loki and
                    self.opponent.attack_range() == 1)
    def before_trigger(self):
        if self.me.possessed_puppet is self.me.eris:
            self.me.advance(range(4))
    @property
    def ordered_before_trigger(self):
        return self.me.possessed_puppet is self.me.eris
    def evaluation_bonus(self):
        loki = self.me.loki
        possible_puppets = [p for p in self.me.puppets
                            if p.position == self.me.position]
        possible_values = []
        if loki in possible_puppets:
            possible_values.append(
                0.5 if abs(loki.position - self.opponent.position) == 1
                else -0.1)
        if self.me.mephisto in possible_puppets:
            possible_values.append(0.03 * self.opponent.expected_stunguard() - 0.03)
        if self.me.eris in possible_puppets:
            possible_values.append(0)
        return max(possible_values)

class Playful(Style):
    maxrange = 1
    preferred_range = 0.5
    mean_priority = 0.2
    suspend_blocking = False
    def end_trigger(self):
        self.me.move_directly(set(xrange(7)) - set([self.me.position]))
    ordered_end_trigger = True
    def get_maxrange_bonus(self):
        return self.opponent.get_maxrange_bonuses()
    def get_power_bonus(self):
        return self.opponent.get_power_bonuses()
    def get_priority_bonus(self):
        return self.opponent.get_priority_bonuses()
    # Blocking bonuses handled by character, stealing them handled
    # above.

class Distressed(Style):
    priority = -1
    def get_preferred_range(self):
        me = self.me.position
        eris = self.me.eris.position
        opp = self.opponent.position
        # If I'm definitely possessing Eris, no extra range.
        if me == eris and not me in (self.me.loki.position,
                                     self.me.mephisto.position):
            return 0
        if (me-opp) * (eris-opp) >= 0:
            return 2
        else:
            return 0
    def start_trigger(self):
        if self.me.possessed_puppet is not self.me.eris:
            me = self.me.position
            opp = self.opponent.position
            eris = self.me.eris.position
            direction = (me - opp) * (eris - opp)
            if direction > 0:
                self.me.pull([1,2])
            elif direction < 0:
                self.me.push([1,2])
            else:
                self.me.move_opponent([1,2])
    @property
    def ordered_start_trigger(self):
        return self.me.possessed_puppet is not self.me.eris
    def end_trigger(self):
        puppets = [p for p in self.me.puppets
                   if p is not self.me.possessed_puppet]
        prompt = "Select a puppet to move:"
        options = [p.name for p in puppets]
        puppet = puppets[self.game.make_fork(len(puppets), self.me,
                                             prompt, options)]
        positions = range(7)
        prompt = "Choose new position for %s:", puppet
        options = []
        if self.me.is_user and self.game.interactive_mode:
            for pos in positions:
                opt = ['.'] * 7
                opt[pos] = puppet.initial
                options.append(''.join(opt))
        pos = positions[
            self.game.make_fork(7, self.me, prompt, options)]
        puppet.position = pos
        if self.game.reporting:
            self.game.report("Tanis moves %s:" % puppet)
            for line in self.game.get_board():
                self.game.report(line)
    def evaluation_bonus(self):
        me = self.me.position
        if (me == self.me.eris.position and
            not me in (self.me.loki.position,
                       self.me.mephisto.position)):
            return -0.2
        else:
            return 0.1

# Making it a card gives it a name, and a link to the game object.
class Puppet(Card):
    pass
class Eris(Puppet):
    initial = 'e'
class Loki(Puppet):
    initial = 'l'
class Mephisto(Puppet):
    initial = 'm'

#Tatsumi

class TsunamisCollide (Finisher):
    minrange = 2
    maxrange = 4
    def can_hit (self):
        return self.me.juto_position is not None and \
               abs(self.opponent.position - self.me.juto_position) > 1
    def reveal_trigger (self):
        tsunami_spaces = \
            (abs (self.me.position - self.me.juto_position) - 1
             if self.me.zone_3()
             else 0)
        self.me.add_triggered_power_bonus(3 * tsunami_spaces)
        self.me.add_triggered_priority_bonus(2 * tsunami_spaces)
    def evaluate_setup (self):
        if self.can_hit() and self.game.distance() > 1 and self.me.zone_3():
            return 1
        else:
            return 0

class GuardianXandjinn (Finisher):
    power = 6
    priority = 5
    standard_range = False
    def special_range_hit (self):
        return self.me.juto_position is not None and \
               abs(self.opponent.position - self.me.juto_position) <= 1
    def hit_trigger (self):
        self.opponent.stun()
        self.me.move_juto(range(7))
    def evaluate_setup(self):
        return 0.5 if self.special_range_hit() else 0

class Whirlpool (Base):
    minrange = 1
    maxrange = 2
    power = 3
    priority = 3
    # Effective maxrange is 1 when I'm pushing opponent, 2.5 when pulling,
    # 1.5 when nothing
    def get_preferred_range (self):
        if self.me.juto_position in (None, self.opponent.position):
            return 1.5
        if self.me.zone_3():
            return 1
        return 2.5
    # move opponent 1 towards juto
    def start_trigger (self):
        if self.me.juto_position in (None, self.opponent.position):
            return
        # juto behind opponent, push opponent.
        if self.me.zone_3():
            self.me.push([1])
        # otherwise, pull.
        else:
            self.me.pull([1])
    @property
    def ordered_start_trigger(self):
        return self.me.juto_position is not None
    def after_trigger (self):
        self.me.move ((0,1,2))
        if self.me.juto_position is not None:
            juto_dests = [d for d in xrange(7) if d >= self.me.juto_position - 2 \
                                        and  d <= self.me.juto_position + 2]
            self.me.move_juto (juto_dests)
    ordered_after_trigger = True

class Siren (Style):
    power = -1
    priority = 1
    tatsumi_attack = True
    juto_attack = False
    def hit_trigger (self):
        self.opponent.stun()
    def end_trigger (self):
        if self.me.juto_position is not None:
            juto_dests = [d for d in xrange(7) if d >= self.me.juto_position - 2 \
                                        and  d <= self.me.juto_position + 2]
            self.me.move_juto (juto_dests)

class Fearless (Style):
    minrange = -1
    priority = 1
    preferred_range = 0
    tatsumi_attack = False
    @property
    def juto_attack(self):
        return self.me.juto_position is not None
    def can_hit (self):
        return self.me.juto_position is not None
    # special range handled by Tatsumi.attack_range()
    def end_trigger (self):
        if self.me.juto_position is None:
            self.me.juto_position = self.me.position
            self.me.juto_life = 4
            if self.game.reporting:
                self.game.report ("Juto is revived:")
                for s in self.game.get_board():
                    self.game.report (s)

class Riptide (Style):
    maxrange = 2
    priority = -1
    preferred_range = 1
    tatsumi_attack = True
    juto_attack = False
    def start_trigger (self):
        self.me.riptide_dodge = self.me.zone_2 ()
    @property
    def ordered_start_trigger(self):
        return self.me.juto_position is not None
    # move juto any number of spaces towards tatsumi
    def end_trigger (self):
        if self.me.juto_position is not None:
            destinations = pos_range(self.me.position,
                                     self.me.juto_position)
            self.me.move_juto(destinations)
    @property
    def ordered_end_trigger(self):
        return self.me.juto_position is not None
    def evaluation_bonus (self):
        return 1 if self.me.zone_2() and self.game.distance() >= 3 else -0.5


class Empathic (Style):
    priority = -1
    tatsumi_attack = True
    juto_attack = False
    # may swap places with juto
    def start_trigger (self):
        if self.me.juto_position not in [None, self.me.position, \
                                         self.opponent.position]:
            if self.game.make_fork (2, self.me,
                                    "Swap positions with Juto?",
                                    ["No","Yes"]):
                old_pos = self.me.position
                self.me.move_directly ([self.me.juto_position])
                if self.me.position == self.me.juto_position:
                    self.me.move_juto([old_pos])
    @property
    def ordered_start_trigger(self):
        return self.me.juto_position is not None
    def end_trigger (self):
        self.opponent.lose_life(self.me.juto_damage_taken)

class WaveStyle (Style):
    name_override = 'Wave'
    minrange = 2
    maxrange = 4
    power = -1
    preferred_range = 3
    tatsumi_attack = True
    juto_attack = False
    def hit_trigger (self):
        self.me.push ((2,1,0))
    ordered_hit_trigger = True
    def after_trigger (self):
        # move Juto towards opponent
        if self.me.juto_position is not None:
            destinations = [d for d in xrange(7) \
                            if (d - self.me.juto_position) * \
                               (self.opponent.position - self.me.position) >= 0]
            self.me.move_juto(destinations)
    @property
    def ordered_after_trigger(self):
        return self.me.juto_position is not None

#Vanaah

class DeathWalks (Finisher):
    minrange = 1
    maxrange = 2
    power = 5
    priority = 6
    def hit_trigger (self):
        self.opponent.stun()
        self.me.priority_penalty_status_effect.activate(-4)
    def evaluate_setup (self):
        return 1 if self.game.distance() <= 2 else 0

class HandOfDivinity (Finisher):
    name_override = "Hand of Divinity"
    minrange = 5
    maxrange = 5
    power = 7
    priority = 3
    soak = 3
    def has_stun_immunity (self):
        return True
    def after_trigger (self):
        self.me.advance (range(6))
    ordered_after_trigger = True
    def evaluate_setup (self):
        return 1 if self.game.distance() == 5 else 0

class Scythe (Base):
    minrange = 1
    maxrange = 2
    power = 3
    priority = 3
    stunguard = 3
    preferred_range = 2 # effective range 1-3
    def before_trigger(self):
        self.me.advance ([1])
    ordered_before_trigger = True
    def hit_trigger(self):
        self.me.pull ((0,1))
    ordered_hit_trigger = True

class Reaping (Style):
    maxrange = 1
    priority = 1
    preferred_range = 0.5
    def hit_trigger (self):
        # try to make opponent discard token
        # if she doesn't, retrieve Divine Rush
        if not self.opponent.discard_token():
            self.me.recover_tokens()

class Judgment (Style):
    minrange = 1
    maxrange = 1
    power = 1
    priority = -1
    preferred_range = 1
    def blocks_opponent_movement (self, initiator, direct):
        # Can't retreat or move past me.
        if direct:
            return set()
        if initiator is self.me:
            return set([self.me.position])
        direction = (self.me.position - self.opponent.position)
        direction /= abs(direction)
        return set([self.me.position,
                    self.opponent.position - direction])

class Glorious (Style):
    power = 2
    preferred_range = 0.5
    def before_trigger (self):
        self.me.advance ([0,1])
    ordered_before_trigger = True
    def can_hit (self):
        return self.me.get_priority() >= self.opponent.get_priority()
    def evaluation_bonus(self):
        return 0.4 if self.me.pool else -0.2

class Paladin (Style):
    maxrange = 1
    power = 1
    priority = -2
    preferred_range = 0.5
    stunguard = 3
    # jump to space adjacent to opponent
    def end_trigger (self):
        self.me.move_directly ((self.opponent.position+1, \
                                self.opponent.position-1))
    ordered_end_trigger = True

class Vengeance (Style):
    power = 2
    stunguard = 4
    def can_hit (self):
        return self.me.get_priority() <= self.opponent.get_priority()
    def evaluation_bonus(self):
        return -0.4 if self.me.pool else 0.2

class DivineRush (Token):
    power = 2
    priority = 2
    value = 2.25
    def discard_evaluation (self, discard_pile):
        return self.value * (-2/3.0)

#Voco

class ZMosh (Finisher):
    name_override = 'Z-Mosh'
    minrange = 1
    maxrange = 6
    stunguard = 4
    def hit_trigger (self):
        pos = self.opponent.position
        self.me.add_triggered_power_bonus (
                        3 * len (self.me.zombies & set ((pos-1,pos,pos+1))))
    ordered_hit_trigger = True
    def evaluate_setup (self):
        pos = self.opponent.position
        return 0.5 * len (self.me.zombies & set ((pos-1,pos,pos+1)))
    # Non-removal of zombies handled by Voco.soak_trigger()

class TheWave (Finisher):
    power = 2
    priority = 5
    standard_range = False
    def special_range_hit (self):
        return self.opponent.position in self.me.zombies
    def hit_trigger (self):
        self.me.zombies -= set([self.opponent.position])
        self.me.move_opponent ([1])
        if self.opponent.position in self.me.zombies:
            self.me.max_attacks += 1
    ordered_hit_trigger = True

class Shred (Base):
    power = 1
    priority = 4
    standard_range = False
    preferred_range = 2 # average so it doesn't interfere
    def special_range_hit (self):
        return self.opponent.position in self.me.zombies
    def hit_trigger (self):
        pos = self.opponent.position
        behind_opponent = set(xrange(0,pos)) if pos < self.me.position \
                          else set(xrange(pos+1,7))
        self.me.add_triggered_power_bonus(
                                    len (self.me.zombies & behind_opponent))
    ordered_hit_trigger = True
    def reduce_soak (self, soak):
        return 0
    def evaluation_bonus (self):
        p = self.me.position
        op = self.opponent.position
        if op in self.me.zombies:
            behind_range = set(xrange(0,op)) if op<p else set(xrange(op+1,7))
            behind = len (self.me.zombies & behind_range)
            return 0.3 + 0.15 * (behind + self.opponent.expected_soak())
        else:
            return -0.3

class Monster (Style):
    priority = -1
    # Crowdsurf - can keep advancing as long as you have zombies to step on.
    # Stop asking if you get into a loop (either jumping around the opponent,
    # or stuck with nowhere to advance).
    def before_trigger (self):
        visited_positions = [self.me.position]
        self.me.advance ([1])
        while self.me.position in self.me.zombies and \
           self.me.position not in visited_positions and \
           self.game.make_fork (2, self.me, "Keep advancing with Monster?",
                                ["No", "Yes"]):
            visited_positions.append(self.me.position)
            self.me.advance ([1])
    ordered_before_trigger = True
    # preferred range and evaluation bonus affected by crowdsurf potential
    def surf_potential (self):
        me = self.me.position
        her = self.opponent.position
        direction = (me-her) / abs(me-her)
        pos = me + direction
        surf = 0
        while pos in self.me.zombies and pos != her:
            pos += direction
            surf += 1
        return surf
    def get_preferred_range (self):
        return 0.5 * (1 + self.surf_potential())
    def evaluatiohn_bonus (self):
        surf = self.surf_potential()
        return 0.1 * surf if surf else -0.2

class Metal (Style):
    minrange = 1
    maxrange = 1
    preferred_range = 3
    def before_trigger (self):
        self.me.advance ([2])
    ordered_before_trigger = True
    def movement_reaction (self, initiator, mover, old_position, direct):
        if mover is self.me and self.me.position != old_position:
            if direct:
                self.me.add_zombies (set([old_position]))
            else:
                self.me.add_zombies ((pos_range (self.me.position,
                                                 old_position)
                                      - set([self.me.position,
                                             self.opponent.position])))

class Hellraising (Style):
    minrange = 1
    maxrange = 2
    power = -1
    priority = -2
    # in corner, no retreat, so extra range is effective
    def get_preferred_range (self):
        return 1.5 if self.me.position in [0,6] else 0.5
    def start_trigger (self):
        self.me.retreat ([1])
        self.me.add_zombies (pos_range (self.me.position,
                                        self.opponent.position) \
                              - set ((self.me.position, self.opponent.position)))
    ordered_start_trigger = True
    def evaluation_bonus(self):
        return 0.3 if self.me.position in [0,6] else -0.15

class Abyssal (Style):
    minrange = 2
    maxrange = 4
    preferred_range = 3
    def after_trigger (self):
        pos_set = self.me.in_standard_range()
        if pos_set:
            self.me.add_zombies(pos_set)
    @property
    def ordered_after_trigger(self):
        return self.me.is_attacking() and self.me.standard_range()

class Thunderous (Style):
    minrange = 1
    maxrange = 2
    power = -1
    def get_preferred_range (self):
        return 1.5 if self.me.position in (0,6) else 1
    def start_trigger (self):
        self.me.zombies.add(self.me.position)
        self.me.advance([2])
    ordered_start_trigger = True
    def damage_trigger (self, damage):
        self.me.push ((2,1,0))
    ordered_damage_trigger = True

#Xenitia

class Exsanguinate(Finisher):
    minrange = 2
    maxrange = 5
    @property
    def power(self):
        return 4 + self.me.strat[2][0]
    printed_power = None
    priority = 1
    def has_stun_immunity(self):
        return True
    def end_trigger(self):
        self.me.gain_life(3)
    def evaluate_setup(self):
        return 0.5 if self.game.distance() in (2,3,4,5) else 0

class Phlebotomy(Finisher):
    @property
    def minrange(self):
        return self.me.strat[2][0]
    @property
    def maxrange(self):
        return self.me.strat[2][0]
    power = 5
    priority = 5
    soak = 2
    def hit_trigger(self):
        self.me.gain_life(4)
    def end_trigger(self):
        self.me.gain_life(2)
    def evaluate_setup(self):
        return 0.5 if self.game.distance() < self.me.life else 0

class Bloodweapon(Base):
    maxrange = 1
    power = 3
    priority = 4
    def get_preferred_range(self):
        return 1 + 0.5 * self.me.max_draw()
    def get_minrange_bonus(self):
        return self.me.strat[2][0]
    def get_maxrange_bonus(self):
        return self.me.strat[2][0]
    def hit_trigger(self):
        self.me.gain_life(1)

class Razor(Style):
    power = 1
    priority = 1
    @property # for eval
    def stunguard(self):
        return self.me.max_draw()
    def get_stunguard(self):
        return self.me.strat[2][0]
    def start_trigger(self):
        draw = self.me.strat[2][0]
        if draw > 0 and draw == self.opponent.get_stunguard():
            self.me.razor_ignore_stunguard = True
            if self.game.reporting:
                self.game.report("Xenitia ignores stun guard this beat")

class Guilty(Style):
    power = 1
    priority = -2
    stunguard = 2
    preferred_range = 0.5
    # Blocking damage handled by Xenitia.take_damage()
    def before_trigger(self):
        self.me.move([1])
    ordered_before_trigger = True

class Draining(Style):
    maxrange = 2
    preferred_range = 1
    stunguard = 2
    def damage_trigger(self, damage):
        if self.me.attack_range() == self.me.strat[2][0]:
            self.me.gain_life(2)
            self.opponent.lose_life(1)
    ordered_damage_trigger = True

class Rending(Style):
    priority = -3
    stunguard = 3
    # If your Draw Value is 1 or higher, Opponents with priority X cannot his you. X is your draw value
    def can_be_hit(self):
        draw = self.me.strat[2][0]
        return draw == 0 or draw != self.opponent.get_priority()

class Purging(Style):
    power = 1
    priority = 1
    def get_preferred_range(self):
        return self.me.max_draw() / 2.0
    def before_trigger(self):
        self.me.move([self.me.strat[2][0]])

#Zaamassal

class OpenTheGate (Finisher):
    name_override = 'Open the Gate'
    minrange = 1
    maxrange = 2
    power = 3
    priority = 7
    def hit_trigger (self):
        self.opponent.stun()
        trios = list (itertools.combinations(self.me.paradigms, 3))
        prompt = "Select 3 paradigms to adopt:"
        options = [', '.join([p.name for p in t]) for t in trios]
        select = self.game.make_fork (len(trios), self.me, prompt, options)
        self.me.set_active_paradigms (trios[select])
    def evaluate_setup (self):
        dist = self.game.distance()
        return 2 if dist <= 2 or \
                    (dist==3 and self.me.fluidity in self.me.paradigms) or \
                    (dist in (3,4) and self.me.distortion in self.me.paradigms) \
               else 0

class PlaneDivider (Finisher):
    minrange = 1
    maxrange = 1
    power = 2
    priority = 4
    def before_trigger (self):
        self.me.move_to_unoccupied()
    def hit_trigger (self):
        self.me.move_opponent_to_unoccupied()
        self.me.add_triggered_power_bonus (self.game.distance() - 1)
        prompt = "Select new paradigm:"
        options = [p.name for p in self.me.paradigms]
        paradigm = self.me.paradigms [self.game.make_fork (5, self.me, prompt,
                                                           options)]
        if paradigm not in self.me.active_paradigms:
            self.me.set_active_paradigms ([paradigm])
            # no paradigm has a hit trigger, so no tricky stuff.
    ordered_hit_trigger = True
    def evaluate_setup (self):
        return 1

class ParadigmShift (Base):
    minrange = 2
    maxrange = 3
    power = 3
    priority = 3
    preferred_range = 2.5
    def after_trigger (self):
        prompt = "Select new paradigm:"
        options = [p.name for p in self.me.paradigms]
        paradigm = self.me.paradigms [self.game.make_fork (5, self.me, prompt,
                                                           options)]
        if paradigm not in self.me.active_paradigms:
            self.me.set_active_paradigms ([paradigm])
            paradigm.after_trigger()
    def discard_evaluation (self, discard_pile):
        return 0.5

# all zaamassal styles allow him to assume the appropriate paradigm
class ZStyle (Style):
    def after_trigger (self):
        # No need to switch if we can switch again anyway.
        if self.me.unique_base in self.me.active_cards:
            return
        # if appropriate paradigm not active, fork to decide on activation
        paradigm = self.me.paradigms [self.order]
        if paradigm not in self.me.active_paradigms:
            prompt = "Adopt the Paradigm of " + paradigm.name + "?"
            if self.game.make_fork (2, self.me, prompt, ["No", "Yes"]):
                self.me.set_active_paradigms ([paradigm])
                # if the new paradigm has an after_trigger, activate it.
                paradigm.after_trigger()

class Malicious (ZStyle):
    power = 1
    priority = -1
    stunguard = 2

class Sinuous (ZStyle):
    priority = 1
    def end_trigger (self):
        self.me.move_to_unoccupied()
    ordered_end_trigger = True

class Urgent (ZStyle):
    maxrange = 1
    power = -1
    priority = 2
    preferred_range = 1
    def before_trigger (self):
        self.me.advance ([0,1])
    ordered_before_trigger = True

class Sturdy (ZStyle):
    def has_stun_immunity (self):
        return True
    def blocks_own_movement (self, initiator):
        # optional negation of own movement implemented by always adding 0
        # to own move options (handled by Zaamassal.execute_move())
        if initiator is self.opponent and self.game.make_fork(2, self.me,
            "Block %s's attempt to move you?" % self.opponent,
            ["No", "Yes"]):
            return set(xrange(7)) #block
        else:
            return set() #don't block

class Warped (ZStyle):
    maxrange = 2
    # in corner, no retreat, so extra range is fully effective
    def get_preferred_range (self):
        return 1 if self.me.position in [0,6] else 0.5
    def start_trigger (self):
        self.me.retreat ([1])
    ordered_start_trigger = True
    def evaluation_bonus(self):
        return 0.3 if self.me.position in [0,6] else -0.15

class Paradigm (Card):
    pass

class Pain (Paradigm):
    shorthand = 'p'
    values = [0,0,0,0,0.1,0.4,0.8]
    def damage_trigger (self, damage):
        self.opponent.lose_life (2)
    # almost 2 damage usually, but less when opponent low on life
    # (can't take last life with life loss)
    def evaluate (self):
        return self.values[min(6, self.opponent.life)]

class Fluidity (Paradigm):
    shorthand = 'f'
    def before_trigger (self):
        self.me.move ([0,1])
    ordered_before_trigger = True
    def end_trigger (self):
        self.me.move ([0,1])
    ordered_end_trigger = True
    def evaluate (self):
        return 1.5

class Haste (Paradigm):
    shorthand = 'h'
    clash_priority = 0.1
    # Opponents adjacent to me can't move.
    def blocks_opponent_movement (self, initiator, direct):
        # if opponent adjacent to me, block everything
        if self.game.distance() == 1:
            return set(xrange(7))
        # otherwise, if move is direct, don't block
        if direct:
            return set()
        # if move is indirect, block switching sides
        # (once they go next to me, they can't continute moving).
        return set([self.me.position])
    # winning clashes is 0.5 priority, or 0.25 value
    def evaluate (self):
        return 1.0 if self.game.distance() == 1 else 0.25

class Resilience (Paradigm):
    shorthand = 'r'
    def start_trigger (self):
        self.me.resilience_soak = 2
    # if Resilience is replaced in the after trigger,
    # its own after trigger can be avoided
    # (style and base triggers precede paradigm triggers)
    def after_trigger (self):
        if self in self.me.active_paradigms:
            self.me.resilience_soak = 0
    def evaluate (self):
        return 1.2

class Distortion (Paradigm):
    shorthand = 'd'
    values = [0,0,0.0,1.0,1.0,0.0,0]
    def can_be_hit (self):
        return (self.opponent.attack_range() != 4)
    def special_range_hit (self):
        return (self.me.attack_range() == 3)
    # value depends on range.
    # completely useless against Heketch, who can jump right back in
    def evaluate (self):
        if isinstance(self.opponent,Heketch):
            return 0
        else:
            return self.values[self.game.distance()]


# Character name => corresponding class
character_dict = {'abarene'     :Abarene,
                  'adjenna'     :Adjenna,
                  'alexian'     :Alexian,
                  'alumis'      :Alumis,
                  'arec'        :Arec,
                  'aria'        :Aria,
                  'baenvier'    :Baenvier,
                  'borneo'      :Borneo,
                  'byron'       :Byron,
                  'cadenza'     :Cadenza,
                  'cesar'       :Cesar,
                  'claus'       :Claus,
                  'clinhyde'    :Clinhyde,
                  'danny'       :Danny,
                  'demitras'    :Demitras,
                  'eligor'      :Eligor,
                  'eustace'     :Eustace,
                  'gerard'      :Gerard,
                  'heketch'     :Heketch,
                  'hikaru'      :Hikaru,
                  'iri'         :Iri,
                  'jager'       :Jager,
                  'juto'        :Juto,
                  'kallistar'   :Kallistar,
                  'karin'       :Karin,
                  'kehrolyn'    :Kehrolyn,
                  'kajia'       :Kajia,
                  'khadath'     :Khadath,
                  'larimore'    :Larimore,
                  'lesandra'    :Lesandra,
                  'lixis'       :Lixis,
                  'luc'         :Luc,
                  'lymn'        :Lymn,
                  'magdelina'   :Magdelina,
                  'marmelee'    :Marmelee,
                  'mikhail'     :Mikhail,
                  'oriana'      :Oriana,
                  'ottavia'     :Ottavia,
                  'rexan'       :Rexan,
                  'rukyuk'      :Rukyuk,
                  'runika'      :Runika,
                  'sarafina'    :Sarafina,
                  'seth'        :Seth,
                  'shekhtur'    :Shekhtur,
                  'tanis'       :Tanis,
                  'tatsumi'     :Tatsumi,
                  'vanaah'      :Vanaah,
                  'voco'        :Voco,
                  'xenitia'     :Xenitia,
                  'zaamassal'   :Zaamassal}

if __name__ == "__main__":
    main()
