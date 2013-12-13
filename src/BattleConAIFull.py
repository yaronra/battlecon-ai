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
#                (can simulate again, but that's even longer.)
# Option 2: Evaluate in simulation the loss of a pair for 3 beats
#     Problems:
#         1. Always the same evaluation of lost pair.  There's also
#            preferred_range of lost/remaining card, but that's 
#            very rough.
#         2. Not clear how to evaluate spent ante, which might be
#            a complete loss, or still useful this beat.


# when a Cancel ends up killing the opponent, points are still deducted
# for spending the special action

# When a clash empties hands, the result is 0.  Actually, cycling should happen.

# Active ante triggers (Heketch's move) are reported twice when
# there's a Cancel

# TO CHECK OPENING DISCARDS
# free_for_all (1, <name>, skip=['kehrolyn'], first_beats=True)

from operator import attrgetter
from optparse import OptionParser
import itertools
import numpy
import os.path
import pstats
import random
import re
import solve
import time

debug_log = []

# MAIN FUNCTIONS

def main():
    parser = OptionParser()
    parser.add_option("-t", "--test",
                      action="store_true", dest="test",
                      default=False,
                      help="run test")
    parser.add_option("-b", "--beta",
                      action="store_true", dest="beta",
                      default=False,
                      help="use beta bases in test")
    parser.add_option("-a", "--ad_hoc",
                      action="store_true", dest="ad_hoc",
                      default=False,
                      help="run ad_hoc")
    options, unused_args = parser.parse_args()
    if options.test:
        test(None, options.beta)
    elif options.ad_hoc:
        ad_hoc()
    else:
        play()
    

def ad_hoc():
    beta_challenge(['luc','rukyuk'],
                   ['voco','zaamassal'],
                   True, True)
#     start_with = 'adjenna'
#     names = [n for n in playable if n >= start_with]
#     names = ['marmelee']
#     free_for_all (repeat=1,
#                   names=names, 
#                   start=start_with, 
#                   skip=[],
#                   raise_exceptions=True, 
#                   first_beats=False)

playable = ['adjenna',
            'alexian',
            'aria',
            'byron',
            'cadenza',
            'cesar',
            'clinhyde',
            'demitras',
            'eligor',
            'heketch',
            'hikaru',
            'kajia',
            'kallistar',
            'karin',
            'kehrolyn',
            'khadath',
            'lixis',
            'luc',
            'lymn',
            'magdelina',
            'marmelee',
            'mikhail',
            'oriana',
            'rexan',
            'rukyuk',
            'runika',
            'seth',
            'shekhtur',
            'tatsumi',
            'vanaah',
            'voco',
            'zaamassal']

def test (first=None, beta_bases=False):
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
        game = Game.from_start (n0, n1, beta_bases, beta_bases, 
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
    names = sorted([k.capitalize() for k in character_dict.keys()])
    while True:
        print "Select your character: [1-%d]\n" %len(names)
        human = names [menu(names)]
        print "Select AI character: [1-%d]\n" %len(names)
        ai = names[menu(names)]
        print "Which set of bases should be used?"
        ans = menu(['Standard bases',
                    'Beta bases',
                    'I use standard, AI uses beta',
                    'I use beta, AI uses standard'])
        ai_beta = ans in (1,2)
        human_beta = ans in (1,3)
        print "Default Discards?"
        default_discards = menu (["No", "Yes"])
        game = Game.from_start (ai, human, ai_beta, human_beta, 
                                default_discards=default_discards,
                                interactive=True)
        game.select_finishers()
        game_log, unused_winner = game.play_game()
        if not os.path.exists ("logs"):
            os.mkdir ("logs")
        if ai_beta:
            ai = ai + '_beta'
        if human_beta:
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

def duel (name0, name1, repeat, beta_bases0=False, beta_bases1=False,
          first_beats=False):
    victories = [0,0]
    beta_str0 = " (beta bases)" if beta_bases0 else ""
    beta_str1 = " (beta bases)" if beta_bases1 else ""
    print name0 + beta_str0, "vs.", name1 + beta_str1
    log = []
    start = time.time()
    for i in range(repeat):
        game = Game.from_start (name0, name1, beta_bases0, beta_bases1, 
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

def play_start_beat (name0, name1):
    game = Game.from_start(name0, name1, default_discards=True)
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
def menu (options):
    # displays numbers 1..n
    for i,o in enumerate (options):
        print '[%d] %s' %(i+1,o)
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
        return set(range (a,b+1))
    else:
        return set(range (b,a+1))
    

# GENERAL CLASSES


# main game class, holds all information and methods
class Game:

    # Constants for indicating special events in result matrix.
    # These will never actually given to the solver - they need to 
    # be dealt with in pre-processing.
    # (all constants are floats, so that the numpy array stays a float array)
    clash_indicator = -1000.0
    cancel_0_indicator = -2000.0
    cancel_1_indicator = -2001.0
    cancel_both_indicator = -2002.0
    cancel_indicators = set ([cancel_0_indicator, \
                                 cancel_1_indicator, \
                                 cancel_both_indicator])
    # Very good/bad result constant (used to prevent certain strats
    # from being chosen by AI:
    extreme_result = 20.0

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
        beta_bases0, beta_bases1 = (False, False)
        if name0.endswith(" (beta bases)"):
            name0 = name0[:-13]
            beta_bases0 = True
        if name1.endswith(" (beta bases)"):
            name1 = name1[:-13]
            beta_bases1 = True
                
        game = Game(name0, name1, beta_bases0, beta_bases1)
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
        game.initialize_simulations()
        return game

    @staticmethod
    def from_start (name0, name1, beta_bases0, beta_bases1,
                    default_discards=True, interactive=False, cheating=0,
                    first_beats=False):
        """Create a game in starting position.
        name0, name1: names of characters
        """
        game = Game(name0, name1, beta_bases0, beta_bases1,
                    interactive, cheating, first_beats=first_beats)
        game.set_starting_setup (default_discards,
                                 use_special_actions = True)
        game.initialize_simulations()
        return game

    def __init__ (self, name0, name1, beta_bases0, beta_bases1,
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
        self.player = [character_dict[name0](self, 0, beta_bases0),
                       character_dict[name1](self, 1, beta_bases1,
                                             is_user=interactive)]
        for i in range(2):
            self.player[i].opponent = self.player[1-i]
            for card in self.player[i].all_cards():
                card.opponent = self.player[1-i]
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
        for p in self.player:
            p.set_preferred_range()
        self.initial_evaluation = [p.evaluate() for p in self.player]

    # Play a game from current situation to conclusion
    def play_game (self):
        full_names = [p.name + 
                      (" (beta bases)" if p.use_beta_bases else "")
                      for p in self.player]
        log = ["\n" + " vs. ".join(full_names)]
        for p in self.player:
            log.append ("Chosen finisher for %s: " % p.name +
                             ', '.join([o.name for o in p.finishers]))
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
            self.reporting = True
            final_state, report = self.solve_and_execute_beat ()
            log.extend (report)
            if final_state.winner != None:
                break
            if self.first_beats and self.current_beat == 2:
                break
            self.full_restore (final_state)
            self.prepare_next_beat ()
        winner = final_state.winner
        if winner == 0.5:
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
        if len(self.results[0]) == 0:
            raise DebugException(self)
        results = numpy.array(self.results)
        s0 = self.player[0].strats
        s1 = self.player[1].strats
        regular_0 = [i for i in range(len(s0))
                   if not isinstance (s0[i][0], SpecialAction)] 
        regular_1 = [i for i in range(len(s1))
                   if not isinstance (s1[i][0], SpecialAction)]
        results = results[regular_0,:][:,regular_1]
        row_values = results.min(1)
        if row_values.max() > 0:
            for i in range(len(row_values)):
                if row_values[i] > 0:
                    log.append ("Unbeatable strategy for %s: %s: %.2f"
                        % (self.player[0].name,
                           self.player[0].get_strategy_name(
                                            s0[regular_0[i]]),
                           row_values[i]))
            log.append ("")
        col_values = results.max(0)
        if col_values.min() < 0:
            for j in range(len(col_values)):
                if col_values[j] < 0:
                    log.append ("Unbeatable strategy for %s: %s: %.2f"
                        % (self.player[1].name,
                           self.player[1].get_strategy_name(
                                            s1[regular_1[j]]),
                           col_values[j]))
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
        return state

    def initial_restore (self, state):
        for i in range(2):
            self.player[i].initial_restore (state.player_states[i])

    # makes full snapshot, mid beat
    def full_save (self, stage):
        state = GameState()
        state.player_states = [p.full_save() for p in self.player]
        state.reports = [s for s in self.reports]
        state.decision_counter = self.decision_counter
        state.winner = self.winner
        state.stage = stage
        state.active = self.active
        state.stop_the_clock = self.stop_the_clock
        return state

    def full_restore (self, state):
        for i in range(2):
            self.player[i].full_restore (state.player_states[i])
        self.reports = [s for s in state.reports]
        self.decision_counter = state.decision_counter
        self.winner = state.winner
        self.active = state.active
        self.stop_the_clock = state.stop_the_clock

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
        self.results = [[float(self.simulate (s0,s1)[0]) \
                         for s1 in self.player[1].strats] \
                         for s0 in self.player[0].strats]

        self.initial_restore (self.initial_state)
        
        # remove redundant finisher strategies
        # (that devolve into identical cancels)
        self.remove_redundant_finishers()

        # Usually this does nothing, but some characters might need to
        # fix the result tables and strategies (e.g. Seth).
        for p in self.player:
            p.post_simulation_processing ()

    def remove_redundant_finishers (self):
        redundant_finishers = []
        s0 = self.player[0].strats
        for i in xrange(len(self.results)):
            if isinstance (s0[i][1], Finisher):
                for ii in xrange(len(self.results)):
                    if isinstance (s0[ii][1], Cancel) and s0[ii][2] == s0[i][2]:
                        if self.results[i] == self.results[ii] :
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
                        if [r[j] for r in self.results] == \
                           [r[jj] for r in self.results]:
                            redundant_finishers.append (j)
                            break
        self.results = [[r[j] for j in xrange(len(r)) \
                               if j not in redundant_finishers] \
                               for r in self.results]
        self.player[1].strats = [s1[j] for j in xrange(len(s1)) \
                                 if j not in redundant_finishers]


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

            for p in range(2):
                self.player[p].ante_trigger()

            # Reveal and set attack pairs

            for p in self.player:
                p.style = p.strat[0]
                p.base = p.strat[1]
                p.set_active_cards()
                if not isinstance (p.style, SpecialAction):
                    # Adding to existing set,
                    # because Vanaah's token might already be there.
                    p.discard[0] |= set ([p.style, p.base])

            # Special Actions

            # Finishers devolve into Cancels above 7 life
            # or if failing to meet their specific conditions
            for p in self.player:
                if isinstance (p.base, Finisher):
                    if p.base.devolves_into_cancel():
                        p.base = p.cancel

            # Cancel - return an appropriate cancel indicator
            # (depending on who cancelled).
            # This will be solved retroactively.
            cancel0 = isinstance (self.player[0].base, Cancel)
            cancel1 = isinstance (self.player[1].base, Cancel)
            if cancel0 or cancel1:
                final_state = self.full_save (None)
            if cancel0 and cancel1:
                return self.cancel_both_indicator, final_state, self.fork_decisions[:]
            if cancel0:
                return self.cancel_0_indicator, final_state, self.fork_decisions[:]
            if cancel1:
                return self.cancel_1_indicator, final_state, self.fork_decisions[:]

            # save state before pulse phase (stage 0)
            state = self.full_save (0)
            
        # catch ForkExceptions and WinExceptions
        try:

            # Stage 0 includes: pulse check, reveal trigger, clash check.
            if state.stage <=0:
                pulsing_players = [p for p in self.player
                                   if isinstance (p.base, Pulse)]
                # With one pulse - fork to decide new player positions.
                # Not using execute_move, because Pulse negates any blocking
                # or reaction effects (including status effects from last beat).
                if len (pulsing_players) == 1:
                    pairs = list(itertools.permutations(xrange(7), 2))
                    prompt = "Choose positions after Pulse:"
                    options = []
                    if pulsing_players[0].is_user and self.interactive_mode:
                        current_pair = (self.player[0].position,
                                        self.player[1].position)
                        for pair in pairs:
                            (self.player[0].position,
                             self.player[1].position) = pair
                            options.append (self.get_basic_board())
                        (self.player[0].position,
                         self.player[1].position) = current_pair
                    (self.player[0].position,
                     self.player[1].position) = pairs [
                                                 self.make_fork (len(pairs),
                                                 pulsing_players[0],
                                                 prompt, options)]
                    if self.reporting:
                        self.report ('Pulse:')
                        for s in self.get_board():
                            self.report (s)
                
                # With double pulse, put special
                # action card in discard so that it returns
                # in 3 beats.
                if len(pulsing_players) == 2:
                    for p in pulsing_players:
                        p.discard[1].add(p.special_action)
                        
                # For any Pulse, skip directly to 
                # cycle and evaluation phase.
                if pulsing_players:
                    self.stop_the_clock = True
                    return self.cycle_and_evaluate()

                for p in range(2):
                    self.player[p].reveal_trigger()

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
                        return self.cancel_both_indicator, final_state, self.fork_decisions[:]
                    else:
                        if self.reporting:
                            self.report ("Clash!\n")
                        final_state = self.full_save (None)
                        if self.clash0:
                            return 0, final_state, self.fork_decisions[:]
                        else:
                            return self.clash_indicator, final_state, self.fork_decisions[:]
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
            if p.can_hit() and p.opponent.can_be_hit():
                if self.reporting:
                    self.report (p.name + " hits")
                p.hit_trigger()
                p.opponent.take_a_hit_trigger()
                # did_hit is set after triggers, so that triggers can check
                # if this is first hit this beat
                p.did_hit = True
                if p.base.deals_damage:
                    p.deal_damage (p.get_power())
            else:
                if self.reporting:
                    self.report (p.name + " misses")
        p.after_trigger()

    def cycle_and_evaluate (self):
        for p in self.player:
            p.cycle ()
            # finishers and pulsers lose their special actions
            # (a cancel doesn't get here)
            if isinstance (p.style, SpecialAction):
                p.special_action_available = False
        
        # If 15 beats have been played, raise WinException 
        if self.current_beat == 15 and not self.stop_the_clock:
            if self.reporting:
                self.report("Game goes to time")
            diff = self.player[0].life - self.player[1].life
            if diff > 0:
                raise WinException (0)
            elif diff < 0:
                raise WinException (1)
            else:
                # A tie.
                raise WinException (0.5)
        
        for p in self.player:
            p.set_preferred_range()
        relative_evaluation = \
            [self.player[i].evaluate() - self.initial_evaluation[i]
             for i in range(2)]
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
                        %(relative_evaluation[0], relative_evaluation[1], \
                          relative_evaluation[0] - relative_evaluation[1]))
        final_state = self.full_save (None)
##        if self.reporting:
##            print "decisions returned: ", self.fork_decisions
        return relative_evaluation[0] - relative_evaluation[1], final_state, \
               self.fork_decisions[:]
        
    def distance (self):
        return abs (self.player[0].position - self.player[1].position)

    # does any character block status effects?
    def status_effects_blocked (self):
        return False

    def expected_beats (self):
        return 0.6 * min ([p.life for p in self.player])

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
            if a is not None:
                if isinstance (a, str):
                    a = [a]
                addenda += a
        return [''] + [self.get_basic_board()] + addenda + ['']

    def get_basic_board (self):
        board = ['.'] * 7
        for p in self.player:
            board[p.position] = p.get_board_symbol()
        return ''.join(board)
        
    # find minmax for results table
    def solve (self):
        self.fix_clashes()
        self.fix_cancels()
        array_results = numpy.array(self.results)
        (mix0, value0) = solve.solve_game_matrix (array_results)
        (mix1, value1) = solve.solve_game_matrix (-array_results.transpose())
        stratmix0 = zip (self.player[0].strats, list(mix0))
        stratmix1 = zip (self.player[1].strats, list(mix1))
        assert abs(value0 + value1) < 0.01, ("Error: value0=%f, value1=%f" %
                                             (value0, value1))
        self.value = value0
        self.player[0].mix = stratmix0
        self.player[1].mix = stratmix1

    def solve_and_execute_beat (self):
        ss0 = self.player[0].strats
        ss1 = self.player[1].strats
        self.initial_restore (self.initial_state)
        self.solve ()
        if self.interactive:
            report = []
            if self.cheating > 0:
                self.dump (self.report_solution())
        else:
            report = self.report_solution()
        s0 = self.player[0].choose_strategy()
        if self.interactive and self.cheating == 2:
            print self.player[0], "plays", self.player[0].get_strategy_name(s0)
        s1 = self.player[1].choose_strategy()
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

        # Solve Cancels
        if value in self.cancel_indicators:
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
            # (record the lost special action/s and any discarded attack pair).
            self.initial_restore (self.initial_state)
            if value == self.cancel_0_indicator:
                cancellers = [self.player[0]]
            elif value == self.cancel_1_indicator:
                cancellers = [self.player[1]]
            else:
                cancellers = self.player[:]
            for c in cancellers:
                c.special_action_available = False
            if self.player[0] not in cancellers:
                self.player[0].discard[1] |= set(s0[:2])
            if self.player[1] not in cancellers:
                self.player[1].discard[1] |= set(s1[:2])
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
            s0 = self.player[0].choose_strategy()
            if self.interactive and self.cheating == 2:
                print self.player[0], "plays", self.player[0].get_strategy_name(s0)
            s1 = self.player[1].choose_strategy()
                
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
        if value != self.clash_indicator:
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
        # if one player ran out of bases in clash, just apply unique abilities
        # and cycle
        if min (len(g0),len(g1)) == 0:
            for p in self.player:
                p.unique_ability_end_trigger()
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
        # make vectors of remaining strategies
        self.player[0].strats = self.player[0].fix_strategies_post_clash(
                                                        [ss0[i] for i in g0],
                                                        s1)
        self.player[1].strats = self.player[1].fix_strategies_post_clash(
                                                        [ss1[j] for j in g1],
                                                        s0)
        # Run this function recrusively with post-clash strategies only.
        if self.interactive:
            self.dump(report)
        recursive_state, recursive_report = self.solve_and_execute_beat ()
        report.extend (recursive_report)
        return recursive_state, report
    

    # fix clash results
    def fix_clashes (self):
        if self.clash0:
            return
        ss0 = self.player[0].strats
        ss1 = self.player[1].strats
        n = len (self.results)
        m = len (self.results[0])
        # if at least one matrix dimension is 1, clashes are final,
        # and approximated with 0
        final_clashes = (min(n,m) == 1)
        # record results before clash fixing
        # when each clash is resolved, other clashes should still be unresolved
        self.pre_clash_results = [[self.results[i][j] for j in range(m)]
                                  for i in range(n)]
        for i in range(n):
            for j in range(m):
                if self.pre_clash_results[i][j] == self.clash_indicator:
                    if final_clashes:
                        # Cutting a corner here:
                        # In fact, when someone runs out of bases, cycling
                        # happens, which might be better for one player.
                        self.results[i][j] = 0
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
                        subresults = [[self.pre_clash_results
                                       [p0.clash_strat_index(ii,jj,i,j)]
                                       [p1.clash_strat_index(jj,ii,j,i)]
                                       for jj in g1]
                                      for ii in g0]
                        # and solve it
                        self.results[i][j] = self.sub_solve (subresults)

    # Fix cancel results in matrix.
    # Until a better solution is found, just cause AI to ignore the 
    # possibility: it doesn't play cancel, or play around cancel.
    def fix_cancels(self):
        n = len (self.results)
        m = len (self.results[0])
        for i in range(n):
            for j in range(m):
                if self.results[i][j] in self.cancel_indicators:
                    if self.results[i][j] == self.cancel_0_indicator:
                        self.results[i][j] = -self.extreme_result
                    elif self.results[i][j] == self.cancel_1_indicator:
                        self.results[i][j] = self.extreme_result
                    else:
                        self.results[i][j] = 0

    # solves 4x4 (or smaller) matrix created by clash
    def sub_solve (self, matrix):
        n = len(matrix)
        m = len(matrix[0])
        # trivial matrix: one dimensional
        if min(n,m) == 1:
            # a final clash ends the beat, approximate as 0:
            for i in range(n):
                for j in range(m):
                    if matrix[i][j] == self.clash_indicator:
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
                if matrix[i][j] == self.clash_indicator:
                    sub_matrix = [[matrix[ii][jj] for jj in range(m) if jj!=j] \
                                                  for ii in range(n) if ii!=i]
                    matrix[i][j] = self.sub_solve(sub_matrix)
        (unused_sol, val) = solve.solve_game_matrix (numpy.array(matrix))
        return val

    # return report of positive strategies
    def report_solution (self):
        report = []
        # for each player
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
        # for each player
        for p in self.player:
            # keep only positive probs
            if p.name.lower() in extra:
                p.filtered_indices = range(len(p.mix))
            else:
                p.filtered_indices = [i for i in range(len(p.mix)) \
                                  if p.mix[i][1]>0.0001 \
                                  or p.get_strategy_name(p.mix[i][0]).lower() in extra]
            p.filtered_mix = [p.mix[j] for j in p.filtered_indices]
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
        print '\n' + self.player[0].name + '\'s Value:', self.value, '\n'
        small_mat = numpy.array ([[self.results[i][j] \
                             for j in self.player[1].filtered_indices] \
                             for i in self.player[0].filtered_indices])
        # if all player 1 strategies displayed, transpose for ease of reading
        if self.player[1].name.lower() in extra and \
           self.player[0].name.lower() not in extra:
            small_mat = small_mat.transpose()
            print "(transposing matrix)"
        print small_mat.round(2)

    # game value if one player uses given strategy, and other player uses
    # calculated mix
    def vs_mix (self, name):
        name = name.lower()
        ii = [i for i in range(len(self.player[0].mix))
              if self.player[0].get_strategy_name(self.player[0].mix[i][0]).lower() == name]
        jj = [j for j in range(len(self.player[1].mix))
              if self.player[1].get_strategy_name(self.player[1].mix[j][0]).lower() == name]
        for i in ii:
            value = 0
            for j in range(len(self.player[1].mix)):
                value += self.results[i][j] * self.player[1].mix[j][1]
            print value
        for j in jj:
            value = 0
            for i in range(len(self.player[0].mix)):
                value += self.results[i][j] * self.player[0].mix[i][1]
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
        s0 = [s for s in self.player[0].get_strategies() \
              if self.player[0].get_strategy_name(s).lower() == name0]
        s1 = [s for s in self.player[1].get_strategies() \
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
                self.log.append (s)
        else:
            self.reports.append (s)
        
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
        b0 = len(set([s[2] for s in ss0]))
        b1 = len(set([s[2] for s in ss1]))
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

    def __init__(self, the_game, n, use_beta_bases=False, is_user=False):
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

        # All Characters have all bases (alpha and beta).
        # This allows conversion of initial discards from alpha to
        # beta.
        self.strike = Strike (the_game,self)
        self.shot = Shot (the_game,self)
        self.drive = Drive (the_game,self)
        self.burst = Burst (the_game,self)
        self.grasp = Grasp (the_game,self)
        self.dash = Dash (the_game,self)
        # Counter and Wave avoid clashing with styles of same name.
        self.counter_base = Counter (the_game,self)
        self.wave_base = Wave (the_game,self)
        self.force = Force (the_game,self)
        self.spike = Spike (the_game,self)
        self.throw = Throw (the_game,self)
        self.parry = Parry (the_game,self)
        # And create conversions:
        self.strike.corresponding_beta = self.counter_base
        self.shot.corresponding_beta = self.wave_base
        self.drive.corresponding_beta = self.force
        self.burst.corresponding_beta = self.spike
        self.grasp.corresponding_beta = self.throw
        self.dash.corresponding_beta = self.parry
        self.unique_base.corresponding_beta = self.unique_base
        
        self.use_beta_bases = use_beta_bases
        if use_beta_bases:
            self.bases = [self.unique_base,
                          self.counter_base,
                          self.wave_base,
                          self.force,
                          self.spike,
                          self.throw,
                          self.parry]
        else:
            self.bases = [self.unique_base,
                          self.strike,
                          self.shot,
                          self.drive,
                          self.burst,
                          self.grasp,
                          self.dash]

        # create attributes for styles and finishers
        for card in self.styles + self.finishers:
            name = card.name.replace('-','_').replace(' ','_').replace("'",'').lower()
            self.__dict__[name] = card
        
        self.styles_and_bases_set = set(self.styles) | set(self.bases)
        self.tokens = []
        self.pool = []
        for i in range(5):
            self.styles[i].order = i
        for i in range(7):
            self.bases[i].order = i
        # special action card (a style) and bases.
        # (finishers are unique to each character)
        self.special_action = SpecialAction (the_game,self)
        self.pulse = Pulse (the_game,self)
        self.cancel = Cancel (the_game,self)
        self.pulse_generators = set ([self.bases[4], self.bases[6]])
        self.cancel_generators = set ([self.bases[2], self.bases[5]])
        self.finisher_generators = set ([self.bases[0], self.bases[1],
                                         self.bases[3]])
        # discard[0] is for cards played this beat.
        # they will cycle into discard[1] at end of beat.
        # (discard [0] is empty between beats)
        self.discard = [set() for i in range(3)]

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

    # used by game to add opponent to all your cards/tokens etc.
    def all_cards (self):
        return self.styles + self.bases + self.tokens + self.finishers + \
               [self.special_action, self.pulse, self.cancel,
                self.null_style, self.null_base]

    def __str__ (self):
        return self.name

    def logfile_name(self):
        name = self.name
        if self.use_beta_bases:
            name = name + "(beta)"
        if self.opponent.is_user:
            name = name + "(AI)"
        return name

    # read character state from list of strings (as written in game log)
    # lines - list of strings for this character's situation report
    # board - board string for position
    # addendum - list of strings for this character's board addendum
    # Returns lines specific to the character, for the benefit of inherited
    # methods.
    def read_my_state (self, lines, board, addendum):
        self.life = int(lines[0].split()[-1])
        # position on board marked with uppercase initial.
        # for player1, it might be marked with lowercase
        # (if both characters have same initial).
        self.position = -1
        if self.my_number == 1:
            self.position = board.find(self.name[0].lower())
        if self.position == -1:
            self.position = board.find(self.name[0])
        cards = self.styles + self.bases
        for d in (1,2):
            self.discard[d] = set([c for c in cards
                                   if c.name in lines[d]])
        if len(lines) > 3 and lines[3] == "special action\n":
            self.special_action_available = True
            return lines[4:]
        else:
            self.special_action_available = False
            return lines[3:]
    
    # set character state for start of a game
    def set_starting_setup (self, default_discards, use_special_actions):
        self.life = self.starting_life
        self.position = (1 if self.my_number==0 else 5)
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
                if self.use_beta_bases:
                    b1 = b1.corresponding_beta
                    b2 = b2.corresponding_beta
            self.discard[1] = set ((s1,b1))
            self.discard[2] = set ((s2,b2))
        
    def situation_report (self):
        report = []
        report.append (self.name + (" (beta bases)" 
                                    if self.use_beta_bases else ""))
        report.append ('-' * len (self.name))
        report.append ('life: %d' %self.life)
        report.append ('discard 1: ' + self.discard_report (self.discard[1]))
        report.append ('discard 2: ' + self.discard_report (self.discard[2]))
        if self.special_action_available:
            report.append ('special action')
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
        return state

    # restore state before setting a new strategy
    def initial_restore (self, state):
        self.life = state.life
        self.position = state.position
        self.discard = [d.copy() for d in state.discard]
        self.special_action_available = state.special_action_available
        self.pool = state.pool[:]

    # reset character to status at start of beat
    def reset (self):
        self.damage_taken = 0
        self.was_stunned = False
        self.did_hit = False
        self.moved = False
        self.was_moved = False
        self.triggered_dodge = False
        self.attacks_executed = 0
        self.max_attacks = 1
        self.ante = []
        self.active_cards = []
        # replaced in reveal phase
        self.style = self.null_style
        self.base = self.null_base
        # accumulated bonuses for misc. events that triggered during the beat
        self.evaluation_bonus = 0
        # accumulated bonuses from triggers this beat
        self.triggered_power_bonus = 0
        self.triggered_priority_bonus = 0
        # if not None, someone has replaced the power of your pair
        # with this number 
        self.alt_pair_power = None

    # snapshot of game state mid-turn, for forks
    def full_save (self):
        state = self.initial_save()
        state.damage_taken = self.damage_taken
        state.was_stunned = self.was_stunned
        state.did_hit = self.did_hit
        state.moved = self.moved
        state.was_moved = self.was_moved
        state.triggered_dodge = self.triggered_dodge
        state.attacks_executed = self.attacks_executed
        state.max_attacks = self.max_attacks
        state.ante = self.ante[:]
        state.active_cards = self.active_cards[:]
        state.style = self.style
        state.base = self.base
        state.evaluation_bonus = self.evaluation_bonus
        state.triggered_power_bonus = self.triggered_power_bonus
        state.triggered_priority_bonus = self.triggered_priority_bonus
        state.alt_pair_power = self.alt_pair_power
        return state

    # restoration of mid-turn snapshot (in a fork)
    def full_restore (self, state):
        self.initial_restore (state)
        self.damage_taken = state.damage_taken
        self.was_stunned = state.was_stunned
        self.did_hit = state.did_hit
        self.moved = state.moved
        self.was_moved = state.was_moved
        self.triggered_dodge = state.triggered_dodge
        self.attacks_executed = state.attacks_executed
        self.max_attacks = state.max_attacks
        self.ante = state.ante[:]
        self.active_cards = state.active_cards[:]
        self.style = state.style
        self.base = state.base
        self.evaluation_bonus = state.evaluation_bonus
        self.triggered_power_bonus = state.triggered_power_bonus
        self.triggered_priority_bonus = state.triggered_priority_bonus
        self.alt_pair_power = state.alt_pair_power

    # switch "next beat" bonuses to "this beat", etc.
    def prepare_next_beat (self):
        pass

    def get_pairs (self):
        unavailable_cards = self.discard[1] \
                          | self.discard[2]
        styles = sorted (set(self.styles) - unavailable_cards,
                         key = attrgetter('order'))
        bases = sorted (set(self.bases) - unavailable_cards,
                        key = attrgetter('order'))
        pairs = [(s,b) for s in styles for b in bases]
        if self.special_action_available:
            if self.pulse_generators - unavailable_cards:
                pairs.append ((self.special_action, self.pulse))
            if self.cancel_generators - unavailable_cards:
                pairs.append ((self.special_action, self.cancel))
            if self.finisher_generators - unavailable_cards:
                for finisher in self.finishers:
                    pairs.append ((self.special_action, finisher))
        return pairs

    # prompt user for a style and a base
    # available cards based on cards in strategy mix
    # (so that they are suitably limited after a clash)
    def input_pair (self):
        styles = sorted(list(set([m[0][0] for m in self.mix])),
                       key=attrgetter('order'))
        bases = sorted(list(set([m[0][1] for m in self.mix])),
                       key=attrgetter('order'))
        # friendly reminder of bases in hand when asking about styles
        if len(styles) > 1:
            print "(Bases in hand: " + \
                  ', '.join([b.name for b in bases
                             if not isinstance (b, SpecialBase)]) + ")\n" 
        style = (styles[menu(styles)] if len(styles) > 1 else styles[0])
        if isinstance (style, SpecialAction):
            bases = [b for b in bases if isinstance (b, SpecialBase)]
        else:
            bases = [b for b in bases if not isinstance (b, SpecialBase)]
        base = (bases[menu(bases)] if len(bases) > 1 else bases[0])
        return (style, base)

    def get_antes (self):
        return [None]

    def input_ante (self):
        return None

    # Used by characters who give ante decisions to their opponents.
    # (Rexan/Alexian)
    def get_induced_antes (self):
        return [None]

    def input_induced_ante (self):
        return None

    # Strategy format:
    # (style, base, (ante, induced_ante))
    # induced_ante is the ante decision you made because of opponent's ability.
    def get_strategies (self):
        return [p+((a,aa),) for p in self.get_pairs() for a in self.get_antes()
                         for aa in self.opponent.get_induced_antes()]

    def set_own_ante (self, ante):
        pass
    def add_induced_ante (self, ante):
        pass

    def input_strategy (self):
        pair = self.input_pair()
        # Look at antes in strategy mix.  If there's just one, we don't need
        # to input antes (either there was just one to begin with, or we are
        # post-clash/cancel.
        possible_antes = list(set([m[0][2] for m in self.mix]))
        if len(possible_antes) == 1:
            ante = possible_antes[0]
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
            possible_own_antes = list(set([a[0] for a in possible_antes]))
            if len(possible_own_antes) > 1:
                own_ante = self.input_ante()
            else:
                own_ante = possible_own_antes[0]
            possible_induced_antes = list(set([a[1] for a in possible_antes]))
            if len(possible_induced_antes) > 1:
                induced_ante = opp.input_induced_ante()
            else:
                induced_ante = possible_induced_antes[0]
            ante = (own_ante, induced_ante)
        return pair + (ante,)

    # choose a random strategy according to mix
    # (or prompt human player for a strategy)
    def choose_strategy (self):
        # If there's only one option, return it.
        if len(self.mix) == 1:
            return self.mix[0][0]
        if self.is_user:
            strats = [m[0] for m in self.mix]
            while True:
                strategy = self.input_strategy ()
                if strategy in strats:
                    break
                else:
                    print "Invalid strategy"
        else:
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

    def set_active_cards (self):
        self.active_cards = [self.style, self.base] + self.get_active_tokens()
    def get_active_tokens (self):
        return ([] if self.opponent.blocks_tokens() else self.ante)

    # Bonus and ability calculation
        
    def get_priority (self):
        priority = self.get_priority_bonus() + self.triggered_priority_bonus + \
                   sum(card.priority+card.get_priority_bonus()
                        for card in self.active_cards) + \
                   self.opponent.give_priority_penalty()
        if self.opponent.blocks_bonuses():
            priority = min (priority, self.style.priority + self.base.priority)
        return priority
    def get_power (self):
        if self.alt_pair_power is None:
            card_power = sum(card.power+card.get_power_bonus()
                             for card in self.active_cards)
        else:
            # When pair power is externally fixed, used it instead
            # of actual pair power on the two cards.
            card_power = (self.alt_pair_power +
                          sum(card.power for card in self.active_cards
                              if card not in (self.style, self.base)) +
                          sum(card.get_power_bonus()
                              for card in self.active_cards))
        power = (card_power + self.get_power_bonus() + 
                 self.triggered_power_bonus + 
                 self.opponent.give_power_penalty())
        if self.opponent.blocks_bonuses():
            power = min (power, self.style.power + self.base.power)
        # power has a minimum of 0
        return max (0, power)
    def get_minrange (self):
        minrange = self.get_minrange_bonus() + \
                   sum(card.minrange+card.get_minrange_bonus()
                        for card in self.active_cards) + \
                   self.opponent.give_minrange_penalty()
        if self.opponent.blocks_bonuses():
            minrange = min (minrange, self.style.minrange + self.base.minrange)
        return minrange
    def get_maxrange (self):
        maxrange = self.get_maxrange_bonus() + \
                   sum(card.maxrange+card.get_maxrange_bonus()
                        for card in self.active_cards) + \
                   self.opponent.give_maxrange_penalty()
        if self.opponent.blocks_bonuses():
            maxrange = min (maxrange, self.style.maxrange + self.base.maxrange)
        return maxrange
    def get_priority_bonus (self):
        return 0
    def get_power_bonus (self):
        return 0
    def get_minrange_bonus (self):
        return 0
    def get_maxrange_bonus (self):
        return 0

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
        self.activate_card_triggers('start_trigger')
    def before_trigger (self):
        self.activate_card_triggers('before_trigger')
    def hit_trigger (self):
        if not self.opponent.blocks_hit_triggers():
            self.activate_card_triggers('hit_trigger')
    def take_a_hit_trigger (self):
        self.activate_card_triggers('take_a_hit_trigger')
    def damage_trigger (self, damage):
        if not self.opponent.blocks_damage_triggers():
            self.activate_card_triggers('damage_trigger', [damage])
        self.opponent.stun (damage)
    def take_damage_trigger (self, damage):
        pass
    def soak_trigger (self, damage_soaked):
        pass
    def after_trigger (self):
        self.activate_card_triggers('after_trigger')
    def end_trigger (self):
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
        n_ordered = len(ordered_cards)
        if n_ordered == 1:
            trigger(ordered_cards[0])(*params)
        elif n_ordered == 2:
            prompt = "Choose trigger to execute first:"
            options = [card.name for card in ordered_cards]
            first = self.game.make_fork (2, self, prompt, options)
            trigger(ordered_cards[first])(*params)
            trigger(ordered_cards[1-first])(*params)
        elif n_ordered == 3:
            prompt = "Choose order of triggers:"
            options = []
            if self.is_user and self.game.interactive_mode:
                for i in range (6):
                    options.append (', '.join([ordered_cards[i%3].name,
                                               ordered_cards[2-i/2].name,
                                               ordered_cards[1-i%3+i/2].name]))
            order = self.game.make_fork (6, self, prompt, options)
            first = order%3             # [012012]
            second = 2 - order/2        # [221100]
            third = 3 - first - second  # [100221]
            trigger(ordered_cards[first])(*params)
            trigger(ordered_cards[second])(*params)
            trigger(ordered_cards[third])(*params)
        elif n_ordered > 3:
            raise Exception("Can't handle %d simultaneous ordered triggers" % n_ordered)
            
            

    # special ability calculation
    def is_attacking (self):
        return self.base.is_attack
    def can_hit (self):
        if self.standard_range():
            distance = self.attack_range()
            if self.game.debugging:
                self.game.report ("%s's range: %d-%d"
                                  %(self.name,
                                    self.get_minrange(),
                                    self.get_maxrange()))
            in_range = distance <= self.get_maxrange() and \
                       distance >= self.get_minrange()
        else:
            in_range = False
        return (in_range or self.special_range_hit()) and \
               all(card.can_hit() for card in self.active_cards)
    def can_be_hit (self):
        return not self.triggered_dodge and \
               all(card.can_be_hit() for card in self.active_cards)
    def standard_range (self):
        return all (card.standard_range for card in self.active_cards)
    def special_range_hit (self):
        return any(card.special_range_hit() for card in self.active_cards)
    
    def reduce_soak (self, soak):
        for card in self.active_cards:
            soak = card.reduce_soak(soak)
        return soak
    def reduce_stunguard (self, stunguard):
        for card in self.active_cards:
            stunguard = card.reduce_stunguard(stunguard)
        return stunguard
    
    # return blocked destinations, that cannot be moved into or through
    def blocks_movement (self, direct):
        blocked = set()
        for card in self.active_cards:
            blocked |= card.blocks_movement (direct)
        return blocked
    def blocks_pullpush (self):
        blocked = set()
        for card in self.active_cards:
            blocked |= card.blocks_pullpush()
        return blocked

    # performance hacks:
    # For rare abilities, return False automatically.
    # Characters that have the ability need to override this.

    # block opponent's power,priority,range from going above printed style+base
    def blocks_bonuses (self):
        return False
    # block effects of tokens anted and spent
    # cannot refer to self.active_cards, because it is used in making it
    def blocks_tokens (self):
        return False
    # react to opponent's execute_move()
    # invoked also for failed movement
    def movement_reaction (self, mover, old_position, direct):
        pass
    def mimics_movement (self):
        return False
    # makes in-beat choices (forks) for opponent
    def controls_opponent (self):
        return False
    def blocks_hit_triggers (self):
        return False
    def blocks_damage_triggers (self):
        return False
    
    
##    def mimics_movement (self):
##        return any (card.mimics_movement() for card in self.active_cards)
##    # block opponent's power,priority,range from increasing above style+base
##    def blocks_bonuses (self):
##        return any (card.blocks_bonuses() for card in self.active_cards)
##    # block effects of tokens anted and spent
##    # cannot refer to self.active_cards, because it is used in making it
##    def blocks_tokens (self):
##        return self.style.blocks_tokens() or self.base.blocks_tokens()
##    def movement_reaction (self, mover, old_position, direct):
##        for card in self.active_cards:
##            card.movement_reaction (mover, old_position, direct)
    def has_stun_immunity (self):
        return any (card.has_stun_immunity() for card in self.active_cards)
    def give_priority_penalty (self):
        return sum (card.give_priority_penalty() for card in self.active_cards)
    def give_power_penalty (self):
        return sum (card.give_power_penalty() for card in self.active_cards)
    def give_minrange_penalty (self):
        return sum (card.give_minrange_penalty() for card in self.active_cards)
    def give_maxrange_penalty (self):
        return sum (card.give_maxrange_penalty() for card in self.active_cards)
    # fraction added to priority to break ties in clash
    # used when a card wins/loses ties without clashing
    def clash_priority (self):
        return sum (card.clash_priority for card in self.active_cards)
    # change this when a character can't go below 1 life (or other value)
    def get_minimum_life (self):
        return 0
    def get_damage_cap (self):
        return min (card.get_damage_cap() for card in self.active_cards)
    
    # execution of game tasks

    # cycle discard piles
    def cycle (self):
        # unless you played your special action
        if not isinstance (self.style, SpecialAction):
            # if the special action cycles back, restore it.
            if self.special_action in self.discard[2]:
                self.special_action_available = True
            # Actual cycling.
            self.discard [2] = self.discard [1].copy()
            self.discard [1] = self.discard [0].copy()
        # but remove cards from the virtual discard[0] in any case
        self.discard [0] = set()

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
        if self.game.debug and soak:
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
        self.opponent.damage_trigger (final_damage)
        self.take_damage_trigger (final_damage)
        # damage_taken updated after triggers, so that triggers can
        # check if this is first damage this beat
        self.damage_taken += final_damage

    # attempts to stun me with given damage
    # if damage == None, it's an auto-stun effect
    def stun (self, damage=None):
        if self.has_stun_immunity():
            return
        if damage == None:
            self.was_stunned = True #auto stun
            if self.game.reporting:
                self.game.report (self.name + " is stunned")
        else:
            stunguard = self.opponent.reduce_stunguard(self.get_stunguard())
            if damage > stunguard:
                self.was_stunned = True
                if self.game.reporting:
                    self.game.report (self.name + " is stunned")

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
        self.pool = [self.tokens[0]] * min (self.max_tokens, len(self.pool) + n)
        gain = len(self.pool) - old_pool
        if self.game.reporting and gain:
            self.game.report ("%s gains %d %s token%s"
                              %(self.name, gain, self.tokens[0].name,
                                ("s" if gain>1 else "")))

    # Various convenience movement functions:
    def advance (self, moves):
        self.execute_move (self, moves)
    def pull (self, moves):
        self.execute_move (self.opponent, moves)
    def retreat (self, moves):
        moves = [-m for m in moves]
        self.execute_move (self, moves)
    def push (self, moves):
        moves = [-m for m in moves]
        self.execute_move (self.opponent, moves)
    def move (self, moves):
        moves = [-m for m in moves if m != 0] + list(moves)
        self.execute_move (self, moves)
    def move_opponent (self, moves):
        moves = [-m for m in moves if m != 0] + list(moves)
        self.execute_move (self.opponent, moves)
    def move_directly (self, dests):
        self.execute_move (self, dests, direct=True)
    def move_opponent_directly (self, dests):
        self.execute_move (self.opponent, dests, direct=True)
    def move_to_unoccupied (self):
        self.move_directly (list(set(xrange(7)) -
                                 set((self.position,
                                      self.opponent.position))))
    def move_opponent_to_unoccupied (self):
        self.move_opponent_directly (list(set(xrange(7)) -
                                          set((self.position,
                                               self.opponent.position))))

    # Handles both moving yourself and moving opponent.
    # Mover is player that is actually being moved.
    # direct: the move skips intervening spaces.
    # Moves are relative for indirect moves, absolute for direct moves.
    def execute_move (self, mover, moves, direct=False):
        self.forced_block = False
        mover_pos = mover.position
        
        self.inner_execute_move (mover, moves, direct)

        # Note and report movement
        if mover_pos != mover.position:
            if mover is self:
                mover.moved = True
            else:
                mover.was_moved = True
            if self.game.reporting:
                if mover == self:
                    self.game.report (self.name + " moves:")
                else:
                    self.game.report (self.name + " moves " + mover.name + ":")
                for s in self.game.get_board():
                    self.game.report (s)
        # reaction not dependent on actual movement, some cards react to
        # failed movement.
        self.opponent.movement_reaction (mover, mover_pos, direct)

    # Does heavy lifting of movement.
    def inner_execute_move (self, mover, moves, direct):
        mover_pos = mover.position
        # obtain set of attempted destinations
        if direct:
            # Moves are given as destinations, just remove opponent's position.
            # Don't remove own position, because some moves say "may".
            dests = (set(moves) - set([mover.opponent.position])) & set(xrange(7))
        else:
            # convert relative moves to destinations
            dests = self.get_destinations(mover, moves)
        # obtain set of spaces blocked by opponent
        if mover == self:
            blocked = self.opponent.blocks_movement (direct)
        else:
            blocked = self.opponent.blocks_pullpush()
        # record this for later inspection
        self.blocked = blocked
        # compute possible destinations after blocking
        if len (blocked) == 0:
            possible = list (dests)
        else:
            # can't block null movement
            blocked.discard(mover_pos)
            # for normal movement, a position is 'unobstructed' if there are
            # no blocked positions between it and the mover
            if not direct:
                unobstructed = set([pos for pos in xrange(7) \
                        if len(pos_range(pos,mover_pos) & blocked) == 0])
            # for direct movement, only directly blocked spaces are obstructed.
            else: 
                unobstructed = set(xrange(7)) - blocked
            possible = list (dests & unobstructed)
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
        # no possible moves
        else:
            # if I had possible destinations, but all were blocked,
            # then I was forced into attempting the block
            if dests:
                self.forced_block = True
            
    # given player moving/being moved, and set of relative moves,
    # return possible destinations, taking into account board size
    # and jumping over opponent
    def get_destinations (self, mover, moves):
        mypos = mover.position
        otherpos = mover.opponent.position
        direction = (otherpos - mypos) / abs (otherpos - mypos)
        destinations = []
        mimic = self.opponent.mimics_movement()
        for m in moves:
            dest = mypos + m * direction
            # forward movement takes into account running into opponent
            if m > 0:
                # if moving yourself against a mimic, stop one short
                if mover == self and mimic:
                    if direction == 1 and dest == 6:
                        break
                    if direction == -1 and dest == 0:
                        break
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

    # determine the range the character wants to be in this beat:
    def set_preferred_range (self):
        unavailable_cards = self.discard[1] \
                          | self.discard[2]
        styles = set(self.styles) - unavailable_cards
        bases = set(self.bases) - unavailable_cards
        self.preferred_range = \
            sum (s.get_preferred_range() for s in styles) / len (styles) + \
            sum (b.get_preferred_range() for b in bases) / len (bases)

    def evaluate_range (self):
        return - self.game.range_weight * \
                      (self.preferred_range - self.game.distance()) ** 2

    # opponent's life (negative)
    # special bonuses accumulated,
    # square difference between my preferred range and current range (negative)
    # bonuses related to cards in hand
    # discard rotation (cards in discard 2 are half penalty)
    # having a special action
    def evaluate (self):
        hand = self.styles_and_bases_set - (self.discard[1] | self.discard[2])
        card_bonus = sum(card.evaluation_bonus() for card in hand)
        discard_penalty = \
            sum (card.discard_penalty() for card in self.discard[1]) \
          + sum (card.discard_penalty() for card in self.discard[2]) / 2.0
        special_action_bonus = \
                    ((self.special_action.value +
                      (max([finisher.evaluate_setup()
                            for finisher in self.finishers])
                       if (self.life <= 7) else 0)) \
                    if self.special_action_available else 0)

        return - self.opponent.effective_life() + self.evaluation_bonus \
               + self.evaluate_range () + card_bonus \
               + discard_penalty + special_action_bonus

    # traits that are useful for AI opponents to know about

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
    
                                  
# One card, style or base
class Card (object):
    minrange = 0
    maxrange = 0
    power = 0 
    priority = 0
    stunguard = 0
    soak = 0
    name_override = None
    # average of minimum and maximum range (including movement)
    preferred_range = 0 
    # fraction added to priority to break ties in clash
    # used when a card wins/loses ties without clashing
    clash_priority = 0

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

    # triggers
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
    standard_range = True #change if attack *only* has special range
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
    def blocks_movement (self, direct):
        return set()
    def blocks_pullpush (self):
        return set()
    
    # note that this is invoked for failed movement as well
    def movement_reaction(self, mover, old_position, direct):
        pass
    def mimics_movement (self):
        return False
    def blocks_bonuses (self):
        return False
    def blocks_tokens (self):
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
    # bonus (usually negative) for this card being in discard 1
    # (halved in discard 2)
    def discard_penalty (self):
        return 0
    
class Style (Card):
    pass

class Base (Card):
    is_attack = True  #change for bases that don't attack
    deals_damage = True #change for bases that do attack, but can't deal damage

# Placeholders used before the reveal step, so that no card abilities
# are activated.
class NullStyle (Style):
    pass
class NullBase (Base):
    pass

# only covers token's effects when anted.
# anything else should be covered by the character
class Token (Card):
    value = 0

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

class Adjenna (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Gaze (the_game, self)
        self.styles = [Alluring     (the_game, self),
                       Arresting    (the_game, self),
                       Pacifying    (the_game, self),
                       Irresistible (the_game, self),
                       Beckoning    (the_game, self)]
        self.finishers = [BasiliskGaze (the_game, self),
                           Fossilize    (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)

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

    # overrides default method, which I set to pass for performance
    def movement_reaction (self, mover, old_position, direct):
        for card in self.active_cards:
            card.movement_reaction (mover, old_position, direct)

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
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Divider (the_game, self)
        self.styles = [Gestalt  (the_game, self), \
                       Regal    (the_game, self), \
                       Stalwart (the_game, self), \
                       Mighty   (the_game, self), \
                       Steeled  (the_game, self)  ]
        self.finishers = [EmpireDivider (the_game, self),
                          HailTheKing (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.chivalry = Chivalry  (the_game, self)
        self.induced_tokens = [self.chivalry]
    
    def choose_initial_discards (self):
        return (self.regal, self.dash,
                self.stalwart, self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.induced_pool = [self.chivalry]

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("Opponent has %d Chivalry tokens" %
                       len(self.induced_pool))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.induced_pool = [self.chivalry] * int(lines[0][13])
        
    def initial_save (self):
        state = Character.initial_save (self)
        state.induced_pool = self.induced_pool[:]
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.induced_pool = state.induced_pool[:]
    
    def reset (self):
        self.damage_soaked = 0
        self.switched_sides = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.damage_soaked = self.damage_soaked
        state.switched_sides = self.switched_sides
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.damage_soaked = state.damage_soaked
        self.switched_sides = state.switched_sides
        
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
        for i in xrange(self.opponent.strat[2][1]):
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

    # Record switching sides, for Divider
    # This tracks switching under Alexian's initiative
    def execute_move (self, mover, moves, direct=False):
        old_direction = self.position - self.opponent.position
        Character.execute_move (self, mover, moves, direct)
        new_direction = self.position - self.opponent.position
        if old_direction * new_direction < 0:
            self.switched_sides = True
    # And this tracks switching under opponent's initiative
    def movement_reaction (self, mover, old_position, direct):
        if mover is self:
            old_direction = old_position - self.opponent.position
        else:
            old_direction = self.position - old_position
        new_direction = self.position - self.opponent.position
        if old_direction * new_direction < 0:
            self.switched_sides = True

    def evaluate (self):
        value = Character.evaluate(self)
        value -= 0.6 * len(self.induced_pool)
        return value


class Aria (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Reconfiguration (the_game, self)
        self.styles = [Photovoltaic (the_game, self), \
                       Ionic        (the_game, self), \
                       Laser        (the_game, self), \
                       Catalyst     (the_game, self), \
                       Dimensional  (the_game, self)  ]
        self.finishers = [LaserLattice (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.dampening = Dampening()
        self.magnetron = Magnetron()
        self.turret = Turret()
        self.droids = [self.dampening,
                       self.magnetron,
                       self.turret]

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
        magnetron = self.magnetron.position
        return (-1
                if magnetron is not None and
                   abs (self.opponent.position - magnetron) <= 1
                else 0)

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
    
class Byron (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Smoke (the_game, self)
        self.styles = [Soulless   (the_game, self), \
                       Deathless  (the_game, self), \
                       Heartless  (the_game, self), \
                       Faceless   (the_game, self), \
                       Breathless (the_game, self)  ]
        self.finishers = [SoulTrap (the_game, self),
                           SoulGate (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.starting_life = 15
    
    def choose_initial_discards (self):
        return (self.soulless, self.strike,
                self.heartless, self.shot)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.mask_emblems = 5
        self.priority_bonus = 0
        
    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Mask emblems" %self.mask_emblems)
        if self.priority_bonus > 0:
            report.append ("Byron has +%d priority this beat" % self.priority_bonus)
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.mask_emblems = int(lines[0][0])
        line = find_start(lines, "Byron has +")
        # assuming priority bonus is one digit
        self.priority_bonus = int(line[11]) if line else 0
        
    def initial_save (self):
        state = Character.initial_save (self)
        state.mask_emblems = self.mask_emblems
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.mask_emblems = state.mask_emblems

    def reset (self):
        self.life_lost = 0
        self.emblems_discarded = 0
        self.heartless_ignore_soak = False
        self.heartless_soak = False
        self.priority_bonus_next_beat = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.life_lost = self.life_lost
        state.emblems_discarded = self.emblems_discarded
        state.heartless_ignore_soak = self.heartless_ignore_soak
        state.heartless_soak = self.heartless_soak
        state.priority_bonus_next_beat = self.priority_bonus_next_beat
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.life_lost = state.life_lost
        self.emblems_discarded = state.emblems_discarded
        self.heartless_ignore_soak = state.heartless_ignore_soak
        self.heartless_soak = state.heartless_soak
        self.priority_bonus_next_beat = state.priority_bonus_next_beat

    def prepare_next_beat (self):
        Character.prepare_next_beat (self)
        self.priority_bonus = self.priority_bonus_next_beat
        
    def get_priority_bonus (self):
        return self.priority_bonus

    def discard_emblem (self):
        if self.mask_emblems > 0:
            self.mask_emblems -= 1
            # recording emblem discards for Soulless
            self.emblems_discarded += 1
            if self.game.reporting:
                self.game.report ("Byron loses a Mask Emblem: now at %d"
                                  % self.mask_emblems)

    def recover_emblem (self):
        if self.mask_emblems < 5:
            self.mask_emblems += 1
            if self.game.reporting:
                self.game.report ("Byron recovers a Mask Emblem: now at %d"
                                  % self.mask_emblems)

    def take_damage_trigger (self, damage):
        if not self.damage_taken:
            self.discard_emblem()

    def lose_life (self, amount):
        Character.lose_life (self, amount)
        self.life_lost += amount

    # Smoke ignores style's range modifier
    def get_maxrange (self):
        if self.base.name == 'Smoke':
            maxrange = self.base.maxrange + \
                       self.opponent.give_maxrange_penalty()
            if self.opponent.blocks_bonuses():
                maxrange = min (maxrange, self.base.maxrange)
        else:
            maxrange = Character.get_maxrange(self)
        return maxrange
            
    def get_minrange (self):
        if self.base.name == 'Smoke':
            minrange = self.base.minrange + \
                       self.opponent.give_minrange_penalty()
            if self.opponent.blocks_bonuses():
                minrange = min (minrange, self.base.minrange)
        else:
            minrange = Character.get_minrange(self)
        return minrange

    # restoring life at end, rather than start of beat
    # this lets opponent's evaluation figure that damage is worthless
    # (it's incorrect in beat 15, where the damage might break the tie)

    # Putting in cycle, to make sure Pulse doesn't stop it
    # (Pulse shouldn't stop it,
    #  because it really happens at the start of next beat)
    def cycle (self):
        Character.cycle (self)
        self.life = max (3 * self.mask_emblems - self.life_lost, 1)

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
        return 20.0 / 15 * max (3 * self.mask_emblems - self.life_lost * life_loss_effect, 1)

class Cadenza (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Press (the_game, self)
        self.styles = [Battery    (the_game, self), \
                       Clockwork  (the_game, self), \
                       Hydraulic  (the_game, self), \
                       Grapnel    (the_game, self), \
                       Mechanical (the_game, self)  ]
        self.finishers = [RocketPress   (the_game, self),
                           FeedbackField (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.iron_body = IronBody  (the_game, self)
        self.tokens = [self.iron_body]

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = [self.iron_body] * 3
        self.priority_bonus = False

    def choose_initial_discards (self):
        return (self.hydraulic, self.burst,
                self.mechanical, self.unique_base)

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Iron Body tokens" %len (self.pool))
        if self.priority_bonus:
            report.append ("+4 priority from last beat")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.iron_body] * int(lines[0][0])
        self.priority_bonus = find_start (lines, "+4 priority")

    def reset (self):
        self.token_spent = False
        self.priority_bonus_next_beat = False
        self.damage_soaked = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.token_spent = self.token_spent
        state.priority_bonus_next_beat = self.priority_bonus_next_beat
        state.damage_soaked = self.damage_soaked
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.token_spent = state.token_spent
        self.priority_bonus_next_beat = state.priority_bonus_next_beat
        self.damage_soaked = state.damage_soaked

    def prepare_next_beat (self):
        Character.prepare_next_beat (self)
        self.priority_bonus = self.priority_bonus_next_beat

    # Anteing Iron Body is "illegal" when there's absolutely no reason.
    # Possible reasons are auto-stun and the possibility of losing the token
    def get_antes (self):
        if self.pool:
            return [0,1]
        else:
            return [0]

    def input_ante (self):
        if self.pool:
            print ("Ante an Iron Body Token?")
            return menu (["No", "Yes"])
        else:
            return 0

    def ante_trigger (self):
        for i in range(self.strat[2][0]):
            self.ante_token(self.iron_body)

    def get_ante_name (self, a):
        return ("Iron Body" if a==1 else "")

    def get_priority_bonus (self):
        return 4 if (self.priority_bonus and
                     not self.game.status_effects_blocked()) else 0

    def get_stunguard (self):
        return (1000000 if self.token_spent else Character.get_stunguard(self))

    # when Cadenza is about to be stunned by damage, he may spend a token
    def stun (self, damage=None):
        if (damage is not None and
            self.can_spend(1) and
            self != self.game.active and
            self.iron_body not in self.ante and
            damage > self.get_stunguard()):
            if self.game.make_fork (2, self, "Spend an Iron Body token?",
                                    ["No", "Yes"]):
                self.token_spent = 1
                self.spend_token()
                if self.game.reporting:
                    self.game.report ("Cadenza gains infinite stunguard")
        Character.stun (self, damage)

    # record damage soaked (for Feedback Field)
    def soak_trigger (self, soaked_damage):
        self.damage_soaked += soaked_damage
        
    def evaluate (self):
        return Character.evaluate(self) + 2.5 * len(self.pool)

class Cesar (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Suppression (the_game, self)
        self.styles = [Phalanx     (the_game, self),
                       Unstoppable (the_game, self),
                       Fueled      (the_game, self),
                       Bulwark     (the_game, self),
                       Inevitable  (the_game, self)  ]
        self.finishers = [Level4Protocol (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        
    def choose_initial_discards (self):
        return (self.fueled, self.strike,
                self.unstoppable, self.grasp)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.threat_level = 0
        self.defeat_immunity = True
        self.power_penalty = False

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("Threat level: %d" % self.threat_level)
        if not self.defeat_immunity:
            report.append("Defeat immunity used-up")
        if self.power_penalty:
            report.append ("Opponent has -3 power this beat")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.threat_level = int(lines[0][-2])
        self.power_penalty = find_start (lines, "Opponent has -3 power this beat")
        self.defeat_immunity = not find_start(lines, "Defeat immunity used")
        
    def initial_save (self):
        state = Character.initial_save (self)
        state.threat_level = self.threat_level
        state.power_penalty = self.power_penalty
        state.defeat_immunity = self.defeat_immunity
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.threat_level = state.threat_level
        self.power_penalty = state.power_penalty
        self.defeat_immunity = state.defeat_immunity

    def reset (self):
        Character.reset (self)
        self.power_penalty_next_beat = False

    def full_save (self):
        state = Character.full_save (self)
        state.power_penalty_next_beat = self.power_penalty_next_beat
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.power_penalty_next_beat = state.power_penalty_next_beat

    def prepare_next_beat (self):
        Character.prepare_next_beat (self)
        self.power_penalty = self.power_penalty_next_beat
        
    def give_power_penalty (self):
        return -3 if (self.power_penalty and
                     not self.game.status_effects_blocked()) else 0

    def ante_trigger (self):
        self.gain_threat_level()

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
        return (self.threat_level if self.threat_level < 4 else 0)

    def get_stunguard(self):
        stunguard = Character.get_stunguard(self)
        if self.threat_level <= 2:
            stunguard += self.threat_level
        return stunguard

    def has_stun_immunity(self):
        return self.threat_level == 3 or Character.has_stun_immunity(self)

    def get_soak(self):
        soak = Character.get_soak(self)
        if self.threat_level == 4:
            soak += 2
        return soak

    # Copying all this to implement Cesar's death immunity.
    def take_damage (self, damage):
        soak = self.opponent.reduce_soak(self.get_soak())
        if self.game.debug and soak:
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

    # overrides default method, which I set to pass for performance
    def movement_reaction (self, mover, old_position, direct):
        for card in self.active_cards:
            card.movement_reaction (mover, old_position, direct)

    def blocks_hit_triggers (self):
        return self.unstoppable in self.active_cards

    def blocks_damage_triggers (self):
        return self.unstoppable in self.active_cards

    # It's best to end the beat with level 4,0,1 and worst to end with 3
    def evaluate (self):
        return (Character.evaluate(self) +
                [1.4, 0.9, -0.35, -2.6, 1.15][self.threat_level] +
                5 * self.defeat_immunity)
         
class Clinhyde (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Frenzy (the_game, self)
        self.styles = [Toxic     (the_game, self),
                       Shock     (the_game, self),
                       Diffusion (the_game, self),
                       Gravity   (the_game, self),
                       Phase     (the_game, self)  ]
        self.finishers = [VitalSilverInfusion (the_game, self),
                           RitherwhyteInfusion (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.packs = [Crizma   (the_game, self),
                      Ehrlite  (the_game, self),
                      Hylatine (the_game, self) ]

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

    def input_ante (self):
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

    def execute_move (self, mover, moves, direct=False):
        # can choose not to move when using Gravity
        new_move = self.position if direct else 0
        if self.style.name == 'Gravity' and \
                       mover == self and \
                       new_move not in moves:
            moves.append(new_move)
        Character.execute_move (self, mover, moves, direct)

    def set_preferred_range (self):
        Character.set_preferred_range (self)
        if self.ritherwhyte_activated:
            self.preferred_range += 1

    activation_values = [0, -0.5, -1.5, -3]
    def evaluate (self):
        value = Character.evaluate(self)
        value += self.activation_values[len(self.active_packs)]
        return value

class Clive (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Wrench (the_game, self)
        self.styles = [Upgradeable (the_game, self), \
                       Rocket      (the_game, self), \
                       Burnout     (the_game, self), \
                       Megaton     (the_game, self), \
                       Leaping     (the_game, self)  ]
        self.finishers = [SystemShock (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.modules = [RocketBoots   (the_game, self), \
                        BarrierChip   (the_game, self), \
                        AtomicReactor (the_game, self), \
                        ForceGloves   (the_game, self), \
                        CoreShielding (the_game, self), \
                        AfterBurner   (the_game, self), \
                        SynapseBoost  (the_game, self), \
                        AutoRepair    (the_game, self), \
                        ExtendingArms (the_game, self)  ]
        self.rocket_boots = self.modules[0]
        self.force_gloves = self.modules[3]
        self.core_shielding = self.modules[4]
        self.extending_arms = self.modules[8]
        
    def all_cards (self):
        return Character.all_cards(self) + self.modules

    def choose_initial_discards (self):
        # Not really checked
        return (self.styles[0], self.grasp,
                self.styles[1], self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.module_stack = self.modules[:]
        self.active_modules = []
        self.removed_modules = []

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

    def input_ante (self):
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
        Character.stun (self, damage)
        if self.is_stunned():
            # Oppnent loses life when playing Burnout
            if self.game.distance() == 1 and isinstance(self.style, Burnout):
                self.opponent.lose_life (len(self.active_modules))
            while self.active_modules:
                self.discard_module(self.active_modules[0])

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

    # record switching sides, for Leaping
    # I assume it only counts if it's my movement effect, and I'm moving
    def execute_move (self, mover, moves, direct=False):
        old_pos = self.position
        Character.execute_move (self, mover, moves, direct)
        if mover == self and \
           ordered (old_pos, self.opponent.position, self.position):
            self.switched_sides = True
        
    def blocks_hit_triggers (self):
        return self.core_shielding in self.active_modules

    def blocks_damage_triggers (self):
        return self.core_shielding in self.active_modules

    def set_preferred_range (self):
        Character.set_preferred_range(self)
        if self.rocket_boots in self.active_modules:
            self.preferred_range += 0.5
        if self.extending_arms in self.active_modules:
            self.preferred_range += 0.5

    # Activating a module is worth 1.0, losing it is worth -1.5
    def evaluate (self):
        return (Character.evaluate(self)
              + 1.5 * len(self.active_modules)
              + 0.5 * len(self.module_stack))
    

class Demitras (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Deathblow (the_game, self)
        self.styles = [Darkside     (the_game, self), \
                       Bloodletting (the_game, self), \
                       Vapid        (the_game, self), \
                       Illusory     (the_game, self), \
                       Jousting     (the_game, self)  ]
        self.finishers = [SymphonyOfDemise (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.crescendo = Crescendo  (the_game, self)
        self.tokens = [self.crescendo]
        self.max_tokens = 5
        
    def choose_initial_discards (self):
        return (self.bloodletting, self.grasp,
                self.illusory, self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 2 * [self.crescendo]

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Crescendo tokens" %len (self.pool))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.crescendo] * int(lines[0][0])

    def reset (self):
        self.deathblow_spending = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.deathblow_spending = self.deathblow_spending
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.deathblow_spending = state.deathblow_spending
        
    def get_antes (self):
        return range (len(self.pool) + 1)

    def input_ante (self):
        n = len(self.pool)
        if n > 0 :
            print "Number of Crescendo tokens [0-%d]: " %n
            return input_number (n+1)
        else:
            return 0

    def ante_trigger (self):
        for i in range(self.strat[2][0]):
            self.ante_token(self.crescendo)

    def get_ante_name (self, a):
        if a == 0:
            return ""
        if a == 1:
            return "1 token"
        return str(a) + " tokens"

    def get_priority_bonus (self): 
        return len(self.pool)
    
    def get_power_bonus (self): 
        return 2 * self.deathblow_spending

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
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Aegis (the_game, self)
        self.styles = [Vengeful     (the_game, self), \
                       CounterStyle (the_game, self), \
                       Martial      (the_game, self), \
                       Chained      (the_game, self), \
                       Retribution  (the_game, self)  ]
        self.finishers = [SweetRevenge   (the_game, self),
                           SheetLightning (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.vengeance = VengeanceT (the_game, self)
        self.tokens = [self.vengeance]
        self.max_tokens = 5
        
    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 2 * [self.vengeance]
        self.opponent_immobilized = False

    def choose_initial_discards (self):
        return (self.martial, self.grasp,
                self.vengeful, self.unique_base)

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Vengeance tokens" %len(self.pool))
        if self.opponent_immobilized:
            report.append (self.opponent.name + "is immobilized")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.vengeance] * int(lines[0][0])
        self.opponent_immobilized = find_end (lines, "is immobilized\n")

    def reset (self):
        self.opponent_immobilized_next_beat = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.opponent_immobilized_next_beat = \
            self.opponent_immobilized_next_beat
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.opponent_immobilized_next_beat = \
            state.opponent_immobilized_next_beat

    def prepare_next_beat (self):
        self.opponent_immobilized = self.opponent_immobilized_next_beat

    def get_antes (self):
        return range (len(self.pool) + 1)

    def input_ante (self):
        n = len(self.pool)
        if n > 0 :
            print "Number of Vengeance tokens [0-%d]: " %n
            return input_number (n+1)
        else:
            return 0

    def ante_trigger (self):
        for i in range(self.strat[2][0]):
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
        
    def blocks_movement (self, direct):
        return (set(xrange(7)) if (self.opponent_immobilized and
                             not self.game.status_effects_blocked())
                else Character.blocks_movement(self, direct))

    def get_minimum_life (self):
        return 1 if self.base.name == 'Sweet Revenge' \
               else Character.get_minimum_life(self)

    def evaluate (self):
        return Character.evaluate (self) + 0.2 * len(self.pool)

class Heketch (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Knives (the_game, self)
        self.styles = [Merciless (the_game, self), \
                       Rasping   (the_game, self), \
                       Critical  (the_game, self), \
                       Assassin  (the_game, self), \
                       Psycho    (the_game, self)  ]
        self.finishers = [MillionKnives   (the_game, self),
                           LivingNightmare (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.dark_force = DarkForce (the_game, self)
        self.tokens = [self.dark_force]
        self.max_tokens = 1
    
    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = [self.dark_force]
        self.assassin_immobilized = False
        self.living_nightmare_active = False

    def choose_initial_discards (self):
        return (self.rasping, self.burst,
                self.critical, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        if len(self.pool) == 1:
            report.append ("Dark Force in pool")
        if self.assassin_immobilized:
            report.append (self.opponent.name + "is immobilized")
        if self.living_nightmare_active:
            report.append ("Living Nightmare active")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = ([self.dark_force] if find_start(lines, 'Dark Force')
                     else [])
        self.assassin_immobilized = find_end(lines, "is immobilized\n")
        self.living_nightmare_active = find_start(lines, "Living Nightmare active")

    def initial_save (self):
        state = Character.initial_save (self)
        state.living_nightmare_active = self.living_nightmare_active
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.living_nightmare_active = state.living_nightmare_active

    def reset (self):
        self.assassin_immobilized_next_beat = False
        self.merciless_immobilized = False
        self.merciless_dodge = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.assassin_immobilized_next_beat = \
            self.assassin_immobilized_next_beat
        state.merciless_immobilized = self.merciless_immobilized
        state.merciless_dodge = self.merciless_dodge
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.assassin_immobilized_next_beat = \
            state.assassin_immobilized_next_beat
        self.merciless_immobilized = state.merciless_immobilized
        self.merciless_dodge = state.merciless_dodge

    def prepare_next_beat (self):
        self.assassin_immobilzied = self.assassin_immobilized_next_beat

    # 0 is no ante
    # -1,1 are ante and going to the left/right of opponent
    # if your destination is impossible, go to the other
    def get_antes (self):
        if self.pool:
            return [0,-1,1]
        else:
            return [0]

    def input_ante (self):
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
        for i in range(abs(ante)):
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

    def blocks_movement (self, direct):
        if (self.assassin_immobilized and
            not self.game.status_effects_blocked()) \
           or self.merciless_immobilized:
            # block everything
            return set(xrange(7))
        else:
            return set()

    # overrides default method, which I set to pass for performance
    def movement_reaction (self, mover, old_position, direct):
        for card in self.active_cards:
            card.movement_reaction (mover, old_position, direct)

    # Apply Merciless when I move opponent.
    def execute_move (self, mover, moves, direct=False):
        old_dir = self.opponent.position - self.position
        Character.execute_move (self, mover, moves, direct)
        if mover == self.opponent and self.merciless in self.active_cards:
            new_dir = self.opponent.position - self.position
            if new_dir * old_dir < 0:
                self.opponent.lose_life (2)
                self.merciless_immobilized = True
                if self.game.reporting:
                    self.game.report (self.opponent.name + " cannot move again this beat")

    # retrieve token at end of beat
    def unique_ability_end_trigger (self):
        if self.game.distance() >= 3 or self.living_nightmare_active:
            self.recover_tokens(1)

    def evaluate (self):
        return Character.evaluate (self) + 2 * len (self.pool)

class Hikaru (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = PalmStrike (the_game,self)
        self.styles = [Trance    (the_game, self), \
                       Focused   (the_game, self), \
                       Advancing (the_game, self), \
                       Sweeping  (the_game, self), \
                       Geomantic (the_game, self)  ]
        self.finishers = [WrathOfElements (the_game, self),
                          FourWinds       (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.tokens = [Earth (the_game, self), \
                       Fire  (the_game, self), \
                       Water (the_game, self), \
                       Wind  (the_game, self)]
        self.water = self.tokens[2]
        
    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = self.tokens[:]

    def choose_initial_discards (self):
        return (self.focused, self.unique_base,
                self.trance, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        tokens = [t.name for t in self.pool]
        report.append ("pool: " + ', '.join(tokens))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [t for t in self.tokens if t.name in lines[0]]

    def get_antes (self):
        return [None] + self.pool

    def input_ante (self):
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
                self.game.report ("Hikaru recovers " + recovered.aname + " token")

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

    def evaluate (self):
        return Character.evaluate(self) + sum(t.value for t in self.pool)


class Kajia (Character):

    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Mandibles (the_game,self)
        self.styles = [Burrowing (the_game,self),
                       Swarming  (the_game,self),
                       Parasitic (the_game,self),
                       Stinging  (the_game,self),
                       Biting    (the_game,self)]
        self.finishers = [Wormwood      (the_game, self),
                           ImagoEmergence (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        
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
        report.append ("%d insects on opponent's discard 1" %
                       self.insects[1])
        report.append ("%d insects on opponent's discard 2" %
                       self.insects[2])
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
        self.infested_piles_on_reveal = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.infested_piles_on_reveal = self.infested_piles_on_reveal
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.infested_piles_on_reveal = state.infested_piles_on_reveal

    def give_insects (self, n, pile=0):
        # can't give more insects than total of 9
        n = min (n, 9 - sum(self.insects))
        if n > 0:
            self.insects[pile] += n
            if self.game.reporting:
                report = "Kajia places %d insect token%s on the oppoent's %s" % \
                         (n,
                          ('s' if n>1 else ''),
                          ('attack pair' if pile==0 else 'disard pile %d' % pile))
                self.game.report (report)
                self.game.report ("(Current insects: %d / %d / %d)" %
                                  tuple(self.insects))

    def infested_piles (self):
        return (self.insects[1] > 0) + (self.insects[2] > 0)
    def total_insects (self):
        return self.insects[1] + self.insects[2]

    # Give counter when Kajia initiates switch
    def execute_move (self, mover, moves, direct=False):
        old_dir = self.opponent.position - self.position
        Character.execute_move (self, mover, moves, direct)
        if mover == self.opponent:
            new_dir = self.opponent.position - self.position
            if new_dir * old_dir < 0:
                self.give_insects(1)

    # Give counter when opponent initiates switch
    def movement_reaction (self, mover, old_position, direct):
        old_dir = (old_position - self.position
                   if mover == self.opponent
                   else self.opponent.position - old_position)
        if mover == self.opponent:
            new_dir = self.opponent.position - self.position
            if new_dir * old_dir < 0:
                self.give_insects(1)

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

    def remove_insects (self):
        insects_removed = 0
        prompt = "Choose number of insects to remove from discard %d"
        for i in (1,2):
            remove = self.game.make_fork (1 + self.insects[i], self,
                                          prompt % i)
            self.insects[i] -= remove
            insects_removed += remove
            if remove and self.game.reporting:
                self.game.report("Kajia removes %d insects from discard %d" %
                                 (remove, i))
        return insects_removed

    def end_trigger (self):
        Character.end_trigger (self)
        if self.imago_emergence_active:
            self.give_insects(1)

    # This just evaluates projected life loss.
    # Style tricks are evaluated by style evaluation bonuses.
    def evaluate (self):
        value = Character.evaluate(self)
        insects = self.insects[1]+self.insects[2]
        value += min (insects, self.opponent.life - 1)
        return value


class Kallistar (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Spellbolt (the_game,self)
        self.styles = [Flare    (the_game,self), \
                       Caustic  (the_game,self), \
                       Volcanic (the_game,self), \
                       Ignition (the_game,self), \
                       Blazing  (the_game,self)  ]
        self.finishers = [Supernova          (the_game, self),
                           ChainOfDestruction (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        
    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.is_elemental = False
        self.priority_penalty = False

    def choose_initial_discards (self):
        return (self.flare, self.strike,
                self.caustic, self.drive)

    def situation_report (self):
        report = Character.situation_report (self)
        if self.is_elemental:
            report.append ("Elemental Form")
        else:
            report.append ("Human Form")
        if self.priority_penalty:
            report.append ("Opponent at -2 priority from last beat")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.is_elemental = find_start (lines, 'Elemental')
        self.priority_penalty = find_start (lines, 'Opponent at -2')

    def initial_save (self):
        state = Character.initial_save (self)
        state.is_elemental = self.is_elemental
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.is_elemental = state.is_elemental

    def reset (self):
        Character.reset (self)
        self.priority_penalty_next_beat = False

    def full_save (self):
        state = Character.full_save (self)
        state.priority_penalty_next_beat = self.priority_penalty_next_beat
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.priority_penalty_next_beat = state.priority_penalty_next_beat

    def prepare_next_beat (self):
        Character.prepare_next_beat (self)
        self.priority_penalty = self.priority_penalty_next_beat
        
    def get_priority_bonus (self):
        return 2*self.is_elemental

    def get_power_bonus (self):
        return 2*self.is_elemental
    
    def get_soak(self):
        soak = Character.get_soak(self)
        if not self.is_elemental:
            soak += 1
        return soak
    
    def give_priority_penalty (self):
        return -2 if (self.priority_penalty and
                     not self.game.status_effects_blocked()) else 0

    def ante_trigger (self):
        if self.is_elemental:
            self.lose_life (1)

    # assumes I'll change to elemental next beat,
    # so value of being elemental is value for one beat,
    # plus value of not having to ignite
    def evaluate (self):
        return Character.evaluate (self) + \
            self.is_elemental * (self.elemental_value() - Ignition.badness)
        
    # value of being elemental (per beat)
    # with low life, the -1 life doesn't hurt
    def elemental_value (self):
        return 0 + (self.life <= 2)

class Karin (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Claw (the_game,self)
        self.styles = [Howling     (the_game,self), \
                       Coordinated (the_game,self), \
                       FullMoon    (the_game,self), \
                       Feral       (the_game,self), \
                       Dual        (the_game,self)  ]
        self.finishers = [RedMoonRage (the_game, self),
                           LunarCross  (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        # give bases different preferred ranges for jager
        # (because jager isn't moved by base)
        self.unique_base.jager_preferred_range = 1.5
        if self.use_beta_bases:
            self.counter_base.jager_preferred_range = 1
            self.wave_base.jager_preferred_range = 3
            self.force.jager_preferred_range = 1
            self.spike.jager_preferred_range = 2
            self.throw.jager_preferred_range = 1
            self.parry.jager_preferred_range = 2
        else:
            self.strike.jager_preferred_range = 1
            self.shot.jager_preferred_range = 2.5
            self.drive.jager_preferred_range = 1
            self.burst.jager_preferred_range = 2.5
            self.grasp.jager_preferred_range = 1
            self.dash.jager_preferred_range = 2

    def choose_initial_discards (self):
        return (self.full_moon, self.strike,
                self.coordinated, self.grasp)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.jager_position = self.position

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

    # self.jager_attacks = True means range is calculated from jager
    def reset (self):
        Character.reset (self)
        self.jager_attacks = False
        self.lunar_swap = False

    def full_save (self):
        state = Character.full_save (self)
        state.jager_attacks = self.jager_attacks
        state.lunar_swap = self.lunar_swap
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.jager_attacks = state.jager_attacks
        self.lunar_swap = state.lunar_swap
        
    board_addendum_lines = 1
    def get_board_addendum (self):
        addendum = ['.'] * 7
        addendum [self.jager_position] = 'j'
        return ''.join(addendum)

    def end_trigger (self):
        Character.end_trigger (self)
        if not self.is_stunned():
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

    # attacks are from Karin or from Jager, depending on style
    # (style sets self.jager_attacks to True if needed)
    def attack_range (self):
        attack_source = self.jager_position if self.jager_attacks else \
                        self.position
        return abs (self.opponent.position - attack_source)

    # overrides default method, which I set to pass for performance
    def movement_reaction (self, mover, old_position, direct):
        for card in self.active_cards:
            card.movement_reaction (mover, old_position, direct)

    # different preferred range for karin and jager, based on the styles
    # that let each of them attack
    # preferred range is now 4-tuple of preferred ranges, and also of
    # weights (number of styles that let each of them attack)
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
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Overload (the_game,self)
        self.styles = [Mutating    (the_game, self), \
                       Bladed      (the_game, self), \
                       Whip        (the_game, self), \
                       Quicksilver (the_game,self), \
                       Exoskeletal (the_game,self)  ]
        self.finishers = [HydraFork (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        
    def choose_initial_discards (self):
        return (self.mutating, self.grasp,
                self.bladed, self.strike)

    # current form is active in addition to normal style and base
    def set_active_cards (self):
        Character.set_active_cards (self)
        # find current form
        for card in self.discard[1]:
            if isinstance (card, Style):
                self.current_form = card
        self.active_cards.append (self.current_form)

    # if mutating is in active cards, replace it with a copy of the other style
    def reveal_trigger (self):
        if self.current_form == self.mutating:
            self.active_cards [self.active_cards.index (self.mutating)] = \
                self.style
        if self.style == self.mutating:
            self.active_cards [self.active_cards.index (self.mutating)] = \
                self.current_form
        Character.reveal_trigger (self)

    # if there's a repeating style in active_cards, lose life for Mutating
    # can't put this in Mutating, because it is replaced at start_trigger
    def end_trigger (self):
        if len (set (self.active_cards)) < len (self.active_cards):
            self.lose_life (1)
        Character.end_trigger (self)

    # if Whip is current form, add Whip bonus (see under Whip)
    # if Mutating is current form, Whip doubling handled by Whip
    def set_preferred_range (self):
        Character.set_preferred_range(self)
        self.preferred_range += (self.whip.preferred_range \
                                 if self.whip in self.discard[1] else 0)

    
class Khadath (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Snare (the_game,self)
        self.styles = [Hunters    (the_game, self), \
                       Teleport   (the_game, self), \
                       Evacuation (the_game, self), \
                       Blight     (the_game,self), \
                       Lure       (the_game,self)  ]
        self.finishers = [DimensionalExile (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        
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
            self.trap_position == None

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

    # non-direct moves can't enter trap and come out on other side
    def blocks_movement (self, direct):
        if direct == True or self.trap_position == self.position \
                          or self.trap_position == None:
            return set()
        return set ([pos for pos in xrange(7) \
                     if ordered (self.opponent.position,
                                 self.trap_position,
                                 pos)])

    def reveal_trigger (self):
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

class Lixis (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Lance (the_game,self)
        self.styles = [Venomous     (the_game, self), \
                       Rooted       (the_game, self), \
                       Naturalizing (the_game, self), \
                       Vine         (the_game,self), \
                       Pruning      (the_game,self)  ]
        self.finishers = [VirulentMiasma (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        
    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.priority_penalty = False
        self.virulent_miasma = False
    
    def choose_initial_discards (self):
        return (self.pruning, self.grasp,
                self.rooted, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        if self.priority_penalty:
            report.append ("Opponent at -2 priority from last beat")
        if self.virulent_miasma:
            report.append ("Virulent Miasma is active")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.priority_penalty = find_start (lines, 'Opponent at -2')
        self.virulent_miasma = find_start (lines, 'Virulent Miasma')

    def initial_save (self):
        state = Character.initial_save (self)
        state.virulent_miasma = self.virulent_miasma
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.virulent_miasma = state.virulent_miasma

    def reset (self):
        Character.reset (self)
        self.priority_penalty_next_beat = False

    def full_save (self):
        state = Character.full_save (self)
        state.priority_penalty_next_beat = self.priority_penalty_next_beat
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.priority_penalty_next_beat = state.priority_penalty_next_beat

    def prepare_next_beat (self):
        Character.prepare_next_beat (self)
        self.priority_penalty = self.priority_penalty_next_beat
        
    def hit_trigger (self):
        Character.hit_trigger (self)
        # opponent discards base
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
        # +2 evaluation for making an opponent's base unavailable for 2 beats
        self.evaluation_bonus += 2

    def give_priority_penalty (self):
        penalty = 0
        if self.priority_penalty and not self.game.status_effects_blocked():
            penalty -= 2
        # If Virulent Miasma is active and opponent started beat with 1 life,
        #she gets a priority penalty
        if self.virulent_miasma and \
           self.game.initial_state.player_states[1-self.my_number].life == 1:
            penalty -= 3
        return penalty

    # when Rooted, add 0 as an option to each self-move you perform
    # (if it's direct, add your own position)
    def execute_move (self, mover, moves, direct=False):
        new_move = self.position if direct else 0
        if self.style.name == 'Rooted' and \
                       mover == self and \
                       new_move not in moves:
            moves.append(new_move)
        Character.execute_move (self, mover, moves, direct)

    def ante_trigger (self):
        if self.virulent_miasma:
            self.opponent.lose_life (3)

    # for performance, the following methods just return False by default
    # so Lixis needs to override them

    # block opponent's power,priority,range from increasing above style+base
    def blocks_bonuses (self):
        return any (card.blocks_bonuses() for card in self.active_cards)
    # block effects of tokens anted and spent
    # cannot refer to self.active_cards, because it is used in making it
    def blocks_tokens (self):
        return self.style.blocks_tokens() or self.base.blocks_tokens()

class Luc (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Flash (the_game, self)
        self.styles = [Eternal  (the_game, self), \
                       Memento  (the_game, self), \
                       Fusion   (the_game, self), \
                       Feinting (the_game, self), \
                       Chrono   (the_game, self)  ]
        self.finishers = [TemporalRecursion (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.time = Time  (the_game, self)
        self.tokens = [self.time]
        self.max_tokens = 5
        # virtual cards help implement ante effects
        self.ante_1_effect = Ante1Effect(the_game, self)
        self.ante_3_effect = Ante3Effect(the_game, self)
        self.ante_5_effect = Ante5Effect(the_game, self)
        
    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 2 * [self.time]

    def choose_initial_discards (self):
        return (self.eternal, self.unique_base,
                self.fusion, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Time tokens" %len(self.pool))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.time] * int(lines[0][0])

    def reset (self):
        self.eternal_spend = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.eternal_spend = self.eternal_spend
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.eternal_spend = state.eternal_spend

    def get_antes (self):
        p = len(self.pool)
        return [a for a in [0,1,3,5] if a <= p]

    def input_ante (self):
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
        self.active_cards = [self.style, self.base]
        n_tokens = self.get_active_tokens().count(self.time)
        if n_tokens == 1:
            self.active_cards.append(self.ante_1_effect)
        elif n_tokens == 3:
            self.active_cards.append(self.ante_3_effect)
        elif n_tokens == 5:
            self.active_cards.append(self.ante_5_effect)
     
    def ante_trigger (self):
        for i in range(self.strat[2][0]):
            self.ante_token(self.time)

    def unique_ability_end_trigger (self):
        self.recover_tokens (1)

    def evaluate (self):
        return Character.evaluate (self) + 0.3 * len(self.pool)

class Lymn (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Visions (the_game,self)
        self.styles = [Maddening  (the_game, self), \
                       Chimeric   (the_game, self), \
                       Surreal    (the_game, self), \
                       Reverie    (the_game,self), \
                       Fathomless (the_game,self)  ]
        self.finishers = [Megrim  (the_game, self),
                           Conceit (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        
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
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Blessing (the_game,self)
        self.styles = [Safety        (the_game, self),
                       Priestess     (the_game, self),
                       Spiritual     (the_game,self),
                       Sanctimonious (the_game, self),
                       Excelsius     (the_game,self)]
        self.finishers = [SolarSoul  (the_game, self),
                          Apotheosis (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        
    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.level = 0
        self.trance = 0

    def choose_initial_discards (self):
        return (self.spiritual, self.drive,
                self.sanctimonious, self.grasp)

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
            self.game.report ("Magdelina gets a +%d bonus to power, priority and stungard" % self.level)

    def get_level_bonus(self):
        if self.spiritual in self.active_cards:
            return 0
        else:
            return self.initial_level

    def get_priority_bonus (self):
        return self.get_level_bonus() + Character.get_priority_bonus(self)
    def get_power_bonus (self):
        return self.get_level_bonus() + Character.get_power_bonus(self)
    def get_stunguard (self):
        return self.get_level_bonus() + Character.get_stunguard(self)

    def unique_ability_end_trigger (self):
        self.gain_trance()
        if self.trance > self.level:
            self.trance = 0
            if self.game.reporting:
                self.game.report("Magdelina discards all Trance counters")
            self.gain_level()

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

    # only half value to dealing damage    
    def evaluate (self):
        # Each trance token is worth 1 point for each level
        # you expect to go up (it hastens the level by one beat).
        return (Character.evaluate(self)
                + self.opponent.effective_life() / 2.0
                + self.level_values[self.level]
                + self.trance * self.trance_values_per_level[self.level])
               
    # importance of going up a level is 15,12,9,6,3
    level_values = [0,4,10,16,20,22.5]
    trance_values_per_level = [0, 3, 2, 1, 0.5, 0]

    # TODO: her preferred range should probably be more about not being at the
    # opponent's preferred range (at least at low levels)

class Marmelee (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Meditation (the_game, self)
        self.styles = [Petrifying  (the_game, self), \
                       Magnificent (the_game, self), \
                       Sorceress   (the_game, self), \
                       Barrier     (the_game, self), \
                       Nullifying  (the_game, self)  ]
        self.finishers = [AstralCannon (the_game, self),
                           AstralTrance (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.concentration = 5
        
    def choose_initial_discards (self):
        return (self.petrifying, self.burst,
                self.barrier, self.grasp)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.concentration = 2

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Concentration counters" % self.concentration)
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.concentration = int(lines[0][0])

    def initial_save (self):
        state = Character.initial_save (self)
        state.concentration = self.concentration
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.concentration = state.concentration

    def reset (self):
        self.counters_spent_by_style = 0
        self.counters_spent_by_base = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.counters_spent_by_style = self.counters_spent_by_style
        state.counters_spent_by_base = self.counters_spent_by_base
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.counters_spent_by_style = state.counters_spent_by_style
        self.counters_spent_by_base = state.counters_spent_by_base

    def unique_ability_end_trigger (self):
        self.recover_counters (1)

    def recover_counters(self, n):
        old = self.concentration
        self.concentration = min (5, old + n)
        diff = self.concentration - old
        if diff and self.game.reporting:
            self.game.report("Marmelee gains %d Concentration Counter%s" %
                             (diff, "s" if diff > 1 else ""))
            
    def discard_counters(self, n):
        old = self.concentration
        self.concentration = max (0, old - n)
        diff = old - self.concentration
        if diff and self.game.reporting:
            self.game.report("Marmelee discards %d Concentration Counter%s" %
                             (diff, "s" if diff > 1 else ""))

    # lose all counters when stunned
    def stun (self, damage=None):
        Character.stun (self, damage)
        if self.is_stunned():
            if self.game.reporting and self.pool:
                self.game.report ("Marmelee loses all counters")
            self.concentration = 0

    def evaluate (self):
        return Character.evaluate (self) + 0.85 * self.concentration

class Mikhail (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Scroll (the_game, self)
        self.styles = [Immutable    (the_game, self), \
                       Transcendent (the_game, self), \
                       Hallowed     (the_game, self), \
                       Apocalyptic  (the_game, self), \
                       Sacred       (the_game, self)  ]
        self.finishers = [MagnusMalleus (the_game, self),
                           TheFourthSeal (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.seal = Seal (the_game, self)
        self.tokens = [self.seal]
        self.max_tokens = 3

    def choose_initial_discards (self):
        return (self.hallowed, self.grasp,
                self.apocalyptic, self.strike)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 3 * [self.seal]

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Seal tokens" %len(self.pool))
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

    def input_ante (self):
        if self.pool:
            print "Ante Seal token?"
            return menu (["No", "Yes"])
        else:
            return 0

    def ante_trigger (self):
        for i in range(self.strat[2][0]):
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
    
    # 0,4,6,8 didn't go well - he steady-stated at 2 tokens, instead of 1.6,
    # and lost more games.
    token_values = (0,3,4,5)
    def evaluate (self):
        return Character.evaluate(self) + self.token_values[len(self.pool)]


class Oriana (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Meteor (the_game, self)
        self.styles = [Celestial   (the_game, self), \
                       Stellar     (the_game, self), \
                       Unstable    (the_game, self), \
                       Metamagical (the_game, self), \
                       Calamity    (the_game, self)  ]
        self.finishers = [NihilEraser   (the_game, self),
                           GalaxyConduit (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.mp = MagicPoint (the_game, self)
        self.tokens = [self.mp]
        self.max_tokens = 10

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 6 * [self.mp]

    def choose_initial_discards (self):
        return (self.stellar, self.grasp,
                self.calamity, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Magic Point tokens" %len(self.pool))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.mp] * int(lines[0][0])

    def get_antes (self):
        return range (len(self.pool) + 1)

    def input_ante (self):
        n = len(self.pool)
        if n > 0 :
            print "Number of Magic Point tokens [0-%d]: " %n
            return input_number (n+1)
        else:
            return 0

    def ante_trigger (self):
        for i in range(self.strat[2][0]):
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

class Rexan (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Malediction  (the_game, self)
        self.styles = [Unyielding   (the_game, self), \
                       Devastating  (the_game, self), \
                       Enervating   (the_game, self), \
                       Vainglorious (the_game, self), \
                       Overlords    (the_game, self)  ]
        self.finishers = [ZeroHour     (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.curse = Curse (the_game, self)
        self.induced_tokens = [self.curse]

    def choose_initial_discards (self):
        return (self.enervating, self.grasp,
                self.devastating, self.dash)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.induced_pool = []

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("Opponent has %d Curse tokens" %
                       len(self.induced_pool))
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.induced_pool = [self.curse] * int(lines[0][13])
        
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
        for i in xrange(self.opponent.strat[2][1]):
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
        if (not self.opponent.did_hit and 
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
        
    # returning tokens on cycle, so that it happens on a Pulse, too
    def cycle (self):
        if not (self.did_hit or self.opponent.did_hit):
            self.give_induced_tokens (self.opponent.ante.count(self.curse))
        Character.cycle (self)

    # There seems to be little punishment for hoarding tokens
    def evaluate (self):
        value = Character.evaluate(self)
        value += 1.0 * len(self.induced_pool)
        return value

class Rukyuk (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Reload  (the_game, self)
        self.styles = [Sniper     (the_game, self), \
                       PointBlank (the_game, self), \
                       Gunner     (the_game, self), \
                       Crossfire  (the_game, self), \
                       Trick      (the_game, self)  ]
        self.finishers = [FullyAutomatic (the_game, self),
                           ForceGrenade   (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.tokens = [APShell        (the_game, self), \
                       ExplosiveShell (the_game, self), \
                       FlashShell     (the_game, self), \
                       ImpactShell    (the_game, self), \
                       LongshotShell  (the_game, self), \
                       SwiftShell     (the_game, self)  ]
        self.flash = self.tokens[2]
        self.impact = self.tokens[3]

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = [t for t in self.tokens]

    def choose_initial_discards (self):
        # AI says: Crossfire Reload, Trick Shot
        return (self.crossfire, self.unique_base,
                self.trick, self.shot)

    def situation_report (self):
        report = Character.situation_report (self)
        tokens = [t.name for t in self.pool]
        report.append ("pool: " + ', '.join(tokens))
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

    def input_ante (self):
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
        if self.base.name in self.finishers:
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
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Tinker  (the_game, self)
        self.styles = [Channeled   (the_game, self),
                       Maintenance (the_game, self),
                       Explosive   (the_game, self),
                       Impact      (the_game, self),
                       Overcharged (the_game, self)]
        self.finishers = [ArtificeAvarice (the_game, self),
                           UdstadBeam      (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.artifacts = [Autodeflector  (the_game, self),
                          Battlefist     (the_game, self),
                          HoverBoots     (the_game, self),
                          ShieldAmulet   (the_game, self),
                          PhaseGoggles   (the_game, self)     ]
        self.autodeflector = self.artifacts[0]
        self.battlefist = self.artifacts[1]
        self.hover_boots = self.artifacts[2]
        self.phase_goggles = self.artifacts[4]

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
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.overcharged_artifact = self.overcharged_artifact
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.overcharged_artifact = state.overcharged_artifact

    def set_active_cards (self):
        Character.set_active_cards(self)
        if self.base.name != 'Udstad Beam':
            self.active_cards += list(self.active_artifacts)
            
    def activate_artifact (self, artifact):
        if artifact in self.deactivated_artifacts:
            self.active_artifacts.add(artifact)
            self.deactivated_artifacts.remove(artifact)
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
        targets = list(self.active_artifacts - set((self.overcharged_artifact,)))
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
        if not self.opponent.did_hit and \
           not self.base.name == 'Artifice Avarice':
            # if Tinker played, Runika chooses artifact to deactivate
            chooser = self if self.unique_base in self.active_cards \
                      else self.opponent
            self.deactivation_fork (chooser)

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
    
class Seth (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Omen  (the_game, self)
        self.styles = [Fools      (the_game, self), \
                       Mimics     (the_game, self), \
                       Vanishing  (the_game, self), \
                       Wyrding    (the_game, self), \
                       Compelling (the_game, self)  ]
        self.finishers = [FortuneBuster (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        
    def choose_initial_discards (self):
        return (self.fools, self.grasp,
                self.wyrding, self.strike)

    # these aren't really possible antes, but rather possible ante results
    # (correct or incorret guess)
    # the actual strategies will be formed in post-processing
    def get_antes (self):
        return [False, True]
        
    def input_ante (self):
        discards = self.opponent.discard[1] | self.opponent.discard[2]
        bases = [b for b in self.opponent.bases if b not in discards]
        print "Name one of the opponent's bases:"
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
        if self.wyrding in self.active_cards:
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
    def movement_reaction (self, mover, old_position, direct):
        for card in self.active_cards:
            card.movement_reaction (mover, old_position, direct)

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

        # For each pair of Seth's strategies (same except for guess correctness),
        # pick one.
        strats_without_guesses = [s for s in fake_strats
                                 if s[2][0] is False]
        # Then, make a new strategy list, with 5 actual guess per strategy.
        full_strats = [(s[0], s[1], (b, s[2][1]))
                       for s in strats_without_guesses for b in bases]

        n = len(full_strats)
        m = len(opp_strats)
        # for each strat combination, take the simulation result corresponding
        # to those pairs, but pick correct/incorrect guess based on whether
        # Seth's guess actually corresponds to the played base
        if self.my_number==0:
            full_results = [[0]*m for _ in xrange(n)]
            for i in xrange(n):
                for j in xrange(m):
                    my_strat = full_strats[i]
                    opp_strat = opp_strats[j]
                    is_correct = (my_strat[2][0] == opp_strat[1])
                    fake_strat = (my_strat[0],
                                     my_strat[1],
                                     (is_correct, my_strat[2][1]))
                    fake_i = fake_strats.index(fake_strat)
                    full_results[i][j] = fake_results[fake_i][j]
        else:
            full_results = [[0]*n for _ in xrange(m)]
            for i in xrange(m):
                for j in xrange(n):
                    my_strat = full_strats[j]
                    opp_strat = opp_strats[i]
                    is_correct = (my_strat[2][0] == opp_strat[1])
                    fake_strat = (my_strat[0],
                                     my_strat[1],
                                     (is_correct, my_strat[2][1]))
                    fake_j = fake_strats.index(fake_strat)
                    full_results[i][j] = fake_results[i][fake_j]

        self.game.results = full_results
        self.strats = full_strats

    # When Seth clashes, the correctness of his guess is still determined by
    # the opponent's original base choice.
    # We simulate this by having him change his guess, to keep it correct (or
    # incorrect, as the case may be)

    # This fixes the index of each of his post-clash strategies,
    # taking into account opponent's new base (so that guess correctness is
    # preserved, and we get the result we need from the simulation results
    # table.
    def clash_strat_index (self, my_new_i, opp_new_i, my_orig_i, opp_orig_i):
        # If the guess is already in boolean form, this is a second clash,
        # and everything is set correctly.
        if isinstance(self.strats[my_new_i][2][0], bool):
            return my_new_i
        
        my_strats = self.strats
        opp_strats = self.opponent.strats
        my_orig_strat = my_strats[my_orig_i]
        opp_orig_strat = opp_strats[opp_orig_i]
        my_new_strat = my_strats[my_new_i]
        opp_new_strat = opp_strats[opp_new_i]
        is_correct = (my_orig_strat[2][0] == opp_orig_strat[1])
        if is_correct:
            # Original guess was correct.  Change guess to keep it correct
            # vs. new base chosen.
            correct_new_strat = (my_new_strat[0],
                                 my_new_strat[1],
                                 (opp_new_strat[1], my_new_strat[2][1]))
            try:
                return my_strats.index(correct_new_strat)
            except Exception as e:
                print "Error"
                print correct_new_strat
                print "is not in list:"
                for s in my_strats:
                    print s
                raise e
        else:
            # Original guess is incorrect.
            # It's still incorrect, so no problem:
            if my_orig_strat[2][0] != opp_new_strat[1]:
                return my_new_i
            # The opponent switched his base into my guess,
            # so I change my guess to his original choice, to keep it wrong.
            else:
                incorrect_new_strat = (my_new_strat[0],
                                       my_new_strat[1],
                                       (opp_orig_strat[1], my_new_strat[2][1]))
                try:
                    return my_strats.index(incorrect_new_strat)
                except Exception as e:
                    print "Error"
                    print incorrect_new_strat
                    print "is not in list:"
                    for s in my_strats:
                        print s
                    raise e
                
    # This actually fixes the strategies themselves, by turning the ante into
    # the boolean version, based on original guess correctness.
    def fix_strategies_post_clash (self, strats, opp_orig):
        # if ante is already boolean, this is not the first clash,
        # and there's no need to touch anything
        if isinstance(strats[0][2][0], bool):
            return strats
        # Otherwise, check original opponent strategy to see if guess was
        # originally correct, and write that into strategies.
        is_correct = (strats[0][2][0] == opp_orig[1])
        new_strats = [(s[0], s[1], (is_correct, s[2][1]))
                      for s in strats]
        return new_strats


class Shekhtur (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Brand (the_game, self)
        self.styles = [Unleashed   (the_game, self), \
                       Combination (the_game, self), \
                       Reaver      (the_game, self), \
                       Jugular     (the_game, self), \
                       Spiral      (the_game, self)  ]
        # Soul Breaker isn't fully implemented, so only Coffin Nails is listed
        self.finishers = [CoffinNails (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.malice = Malice (the_game, self)
        self.tokens = [self.malice]
        self.max_tokens = 5
        
    def choose_initial_discards (self):
        return (self.combination, self.unique_base,
                self.jugular, self.grasp)

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = 3 * [self.malice]
        self.did_hit_last_beat = False
        self.unleashed_bonus = False
        self.coffin_nails_hit = False

    def situation_report (self):
        report = Character.situation_report (self)
        report.append ("%d Malice tokens" %len(self.pool))
        if self.unleashed_bonus:
            report.append ("+1 Power from playing Unleashed last beat")
        if self.coffin_nails_hit:
            report.append ("Opponent has no soak or stunguard (Coffin Nails)")
        if self.did_hit_last_beat:
            report.append ("Shekhtur hit the opponent last beat")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.pool = [self.malice] * int(lines[0][0])
        self.did_hit_last_beat = find_start (lines, 'Shekhtur hit the opponent')
        self.unleashed_bonus = find_start (lines, '+1 Power')
        self.coffin_nails_hit = find_end (lines, '(Coffin Nails)\n')

    def initial_save (self):
        state = Character.initial_save (self)
        state.coffin_nails_hit = self.coffin_nails_hit
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.coffin_nails_hit = state.coffin_nails_hit

    def reset (self):
        self.unleashed_bonus_next_beat = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.unleashed_bonus_next_beat = self.unleashed_bonus_next_beat
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.unleashed_bonus_next_beat = state.unleashed_bonus_next_beat

    def prepare_next_beat (self):
        Character.prepare_next_beat (self)
        self.did_hit_last_beat = self.did_hit
        self.unleashed_bonus = self.unleashed_bonus_next_beat

    def get_antes (self):
        n = len(self.pool)
        return range (0, n + 1)

    def input_ante (self):
        n = len(self.pool)
        if n > 0 :
            print "Number of Malice tokens [0-%d]: " %n
            return input_number (n+1)
        else:
            return 0

    def ante_trigger (self):
        for i in range(self.strat[2][0]):
            self.ante_token(self.malice)

    def get_ante_name (self, a):
        if a == 0:
            return ""
        if a == 1:
            return "1 token"
        return str(a) + " tokens"

    def get_power_bonus (self):
        return 1 if (self.unleashed_bonus and
                     not self.game.status_effects_blocked()) else 0

    # shortcut: no other way for opponent to lose all soak
    def reduce_soak (self, soak):
        return (0 if self.coffin_nails_hit and
                     not self.game.status_effects_blocked()
                else soak)
    # shortcut: no other way for opponent to lose all stunguard
    def reduce_stunguard (self, stunguard):
        return (0 if self.coffin_nails_hit and
                     not self.game.status_effects_blocked()
                else stunguard)

    def damage_trigger (self, damage):
        self.recover_tokens (damage)
        Character.damage_trigger (self, damage)

    def evaluate (self):
        return Character.evaluate (self) + 0.3 * len(self.pool)

# Add special case juto and style evaluation
class Tatsumi (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Whirlpool  (the_game, self)
        self.styles = [Siren     (the_game, self), \
                       Fearless  (the_game, self), \
                       Riptide   (the_game, self), \
                       Empathic  (the_game, self), \
                       WaveStyle (the_game, self)  ]
        self.finishers = [TsunamisCollide (the_game, self),
                           BearArms (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        # give bases different preferred ranges for juto
        # (because juto isn't moved by base)
        self.unique_base.juto_preferred_range = 1.5
        if self.use_beta_bases:
            self.counter_base.juto_preferred_range = 1
            self.wave_base.juto_preferred_range = 3
            self.force.juto_preferred_range = 1
            self.spike.juto_preferred_range = 2
            self.throw.juto_preferred_range = 1
            self.parry.juto_preferred_range = 2
        else:
            self.strike.juto_preferred_range = 1
            self.shot.juto_preferred_range = 2.5
            self.drive.juto_preferred_range = 1
            self.burst.juto_preferred_range = 2.5
            self.grasp.juto_preferred_range = 1
            self.dash.juto_preferred_range = 2

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
        self.riptide_zone_2 = False
        self.juto_damage_taken = 0
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.riptide_zone_2 = self.riptide_zone_2
        state.juto_damage_taken = self.juto_damage_taken 
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.riptide_zone_2 = state.riptide_zone_2
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

    def get_soak (self):
        return Character.get_soak(self) + 1 * self.zone_2() + 3 * self.zone_1()

    # Reduce Juto's life when he soaks damage.
    def soak_trigger (self, soaked_damage):
        self.juto_damage_taken += soaked_damage
        self.juto_life -= soaked_damage
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
        if isinstance (self.style, Fearless) and self.juto_position != None:
            return abs (self.juto_position - self.opponent.position)            
        return abs (self.position - self.opponent.position)

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

class Vanaah (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Scythe  (the_game, self)
        self.styles = [Reaping   (the_game, self), \
                       Judgment  (the_game, self), \
                       Glorious  (the_game, self), \
                       Paladin   (the_game, self), \
                       Vengeance (the_game, self)  ]
        self.finishers = [DeathWalks     (the_game, self),
                           HandOfDivinity (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        self.divine_rush = DivineRush (the_game, self)
        self.tokens = [self.divine_rush]

    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.pool = [self.divine_rush]
        self.priority_penalty = False

    def choose_initial_discards (self):
        return (self.vengeance, self.burst,
                self.reaping, self.strike)

    def situation_report (self):
        report = Character.situation_report (self)
        if self.pool:
            report.append ("Divine Rush token in pool")
        if self.priority_penalty:
            report.append ("Opponent at -4 priority from last beat")
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
        self.priority_penalty = find_start (lines, 'Opponent at -4')

    def reset (self):
        self.judgment_catch_in_reveal_phase = False
        self.priority_penalty_next_beat = False
        Character.reset (self)

    def full_save (self):
        state = Character.full_save (self)
        state.judgment_catch_in_reveal_phase = \
                                            self.judgment_catch_in_reveal_phase
        state.priority_penalty_next_beat = self.priority_penalty_next_beat
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.judgment_catch_in_reveal_phase = \
                                            state.judgment_catch_in_reveal_phase
        self.priority_penalty_next_beat = state.priority_penalty_next_beat

    def prepare_next_beat (self):
        Character.prepare_next_beat (self)
        self.priority_penalty = self.priority_penalty_next_beat

    def get_antes (self):
        return range (len(self.pool) + 1)

    def input_ante (self):
        if self.pool:
            print "Ante Divine Rush token?"
            return menu (["No", "Yes"])
        else:
            return 0

    def ante_trigger (self):
        for i in range(self.strat[2][0]):
            self.ante_token(self.divine_rush)

    def get_ante_name (self, a):
        return ("Divine Rush" if a==1 else "")
        
    def give_priority_penalty (self):
        return -4 if (self.priority_penalty and
                     not self.game.status_effects_blocked()) else 0

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

    # token will cycle with cards from 0 to 1 to 2, but needs some help...
    def cycle (self):
        # ... coming back from discard 2
        if self.divine_rush in self.discard[2] and \
           not isinstance (self.style, SpecialAction):
            self.pool = [self.divine_rush]
        # ... and moving from 0 to 1 with finisher (which doesn't cycle)
        if self.finishers[0] in self.discard[0] and  \
           self.divine_rush in self.discard[0]:
            self.discard[1].add(self.divine_rush)
        Character.cycle (self)


class Voco (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
        self.unique_base = Shred  (the_game, self)
        self.styles = [Monster     (the_game, self), \
                       Metal       (the_game, self), \
                       Hellraising (the_game, self), \
                       Abyssal     (the_game, self), \
                       Thunderous  (the_game, self)  ]
        self.finishers = [ZMosh   (the_game, self),
                           TheWave (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        
    def set_starting_setup (self, default_discards, use_special_actions):
        Character.set_starting_setup (self, default_discards, use_special_actions)
        self.zombies = set()
        self.monster_power_bonus = False

    def choose_initial_discards (self):
        return (self.thunderous, self.drive,
                self.monster, self.burst)
    
    def situation_report (self):
        report = Character.situation_report (self)
        if self.monster_power_bonus:
            report.append ("+2 power from last beat")
        return report

    def read_my_state (self, lines, board, addendum):
        lines = Character.read_my_state (self, lines, board, addendum)
        self.zombies = set([i for i in xrange(7) if addendum[0][i]=='z'])
        self.monster_power_bonus = find_start (lines, '+2 power')

    def initial_save (self):
        state = Character.initial_save (self)
        state.zombies = self.zombies.copy()
        return state

    def initial_restore (self, state):
        Character.initial_restore (self, state)
        self.zombies = state.zombies.copy()

    def reset (self):
        Character.reset (self)
        self.monster_power_bonus_next_beat = False

    def full_save (self):
        state = Character.full_save (self)
        state.monster_power_bonus_next_beat = self.monster_power_bonus_next_beat
        return state

    def full_restore (self, state):
        Character.full_restore (self, state)
        self.monster_power_bonus_next_beat = state.monster_power_bonus_next_beat

    def prepare_next_beat (self):
        Character.prepare_next_beat (self)
        self.monster_power_bonus = self.monster_power_bonus_next_beat

    # this was buggy until 6/2/13 - no bonus was given.
    def get_power_bonus (self):
        return 2 if (self.monster_power_bonus and
                     not self.game.status_effects_blocked()) else 0

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
        return len (self.zombies & self.soak_positions())
    
    # eliminates all zombies next to attacker after soaking
    def soak_trigger (self, damage_soaked):
        if self.z_mosh not in self.active_cards:
            # all zombies are removed, regardless of damage soaked
            self.remove_zombies(self.soak_positions(), 2)

    def unique_ability_end_trigger (self):
        opp = self.opponent.position
        if opp in self.zombies:
            if self.game.make_fork (2, self, "Remove zombie to cause 1 life loss?",
                                    ["No", "Yes"]):
                self.opponent.lose_life (1)
                self.remove_zombies(set([opp]), 1)

    # Metal style leaves zombies behind on my move
    def execute_move (self, mover, moves, direct=False):
        old_pos = self.position
        Character.execute_move (self, mover, moves, direct)
        if mover==self and isinstance (self.style, Metal):
            # Voco has no direct moves, so assume he passed through all
            # spaces between old position and current position
            # (except the opponent's)
            self.add_zombies (pos_range(old_pos,self.position) \
                                - set((self.position, self.opponent.position)))

    # overrides default method, which I set to pass for performance
    def movement_reaction (self, mover, old_position, direct):
        for card in self.active_cards:
            card.movement_reaction (mover, old_position, direct)

    # zombies in the soak range or on opponent are .5
    # zombies anywhere else are .25
    def evaluate (self):
        op = self.opponent.position
        better = self.soak_positions() | set([op])
        value =  Character.evaluate(self) + 0.25 * len(self.zombies) \
                                         + 0.25 * len(self.zombies & better)
        return value

class Zaamassal (Character):
    def __init__ (self, the_game, n, use_beta_bases=False, is_user=False):
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
        self.fluidity = self.paradigms[1]
        self.resilience = self.paradigms[3]
        self.distortion = self.paradigms[4]
        self.finishers = [OpenTheGate  (the_game, self),
                           PlaneDivider (the_game, self)]
        Character.__init__ (self, the_game, n, use_beta_bases, is_user)
        
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
            if len(paradigms) == 1:
                self.game.report ("Zaamassal assumes the Paradigm of " + paradigms[0].name)
            else:
                self.game.report ("Zaamassal assumes the following paradigms:")
                for paradigm in paradigms:
                    self.game.report ("   " + paradigm.name)
        self.set_active_cards ()

    def set_active_cards (self):
        Character.set_active_cards (self)
        self.active_cards.extend (list(self.active_paradigms))

    # Resilience's Soak checked here, becuase it works even if Resilience was
    # already replaced.
    def get_soak (self):
        return self.resilience_soak + Character.get_soak (self)

    # lose all paradigms when stunned
    def stun (self, damage=None):
        Character.stun (self, damage)
        if self.is_stunned():
            self.set_active_paradigms ([])
            
    # when Sturdy, add 0 as an option to each self-move you perform
    # (if it's direct, add your own position)
    def execute_move (self, mover, moves, direct=False):
        new_move = self.position if direct else 0
        if self.style.name == 'Sturdy' and \
                       mover == self and \
                       new_move not in moves:
            moves.append(new_move)
        Character.execute_move (self, mover, moves, direct)

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

# Generic Bases

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
    def get_preferred_range (self):
        return 2.5 if self.me.position in [0,6] else 1.5
    def start_trigger(self):
        self.me.retreat ([1,2])
    ordered_start_trigger = True

class Grasp (Base):
    minrange = 1
    maxrange = 1
    power = 2
    priority = 5
    preferred_range = 1
    def hit_trigger(self):
        self.me.move_opponent([1])
    ordered_hit_trigger = True
        
class Dash (Base):
    power = None
    priority = 9
    preferred_range = 2 # dash can switch sides at range 1-3
    is_attack = False
    def after_trigger(self):
        old_pos = self.me.position
        self.me.move([1,2,3])
        if ordered (old_pos, self.opponent.position, self.me.position):
            self.me.triggered_dodge = True
    ordered_after_trigger = True
    # Dash is usually a strong out, that's worth keeping in hand
    def discard_penalty(self):
        return -0.5

# Beta Bases

class Counter (Base):
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
    minrange = 2
    maxrange = 4
    power = 3
    priority = 4
    # extra range in corner
    def get_preferred_range (self):
        return 3 if self.me.position in [0,6] else 2
    def before_trigger(self):
        self.me.retreat ([1,2])
    ordered_before_trigger = True

class Force (Base):
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

class Spike (Base):
    minrange = 2
    maxrange = 2
    power = 3
    priority = 3
    # range restriction in corner
    def get_preferred_range (self):
        return 2.5 if self.me.position in [0,6] else 2
    def start_trigger(self):
        self.me.move([0, 1])
    ordered_start_trigger = True
    def hit_trigger(self):
        self.me.triggered_dodge = True

class Throw (Base):
    minrange = 1
    maxrange = 1
    power = 2
    priority = 5
    preferred_range = 1
    def hit_trigger(self):
        self.me.execute_move(self.opponent, [-1,2])
    ordered_hit_trigger = True

class Parry (Base):
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


# Special Action style and bases

class SpecialAction (Style):
    name_override = 'Special'
    order = 10 # presented after regular styles in lists
    # value of keeping this in hand
    value = 5.0
    # this is only relevant for finishers -
    # pulses and cancels never get to the reveal phase
    def reveal_trigger (self):
        self.special_action_available = False

# Cancel, Pulse and Finisher inherit from this
class SpecialBase (Base):
    pass

# classes for specific finishers inherit from this
class Finisher (SpecialBase):
    order = 10
    # finishers win priority, even vs. characters that win priority
    clash_priority = 0.2
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
    def get_soak (self):
        return self.me.petrification
    # back push handled by Adjenna.soak_trigger()
        
class Pacifying (Style):
    power = -1
    def after_trigger (self):
        self.me.move ((1,2))
    ordered_after_trigger = True
    def movement_reaction (self, mover, old_pos, direct):
        if mover is self.opponent and mover.position!=old_pos:
            self.opponent.lose_life (self.me.petrification)

class Irresistible (Style):
    stunguard = 3
    def before_trigger (self):
        if self.opponent.did_hit:
            self.me.pull ((3,2,1,0))
    @property
    def ordered_before_trigger(self):
        return self.opponent.did_hit

class Beckoning (Style):
    minrange = 2
    maxrange = 5
    power = -2
    priority = -1
    preferred_range = 3.5
    def get_stunguard (self):
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
        # advance up to 5, get +1 power per advance
        old_pos = self.me.position
        self.me.advance ((0,1,2,3,4,5))
        spaces_advanced = abs(self.me.position-old_pos)
        # If I switched sides, actual advance is one less then distance moved
        if ordered (old_pos, self.opponent.position, self.me.position):
            spaces_advanced -= 1
        self.me.add_triggered_power_bonus (spaces_advanced)
    ordered_before_trigger = True
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
    def blocks_pullpush (self):
        return set(xrange(7))
    def start_trigger (self):
        self.me.advance ([0,1])
    ordered_start_trigger = True
    
class Regal (Style):
    power = 1
    priority = 1
    stunguard = 3
    # Opponents at range 1 can't retreat
    def blocks_movement (self, direct):
        if direct:
            return set()
        if self.game.distance() > 1:
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
        old_pos = self.me.position
        # advance up to 3, but don't switch sides
        self.me.advance (range(min(self.game.distance(), 3)))
        spaces_advanced = abs(self.me.position-old_pos)
        self.me.add_triggered_power_bonus (spaces_advanced)
    ordered_before_trigger = True

class Mighty (Style):
    priority = 1
    soak = 1
    stunguard = 2
    # Opponent ignores style range modifier.
    # Hack: give a penalty equal to the style bonus.
    # This works because no one makes their own range equal to printed value
    # TODO: check if non printed style bonuses are also ignored.
    def give_minrange_penalty (self):
        return -self.opponent.style.minrange
    def give_maxrange_penalty (self):
        return -self.opponent.style.maxrange

class Steeled (Style):
    power = 2
    priority = -1
    preferred_range = 2 # can usually advance 3, and higher range gives more soak
    def get_soak (self):
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
        # This implementation is hacky, but should work as long as opponents
        # don't reference their own printed priority
        return 3 - self.opponent.base.priority
    # Pull opponent up to 1 space towards Magnetron.
    def start_trigger (self):
        # Can't pull if no Magnetron, or if Magnetron on opponent.
        if self.me.magnetron.position in [None, self.opponent.position]:
            return
        if ordered (self.me.position,
                    self.opponent.position,
                    self.me.magnetron.position):
            self.me.push((1,))
        else:
            self.me.pull((1,))
    @property
    def ordered_start_trigger(self):
        return self.me.magnetron.position is not None
    def evaluation_bonus (self):
        return (0.1 if self.me.magnetron.position not in (None,
                                                          self.opponent.position)
                else -0.1)
                                    
class Laser (Style):
    # make the choice about origin of attack - just before attacking
    # (Arias's before trigger calls base before style)
    def before_trigger (self):
        sources = [self.me] + [d for d in self.me.droids
                               if d.position is not None]
        ans = self.game.make_fork (len(sources), self.me,
                                   "Choose source of attack:",
                                   sources)
        self.me.attacker = sources[ans]
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

class Catalyst (Style):
    minrange = 1
    maxrange = 2
    power = -1
    priority = 1
    preferred_range = 1.5
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
    
#Byron

class SoulTrap (Finisher):
    minrange = 1
    maxrange = 3
    power = 7
    priority = 2
    def has_stun_immunity (self):
        return True
    def hit_trigger (self):
        self.me.recover_emblem()
    def evaluate_setup (self):
        return 1 if self.game.distance() in (1,2,3) else 0

class SoulGate (Finisher):
    minrange = 3
    maxrnage = 4
    power = 25
    def can_be_hit (self):
        return self.opponent.moved
    def evaluate_setup (self):
        return 1 if self.game.distance() in (3,4) else 0

class Smoke (Base):
    minrange = 1
    maxrange = 5
    power = 2
    priority = 3
    preferred_range = 2.5 # ignores style range
    def hit_trigger (self):
        self.me.move_opponent((0,1))
    ordered_hit_trigger = True
    # Ignoring style range handled by Byron.get_maxrange(), Byron.get_minrange()

class Soulless (Style):
    power = 1
    priority = -4
    def has_stun_immunity (self):
        return True
    def get_soak (self):
        return 5 - self.me.mask_emblems
    def reveal_trigger (self):
        if self.game.distance() == 1:
            self.opponent.lose_life (2)
            self.me.add_triggered_power_bonus (2)
    def end_trigger (self):
        if self.me.emblems_discarded == 0:
            self.me.discard_emblem()
    def evaluation_bonus (self):
        return 1 if self.game.distance() == 1 else -1

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
    def reduce_soak (self, soak):
        return 0 if self.me.heartless_ignore_soak else soak
    def damage_trigger (self, damage):
        self.me.heartless_soak = damage
    def get_soak (self):
        return self.me.heartless_soak

class Faceless (Style):
    maxrange = 1
    power = 1
    preferred_range = 0.5
    def hit_trigger (self):
        self.me.triggered_dodge = True
    def damage_trigger (self, damage):
        if self.me.attack_range() == 2:
            self.me.priority_bonus_next_beat = damage
            self.me.evaluation_bonus += damage / 3.0
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
    def end_trigger (self):
        self.me.priority_bonus_next_beat = True
        self.me.evaluation_bonus += 1.3

class Clockwork (Style):
    power = 3
    priority = -3
    soak = 3

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
    def discard_penalty (self):
        return 0.5

class Mechanical (Style):
    power = 2
    priority = -2
    def end_trigger (self):
        self.me.advance ((0,1,2,3))
    ordered_end_trigger = True
    # discard bonus to encourage use of weak style
    def discard_penalty (self):
        return 0.5

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
    def blocks_movement(self, direct):
        if direct:
            return set()
        if self.opponent.position > self.me.position:
            return set(xrange(self.me.position))
        else:
            return set(xrange(self.me.position+1,7))
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
        if self.opponent.did_hit:
            self.me.advance (range(self.me.damage_taken+1))
    @property
    def ordered_before_trigger(self):
        return self.opponent.did_hit

class Unstoppable (Style):
    priority = -2
    # for now, I assume this requires the opponent to actually move
    def movement_reaction (self, mover, old_pos, direct):
        if mover == self.opponent and self.opponent.position != old_pos:
            self.me.advance ((1,2))
    # blocking hit/damage triggers handled by Cesar

class Fueled (Style):
    power = 1
    priority = -1
    preferred_range = 1
    def start_trigger (self):
        self.me.gain_threat_level()
    def before_trigger (self):
        self.me.advance ((1,2))
    ordered_before_trigger = True
    # You want to use it to skip low levels
    # (or 4, which is like skipping 0, but loses you the soak)
    def evaluation_bonus (self):
        return 0.25 * (1 - self.me.threat_level if self.me.threat_level < 4
                       else 0)

class Bulwark (Style):
    priority = -2
    stunguard = 3
    def blocks_pullpush (self):
        return set(xrange(7))
    def after_trigger (self):
        self.me.move ([0,1,2,3])
    ordered_after_trigger = True

class Inevitable (Style):
    stunguard = 3
    def hit_trigger (self):
        self.opponent.stun()
        self.me.power_penalty_next_beat = True
        self.me.evaluation_bonus += 1
    

#Clinhyde

class VitalSilverInfusion (Finisher):
    is_attack = False
    power = None
    priority = 4
    def after_trigger (self):
        self.me.gain_life (10)
        self.me.vital_silver_activated = True
        self.me.evaluation_bonus -= 4 # for losing life in the future

class RitherwhyteInfusion (Finisher):
    is_attack = False
    power = None
    priority = 4
    def after_trigger (self):
        self.me.ritherwhyte_activated = True
        self.me.evaluation_bonus += 4 # for losing life and gaining range

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
    def get_preferred_range (self):
        # we might activate a stim before start of beat.
        projected_stims = min (3, len(self.me.active_packs)+0.5)
        return projected_stims / 2.0
    def start_trigger (self):
        for i in xrange(len(self.me.active_packs)):
            self.me.advance([1])
    @property
    def ordered_start_trigger(self):
        return self.me.active_packs
    # doubling of stims handled by stims themselves
    def evaluation_bonus (self):
        # if we have 2 stims at end of beat, we can easily go to 3 next beat
        n = min (2, len(self.me.active_packs))
        return 0.3 * (n-2)

class Shock (Style):
    power = -1
    priority = 1
    preferred_range = 1.5 # this is wrong, but not sure how to fix it
    def before_trigger (self):
        self.me.move ([0,3,4,5])
    ordered_before_trigger = True
    def damage_trigger (self, damage):
        if self.game.distance() == 1:
            self.opponent.stun()
    ordered_damage_trigger = True    
    def evaluation_bonus (self):
        dist = self.game.distance()
        if dist >= 4 or (dist==3 and self.opponent.position not in (0,6)):
            return 1
        else:
            return -0.5

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
    def blocks_pullpush (self):
        if self.game.make_fork(2, self.me,
                               "Block opponent's attempt to move you?",
                               ["No", "Yes"]) == 0:
            return set() #don't block
        else:
            return set(xrange(7)) #block
    # ignoring own movement handled by self.execute_move

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
        return 2 if self.me.style.name=='Toxic' else 1
class Ehrlite (StimPack):
    def get_priority_bonus (self):
        return 2 if self.me.style.name=='Toxic' else 1
class Hylatine (StimPack):
    def get_stunguard (self):
        return 4 if self.me.style.name=='Toxic' else 2

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
        returned_modules = []
        for module in self.me.active_modules:
            if self.game.make_fork (2, self.me,
                            "Return %s to your module stack?" % module.name,
                                    ["No", "Yes"]):
                returned_modules.append(module)
        for module in returned_modules:
            self.me.return_module(module)

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
        me = self.me.position
        opp = self.opp.position
        # This doesn't really retreat as far as possible, when the
        # full retreat is blocked.
        max_retreat = me if me < opp else 6-me
        self.me.retreat ([max_retreat])
        self.me.push ([1])
    ordered_hit_trigger = True

class Leaping (Style):
    priority = 1
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
    # Clive.execute_move() keeps track of switching sides

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

class AfterBurner(Module):
    def after_trigger(self):
        self.me.retreat([1])
    ordered_after_trigger = True

class SynapseBoost (Module):
    priority = 1

class AutoRepair (Module):
    def end_trigger (self):
        self.me.gain_life(1)
        # TODO: return this to module stack

class ExtendingArms (Module):
    def before_trigger(self):
        self.me.pull([1])
    ordered_before_trigger = True

    
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

# NOT IMPLEMENTED
class Accelerando (Finisher):
    minrange = 1
    maxrange = 2
    power = 2
    priority = 4
    def reduce_stunguard (self, stunguard):
        return 0
    def before_trigger (self):
        pass
        # Advance as far as possible - needs a change in execute_move.
        # Can't just try decreasing amounts, because that might trigger
        # multiple movement reactions.
    ordered_before_trigger = True
    def hit_trigger (self):
        if self.me.can_spend (1):
            spend = self.game.make_fork (len(self.me.pool)+1, self.me, \
                                    "Choose number of Crescendo tokens to spend:")
            # not really added to ante, but only Bloodletting cares
            for i in range (spend):
                self.me.ante_token()
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
    power = 0
    priority = 8
    def hit_trigger (self):
        if self.me.can_spend (1):
            self.me.deathblow_spending = self.game.make_fork (
                len(self.me.pool)+1, self.me, \
                "Choose number of Crescendo tokens to spend:")
            for i in range (self.me.deathblow_spending):
                self.me.spend_token()
    def after_trigger(self):
        if self.me.did_hit:
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
        return 0.1 * (len(self.me.pool) - 3)

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
               3 * (self.opponent.style.printed_power +
                    self.opponent.base.printed_power)
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
        self.opponent_immobilized_next_beat = True
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
               (self.opponent.style.printed_power +
                self.opponent.base.printed_power)
    printed_power = None
    def get_soak (self):
        return self.me.ante.count(self.me.vengeance)
    
class Vengeful (Style):
    power = 1
    stunguard = 3
    preferred_range = 0.5
    def before_activating (self):
        self.me.advance([1])
    def hit_trigger (self):
        self.me.recover_tokens(2)
    def blocks_pullpush (self):
        return set(xrange(7))

class CounterStyle (Style):
    name_override = 'Counter'
    power = 1
    priority = -1
    def start_trigger (self):
        if self.me.base.name == self.opponent.base.name:
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
            for i in range (pull):
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
        if self.opponent.did_hit:
            self.me.move_directly ((self.me.position,
                                    self.opponent.position-1,
                                    self.opponent.position+1))
    @property
    def ordered_before_trigger(self):
        return self.opponent.did_hit
    # regaining tokens for soak handled by Eligor.soak_trigger()
            
class VengeanceT (Token):
    name_override = 'Vengeance'
    stunguard = 2


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
        self.me.evaluation_bonus += 7
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
    # blocking your own stun handled by Heketch.damage_trigger()
    def discard_penalty (self):
        return -0.5

class Merciless (Style):
    maxrange = 1
    power = -1
    preferred_range = 0.5
    def movement_reaction (self, mover, old_pos, direct):
        if mover == self.opponent and not direct and \
           (mover.position - self.me.position) * \
           (old_pos - self.me.position) < 0:
            self.opponent.lose_life (2)
            self.me.merciless_immobilized = True
            if self.game.reporting:
                self.game.report (self.opponent.name + " cannot move again this beat")
    def after_trigger (self):
        if len (self.me.pool) == 1:
            self.me.triggered_dodge = True
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
                self.me.assassin_immobilized_next_beat = True
                self.me.spend_token ()
                self.me.evaluation_bonus += 3.5
                if self.game.reporting:
                    self.game.report (self.opponent.name + " immobilized next beat")
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
        me = self.me
        her = self.opponent
        old_pos = me.position
        self.me.advance ((1,))
        if ordered (old_pos, her.position, me.position):
            me.add_triggered_power_bonus(1)
    ordered_start_trigger = True

class Sweeping (Style):
    power = -1
    priority = 3
    # Increasing incoming damage handled by Hikaru.take_damage()
    # positive penalty to encourage use
    def discard_penalty (self):
        return 0.5

class Geomantic (Style):
    power = 1
    def start_trigger (self):
        # fork to decide which token is anted (or none)
        n_options = len(self.me.pool) + 1
        prompt = "Select a token to ante with Geomantic:"
        options = [t.name for t in self.me.pool] + ["None"]
        token_number = self.game.make_fork (n_options, self.me, prompt, options)
        if token_number < len (self.me.pool):
            self.me.ante_token (self.me.pool[token_number])

class Earth (Token):
    soak = 3
    value = 1.2

class Fire (Token):
    power = 3
    value = 1

class Water (Token):
    minrange = -1
    maxrange = 1
    value = 0.8

class Wind (Token):
    priority = 2
    value = 0.6


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
        self.me.evaluation_bonus += 3
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
        self.me.infested_piles_on_reveal = self.me.infested_piles()
    def reduce_soak (self, soak):
        return 0 if self.me.infested_piles_on_reveal > 0 else soak
    def reduce_stunguard (self, stunguard):
        return 0 if self.me.infested_piles_on_reveal > 1 else stunguard
    def hit_trigger (self):
        self.me.give_insects(1)
    def evaluation_bonus (self):
        return 0.25 * (self.me.infested_piles() - 1)

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
        return Finisher.devolves_into_cancel (self) or not self.me.is_elemental
    # if you got here, you lose
    def end_trigger (self):
        raise WinException (self.opponent.my_number)
    def evaluate_setup (self):
        return 1 if self.me.is_elemental and self.game.distance() <=2 and \
               self.opponent.life <= 8 else 0

class ChainOfDestruction (Finisher):
    name_override = 'Chain of Destruction'
    minrange = 4
    maxrange = 6
    power = 4
    priority = 5
    # devolves to cancel if I'm elemental
    def devolves_into_cancel (self):
        return Finisher.devolves_into_cancel (self) or self.me.is_elemental
    def hit_trigger (self):
        if self.me.life >= 4:
            if self.game.make_fork (2, self.me,
                                    "Spend 3 life to repeat this attack?",
                                    ("No", "Yes")):
                self.me.lose_life (3)
                self.me.max_attacks += 1
    def evaluate_setup (self):
        if self.game.distance() >= 4:
            return (2.5 if self.me.life >= 7 else
                    (1.5 if self.me.life >= 4 else 0.5))
   
class Spellbolt (Base):
    minrange = 2
    maxrange = 6
    power = 2
    priority = 3
    preferred_range = 4
    def hit_trigger (self):
        if self.me.is_elemental:
            self.me.pull ((0,1,2))
        else:
            self.opponent.add_triggered_power_bonus(-2)
    @property
    def ordered_hit_trigger(self):
        return self.me.is_elemental

class Flare (Style):
    power = 3
    def reveal_trigger (self):
        if not self.me.is_elemental:
            self.me.lose_life (3)
            self.me.add_triggered_priority_bonus(3)
    def end_trigger (self):
        self.me.is_elemental = False
        
class Caustic (Style):
    power = 1
    priority = -1
    def hit_trigger (self):
        if self.me.is_elemental:
            self.opponent.stun()
    def get_soak (self):
        return 2 * (not self.me.is_elemental)

class Volcanic (Style):
    minrange = 2
    maxrange = 4
    preferred_range = 3
    def hit_trigger (self):
        if self.me.is_elemental:
            self.me.priority_penalty_next_beat = True
            self.me.evaluation_bonus += 0.7
    def end_trigger (self):
        if not self.me.is_elemental:
            self.me.move_to_unoccupied()
    @property
    def ordered_end_trigger(self):
        return not self.me.is_elemental
    
class Ignition (Style):
    power = 1
    priority = -1
    # Ignition.badness adds to value of being elemental,
    # because being elemental saves me another use of ignition
    badness = -1.0 
    def reveal_trigger (self):
        if self.me.is_elemental:
            self.me.lose_life (3)
            self.me.add_triggered_power_bonus(3)
    def end_trigger (self):
        self.me.is_elemental = True
    # if Kallistar is human and Ignition is in discard, that delays ignition
    def discard_penalty (self):
        return 0 if self.me.is_elemental else -2 * self.me.elemental_value()
        
class Blazing (Style):
    priority = 1
    def get_maxrange_bonus (self):
        return 1 * self.me.is_elemental
    def get_preferred_range (self):
        return 0.5 * self.me.is_elemental
    def after_trigger(self):
        if not self.me.is_elemental:
            self.me.move ((1,2))
    @property
    def ordered_after_trigger(self):
        return not self.me.is_elemental

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
    def standard_range (self):
        return False
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
    def before_trigger (self):
        self.me.jager_attacks = True
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
    def blocks_movement (self, direct):
        if self.me.position == self.me.jager_position:
            return set()
        else:
            return set([self.me.jager_position])
    # deals damage if opponent attempted to enter Jager's position
    def movement_reaction (self, mover, old_position, direct):
        if mover == self.opponent:
            if self.opponent.forced_block:
                if self.game.reporting:
                    self.game.report("%s is blocked by Jager"
                                     % self.opponent.name)
                self.me.deal_damage (2)
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
    def get_soak (self):
        return 2 if ordered(self.me.position,
                            self.me.jager_position,
                            self.opponent.position) else 0
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
    def start_trigger (self):
        # list 2 available styles to play
        available_styles = set (self.me.styles) - set([self.me.style]) \
                         - self.me.discard[1] - self.me.discard[2]
        available_styles = sorted (list(available_styles), \
                                   key=attrgetter('order'))
        # choose one, and add it to active styles
        # if the choice is Mutating, re-add the current form instead.
        prompt = "Select a style to overload:"
        options = [s.name for s in available_styles]
        choice = self.game.make_fork (len(available_styles), self.me, prompt,
                                      options)
        if self.game.reporting:
            self.game.report ("Kehrolyn overloads " + available_styles[choice].name)
        if available_styles[choice] == self.me.mutating:
            self.me.active_cards.append (self.me.current_form)
        else:
            self.me.active_cards.append (available_styles[choice])

# replaced at reveal, never does anything
class Mutating (Style):
    # doubles as Whip if Whip is current form
    def get_preferred_range (self):
        return (self.me.whip.preferred_range if self.me.whip in self.me.discard[1] else 0)
    def discard_penalty (self):
        return 0.5

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
    soak = 2
    # may block pull/push  - fork to decide whether to block
    def blocks_pullpush (self):
        if self.game.make_fork(2, self.me,
                               "Block opponent's attempt to move you?",
                               ["No", "Yes"]) == 0:
            return set() #don't block
        else:
            return set(xrange(7)) #block

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
    def reveal_trigger (self):
        if self.me.trap_position is not None and \
           abs(self.me.trap_position - self.me.opponent.position) <= 1:
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
        if self.me.base.is_attack and self.me.base.standard_range:
            ranges = range (self.me.get_minrange(), 1 + self.me.get_maxrange())
            positions = [pos for pos in xrange(7)\
                         if abs (pos - self.me.position) in ranges
                         and pos != self.opponent.position]
            if len (positions) > 0:
                self.me.move_trap (positions)
    ordered_start_trigger = True

class Lure (Style):
    maxrange = 5
    power = -1
    priority = -1
    preferred_range = 2.5
    def hit_trigger (self):
        self.me.pull (range(self.game.distance()))
    ordered_hit_trigger = True

#Lixis

class VirulentMiasma (Finisher):
    minrange = 1
    maxrange = 3
    power = 4
    priority = 5
    def hit_trigger (self):
        self.me.virulent_miasma = True
        self.me.evaluation_bonus += 6
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
    def blocks_movement (self, direct):
        pos = self.me.position
        return set ([pos+1, pos-1]) & set(xrange(7))

class Venomous (Style):
    power = 1
    def before_trigger (self):
        self.me.advance ([0,1])
    ordered_before_trigger = True
    def hit_trigger (self):
        self.me.priority_penalty_next_beat = True
        self.me.evaluation_bonus += 0.7

class Rooted (Style):
    minrange = -1
    power = 1
    priority = -2
    soak = 2
    # preferred_range is 0, reducing minrange isn't usually effective
    # may block pull/push  - fork to decide whether to block
    def blocks_pullpush (self):
        if self.game.make_fork(2, self.me,
                               "Block opponent's attempt to move you",
                               ["No", "Yes"]) == 0:
            return set() #don't block
        else:
            return set(xrange(7)) #block
    # optional negation of own movement implemented by always adding 0
    # to Lixis' move options (handled by Lixis.execute_move())

class Naturalizing (Style):
    maxrange = 1
    power = -1
    priority = 1
    preferred_range = 0.5
    def blocks_bonuses (self):
        return True
    def blocks_tokens (self):
        return True

class Vine (Style):
    maxrange = 2
    priority = -2
    stunguard = 3
    preferred_range = 1
    def hit_trigger (self):
        self.me.pull ([0,1,2])
    ordered_hit_trigger = True
    
class Pruning (Style):
    power = -1
    priority = -2
    def reveal_trigger (self):
        bases = self.count_bases()
        self.me.add_triggered_power_bonus(bases)
        self.me.add_triggered_priority_bonus(bases)
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
            for i in range (extra_attacks):
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
    def start_trigger (self):
        self.me.advance ([1])
    ordered_start_trigger = True
    def damage_trigger (self, damage):
        self.opponent.stun()

class Eternal (Style):
    priority = -4
    soak = 1
    def take_a_hit_trigger (self):
        if self.me.can_spend (1):
            self.me.eternal_spend = \
                self.game.make_fork (1+len(self.me.pool), self.me,
                                     "Spend how many tokens (+1 Soak each)?")
            for i in range (self.me.eternal_spend):
                self.me.spend_token()
            if self.me.eternal_spend and self.game.reporting:
                self.game.report ("Luc gains +%d soak (%d total)"
                                  %(self.me.eternal_spend,
                                    self.me.eternal_spend+1))
    def get_soak (self):
        return 1 + self.me.eternal_spend

class Memento (Style):
    priority = -1
    def after_trigger (self):
        if self.me.max_attacks == 1 and self.me.can_spend (2):
            if self.game.make_fork (2, self.me, \
                                    "Spend 2 Time tokens for an extra attack?",
                                    ["No", "Yes"]):
                for i in range (2):
                    self.me.spend_token ()
                self.me.max_attacks += 1
                if self.game.reporting:
                    self.game.report ("Luc attacks again")

class Fusion (Style):
    power = -1
    priority = 1
    def damage_trigger (self, damage):
        blow_out = self.me.get_destinations (self.opponent, (-damage,)) == set()
        self.me.push ([damage])
        # if moving opponent back is illegal and opponent doesn't block
        # (assumes oppoent either blocks all pushes or none)
        if blow_out and self.me.blocked == set():
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
            for i in range (advance):
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
        self.me.move(range(self.me.disparity + 1))
    ordered_before_trigger = True
    def after_trigger (self):
        self.me.move([1,2])
    ordered_after_trigger = True

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
    @property
    def maxrange (self):
        return self.me.disparity
    priority = 1
    preferred_range = 1.5
    def after_trigger (self):
        if self.me.base.is_attack and self.me.base.standard_range:
            minrange = self.me.get_minrange()
            maxrange = self.me.get_maxrange()
            pos = self.me.position
            dests = [d for d in xrange(7) if abs(d-pos) >= minrange and
                                             abs(d-pos) <= maxrange and
                                             d != self.me.position]
            if dests:
                self.me.move_directly(dests)
    @property
    def ordered_after_trigger(self):
        return self.me.base.is_attack and self.me.base.standard_range

class Reverie (Style):
    minrange = 1
    maxrange = 3
    power = 3
    priority = -1
    preferred_range = 2
    clash_priority = 0.1
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
    preferred_range = 0.5
    def get_stunguard(self):
        return 2 * (self.me.disparity >= 3)
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
    def evaluate(self):
        # Good against Dash
        if self.opponent.use_beta_bases:
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
        return 4
    def end_trigger (self):
        if self.opponent.did_hit:
            self.me.move ((0,1,2,3))
    @property
    def ordered_end_trigger(self):
        return self.opponent.did_hit
    
class Priestess (Style):
    power = -1
    priority = -1
    def hit_trigger (self):
        self.opponent.add_triggered_power_bonus (-2)
    def after_trigger (self):
        self.me.gain_life (1)

class Spiritual (Style):
    maxrange = 2
    power = 2
    priority = 1
    preferred_range = 1
    def before_trigger(self):
        if self.me.trance:
            self.me.trance -= 1
        else:
            self.opponent.triggered_dodge = True
    # Disabling level bonuses is handled by Magdelina.get_level_bonus() 
    def evaluation_bonus(self):
        return (0.3 * (1.8 - self.me.level)
                if self.me.trance
                else -1)
        
class Sanctimonious (Style):
    power = -1
    priority = -2
    def get_maxrange_bonus (self):
        return self.me.level
    def get_preferred_range (self):
        return self.me.level / 2.0
    def evaluation_bonus (self):
        return 0.3 * (self.me.level - 2)

class Excelsius (Style):
    maxrange = 1
    power = -1
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
        n = self.me.concentration
        self.me.discard_counters(n)
        self.me.add_triggered_power_bonus(2*n)
    def evaluate_setup (self):
        return 1 if self.me.concentration >= 3 and self.game.distance() in (2,3,4) \
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
        if self.me.concentration >= 1:
            self.me.counters_spent_by_base = \
                self.game.make_fork (1 + self.me.concentration, self.me,
                        "Spend how many counters? [Soak 1 per counter spent]")
            self.me.discard_counters(self.me.counters_spent_by_base)
            if self.game.reporting and self.me.counters_spent_by_base:
                self.game.report ("Marmelee gains Soak %d"
                                  %self.me.counters_spent_by_base)
    def get_soak (self):
        return self.me.counters_spent_by_base
    def end_trigger (self):
        self.me.recover_counters (1)
    # When pool is empty, Meditation/Sorceress order matters. 
    @property
    def ordered_end_trigger(self):
        return self.me.concentration == 0

class Petrifying (Style):
    power = 1
    priority = -1
    def start_trigger (self):
        if self != self.game.active and self.me.concentration >= 3:
            if (self.game.make_fork (2, self.me,
                    "Spend 3 Concentration counters to become Active Player?",
                                     ["No", "Yes"])):
                self.me.discard_counters(3)
                self.me.become_active_player()
    def hit_trigger (self):
        if self.me.concentration >= 2 and not self.opponent.has_stun_immunity():
            if (self.game.make_fork (2, self.me,
                                     "Spend 2 Concentration counters to stun %s?"%
                                     self.opponent.name,
                                     ["No", "Yes"])):
                self.me.discard_counters(2)
                self.opponent.stun()
    def evaluation_bonus (self):
        value = 0.2 if self.me.concentration >=3 else -0.3
        if self.me.concentration == 2 or self.me.concentration == 5:
            value += 0.1
        return value

class Magnificent (Style):
    minrange = 1
    maxrange = 2
    power = -1
    preferred_range = 1.5
    def hit_trigger (self):
        if self.me.concentration >= 1:
            spend = self.game.make_fork (1+self.me.concentration, self.me,
                                     "Spend how many counters (+1 Power each)?")
            self.me.discard_counters(spend)
            self.me.add_triggered_power_bonus(spend)
    def after_trigger (self):
        if self.me.concentration >= 2:
            if self.game.make_fork (2, self.me,
                                    "Spend 2 counters to move anywhere?",
                                    ["No","Yes"]):
                self.me.discard_counters(2)
                self.me.move_to_unoccupied ()
    @property
    def ordered_after_trigger(self):
        return self.me.concentration >= 2
    def evaluation_bonus(self):
        return 0.1 * (self.me.concentration - 2)

class Sorceress (Style):
    priority = -1
    preferred_range = 0.5 # 0-2 is 1, but it costs a token
    def before_trigger (self):
        if self.me.is_attacking() and self.me.concentration >= 1:
            self.me.counters_spent_by_style = \
                self.game.make_fork (2, self.me,
                    "Spend a Concentration counter for +0~2 range?",
                                     ["No", "Yes"])
            self.me.discard_counters(self.me.counters_spent_by_style)
            if self.me.counters_spent_by_style and self.game.reporting:
                self.game.report ("Marmelee gains +0~2 range")
    def end_trigger (self):
        if self.me.concentration >= 1:
            if (self.game.make_fork (2, self.me,
                    "Spend a Concentration counter to move 1 space?",
                                     ["No", "Yes"])):
                self.me.discard_counters(1)
                self.me.move ([1])
    # With counters, order matters vs. Parry.  
    # Without, it matters vs. Meditation.
    ordered_end_trigger = True
    def get_maxrange_bonus (self):
        return 2 * self.me.counters_spent_by_style
    def evaluation_bonus(self):
        n = min(2, self.me.concentration)
        return 0.1 * (n-1)

class Barrier (Style):
    maxrange = 1
    power = -1
    preferred_range = 0.5
    def start_trigger (self):
        if self.me.concentration >= 4 and self.game.make_fork (
                2, self.me,
                "Spend 4 Concentration counters to dodge all attacks?",
                                             ["No", "Yes"]):
            self.me.discard_counters(4)
            self.me.triggered_dodge = True
    def before_trigger (self):
        if self.me.concentration >= 1 and self.game.distance() == 1:
            if self.me.position > self.opponent.position:
                max_push = self.opponent.position
            else:
                max_push = 6 - self.opponent.position
            max_push = min (max_push, len(self.me.pool))
            spend = self.game.make_fork (1 + max_push, self.me,
                                     "Spend how many counters to push opponent?")
            self.me.discard_counters(spend)
            if spend:
                self.me.push([spend])
    @property
    def ordered_before_trigger(self):
        return self.me.concentration >= 1
    def evaluation_bonus (self):
        return 0.3 if self.me.concentration >=4 else -0.1

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
        if self.me.concentration >= 1:
            spend = self.game.make_fork (self.me.concentration + 1, self.me,
                "Spend how many counters? [opponent has -1 power per counter spent]")
            self.me.discard_counters(spend)
            self.opponent.add_triggered_power_bonus (-spend)
    def evaluation_bonus(self):
        return 0.05 * (self.me.concentration - 2)

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
            for i in range (spend):
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
        if not self.opponent.did_hit and self.me.discard_token():
            self.me.immutable_soak = 3
    def get_soak (self):
        return self.me.immutable_soak

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
            if self.me.is_user and self.game.interactive_mode:
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
        # +4 evaluation for making two bases unavailable for 2 beats
        self.me.evaluation_bonus += 4
        
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
    preferred_range = 0.75 # can be anything, but long ranges lose too much power
    def reveal_trigger (self):
        self.opponent.add_triggered_priority_bonus(-len(self.me.induced_pool))
    def before_trigger (self):
        if self.me.base.is_attack and self.me.base.deals_damage:
            old_pos = self.opponent.position
            power = self.me.get_power()
            self.me.pull(range(power+1))
            spaces_pulled = abs(self.opponent.position-old_pos)
            # If opponent switched sides, actual pull is one less then distance
            if ordered (old_pos, self.me.position, self.opponent.position):
                spaces_pulled -= 1
            self.me.add_triggered_power_bonus(-spaces_pulled)
    @property
    def ordered_before_trigger(self):
        return self.me.base.is_attack and self.me.base.deals_damage

class Curse (Token):
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
            for i in range (extra_attacks):
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
    def hit_trigger (self):
        self.me.push ((2,1,0))
    ordered_hit_trigger = True

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
    value = 0
    def reduce_soak (self, soak):
        return 0

class ExplosiveShell (Token):
    name_override = 'Explosive'
    value = 0.4
    power = 2

class FlashShell (Token):
    name_override = 'Flash'
    value = 0.1
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
        for artifact in self.me.deactivated_artifacts.copy():
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
        self.me.activation_fork()
    # Deactivation choice handled by Runika().take_a_hit_trigger

class Channeled (Style):
    power = 1
    priority = 1
    def start_trigger (self):
        self.me.activation_fork ()
    def end_trigger (self):
        # At end of beat, the only thing that matters is artifact value,
        # so might as well make it a fake fork.
        self.me.deactivation_fork (self.me, fake=True)
    # Forced deactivation choice handled by Runika.take_a_hit_trigger()
    
class Maintenance (Style):
    power = -1
    priority = -1
    def get_soak (self):
        return len(self.me.deactivated_artifacts)
    def after_trigger (self):
        self.me.retreat ([2,1])
    ordered_after_trigger = True

class Explosive (Style):
    power = -1
    preferred_range = 0.5
    def start_trigger (self):
        self.me.pull ([0,1])
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
    def blocks_pullpush (self):
        return set(xrange(7)) if self.me.overcharged_artifact else set()

class Battlefist (Artifact):
    value = 0.6
    def get_power_bonus (self):
        if self is self.me.overcharged_artifact:
            return 3
        return 1
    def reduce_soak(self, soak):
        return 0 if self is self.me.overcharged_artifact else soak
    

#Seth

# Not Implemented
class ReadingFate (Finisher):
    minrange = 1
    maxrange = 3
    priority = 6
    def hit_trigger (self):
        self.opponent.stun()
        self.me.evaluation_bonus += 8 # for reading styles
    def evaluate_setup (self):
        return 2 if self.game.distanc() <= 3 else 0

class FortuneBuster (Finisher):
    deals_damage = False
    minrange = 1
    maxrange = 6
    power = None
    priority = 13
    # Wins priority, even against other finishers (that get 0.2)
    clash_priority = 0.3
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
    # Reduces opponent's maximum range by 1.
    # Their minimum range is usually 1 anyway, so it's not really affected.
    preferred_range = 0.5 
    def give_minrange_penalty (self):
        return -1
    def give_maxrange_penalty (self):
        return -1

class Mimics (Style):
    name_override = "Mimic's"
    power = 1
    def mimics_movement (self):
        return True
    # move the same as opponent
    def movement_reaction (self, mover, old_position, direct):
        if mover == self.opponent and not direct:
            self.me.position += (self.opponent.position - old_position)
            if self.me.position > 6:
                self.me.position = 6
            if self.me.position < 0:
                self.me.position = 0

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
    # The built in retreat makes the dodge work from range 3
    def evaluation_bonus (self):
        return 0.25 if self.game.distance() >= 3 else -0.25

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
                self.me.evaluation_bonus -= 2 # for discarding extra base
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
    prioirity = 3
    def damage_trigger (self, damage):
        self.opponent.stun()
        # MISSING: opponent can't ante, except for mandatory.

class CoffinNails (Finisher):
    minrange = 1
    maxrange = 1
    power = 3
    prioirity = 3
    def damage_trigger (self, damage):
        self.opponent.stun()
        self.me.coffin_nails_hit = True
        # This should really depend on specific opponent for soak,
        # but ignoring stunguard is always very good
        self.me.evaluation_bonus += 5

class Brand (Base):
    minrange = 1
    maxrange = 2
    power = 3
    priority = 2
    preferred_range = 1.5
    def reduce_stunguard (self, stunguard):
        return 0 if self.me.get_priority() >= 6 else stunguard
    def after_trigger (self):
        if self.me.did_hit and self.me.can_spend(2):
            max_leech = min(4,len(self.me.pool)) / 2
            leech = self.game.make_fork (max_leech+1, self.me, \
                      "How much life to leech (spend 2 tokens per life point)?")
            for i in range (2*leech):
                self.me.spend_token()
            self.opponent.lose_life (leech)
            self.me.gain_life (leech)

class Unleashed (Style):
    maxrange = 1
    power = -1
    preferred_range = 0.5
    def after_trigger (self):
        self.me.retreat ((2,1))
    ordered_after_trigger = True
    def end_trigger (self):
        self.me.recover_tokens(2)
        self.me.unleashed_bonus_next_beat = True
        self.me.evaluation_bonus += 0.3

class Combination (Style):
    power = 2
    def reduce_soak (self, soak):
        return 0 if self.me.get_priority() >= 7 else soak
    def can_hit (self):
        return self.me.attack_range() < 3
    def hit_trigger (self):
        if self.me.did_hit_last_beat:
            self.me.add_triggered_power_bonus(2)
    def evaluation_bonus (self):
        return (0.4 if self.game.distance() <=2 else -0.4) + \
               (0.2 if self.me.did_hit else -0.2)
            
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
        old_pos = self.me.position
        self.me.advance ((0,1,2,3))
        spiral_move = abs(old_pos-self.me.position)
        # take one space off if jumped over opponent
        if ordered (old_pos, self.opponent.position, self.me.position):
            spiral_move -= 1
        self.me.add_triggered_power_bonus (-spiral_move)
    ordered_before_trigger = True
            
class Malice (Token):
    priority = 1

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

class BearArms (Finisher):
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
    # Effective maxrange is only 1 when I'm pushing opponent.
    def get_preferred_range (self):
        return (1 if self.me.zone_3 () and
                     not self.me.juto_position == self.opponent.position
                else 1.5)
    # move opponent 1 towards juto
    def start_trigger (self):
        if self.me.juto_position is not None:
            # juto before opponent, pull opponent
            if not self.me.zone_3 ():
                self.me.pull([1])
            # otherwise, if juto not with opponent, push opponent
            elif self.me.juto_position != self.opponent.position:
                self.me.push([1])
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
        return self.me.juto_position != None
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
        self.me.riptide_zone_2 = self.me.zone_2 ()
    @property
    def ordered_start_trigger(self):
        return self.me.juto_position is not None
    def can_be_hit (self):
        return not (self.me.riptide_zone_2 and
                    self.opponent.attack_range() >= 3)
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
        self.me.priority_penalty_next_beat = True
        self.me.evaluation_bonus += 1.3
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
    def hit_trigger (self):
        self.me.advance ((0,1,2,3,4,5))
    ordered_hit_trigger = True
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
    preferred_range = 0.5
    def blocks_movement (self, direct):
        # Can't retreat or move past me.
        if direct:
            return set()
        direction = (self.me.position - self.opponent.position)
        direction /= abs(direction)
        return set([self.me.position + direction,
                    self.opponent.position - direction])

class Glorious (Style):
    power = 2
    preferred_range = 0.5
    def before_trigger (self):
        self.me.advance ([1])
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
            return 0.3 + 0.1 * behind
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
    # zombie placement when Voco is moved by opponent
    def movement_reaction (self, mover, old_position, direct):
        if mover==self.me and self.me.position != old_position:
            if direct:
                self.me.add_zombies (set([old_position]))
            else:
                self.me.add_zombies ((pos_range (self.me.position,
                                                 old_position)
                                      - set([self.me.position,
                                             self.opponent.position])))
    # zombie placement on self-powered moves handled by Voco.execute_move

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
    def hit_trigger (self):
        self.me.pull ([1])
    ordered_hit_trigger = True

class Abyssal (Style):
    minrange = 2
    maxrange = 4
    preferred_range = 3
    def after_trigger (self):
        if self.me.base.is_attack and self.me.base.standard_range:
            ranges = range (self.me.get_minrange(), 1+self.me.get_maxrange())
            self.me.add_zombies (set([r for r in xrange(7)
                                    if abs(r-self.me.position) in ranges]))
    @property
    def ordered_after_trigger(self):
        return self.me.base.is_attack and self.me.base.standard_range

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
    def hit_trigger (self):
        self.me.push ((2,1,0))
    ordered_hit_trigger = True

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
    # may block pull/push  - fork to decide whether to block
    def blocks_pullpush (self):
        if self.game.make_fork(2, self.me,
                               "Block opponent's attempt to move you?",
                               ["No", "Yes"]) == 0:
            return set() #don't block
        else:
            return set(xrange(7)) #block
    # optional negation of own movement implemented by always adding 0
    # to own move options (handled by Zaamassal.execute_move())

class Warped (ZStyle):
    maxrange = 2
    # in corner, no retreat, so extra range is fully effective
    def get_preferred_range (self):
        return 1 if self.me.position in [0,6] else 0.5
    def start_trigger (self):
        self.me.retreat ([1])
    ordered_start_trigger = True

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
    def blocks_movement (self, direct):
        # if opponent adjacent to me, block everything
        if self.game.distance() == 1:
            return set(xrange(7))
        # otherwise, if move is direct, don't block
        if direct:
            return set()
        # if move is indirect, block switching sides
        # (once they go next to me, they can't continute moving).
        if self.me.position > self.opponent.position:
            return set ([self.me.position + 1])
        else:
            return set ([self.me.position - 1])
    # winning clashes is 0.5 priority, or 0.25 value
    def evaluate (self):
        return 1.0 if self.game.distance() == 1 else 0.25

class Resilience (Paradigm):
    shorthand = 'r'
    def start_trigger (self):
        self.resilience_soak = 2
    # if Resilience is replaced in the after trigger,
    # it's own after trigger can be avoided
    # (style and base triggers precede paradigm triggers)
    def after_trigger (self):
        if self in self.me.active_paradigms:
            self.resilience_soak = 0
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
character_dict = {'adjenna'  :Adjenna,
                  'alexian'  :Alexian,
                  'aria'     :Aria,
                  'byron'    :Byron,
                  'cadenza'  :Cadenza,
                  'cesar'    :Cesar,
                  'clinhyde' :Clinhyde,
                  'demitras' :Demitras,
                  'eligor'   :Eligor,
                  'heketch'  :Heketch,
                  'hikaru'   :Hikaru,
                  'kallistar':Kallistar,
                  'karin'    :Karin,
                  'kehrolyn' :Kehrolyn,
                  'kajia'    :Kajia,
                  'khadath'  :Khadath,
                  'lixis'    :Lixis,
                  'luc'      :Luc,
                  'lymn'     :Lymn,
                  'magdelina':Magdelina,
                  'marmelee' :Marmelee,
                  'mikhail'  :Mikhail,
                  'oriana'   :Oriana,
                  'rexan'    :Rexan,
                  'rukyuk'   :Rukyuk,
                  'runika'   :Runika,
                  'seth'     :Seth,
                  'shekhtur' :Shekhtur,
                  'tatsumi'  :Tatsumi,
                  'vanaah'   :Vanaah,
                  'voco'     :Voco,
                  'zaamassal':Zaamassal}

if __name__ == "__main__":
    main()
