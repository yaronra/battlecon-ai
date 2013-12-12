#!/usr/bin/python

import os
from operator import itemgetter
import csv
from numpy import array

phrases = {  'adjenna' : ['Basilisk Gaze'],
             'alexian' : ['Empire Divider', 'Hail The King'],
             'aria' : ['Laser Lattice'],
             'byron' : ['Soul Trap', 'Soul Gate'],
             'cadenza' : ['Rocket Press', 'Feedback Field'],
             'cesar' : ['Level 4 Protocol'],
             'clinhyde' : ['Vital Silver Infusion', 'Ritherwhyte Infusion'],
             'demitras' : ['Symphony of Demise'],
             'eligor' : ['Sweet Revenge', 'Sheet Lightning'],
             'heketch' : ['Million Knives', 'Living Nightmare'],
             'hikaru' : ['Wrath of Elements', 'Four Winds', 'Palm Strike'],
             'kajia' : ['Imago Emergence'],
             'kallistar' : ['Chain of Destruction'],
             'karin' : ['Red Moon Rage', 'Lunar Cross', 'Full Moon'],
             'kehrolyn' : ['Hydra Fork'],
             'khadath' : ['Dimensional Exile'],
             'lixis' : ['Virulent Miasma'],
             'luc' : ['Temporal Recursion'],
             'lymn' : [],
             'magdelina' : ['Solar Soul'],
             'marmelee' : ['Astral Cannon', 'Astral Trance'],
             'mikhail' : ['Magnus Malleus', 'The Fourth Seal'],
             'oriana' : ['Nihil Eraser', 'Galaxy Conduit'],
             'rexan' : ['Zero Hour'],
             'rukyuk' : ['Point Blank', 'Fully Automatic', 'Force Grenade'],
             'runika' : ['Artifice Avarice', 'Udstad Beam'],
             'seth' : ['Fortune Buster'],
             'shekhtur' : ['Coffin Nails'],
             'tatsumi' : ['Tsunamis Collide', 'Bear Arms'],
             'vanaah' : ['Death Walks', 'Hand of Divinity'],
             'voco' : ['The Wave'],
             'zaamassal' : ['Paradigm Shift', 'Open the Gate', 'Plane Divider']}

devastation = ['adjenna', 'alexian', 'aria', 'byron', 'cesar', 'clinhyde',
               'eligor', 'kajia', 'lymn', 'marmelee', 'mikhail', 'rexan',
               'runika', 'shekhtur', 'voco']

all_names = sorted(phrases.keys())

def list_files (logdir, name=None):
    fulldir = 'logs/' + logdir + '/'
    filenames = [filename for filename in os.listdir(fulldir)]
    return [fulldir+fn for fn in filenames
            if (name is None or name in fn) and os.path.isfile(fulldir+fn)]

def all_victories(logdir='main', devastation_only=False):
    names = devastation if devastation_only else all_names
    name_power = []
    for name in names:
        vics = victories(name, logdir, silent=True)
        if vics is not None:
            name_power.append((name, vics))
    name_power = sorted(name_power, key=itemgetter(1))
    for np in name_power:
        print "%s: %s" %(np[0], percentify(np[1]))

def check_redundancies():
    for name0 in all_names:
        for name1 in all_names:
            if name0 < name1:
                file01 = 'logs/main/'+name0+'_'+name1+'_log.txt'
                file10 = 'logs/main/'+name1+'_'+name0+'_log.txt'
                exist01 = os.path.isfile (file01)
                exist10 = os.path.isfile (file10)
                if exist01 and exist10:
                    print name0, name1
                if not exist01 and not exist10:
                    print "missing:", name0, name1

def analyze (name, logdir="main"):
    victories (name,logdir)
    print "Average game length: %.1f"% one_beat(name, logdir)
    strategies(name, logdir)
    hitting (name, logdir)

def strategies (name, logdir="main", beat=None,
                        condition='',reverse_condition=False):
    pair_dict = parse (name, logdir, beat, condition, reverse_condition)
    if name == 'seth':
        pair_dict = seth_consolidation (pair_dict)
    count, styles, bases = pair_count (pair_dict, True) 
    style_count = count.sum(axis=1)
    base_count = count.sum(axis=0)
    total = count.sum()
    print "total beats:", total
    print '--------------'
    sc = [(styles[i],style_count[i]) for i in range(len(styles))]
    sc = sorted (sc, key=itemgetter(1), reverse=True)
    for s in sc:
        print percentify(s[1]/float(total)), s[0]
    print '--------------'
    bc = [(bases[i],base_count[i]) for i in range(len(bases))]
    bc = sorted (bc, key=itemgetter(1), reverse=True)
    for b in bc:
        print percentify(b[1]/float(total)), b[0]
    print '--------------'
    count, styles, antes = style_ante_count (pair_dict, True)
    if len (antes) > 1:
        ante_count = count.sum(axis=0)
        ac = [(antes[i],ante_count[i]) for i in range(len(antes))]
        ac = sorted (ac, key=itemgetter(1), reverse=True)
        for a in ac:
            print percentify(a[1]/float(total)), a[0]
        print '--------------'
    for func in [pair_count, base_ante_count, style_ante_count]:
        count, first, second = func (pair_dict, False)
        if len (second) > 1:
            first_count = count.sum(axis=1)
            second_count = count.sum(axis=0)
            total = count.sum()
            for x in range(len(first)):
                for y in range(len(second)):
                    cxy = count[x][y]
                    print first[x], second[y], \
                          percentify(cxy/float(first_count[x])), \
                          percentify(cxy/float(second_count[y])),
                    ratio = float(cxy * total) / (first_count[x]*second_count[y])
                    if ratio > 1.5:
                        print "STRONG - %.2f" %ratio
                    elif ratio < 0.5:
                        print "WEAK - %.2f" %ratio
                    else:
                        print                                     
            print '-----------'

class HitRecord (object):
    def __init__(self, name):
        self.name = name
        self.beats = 0
        self.hits = 0
        self.misses = 0
        self.stunned = 0
        self.damage = 0
        self.opp_hits = 0
        self.opp_misses = 0
        self.opp_stunned = 0
        self.opp_damage = 0

def hitting (name, logdir="main", printing=True):
    replacement = {p : p.replace(' ','') for p in phrases[name]} 
    record = {}
    cap_name = name.capitalize()
    style = None
    for filename in list_files(logdir, name):
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if line.startswith('Beat '):
                style = None
            if style is None and line.startswith (name.capitalize()+": "):
                for p in replacement:
                    line = line.replace(p,replacement[p])
                style = line.split(' ')[1]
                if style not in record.keys():
                    record[style] = HitRecord(style)
                record[style].beats += 1
            if line.endswith(" is active\n"):
                me_active = line.startswith(cap_name)
            if line.endswith(" is stunned\n"):
                me_stunned = line.startswith(cap_name)
                if me_active and not me_stunned:
                    record[style].opp_stunned += 1
                if not me_active and me_stunned:
                    record[style].stunned += 1
            if line.endswith(" misses\n"):
                if line.startswith(cap_name):
                    record[style].misses += 1
                else:
                    record[style].opp_misses += 1
            if line.endswith(" hits\n"):
                if line.startswith(cap_name):
                    record[style].hits += 1
                else:
                    record[style].opp_hits += 1
            if " damage (now at " in line:
                if line.startswith(cap_name):
                    record[style].opp_damage += int(line.split(' ')[2])
                else:
                    record[style].damage += int(line.split(' ')[2])
    total = HitRecord("All")
    for key in total.__dict__.keys():
        if key != 'name':
            total.__dict__[key] = sum([rec.__dict__[key] for rec in record.itervalues()])
    all_recs = record.values() + [total]
    print "stunned/missed/hit/dam per hit/dam per beat"
    for rec in all_recs:
        if rec.name != 'Special':
            # a stunned beat is not necessarily an attack beat,
            # but dashes don't usually get stunned
            attack_beats = float(rec.stunned + rec.misses + rec.hits)
            opp_attack_beats = float(rec.opp_stunned + rec.opp_misses + rec.opp_hits)
            print "%.2f %s %s %s %s %.1f %.1f | %s %s %s %.1f %.1f" % \
                  (rec.damage/attack_beats/(rec.opp_damage/opp_attack_beats),
                   rec.name + (' ' * (12-len(rec.name))),
                   percentify(rec.stunned/attack_beats),
                   percentify(rec.misses/attack_beats),
                   percentify(rec.hits/attack_beats),
                   rec.damage/float(rec.hits),
                   rec.damage/attack_beats,
                   percentify(rec.opp_stunned/opp_attack_beats),
                   percentify(rec.opp_misses/opp_attack_beats),
                   percentify(rec.opp_hits/opp_attack_beats),
                   rec.opp_damage/float(rec.opp_hits),
                   rec.opp_damage/opp_attack_beats)

class HitWinCorrelationRecord (object):
    def __init__ (self):
        self.win_hits = [0] * 6
        self.tie_hits = [0] * 6
        self.lose_hits = [0] * 6
        
def style_win_correlation (name, logdir="main"):
    replacement = {p : p.replace(' ','') for p in phrases[name]} 
    record = {}
    cap_name = name.capitalize()
    upper_name = name.upper()
    style = None
    style_hits = {}
    for filename in list_files(logdir, name):
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if line.startswith('GAME '):
                style_hits = {}
            if line.startswith('Beat '):
                style = None
            if style is None and line.startswith (name.capitalize()+": "):
                for p in replacement:
                    line = line.replace(p,replacement[p])
                style = line.split(' ')[1]
                if style not in style_hits.iterkeys():
                    style_hits[style] = 0
            if line.endswith(" hits\n") and line.startswith(cap_name):
                style_hits[style] += 1
            win = line.endswith ('WINS\n')
            tie = line.endswith ('TIED!\n')
            if  win or tie:
                styles = style_hits.keys()
                for style in styles:
                    if style not in record.iterkeys():
                        record[style] = HitWinCorrelationRecord()
                if win:
                    if upper_name in line:
                        for style in styles:
                            record[style].win_hits[style_hits[style]] += 1
                    else:
                        for style in styles:
                            record[style].lose_hits[style_hits[style]] += 1
                else:
                    for style in styles:
                        record[style].tie_hits[style_hits[style]] += 1
    for style in record.iterkeys():
        print style
        rec = record[style]
        for hits in range(6):
            win = rec.win_hits[hits]
            lose = rec.lose_hits[hits]
            tie = rec.tie_hits[hits]
            total = float (win+lose+tie)
            if total > 20:
                print "%d: %s / %s (%d)" % (hits,
                                            percentify (win/total),
                                            percentify (lose/total),
                                            total)
##                print "%d: %d / %d (%d)" % (hits,
##                                            win,
##                                            lose,
##                                            tie)

def specials (name, logdir="main"):
    stats = {'all': [0]*16,
             'pulse': [0]*16,
             'cancel': [0]*16,
             'finisher': [0]*16}
    # count first turns to figure out number of games
    d = parse(name, logdir, 1)
    games = sum ([d[k] for k in d.keys()])
    for beat in range (1,16):
        d = parse (name, logdir, beat)
        for key in d.keys():
            if key[0] == 'Special':
                stats['all'][beat] += d[key]
                if key[1] == 'Pulse':
                    stats['pulse'][beat] += d[key]
                elif key[1] == 'Cancel':
                    stats['cancel'][beat] += d[key]
                else:
                    stats['finisher'][beat] += d[key]
    for stat in sorted(stats.keys()):
        s = stats[stat]
        total = 0
        print stat.upper(), percentify(sum(s)/float(games)) 
        for beat in range(1,16):
            total += s[beat]
            print "%d: %s - %s" %(beat, percentify(s[beat]/float(games)),
                                  percentify(total/float(games)))
        
def anomalies():
    for name in all_names:
        count, styles, bases = pair_count (parse (name), False)
        style_count = count.sum(axis=1)
        base_count = count.sum(axis=0)
        total = float(count.sum())
        print '----------'
        print name.capitalize()
        print '----------'
        for s in range(len(style_count)):
            fraction = style_count[s] / total
            if fraction >= .25 or fraction <= .12:
                print percentify(fraction), styles[s]
        for b in range(len(base_count)):
            fraction = base_count[b] / total
            if fraction >= .25 or fraction <= .08:
                print percentify(fraction), bases[b]
        for s in range(len(style_count)):
            for b in range(len(base_count)):
                sb = count[s][b]
                fraction = sb / float(style_count[s])
                if fraction >= 0.35 or fraction <= 0.035:
                    print "%s attached to %s %d of %d times (%s)" \
                          %(bases[b], styles[s], sb, style_count[s],
                            percentify (sb/float(style_count[s])))
                fraction = sb / float(base_count[b])
                if fraction >= 0.5 or fraction <= 0.05:
                    print "%s attached to %s %d of %d times (%s)" \
                          %(styles[s], bases[b], sb, base_count[b],
                            percentify (sb/float(base_count[b])))
        
def total_bases(specials=True, logdir="main"):
    base_dict = {"Strike" : 0,
                 "Shot" : 0,
                 "Drive" : 0,
                 "Burst" : 0,
                 "Grasp" : 0,
                 "Dash" : 0,
                 "Pulse" : 0,
                 "Cancel" : 0}
    total_total = 0
    for name in all_names:
        count, unused_styles, bases = pair_count (parse (name, logdir), specials)
        base_count = count.sum(axis=0)
        total = (count.sum())
        for b in range(len(bases)):
            if bases[b] in base_dict.keys():
                base_dict[bases[b]] += base_count[b]
        total_total += total
    sub_total = sum ([base_dict[k] for k in base_dict])
    base_dict['Unique'] = total_total - sub_total
    bases_counts = sorted(base_dict.items(), key=itemgetter(1),
                          reverse=True)
    for b in bases_counts:
        print percentify(b[1]/float(total_total)), b[0]

def one_beat (name, logdir="main"):
    return pair_count(parse(name,logdir),True)[0].sum() / \
           (10.0 * (len(list_files(logdir, name))))
    
def beats (logdir="main"):
    res = []
    for name in all_names:
        res.append ((name, one_beat(name, logdir),
                     victories(name,logdir,return_timeouts=True,silent=True)))
    res = sorted(res,key=itemgetter(2))
    print "mean game length: %.1f" %(sum([r[1] for r in res])/len(res))
    print "mean timeout percent: %s" %percentify((sum([r[2] for r in res])/len(res))) 
    for r in res:
        print "%s   %.1f   %s" %(r[0], r[1], percentify(r[2]))

def pair_count (strat_dict, specials):
    styles = sorted(list(set([p[0] for p in strat_dict.keys()
                       if p[0]!= 'Special' or specials])))
    bases = sorted(list(set([p[1] for p in strat_dict.keys()
                      if p[0]!= 'Special' or specials])))
    antes = sorted(list(set([p[2] for p in strat_dict.keys()
                      if p[0]!= 'Special' or specials])))
    count = array([[sum([strat_dict[(s,b,a)] if (s,b,a) in strat_dict.keys()
                                             else 0
                         for a in antes])
                    for b in bases]
                   for s in styles])
    return count, styles, bases

def style_ante_count (strat_dict, specials=False):
    styles = sorted(list(set([p[0] for p in strat_dict.keys()
                       if p[0]!= 'Special' or specials])))
    bases = sorted(list(set([p[1] for p in strat_dict.keys()
                      if p[0]!= 'Special' or specials])))
    antes = sorted(list(set([p[2] for p in strat_dict.keys()
                      if p[0]!= 'Special' or specials])))
    count = array([[sum([strat_dict[(s,b,a)] if (s,b,a) in strat_dict.keys()
                                             else 0
                         for b in bases])
                    for a in antes]
                   for s in styles])
    return count, styles, antes

def base_ante_count (strat_dict, specials=False):
    styles = sorted(list(set([p[0] for p in strat_dict.keys()
                       if p[0]!= 'Special' or specials])))
    bases = sorted(list(set([p[1] for p in strat_dict.keys()
                      if p[0]!= 'Special' or specials])))
    antes = sorted(list(set([p[2] for p in strat_dict.keys()
                      if p[0]!= 'Special' or specials])))
    count = array([[sum([strat_dict[(s,b,a)] if (s,b,a) in strat_dict.keys()
                                             else 0
                         for s in styles])
                    for a in antes]
                   for b in bases])
    return count, bases, antes

def victories (name, logdir="main", return_timeouts=False, silent=False,
               per_name=False):
    (win,lose,draw,time_win,time_lose) = (0,0,0,0,0)
    next_result_is_timeout = False
    for filename in list_files(logdir, name):
        (_win,_lose,_draw) = (0,0,0)
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if line.endswith ('WINS!\n'):
                if line.find(name.upper()) > -1:
                    _win += 1
                    if next_result_is_timeout:
                        time_win += 1
                else:
                    _lose += 1
                    if next_result_is_timeout:
                        time_lose += 1
                next_result_is_timeout = False
            if line.endswith ('TIED!\n'):
                _draw += 1
                next_result_is_timeout = False
            if line.endswith ('Game goes to time\n'):
                next_result_is_timeout = True
        if per_name:
            names = filename.split('/')[-1].split('_')
            other_name = names[0] if names[0] != name else names[1]
            print other_name, _win + _draw/2.0
        win += _win
        lose += _lose
        draw += _draw

    games = float(win+lose+draw)
    if games == 0:
        return None
    if not silent:
        print "power:", percentify ((win+draw/2.0)/games)
        print "win: %d (%s) - %s by time out" %(win, percentify(win/games),
                                                percentify(time_win/float(win)))
        print "lose: %d (%s) - %s by time out" %(lose, percentify(lose/games),
                                                 percentify(time_lose/float(lose)))
        print "draw: %d (%s)" %(draw, percentify(draw/games))
        print "total: %s by timeout" %percentify((time_win+time_lose)/games)

    if return_timeouts:
        return (time_win+time_lose)/games
    else:
        return (win+draw/2.0)/games

def victory_csv (logdir="main"):
    index = {name:i for (i,name) in enumerate(all_names)}
    wins = [[0 for i in all_names] for j in all_names]
    logdir = "logs/" + logdir + "/"
    filenames = os.listdir (logdir)
    for filename in filenames:
        split = filename.split('_')
        i = index [split[0]]
        j = index [split[1]]
        with open (logdir+filename) as f:
            log = [line for line in f]
        for line in log:
            if line.endswith ('WINS\n'):
                if line.find(split[0].upper()) > -1:
                    wins[i][j] += 1
                else:
                    wins[j][i] += 1
            if line.endswith ('TIED!\n'):
                wins[i][j] += 0.5
                wins[j][i] += 0.5
    for i in xrange(len(wins)):
        for j in xrange(i):
            if wins[i][j]+wins[j][i] != 10:
                print all_names[i], all_names[j], wins[i][j], wins[j][i]
    # sort names by total wins
    sums = [sum(row) for row in wins]
    names_sums = sorted (zip (all_names, sums), key=itemgetter(1),
                         reverse=True)
    names = [ns[0].capitalize() for ns in names_sums]
    # sort rows by total wins
    wins = sorted(wins, key=lambda x: sum(x), reverse=True)
    # sort columns by total wins
    wins = zip(*wins)
    wins = sorted(wins, key=lambda x: sum(x))
    wins = zip(*wins)
    for i in xrange(len(wins)):
        wins[i] = list(wins[i])
        wins[i][i]=''
    csv.register_dialect('my_excel', delimiter=',', lineterminator='\n')
    with open ('victories.csv','w') as csvfile:
        writer = csv.writer (csvfile,dialect='my_excel')
        writer.writerow(['']+names)
        for (i,name) in enumerate(names):
            writer.writerow([name]+wins[i])

def first_beats_percents (name, logdir="first beats"):
    solution_percents (name, logdir, 1)
    print '----------'
    solution_percents (name, logdir, [1,2])

def solution_percents (name, logdir="first beats", beat=None):
    if isinstance (beat, int):
        beat = [beat]
    style_dict = {}
    base_dict = {}
    n_beats = 0
    replacement = {p : p.replace(' ','') for p in phrases[name]}
    for filename in list_files(logdir, name):
        with open (filename) as f:
            log = [line for line in f]
        i=0
        while True:
            # forward to start of beat
            while i < len (log) and not log[i].startswith("Beat"):
                i += 1
            if i == len (log):
                break
            current_beat = int(log[i].split(' ')[1])
            # look for first strategy percentage breakdown
            while not log[i] == name.capitalize() + ":\n":
                i += 1
            i += 1
            if beat==None or current_beat in beat:
                n_beats += 1
                while '%' in log[i]:
                    line = log[i][:-1] # snip newline
                    for p in replacement:
                        line = line.replace(p,replacement[p])
                    split = line.split(' ')
                    percent = int (split[0][:-1])
                    if split[1] in style_dict.keys():
                        style_dict[split[1]] += percent
                    else:
                        style_dict[split[1]] = percent
                    if split[2] in base_dict.keys():
                        base_dict[split[2]] += percent
                    else:
                        base_dict[split[2]] = percent
                    i += 1
    bases = sorted ([b for b in base_dict.items()
                     if b[0] not in ['Cancel','Pulse']],
                    key=itemgetter(1), reverse=True)
    for b in bases:
        print b[0], percentify(b[1]/100.0/n_beats)
    if len(bases) < 7:
        print 7-len(bases), "MISSING"
    print '----------------'
    styles = sorted ([s for s in style_dict.items()if s[0] != 'Special'],
                     key=itemgetter(1), reverse=True)
    for s in styles:
        print s[0], percentify(s[1]/100.0/n_beats)
    if len(styles) < 5:
        print 5-len(styles), "MISSING"

def string_is_float (s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def unbeatable_strategies (name, thresh=5, life_thresh=8,
                           logdir="main", outdir="logs/unbeatables/"):
    winfile = open (outdir+'/'+name+'_win_by_%d.txt'%thresh,'w')
    total = 0
    for filename in list_files(logdir, name):
        names = filename.split('/')[-1].split('_')
        count = 0
        with open (filename) as f:
            log = [line for line in f]
        i=0
        while True:
            # search for unbeatable strategy reports
            while i < len (log) and \
                  not log[i].startswith("Unbeatable strategy for %s"
                                        % name.capitalize()):
                i += 1
            if i == len (log):
                break
            if len(log[i].split(' ')) < 6:
                    i += 1
            diffs = []
            first_unbeatable_line = i
            while string_is_float(log[i][:-1].split(' ')[-1]):
                    diffs.append (abs(float(log[i][:-1].split(' ')[-1])))
                    i+=1
            if max(diffs) >= thresh:
                # go back to start of beat
                start = i
                while not log[start].startswith('Beat '):
                    start -= 1
                # find opponent's life total
                j = start
                while True:
                    j += 1
                    if log[j].startswith('life: ') and \
                       not log[j-2].startswith(name.capitalize()):
                        life = int(log[j].split()[-1])
                        break
                if life >= life_thresh:
                    for j in range(start, first_unbeatable_line):
                        winfile.write (log[j])
                    for j in range(len (diffs)):
                        if diffs[j] >= thresh:
                            winfile.write (log[first_unbeatable_line+j])
                    winfile.write("###############################\n")
                    count += 1
        print count, "vs.", \
              (names[0] if names[0].lower()!=name else names[1])
        total += count
    winfile.close()
    print "total:", total


def parse (name, logdir="main", beat=None, condition='',
           reverse_condition=False):
    if isinstance (beat, int):
        beat = [beat]
    replacement = {p : p.replace(' ','') for p in phrases[name]} 
    strat_dict = {}
    for filename in list_files(logdir, name):
        with open (filename) as f:
            log = [line for line in f]
        i=0
        while True:
            # forward to start of beat
            while i < len (log) and not log[i].startswith("Beat"):
                i += 1
            if i == len (log):
                break
            condition_active = reverse_condition
            current_beat = int(log[i].split(' ')[1])
            # advance to first pair used, looking for condition along way
            while not log[i].startswith(name.capitalize() + ": "):
                if condition in log[i]:
                    condition_active = not reverse_condition
                i += 1
            # look at pair
            if (beat==None or current_beat in beat) and condition_active:
                line = log[i][:-1] # snip newline
                # remove antes of opponent's token (rexan/alexian)
                line = line.split(' | ')[0]
                for p in replacement:
                    line = line.replace(p,replacement[p])
                split = line.split(' ')
                strat = (split[1],split[2],' '.join(split[3:]))
                if strat in strat_dict.keys():
                    strat_dict[strat] +=1
                else:
                    strat_dict[strat] = 1
    return strat_dict

def text (text, name=None, logdir="main", show=False):
    total = 0
    for filename in list_files(logdir, name):
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if line.find(text) != -1:
                total += 1
                if show:
                    print filename, line
    return total
  
def percentify (fraction):
    if fraction > 0.05:
        return str (int (fraction*100+.5)) + '%'
    else:
        return str(int(fraction*1000+.5)/10.0)+'%'

def adjenna_marker_beat (target=1, logdir="main"):
    # 1-15 is beat in which opponent got to target number of markers
    # 0 means never
    gain_beat = [0 for _ in range (16)]
    victories = 0
    petrifications = 0
    victories_with_target = 0
    for filename in list_files(logdir, 'adjenna'):
        with open (filename) as f:
            log = [line for line in f]
        markers = None # indication that first game wasn't started
        for line in log:
            if line.startswith('ADJENNA WINS'):
                victories += 1
                if markers >= target:
                    victories_with_target +=1
            if line.endswith ('has 6 Petrification Markers\n'):
                petrifications += 1
            # mew game
            if line.startswith('GAME ') and \
               not line.endswith('TIED!\n'):
                if markers is None:
                    markers = 0
                elif markers < target:
                    gain_beat[0] += 1
                markers = 0
            elif line.startswith('Beat '):
                beat = int(line.split(' ')[1])
            else:
                if line.endswith('receives a Petrification Marker\n'):
                    markers += 1
                    if markers == target:
                        gain_beat [beat] += 1
        if markers < target:
            gain_beat[0] += 1
    print "total games:", sum(gain_beat)
    print "victories:", victories
    print "petrifications:", petrifications
    games_with_target = sum(gain_beat) - gain_beat[0]
    print "victories with at least %d markers: %d/%d (%s)"\
          %(target, victories_with_target, games_with_target,
            percentify (victories_with_target/float(games_with_target)))
    for beat,count in enumerate(gain_beat):
        print beat,count
                        
def adjenna_marker_and_wins (target_beat, logdir="main"):
    games_with_x_markers = [0,0,0,0,0,0,0]
    victories_with_x_markers = [0,0,0,0,0,0,0]
    for filename in list_files(logdir, 'adjenna'):
        with open (filename) as f:
            log = [line for line in f]
        # number of markers at target beat
        markers = None # indication that first game wasn't started
        for line in log:
            # new game
            if line.startswith('GAME ') and \
               not line.endswith('TIED!\n'):
                if markers is None:
                    markers = 0
                markers = 0
                beat = 0
            elif line.startswith('Beat '):
                beat = int(line.split(' ')[1])
            elif beat <= target_beat:
                if line.endswith('receives a Petrification Marker\n'):
                    markers += 1
            if line.startswith('ADJENNA WINS'):
                victories_with_x_markers[markers] += 1
                games_with_x_markers[markers] += 1
            elif line.find('WINS') > -1 or line.find('TIED') > -1:
                games_with_x_markers[markers] += 1
    total = sum (games_with_x_markers)
    print "total games:", total
    print "or more:"
    for i in range(7):
        print i, percentify (sum(games_with_x_markers[i:])/float(total)), \
              percentify (float(sum(victories_with_x_markers[i:])) /\
                          (0.000001+sum(games_with_x_markers[i:])))
    print "or less:"
    for i in range(7):
        print i, percentify (sum(games_with_x_markers[:i+1])/float(total)), \
              percentify (float(sum(victories_with_x_markers[:i+1])) /\
                          (0.000001+sum(games_with_x_markers[:i+1])))

def adjenna_by_markers (logdir='main'):
    for markers in range (6):
        print '#############################'
        print markers, "MARKERS"
        print '---------'
        strategies ('adjenna', logdir,
                            condition='%d Petrification markers on'%markers)
        print '#############################'
        
def alexian_tokens (logdir="main"):
    tokens = [0,0,0,0]
    antes = [0,0,0,0]
    post_ante = [0,0,0,0]
    beat_sums = [0]*16
    beat_counts = [0]*16
    current_ante_count = None
    filenames = list_files(logdir, 'alexian')
    for filename in filenames:
        with open (filename) as f:
            log = [line for line in f]
        for i,line in enumerate(log):
            if line.startswith ("Beat "):
                current_beat = int(line[5:-1])
            # look for consecutive ante lines
            # (a second batch can be a repeat due to cancel)
            if current_ante_count == 0 and \
               line.endswith (" antes a Chivalry token\n"):
                j = 1
                while log[i+j].endswith (" antes a Chivalry token\n"):
                    j += 1
                current_ante_count = j
            if line.startswith("Opponent has ") and \
               line.endswith("Chivalry tokens\n"):
                current_tokens = int(line[13])
                if current_ante_count is not None:
                    antes[current_ante_count] += 1
                    post_ante[current_tokens - current_ante_count] += 1
                current_ante_count = 0
                tokens[current_tokens] += 1
                beat_sums[current_beat] += 1
                beat_counts[current_beat] += current_tokens
    antes[current_ante_count] += 1
    post_ante[current_tokens - current_ante_count] += 1
    print "Chivalry tokens at start of beat:"
    for i, count in enumerate(tokens):
        print i, percentify(count/float(sum(tokens)))
    print "total:", sum(tokens)

    print "Anted chivalry tokens"
    for i, count in enumerate(antes):
        print i, percentify(count/float(sum(antes)))
    print "total:", sum(antes)

    print "Un-anted chivalry tokens"
    for i, count in enumerate(post_ante):
        print i, percentify(count/float(sum(antes)))
    print "total:", sum(post_ante)

    print "Chivalry token accumulation by beat"
    for i in range(1,16):
        print "%d: %.1f" %(i, beat_counts[i]/float(beat_sums[i]))

    print
    parse_base_vs_induced_ante('alexian', logdir)

def aria_droids (logdir="main"):
    n_beats = 0.0
    # last slot is for 'None'
    dampening = [0] * 8
    magnetron = [0] * 8
    turret = [0] * 8
    beat = None
    for filename in list_files(logdir, 'aria'):
        with open (filename) as f:
            log = [line for line in f]
        # figure out what side aria starts on in this matchup
        left = 'aria' in filename.split('_')[0]
        for i, line in enumerate(log):
            if line.startswith ("Beat "):
                beat = int(line.split()[-1])
                n_beats += 1
            # board will alway have '..' in it
            elif beat is not None and '..' in line:
                beat = None # stop looking for droids this beat
                d = log[i+1].find('d')
                if d == -1:
                    d = 7
                else:
                    d = d if left else 6-d
                dampening[d] += 1 
                m = log[i+2].find('m')
                if m == -1:
                    m = 7
                else:
                    m = m if left else 6-m
                magnetron[m] += 1 
                t = log[i+3].find('t')
                if t == -1:
                    t = 7
                else:
                    t = t if left else 6-t
                turret[t] += 1 
    print "dampening: %s" % percentify(sum(dampening[:-1])/n_beats)
    for i in range(7):
        print percentify(dampening[i]/n_beats),
    print
    print "magnetron: %s" % percentify(sum(magnetron[:-1])/n_beats)
    for i in range(7):
        print percentify(magnetron[i]/n_beats),
    print
    print "turret: %s" % percentify(sum(turret[:-1])/n_beats)
    for i in range(7):
        print percentify(turret[i]/n_beats),
    print

def byron_by_emblems (logdir='main'):
    for emblems in range (6):
        print '#############################'
        print emblems, "Mask Emblems"
        print '---------'
        strategies ('byron', logdir,
                            condition='%d Mask emblems'%emblems)
        print '#############################'
        
def cesar_cycle (logdir='main'):
    for threat_level in range (5):
        print '#############################'
        print "THREAT LEVEL", threat_level
        print '---------'
        before_ante = (threat_level - 1)%5
        strategies ('cesar', logdir,
                            condition='Threat level: %d'%before_ante)
        print '#############################'
        print "This refers to threat level AFTER ANTE."
    
    styles = ["Unstoppable", "Bulwark", "Phalanx", "Inevitable", "Fueled",
              "Special"]
    bases = ['Burst','Dash','Drive','Grasp','Shot','Strike', 'Suppression',
             'Pulse','Cancel', 'Level4Protocol']

    style_dict = {style: [0]*5 for style in styles}
    base_dict = {base: [0]*5 for base in bases}
    replacement = {p : p.replace(' ','') for p in phrases['cesar']} 
    for filename in list_files(logdir, 'cesar'):
        with open (filename) as f:
            log = [line for line in f]
        i=0
        while True:
            # forward to start of beat, threat level
            while i < len (log) and not log[i].startswith("Threat level: "):
                i += 1
            if i == len (log):
                break
            threat_level = int(log[i].split(' ')[2])
            after_ante = (threat_level + 1)%5
            # advance to first pair used
            while not log[i].startswith("Cesar: "):
                i += 1
            line = log[i][:-1] # snip newline
            # look at pair
            for p in replacement:
                line = line.replace(p,replacement[p])
            split = line.split(' ')
            style, base = split[1], split[2]
            style_dict[style][after_ante] += 1
            base_dict[base][after_ante] += 1
    threat_sums = [float(sum([style_dict[style][i] for style in styles]))
                   for i in xrange(5)]
    threat_sums2 = [float(sum([base_dict[base][i] for base in bases]))
                    for i in xrange(5)]
    assert threat_sums == threat_sums2
    for style in style_dict:
        print percentify(sum(style_dict[style])/sum(threat_sums)), style,
        for i in xrange(5):
            print percentify(style_dict[style][i]/threat_sums[i]),
        print
    print
    for base in base_dict:
        print percentify(sum(base_dict[base])/sum(threat_sums)), base,
        for i in xrange(5):
            print percentify(base_dict[base][i]/threat_sums[i]),
        print
    print
    
        
    
def clinhyde_stims (logdir="main"):
    stims = {}
    n_stims = [0,0,0,0]
    for filename in list_files(logdir, 'clinhyde'):
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if line.startswith('Active Stim Packs:'):
                s = line.strip().split(' ')
                n_stims[len(s)-3] += 1
                for p in s[3:]:
                    p = p.strip(',\n')
                    if p in stims.keys():
                        stims[p] += 1
                    else:
                        stims[p] = 1
    total = float(sum(n_stims))
    print "Number of Stims:"
    for i,n in enumerate (n_stims):
        print i, percentify(n/total)
    for p in stims.keys():
        print p, percentify (stims[p]/total)
                

def eligor_tokens (logdir="main"):
    tokens = [0,0,0,0,0,0]
    for filename in list_files(logdir, 'eligor'):
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if line[2:] == "Vengeance tokens\n":
                tokens[int(line[0])] += 1
    for i, count in enumerate(tokens):
        print i, percentify(count/float(sum(tokens)))
    print "total:", sum(tokens)

def hikaru_tokens (logdir="main"):
    token_dict = {'Earth':0,
                  'Fire':0,
                  'Water':0,
                  'Wind':0}
    beats = 0
    for filename in list_files(logdir, 'hikaru'):
        with open (filename) as f:
            log = [line for line in f]
        hikaru_reporting = False
        previous_line = ''
        for line in log:
            if line.startswith('---'):
                hikaru_reporting = (previous_line == 'Hikaru\n')
            elif hikaru_reporting and line.startswith ('pool:'):
                tokens = line[6:-1].split(', ')
                for t in tokens:
                    if t!='':
                        token_dict[t] += 1
                beats += 1
            previous_line = line
    for token, count in token_dict.items():
        print token, percentify(count/float(beats))
    print "total:", sum([token_dict[t] for t in token_dict])/float(beats)

def kajia_insects(logdir='main'):
    insect_count = [[0]*10 for _ in (0,1,2)]
    totals_count = [0] * 10
    piles_count = [0] * 3
    for filename in list_files(logdir, 'kajia'):
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if " insects on opponent's discard " in line:
                discard = int(line[-2])
                insects = int(line[0])
                insect_count[discard][insects] += 1
                if discard == 1:
                    discard1 = insects
                else:
                    total = discard1 + insects
                    n_piles = (discard1 > 0) + (insects > 0)
                    totals_count[total] += 1
                    piles_count[n_piles] += 1
    print "Total insects:"
    beats = float(sum(totals_count))
    for i in xrange(10):
        if totals_count[i]:
            print "%d: %s" % (i, percentify(totals_count[i]/beats))
    print "Discard piles with insects:"
    assert beats == float(sum(piles_count))
    for i in xrange(3):
        print "%d: %s" % (i, percentify(piles_count[i]/beats))
    print "Discard 1 insects:"
    assert beats == float(sum(insect_count[1]))
    for i in xrange(10):
        if insect_count[1][i]:
            print "%d: %s" % (i, percentify(insect_count[1][i]/beats))
    print "Discard 2 insects:"
    assert beats == float(sum(insect_count[2]))
    for i in xrange(10):
        if insect_count[2][i]:
            print "%d: %s" % (i, percentify(insect_count[2][i]/beats))

def kallistar_elemental(logdir="main"):
    print "ELEMENTAL"
    print "---------"
    strategies ('kallistar', logdir, condition="Elemental Form")
    print "###########################################"
    print "HUMAN"
    print "-----"
    strategies ('kallistar', logdir, condition="Human Form")

def karin_jager (logdir="main"):
    karin_dists = [0,0,0,0,0,0,0]
    opp_dists = [0,0,0,0,0,0,0]
    for filename in list_files(logdir, 'karin'):
        names = filename.split('/')[-1].split('_')
        initial0 = names[0][0].upper()
        initial1 = names[1][0].upper()
        if initial0 == initial1:
            initial1 = initial1.lower()
        if names[0] == 'karin':
            karin_init, opp_init = initial0, initial1
        else:
            karin_init, opp_init = initial1, initial0
        with open (filename) as f:
            log = [line for line in f]
        i=0
        while True:
            # forward to start of beat
            while i < len (log) and not log[i].startswith("Beat"):
                i += 1
            if i == len (log):
                break
            # look for board
            while sum([c=='.' for c in log[i]]) != 5:
                i += 1
            karin_pos = log[i].find(karin_init)
            opp_pos = log[i].find(opp_init)
            j = i + 1
            while True:
                jager_pos = log[j].find('j')
                if jager_pos > -1:
                    break
                j += 1
            karin_dists [abs(karin_pos-jager_pos)] += 1
            opp_dists [abs(opp_pos-jager_pos)] += 1
    total = float(sum(karin_dists))
    print "Karin-Jager distances:"
    for i,d in enumerate(karin_dists):
        print "%d: %s" %(i, percentify(d/total))
    print "Opponent-Jager distances:"
    for i,d in enumerate(opp_dists):
        print "%d: %s" %(i, percentify(d/total))

def luc_tokens (logdir="main"):
    tokens = [0,0,0,0,0,0]
    for filename in list_files(logdir, 'luc'):
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if line[2:] == "Time tokens\n":
                tokens[int(line[0])] += 1
    for i, count in enumerate(tokens):
        print i, percentify(count/float(sum(tokens)))
    print "total:", sum(tokens)

def lymn_disparity (logdir='main'):
    cards = ["Surreal", "Reverie", "Megrim", "Conceit", "Chimeric",
              "Maddening", "Fathomless", "Visions"]
    disparities = {card: [0]*20 for card in cards}
    replacement = {p : p.replace(' ','') for p in phrases['lymn']} 
    for filename in list_files(logdir, 'lymn'):
        with open (filename) as f:
            log = [line for line in f]
        i=0
        while True:
            # forward to start of beat
            while i < len (log) and not log[i].startswith("Beat "):
                i += 1
            if i == len (log):
                break
            # advance to first pair used
            while not log[i].startswith("Lymn: "):
                i += 1
            line = log[i][:-1] # snip newline
            # look at pair
            for p in replacement:
                line = line.replace(p,replacement[p])
            split = line.split(' ')
            style, base = split[1], split[2]
            # advance to disparity
            while not log[i].startswith("Disparity is "):
                i += 1
            disparity = int(log[i][13:-1])
            for card in (style,base):
                if card in cards:
                    disparities[card][disparity] += 1
    for card in sorted(cards):
        total = float(sum(disparities[card]))
        print card
        for i in range(20):
            if disparities[card][i]:
                print i, percentify (disparities[card][i]/total)

def marmelee_tokens (logdir="main"):
    tokens = [0,0,0,0,0,0]
    for filename in list_files(logdir, 'marmelee'):
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if line[2:] == "Concentration tokens\n":
                tokens[int(line[0])] += 1
    for i, count in enumerate(tokens):
        print i, percentify(count/float(sum(tokens)))
    print "total:", sum(tokens)

def marmelee_spending (logdir="main"):
    style_spend_dict = {}
    for filename in list_files(logdir, 'marmelee'):
        names = filename.split('/')[-1].split('_')
        with open (filename) as f:
            log = [line for line in f]
        style = None
        spend = 0
        for i in range(len(log)):
            line = log[i][:-1]
            if line.startswith("Marmelee: "):
                split = line.split(' ')
                style = split[1]
                if split[2]=='Meditation':
                    style += ' Meditation'
                if split[2]=='Dash':
                    style += ' Dash'
            if style is not None and line.startswith('Beat '):
                key = (style, spend)
                if spend >=6:
                    name = [n for n in names if n!='Marmelee'][0]
                    print "spent 6 tokens against %s: line %d" %(name, i)
                if key in style_spend_dict.keys():
                    style_spend_dict[key] += 1
                else:
                    style_spend_dict[key] = 1
                spend = 0
            if line.startswith('Marmelee spends a Concentration token'):
                spend += 1
                if log[i+1].startswith ('Marmelee moves:'):
                    spend += .1 # additional +.1 marks spending for move
            if line.startswith('Marmelee gains Soak'):
                spend += .01 * int (line[-1])
            if line.startswith('Marmelee becomes the active player'):
                spend += .001
        key = (style, spend)
        if key in style_spend_dict.keys():
            style_spend_dict[key] += 1
        else:
            style_spend_dict[key] = 1
    for key in sorted (style_spend_dict.keys()):
        total = 0
        for otherkey in style_spend_dict.keys():
            if otherkey[0]==key[0]:
                total += style_spend_dict[otherkey]
        print key, style_spend_dict[key], \
              percentify(style_spend_dict[key]/float(total))

def mikhail_pairs_with_tokens (logdir="main"):
    d = parse('mikhail', logdir)
    keys = d.keys()
    for k in keys:
        if k[2]=='':
            del d[k]
    count, first, second = pair_count (d, False)
    if len (second) > 1:
        first_count = count.sum(axis=1)
        second_count = count.sum(axis=0)
        total = count.sum()
        for x in range(len(first)):
            for y in range(len(second)):
                cxy = count[x][y]
                print first[x], second[y], \
                      percentify(cxy/float(first_count[x])), \
                      percentify(cxy/float(second_count[y])),
                ratio = float(cxy * total) / (first_count[x]*second_count[y])
                if ratio > 1.5:
                    print "STRONG - %.2f" %ratio
                elif ratio < 0.5:
                    print "WEAK - %.2f" %ratio
                else:
                    print                                     
          

def mikhail_tokens (logdir="main"):
    beat_tokens = [[0] * 4 for i in range(16)]
    antes = [[0] * 2 for i in range(16)]
    ante_this_beat = None
    for filename in list_files(logdir, 'mikhail'):
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if line.startswith('Beat '):
                beat = int(line.split(' ')[1])
                if ante_this_beat is not None:
                    antes[beat][int(ante_this_beat)] += 1
                ante_this_beat = False
            elif len(line) == 14 and line.endswith (' Seal tokens\n'):
                beat_tokens[beat][int(line[0])] += 1
            elif line.startswith('Mikhail: ') and ' (Seal)' in line:
                ante_this_beat = True
    tokens = [sum([beat_tokens[b][t] for b in range(16)]) for t in range(4)]
    total = sum(tokens)
    for t in range(4):
        print t, "tokens:", percentify(tokens[t]/float(total))
    for b in range(1,16):  
        print "beat %d: %.1f tokens;  %s ante" \
              %(b,
                sum([beat_tokens[b][t]*t for t in range(4)]) /
                   float(sum(beat_tokens[b])),
                percentify(antes[b][1]/float(sum(antes[b]))))

def oriana_meteor (logdir="main"):
    d = parse('oriana', logdir)
    styles = sorted(list(set([k[0] for k in d])))
    bases = sorted(list(set([k[1] for k in d])))
    tokens = sorted(list(set([k[2] for k in d])))
    for s in styles:
        if s != "Special":
            print s, "Meteor"
            total = float(sum([d.get((s,"Meteor",t),0) for t in tokens]))
            for t in tokens:
                print t, percentify(d.get((s,"Meteor",t),0)/total)
            print s, "(no Meteor)"
            total = float(sum([d[k] for k in d if k[0]==s and k[1]!="Meteor"]))
            for t in tokens:
                tmp = 0
                for b in bases:
                    if b != "Meteor":
                        tmp += d.get((s,b,t),0)
                print t, percentify(tmp/total)
        
def oriana_tokens (logdir="main"):
    tokens = [0,0,0,0,0,0,0,0,0,0,0]
    for filename in list_files(logdir, 'oriana'):
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if line[2:] == "Magic Point tokens\n":
                tokens[int(line[0])] += 1
            if line[3:] == "Magic Point tokens\n":
                tokens[int(line[0:2])] += 1
    for i, count in enumerate(tokens):
        print i, percentify(count/float(sum(tokens)))
    print "total:", sum(tokens)

def rexan_tokens (logdir="main"):
    tokens = [0,0,0,0]
    antes = [0,0,0,0]
    post_ante = [0,0,0,0]
    beat_sums = [0]*16
    beat_counts = [0]*16
    current_ante_count = None
    filenames = list_files(logdir, 'rexan')
    for filename in filenames:
        with open (filename) as f:
            log = [line for line in f]
        for i,line in enumerate(log):
            if line.startswith ("Beat "):
                current_beat = int(line[5:-1])
            # look for consecutive ante lines
            # (a second batch can be a repeat due to cancel)
            if current_ante_count == 0 and \
               line.endswith (" antes a Curse token\n"):
                j = 1
                while log[i+j].endswith (" antes a Curse token\n"):
                    j += 1
                current_ante_count = j
            if line.startswith("Opponent has ") and \
               line.endswith("Curse tokens\n"):
                current_tokens = int(line[13])
                if current_ante_count is not None:
                    antes[current_ante_count] += 1
                    post_ante[current_tokens - current_ante_count] += 1
                current_ante_count = 0
                tokens[current_tokens] += 1
                beat_sums[current_beat] += 1
                beat_counts[current_beat] += current_tokens
    antes[current_ante_count] += 1
    post_ante[current_tokens - current_ante_count] += 1
    print "Curse tokens at start of beat:"
    for i, count in enumerate(tokens):
        print i, percentify(count/float(sum(tokens)))
    print "total:", sum(tokens)

    print "Anted curse tokens"
    for i, count in enumerate(antes):
        print i, percentify(count/float(sum(antes)))
    print "total:", sum(antes)

    print "Un-anted curse tokens"
    for i, count in enumerate(post_ante):
        print i, percentify(count/float(sum(antes)))
    print "total:", sum(post_ante)

    gains = text ('gains 1 Curse token', 'rexan', logdir)
    games = 10 * len (filenames)
    print "%d gains in %d games - %f per game." %(gains, games,
                                                  float(gains)/games)

    print "Curse token accumulation by beat"
    for i in range(1,16):
        print "%d: %.1f" %(i, beat_counts[i]/float(beat_sums[i]))

    print
    parse_base_vs_induced_ante('rexan', logdir)
    
def runika_artifacts (logdir="main"):
    artifacts = ['Hover Boots',
                 'Battlefist',
                 'Autodeflector',
                 'Shield Amulet',
                 'Phase Goggles']
    active = {a:0 for a in artifacts}
    removed = {a:0 for a in artifacts}
    beats = 0
    for filename in list_files(logdir, 'runika'):
        with open (filename) as f:
            log = [line for line in f]
        for i in range(len(log)):
            if log[i].startswith('Active artifacts:'):
                beats += 1
                k = i + 1
                while log[k].startswith('  '):
                    active[log[k][2:-1]] += 1
                    k += 1
            if log[i].startswith('Removed artifacts:'):
                k = i + 1
                while log[k].startswith('  '):
                    removed[log[k][2:-1]] += 1
                    k += 1
    print "Active:"
    for artifact in active:
        print "%s: %s" % (artifact, percentify(active[artifact]/float(beats)))
    print
    print "Removed:"
    for artifact in removed:
        print "%s: %s" % (artifact, percentify(removed[artifact]/float(beats)))
    print
    games = float(10 * (len(phrases)-1))
    activations = {a : text("%s is activated"%a, 'runika', logdir) / games
                   for a in artifacts}
    deactivations = {a : text("%s is de-activated"%a, 'runika', logdir) / games
                     for a in artifacts}
    removals = {a : text("%s is removed"%a, 'runika', logdir) / games
                for a in artifacts}
    overcharges = {a : text("Runika overcharges her %s"%a, 'runika', logdir) / games
                   for a in artifacts}
    print "Activations:"
    for a in artifacts:
        print "%s: %.2f" %(a, activations[a])
    print
    print "Deactivations (without overcharged):"
    for a in artifacts:
        print "%s: %.2f" %(a, deactivations[a])
    print
    print "Overcharges (including Artifice Avarice):"
    for a in artifacts:
        print "%s: %.2f" %(a, overcharges[a])
    print
    print "removals:"
    for a in artifacts:
        print "%s: %.2f" %(a, removals[a])
    print
    print "Net Loss:"
    for a in artifacts:
        print "%s: %.2f" %(a, deactivations[a]+removals[a]-activations[a])
    print
    print "Total:"
    print "activations: %.2f"% sum(activations.itervalues())
    print "deactivations (no overcharge): %.2f"% (sum(deactivations.itervalues()))
    print "removals: %.2f"% sum(removals.itervalues())
    print "net loss: %.2f"% (sum(deactivations.itervalues())
                           + sum(removals.itervalues())
                           - sum(activations.itervalues()))
    

def seth_consolidation (pair_dict):
    standard_antes = ['(Strike)','(Shot)','(Dash)',
                      '(Burst)','(Drive)','(Grasp)','']
                      
    new_dict = {}
    for strat, count in pair_dict.iteritems():
        if strat[2] in standard_antes:
            new_dict[strat] = count
        else:
            new_strat = (strat[0], strat[1], '(Unique)')
            new_dict[new_strat] = new_dict.get(new_strat,0) + count
    return new_dict
    
def shekhtur_tokens (logdir="main"):
    tokens = [0,0,0,0,0,0]
    for filename in list_files(logdir, 'shekhtur'):
        with open (filename) as f:
            log = [line for line in f]
        for line in log:
            if line[2:] == "Malice tokens\n":
                tokens[int(line[0])] += 1
    for i, count in enumerate(tokens):
        print i, percentify(count/float(sum(tokens)))
    print "total:", sum(tokens)

def voco_zombies (logdir="main"):
    zombies = [0,0,0,0,0,0,0,0]
    zom_pos = [0,0,0,0,0,0,0]
    zom_opp = 0
    for filename in list_files(logdir, 'voco'):
        names = filename.split('/')[-1].split('_')
        voco_on_right = (names[1] == 'voco')
        voco_initial = 'v' if voco_on_right and names[0][0]=='v' else 'V'
        with open (filename) as f:
            log = [line for line in f]
        i=0
        while True:
            # forward to start of beat
            while i < len (log) and not log[i].startswith("Beat"):
                i += 1
            if i == len (log):
                break
            # look for board
            while sum([c=='.' for c in log[i]]) != 5:
                i += 1
            for p in range(7):
                if log[i][p] not in ['.',voco_initial]:
                    opponent = p
                    break
            zombies[sum([c=='z' for c in log[i+1]])] += 1
            if len (log[i+1]) == 8:
                for pos in range(7):
                    if log[i+1][pos] == 'z':
                        if voco_on_right:
                            pos = 6-pos
                        zom_pos[pos] += 1
                if log[i+1][opponent] == 'z':
                    zom_opp += 1
    total = float(sum(zombies))
    for i,z in enumerate(zombies):
        print "%d: %s" %(i, percentify(z/total))
    print "average: %.2f" %sum([i*z/total for i,z in enumerate(zombies)])
    print "positions:"
    for p,z in enumerate(zom_pos):
        print "%d: %s" %(p, percentify(z/total))
    print "on opponent:", percentify(zom_opp/total)

def zaamassal_paradigms (logdir="main"):
    for paradigm in ['Pain','Fluidity','Haste','Resilience','Distortion']:
        print paradigm
        print text("Zaamassal assumes the Paradigm of %s"%paradigm, 'zaamassal',
             logdir)

def parse_base_vs_induced_ante (name, logdir="main"):
    replacement = {p : p.replace(' ','') for opp_name in phrases.keys()
                                         for p in phrases[opp_name]} 
    strat_dict = {}
    for filename in list_files(logdir, name):
        names = filename.split('/')[-1].split('_')
        opp_name = names[0] if names[0] != name else names[1]
        with open (filename) as f:
            log = [line for line in f]
        i=0
        while True:
            # forward to start of beat
            while i < len (log) and not log[i].startswith("Beat"):
                i += 1
            if i == len (log):
                break
            # look for first pair used
            while not log[i].startswith(opp_name.capitalize() + ": "):
                i += 1
            line = log[i][:-1] # snip newline
            if ' | ' in line:
                induced_ante = line.split(' | ')[1]
            else:
                induced_ante = ''
            for p in replacement:
                line = line.replace(p,replacement[p])
            split = line.split(' ')
            strat = (split[1], split[2], induced_ante)
            if strat in strat_dict.keys():
                strat_dict[strat] +=1
            else:
                strat_dict[strat] = 1

    bases = ['Burst','Dash','Drive','Grasp','Shot','Strike','Cancel','Pulse']
    all_bases = bases + ['Finisher','Unique']
    antes = sorted(list(set([p[2] for p in strat_dict.keys()])))
    new_dict = {}
    for (s,b,a) in strat_dict.keys():
        if b in bases:
            new_dict[(b,a)] = new_dict.get((b,a),0) + strat_dict[(s,b,a)]
        else:
            if s == 'Special':
                new_dict[('Finisher',a)] = \
                        new_dict.get(('Finisher',a),0) + strat_dict[(s,b,a)]
            else:
                new_dict[('Unique',a)] = \
                        new_dict.get(('Unique',a),0) + strat_dict[(s,b,a)]

    count = array([[new_dict[(b,a)] if (b,a) in new_dict.keys()
                                             else 0
                    for a in antes]
                   for b in all_bases])

    base_count = count.sum(axis=1)
    ante_count = count.sum(axis=0)
    total = count.sum()

    bc = [(all_bases[i],base_count[i]) for i in range(len(all_bases))]
    bc = sorted (bc, key=itemgetter(1), reverse=True)
    for b in bc:
        print percentify(b[1]/float(total)), b[0]
    print '--------------'
    ac = [(antes[i],ante_count[i]) for i in range(len(antes))]
    ac = sorted (ac, key=itemgetter(1), reverse=True)
    for a in ac:
        print percentify(a[1]/float(total)), a[0]
    print '--------------'
    for x in range(len(all_bases)):
        if base_count[x]:
            for y in range(len(antes)):
                if ante_count[y]:
                    cxy = count[x][y]
                    print all_bases[x], antes[y], \
                          percentify(cxy/float(base_count[x])), \
                          percentify(cxy/float(ante_count[y])),
                    ratio = float(cxy * total) / (base_count[x]*ante_count[y])
                    if ratio > 1.5:
                        print "STRONG - %.2f" %ratio
                    elif ratio < 0.5:
                        print "WEAK - %.2f" %ratio
                    else:
                        print                                     
