    def __init__(self, the_game, n, is_user=False):
    def all_cards (self): # change if there are extra cards beyond tokens
    def choose_initial_discards (self):
    def set_starting_setup (self, default_discards, use_special_actions):
    def situation_report (self):
    def get_board_addendum (self): # and set board_addendum_lines
    def read_my_state (self, lines, board, addendum):
    def initial_save (self):
    def initial_restore (self, state):
    def reset (self):
    def full_save (self):
    def full_restore (self, state):
    def prepare_next_beat (self):
    def get_antes (self): (or induced antes)
    def input_ante (self): (or induced ante)
    def ante_trigger (self):
    def get_ante_name (self, ante): (or induced ante)
    def set_active_cards (self):
        # characters with tokens without ante effects don't need to check them

    def recover_tokens (self, n): (or induced tokens)
    def set_preferred_range (self): # if character can get non-style/base range
    def evaluate (self):

    mean_priority_bonus
    expected_stunguard/soak

    # special evaluations for cards (situation dependent, none )
   
    
