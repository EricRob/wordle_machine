#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
import pdb
import random
import pickle
import argparse
import os
import string
from termcolor import cprint
import collections
import enum
import json
import requests
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from collections import deque, Counter

# Global variables
SMALL_SIZE = 6
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

# Class declarations
class Tip(enum.Enum):
    ABSENT = 0
    PRESENT = 1
    CORRECT = 2

class LetterStates(enum.Enum):
    NOTGUESSEDYET = 0
    NOTPRESENT = 1
    INCORRECTPOSITION = 2
    CORRECTPOSITION = 3

class Config:
    RESET = "\x1b[0m"

    WARN = "\x1b[33m"
    WIN = "\x1b[1;32m"
    LOSE = "\x1b[1;31m"
    HI = "\x1b[1m"
    DIM = "\x1b[90m"

    STATE_COLORS = {
        LetterStates.CORRECTPOSITION:    "\x1b[42;30m",
        LetterStates.INCORRECTPOSITION:  "\x1b[43;30m",
        LetterStates.NOTPRESENT:         "\x1b[40;37m",
        LetterStates.NOTGUESSEDYET:      "\x1b[90m"
        }

class Word(object):
    """docstring for Word"""
    def __init__(self, word):
        super(Word, self).__init__()
        self.word = word
        self.green = 0
        self.yellow = 0
        self.siblings = dict()
        self.counts = dict()
        for i, ch in enumerate(self.word):
            self.siblings[f'{ch}{i}'] = set()
            if ch in self.counts:
                self.counts[ch] += 1
            else:
                self.counts[ch] = 1
        self.valid = True

    def eval(self, letter):
        if letter in self.word:
            return True
        else:
            return False

    def is_sibling(self, sib):
        val = sum(c1 == c2 for c1, c2 in zip(sib, self.word))
        if val > 0:
            return True
        else:
            return False

    def add_sibling(self, sib):
        for i, v in enumerate(self.word):
            if self.word[i] == sib[i]:
                self.siblings[f'{v}{i}'].add(sib)

class Round(object):
    """docstring for Round"""
    def __init__(self, wordle, start=None, answer=None, verbose=False):
        super(Round, self).__init__()
        self.board = []
        self.vb = verbose
        for x in range(6):
            self.board.append([' _ ', ' _ ', ' _ ', ' _ ', ' _ '])
        self.wordle = wordle
        self.living = set(wordle.fives)
        if answer:
            self.answer = world.body[answer]
        else:
            self.answer = wordle.body[random.choice(tuple(self.living))]
        self.guesses = 0
        # self.answer = wordle.body['robot']
        if self.answer.word not in self.living:
            cprint('Answer not in possible guesses!', 'red', 'on_yellow')
            pdb.set_trace()
        
        if self.vb:
            cprint(f'Answer: {self.answer.word}', 'yellow')
            print(f'Starting living: {len(self.living)}\n~~~\n')
        self.known = ['_', '_', '_', '_', '_']
        self.valids = []
        self.untrimmed_valids = []
        
        if start:
            starting_word = wordle.body[start]
        else:
            starting_word = wordle.body[random.choice(tuple(self.living))]
        self.play(starting_word)

    def play(self, starting_word):
        wordle_unknown = True
        while(wordle_unknown):
            if len(self.living) == 0:
                cprint('Ran out of guesses! Did something break?', 'yellow')
                pdb.set_trace()
                break
            self.guesses += 1
            if self.guesses == 1:
                wordle_unknown = self.eval(starting_word)
                self.first_trim = len(self.living)
            else:
                random_guess = self.wordle.body[random.choice(tuple(self.living))]
                wordle_unknown = self.eval(random_guess)
        return

    def eval(self, guess):
        if self.vb:
            cprint(f'New guess: {guess.word}', 'green')
        if guess.word == self.answer.word:
            if self.vb:
                cprint(f'Success!! Correctly guessed in {self.guesses} tries!', 'red', 'on_green')
            return False
        else:
            valid_counts = self.answer.counts.copy()
            guessed_letters = set()

            # Evaluate for greens
            for i, k in enumerate(guess.word):
                if self.answer.word[i] == k:
                    if self.vb:
                        print(f'Correct letter: {k} at {i+1}')
                    self.known[i] = k
                    valid_counts[k] -= 1
                    guessed_letters.add(k)

            # Evaluate for yellows
            for i, k in enumerate(guess.word):
                letter_key = f'{k}{i}'
                if k in self.answer.word:
                    if self.answer.word[i] == k:
                        continue
                    elif valid_counts[k] > 0:
                        if k in guessed_letters:
                            if self.vb:
                                print(f'Valid second letter: {k}, not in position {i+1}')
                            self.valids.append(k)
                        else:
                            if self.vb:
                                print(f'Valid letter: {k}, not in position {i+1}')
                            self.valids.append(k)
                        if self.answer.word in guess.siblings[letter_key]:
                            cprint(f'Warning: about to remove answer', 'red')
                            pdb.set_trace()
                        self.living -= guess.siblings[letter_key]
                    else:
                        if self.vb:
                            cprint(f'Invalid repeat letter: {k}, removing {k} repeats', 'yellow')
                        if guess.counts[k] == 2:
                            if self.answer.word in self.wordle.double_letter[k]:
                                cprint(f'Warning: about to remove answer', 'red')
                                pdb.set_trace()
                            self.living -= self.wordle.double_letter[k]
                        if guess.counts[k] == 3:
                            if self.answer.word in self.wordle.triple_letter[k]:
                                cprint(f'Warning: about to remove answer', 'red')
                                pdb.set_trace()
                            self.living -= self.wordle.triple_letter[k]
                else:
                    if self.vb:
                        print(f'Removing all words with {k}')
                    if self.answer.word in self.wordle.letters[k]:
                        cprint(f'Warning: about to remove answer', 'red')
                        pdb.set_trace()
                    self.living -= self.wordle.letters[k]
            if self.vb:
                print(f'potentially living #: {len(self.living)}')
                self.trim()
                print(f'actually living #: {len(self.living)}')
                print(f'current state: {self.known}')
                print(f'valids: {self.valids}')
                print('~~~')
            else:
                self.trim()
            if len(self.living) == 0:
                cprint('Ran out of guesses! Did something break?', 'yellow')
                pdb.set_trace()
            return True

    # def eval_letters(self, guess):
    #     # Need to fix double letters!!
    #     valid_counts = self.answer.counts
    #     guessed_letters = set()
    #     self.eval_for_greens(guess, valid_counts)
    #     for i, k in enumerate(guess.word):
    #         if k in self.answer.word:
    #             if self.answer.word[i] == k:
    #                 print(f'Correct letter: {k} at {i+1}')
    #                 self.known[i] = k
    #                 valid_counts[k] -= 1
    #                 guessed_letters.add(k)
    #             elif valid_counts[k] > 0:
    #                 if k in guessed_letters:
    #                     print(f'Valid second letter: {k}, not in position {i+1}')
    #                     self.valids.append(k)
    #                 else:
    #                     print(f'Valid letter: {k}, not in position {i+1}')
    #                     self.valids.append(k)
    #                 self.living -= guess.siblings[k]
    #                 self.untrimmed_valids.append(k)
    #                 guess.counts[k] -= 1
    #             else:
    #                 cprint(f'Invalid DOUBLE letter: {k}, removing double {k}', 'yellow')
    #                 self.living -= self.wordle.double_letter[k]
    #         else:
    #             print(f'Removing all words with {k}')
    #             self.living -= self.wordle.letters[k]
    #     print(f'potentially living #: {len(self.living)}')
    #     self.trim()
    #     print(f'actually living #: {len(self.living)}')
    #     print(f'current state: {self.known}')
    #     print(f'valids: {self.valids}')
    #     print('~~~')

    def eval_for_greens(self, guess, valid_counts, guessed_letters):
        for i, k in enumerate(guess.word):
            if self.answer.word[i] == k:
                self.known[i] = k
                valid_counts[k] -= 1
                guessed_letters.add(k)


    def trim(self):
        to_trim = set()
        for i, v in enumerate(self.known):
            if v != "_":
                for word in self.living:
                    if word[i] != v:
                        to_trim.add(word)
        self.living -= to_trim

class Helper(object):
    """docstring for Helper"""
    def __init__(self, wordle):
        super(Helper, self).__init__()
        self.world = wordle
        self.living = wordle.fives
        

class Wordle(object):
    """docstring for Wordle"""
    def __init__(self, words_path, process_all=True):
        super(Wordle, self).__init__()
        self.corncob_path = words_path
        self.fives_path = 'fives.txt'
        self.invalids_path = 'invalids.txt'
        self.past_wordles_path = 'past_wordles.txt'
        self.body = {}
        self.letters = {}
        self.double_letter = {}
        self.triple_letter = {}
        self.atts = 0
        self.process_all = process_all
        self.has_repeat_letter = 0

        self.starts = dict()
        self.last_two = dict()
        self.first_two = dict()
        self.letter_positions = dict()
        self.letter_counts = dict()
        
        self.build_body()
        self.build_siblings()
        self.build_letters()
        # self.run_analytics()

    def build_body(self):
        print('building body...')
        if self.process_all:
            with open(self.corncob_path, 'r') as f:
                all_words = f.read().splitlines()
            with open(self.fives_path, 'r') as f:
                self.fives = f.read().splitlines()
            with open(self.invalids_path, 'r') as f:
                self.invalids = f.read().splitlines()
            with open(self.past_wordles_path, 'r') as f:
                self.past_wordles = f.read().splitlines()
            for word in all_words:
                if word in self.fives:
                    self.body[word] = Word(word)
                    count = Counter(word)
                    if len(count) < 5:
                        self.has_repeat_letter += 1
                elif len(word) == 5:
                    if self.check_valid(word):
                        self.body[word] = Word(word)
                        self.fives.append(word)
                        self.save_fives(word)
                        count = Counter(word)
                        if len(count) < 5:
                            self.has_repeat_letter += 1
        else:
            with open(self.past_wordles_path, 'r') as f:
                self.fives = f.read().splitlines()
                all_words = self.fives
                self.past_wordles = self.fives
            with open(self.invalids_path, 'r') as f:
                self.invalids = f.read().splitlines()
            for word in all_words:
                self.body[word] = Word(word)
                count = Counter(word)
                if len(count) < 5:
                    self.has_repeat_letter += 1
    def build_siblings(self):
        print('building siblings...')
        for word in self.fives:
            self.build_stats(word)
            for sib in self.fives:
                if word == sib:
                    continue
                elif self.body[word].is_sibling(sib):
                    self.body[word].add_sibling(sib)

    def build_stats(self, word):
        first_two = word[:2]
        last_two = word[3:]
        if last_two in self.last_two:
            self.last_two[last_two] += 1
        else:
            self.last_two[last_two] = 1
        
        if first_two in self.first_two:
            self.first_two[first_two] += 1
        else:
            self.first_two[first_two] = 1
        start = word[0]
        if start in self.starts:
            self.starts[start] += 1
        else:
            self.starts[start] = 1

        for i, v in enumerate(word):
            if v in self.letter_counts:
                self.letter_counts[v] += 1
            else:
                self.letter_counts[v] = 1
            letter_pos = f'{v}{i+1}'
            if letter_pos in self.letter_positions:
                self.letter_positions[letter_pos] += 1
            else:
                self.letter_positions[letter_pos] = 1


    
    def build_letters(self):
        print('building letters and stats...')
        alphabet = list('abcdefghijklmnopqrstuvwxyz')
        for letter in alphabet:
            self.letters[letter] = set()
            self.double_letter[letter] = set()
            self.triple_letter[letter] = set()
            for word in self.fives:
                if letter in word:
                    self.letters[letter].add(word)
                    if word.count(letter) == 2:
                        self.double_letter[letter].add(word)
                    if word.count(letter) == 3:
                        self.triple_letter[letter].add(word)
                



    def check_valid(self, word):
        if word in self.invalids:
            return False
        url = f'https://api.dictionaryapi.dev/api/v2/entries/en/{word}'
        payload={}
        headers = {}
        self.atts += 1
        response = requests.request("GET", url, headers=headers, data=payload)
        if self.atts % 10 == 0:
            cprint(f'atts: {self.atts}', 'green')
        try:
            output = json.loads(response.text)
        except Exception as e:
            pdb.set_trace()
        for out in output:
            if 'word' in out:
                try:
                    if out['word'] == word:
                        cprint(f'valid:   {word}', 'green')
                        return True
                except Exception as e:
                    pdb.set_trace()
        cprint(f'invalid: {word}', 'red')
        self.invalids.append(word)
        with open(self.invalids_path, 'a') as f:
            f.write(f'{word}\n')
        return False

    def save_fives(self, word):
        with open(self.fives_path, 'a') as f:
            f.write(f'{word}\n')

    def stats(self):
        N = 20
        width = 1.0
        top_lasttwo = dict(sorted(self.last_two.items(), key = itemgetter(1), reverse = True)[:N])
        top_lettercounts = dict(sorted(self.letter_counts.items(), key = itemgetter(1), reverse = True)[:N])
        top_starts = dict(sorted(self.starts.items(), key = itemgetter(1), reverse = True)[:N])
        top_firsttwo = dict(sorted(self.first_two.items(), key = itemgetter(1), reverse = True)[:N])
        top_letpos = dict(sorted(self.letter_positions.items(), key = itemgetter(1), reverse=True)[:2*N])

        fig, axs = plt.subplots(2,2, tight_layout=True)
        axs[0,0].bar(top_lasttwo.keys(), top_lasttwo.values(), width, color='b')
        axs[0,0].set_title('final two letters')
        axs[0,1].bar(top_starts.keys(), top_starts.values(), width, color='green')
        axs[0,1].set_title('starting letter')
        axs[1,0].bar(top_lettercounts.keys(), top_lettercounts.values(), width, color='orange')
        axs[1,0].set_title('letter frequency (any position)')
        axs[1,1].bar(top_firsttwo.keys(), top_firsttwo.values(), width, color='red')
        axs[1,1].set_title('first two')
        # axs[2,1].bar(top_letpos.keys(), top_letpos.values(), width, color='goldenrod')
        plt.show()

# Function declarations

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        print(f'Unpickling {pickle_path}...')
        wordle = pickle.load(f)
        return wordle

def save_pickle(wordle, pickle_path):
    with open(pickle_path, 'wb') as f:
        print(f'Pickling {pickle_path}...')
        pickle.dump(wordle, f, pickle.HIGHEST_PROTOCOL)

def compare_wordles(game1, game2):
    pdb.set_trace()
    rotated = deque(game2.past_wordles)
    rotated.rotate(1)
    game2.repeated_firsts = []
    for i, word in enumerate(game2.past_wordles):
        if i == 0:
            continue
        else:
            if game2.past_wordles[i][0] == rotated[i][0]:
                game2.repeat_first_letter += 1
                game2.repeated_firsts.append((game2.past_wordles[i], rotated[i]))
    return

def testing(game, ars):
    cprint('Running tests', 'green')
    random_guesses = dict()
    random_trim = dict()
    all_words = list(game.body.keys())
    lowest_mean = (None, 100)
    median_list = []
    lowest_median = (median_list, 100)
    min_trim = (None, 1000000)
    med_trim_list = []
    med_trim = (med_trim_list, 1000000)
    for i, start in enumerate(all_words):
        random_guesses[start] = []
        random_trim[start] = []
        for x in range(ars.tests):
            new_round = Round(game, start=start, verbose=ars.v)
            random_guesses[start].append(new_round.guesses)
            random_trim[start].append(new_round.first_trim)
        start_mean = np.mean(random_guesses[start])
        start_median = np.median(random_guesses[start])
        mean_trim = np.mean(random_trim[start])
        median_trim = np.median(random_trim[start])
        
        if start_mean < lowest_mean[1]:
            lowest_mean = (start, start_mean)
        if start_median < lowest_median[1]:
            median_list = [start]
            lowest_median = (median_list, start_median)
        elif start_median == lowest_median[1]:
            median_list.append(start)
            lowest_median = (median_list, start_median)
        if mean_trim < min_trim[1]:
            min_trim = (start, mean_trim)
        if median_trim < med_trim[1]:
            med_trim_list = [start]
            med_trim = (med_trim_list, median_trim)
        elif median_trim == med_trim[1]:
            med_trim_list.append(start)
            med_trim = (median_list, median_trim)

        if i % 50 == 0 and i > 0:
            print(f'{i} -- Mean guesses: {lowest_mean[0]} ({lowest_mean[1]:.2f}) -- Median guesses: {lowest_median[1]:.1f} -- Mean trim: {min_trim[0]} ({min_trim[1]:.1f}) -- Median trim: {med_trim[0]} ({med_trim[1]:.1f})')
    print(f'Low medians: {lowest_median[0]}')
    test_results = (random_guesses, random_trim)
    try:
        save_pickle(test_results, 'test_results.pickle')
    except Exception as e:
        pdb.set_trace()
    pdb.set_trace()

def main(ars):
    pickle_path = os.path.join('pickled_wordle.pickle')
    word_list = '20k.txt'

    if ars.load and os.path.exists(pickle_path):
        game = load_pickle(pickle_path)
    else:
        game = Wordle(word_list)
        if not ars.no_save:
            save_pickle(game, pickle_path)
        else:
            print('~~~* Pickling skipped')
    # past_games = Wordle(word_list, process_all=False)
    # game.stats()
    # past_games.stats()
    # compare_wordles(game, past_games)
    if ars.tests > 0:
        testing(game, ars)
    


# Main body
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve a wordle puzzle, hopefully in an efficient way')
    parser.add_argument('--load', action='store_true', default=False, help='Load pickled data to reduce processing time')
    parser.add_argument('--no_save', action='store_true', default=False, help='Skip saving pickle')
    parser.add_argument('--tests', type=int, default=0, help='Number of tests to run for starting words')
    parser.add_argument('-v', action='store_true', default=False, help='verbose output of rounds')
    parser.add_argument('--start', type=str, default=None, help='Starting word to use')
    parser.add_argument('--answer', type=str, default=None, help='Answer word to use')
    ars = parser.parse_args()
    main(ars)
