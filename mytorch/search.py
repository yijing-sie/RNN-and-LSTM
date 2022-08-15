import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)
    forward_path_list = []
    forward_prob_list = []
    forward_path_list_unsqueezed = []
    SymbolSets.insert(0, ' ') #inserting b_l_a_n_k 
    for i in range(y_probs.shape[2]):
        seq = ""
        prob = 1
        for j in range(y_probs.shape[1]):
            max_prob = max(y_probs[:, j, i])
            prob = prob * max_prob
            seq += SymbolSets[np.where(y_probs[:, j, i] == max_prob)[0][0]]
        forward_prob_list.append(prob)
        forward_path_list_unsqueezed.append(seq)
    for i in range(y_probs.shape[2]):
        squeezed = ""
        pre_word = None
        for seq in forward_path_list_unsqueezed[i]:
            if seq == ' ': 
                pre_word = None
                continue
            if pre_word == seq:
                continue
            else: 
                pre_word = seq
                squeezed += seq
        forward_path_list.append(squeezed)
    if y_probs.shape[2] == 1:
        return (forward_path_list[0], forward_prob_list[0])
    else:
        return (forward_path_list, forward_prob_list)
                
##############################################################################
def InitializePaths(SymbolSet, y):
    InitialBlankPathScore = {}
    InitialPathScore = {}
    # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet 
    path = "" 
    InitialBlankPathScore[path] = y[0] # Score of blank at t=1 
    InitialPathsWithFinalBlank = {path}
    # Push rest of the symbols into a path-ending-with-symbol stack 
    InitialPathsWithFinalSymbol = set() 
    for i, c in enumerate(SymbolSet): # This is the entire symbol set, without the blank 
        path = c 
        InitialPathScore[path] = y[i+1] # Score of symbol c at t=1 
        InitialPathsWithFinalSymbol.add(path) # Set addition    
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore
##############################################################################

def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore) :
    UpdatedPathsWithTerminalBlank = set() 
    UpdatedBlankPathScore = {} 
    # First work on paths with terminal blanks 
    #(This represents transitions along horizontal trellis edges for blanks) 
    for path in PathsWithTerminalBlank: # Repeating a blank doesn’t change the symbol sequence 
        UpdatedPathsWithTerminalBlank.add(path) # Set addition 
        UpdatedBlankPathScore[path] = BlankPathScore[path]*y[0]

    # Then extend paths with terminal symbols by blanks 
    for path in PathsWithTerminalSymbol: 
        # If there is already an equivalent string in UpdatesPathsWithTerminalBlank 
        # simply add the score. If not create a new entry 
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path]* y[0]
        else:
            UpdatedPathsWithTerminalBlank.add(path) # Set addition 
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]
    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore



def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalSymbol = set() 
    UpdatedPathScore = {}
# First extend the paths terminating in blanks. This will always create a new sequence 
    for path in PathsWithTerminalBlank: 
        for i, c in enumerate(SymbolSet): # SymbolSet does not include blanks 
            newpath = path + c # Concatenation
            UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[i + 1]

# Next work on paths with terminal symbols 
    for path in PathsWithTerminalSymbol: # Extend the path with every symbol other than blank 
        for i, c in enumerate(SymbolSet): # SymbolSet does not include blanks 
            #newpath = (c == path[-1]) ? path : path + c # Horizontal transitions don’t extend the sequence
            newpath =  path if(c == path[-1]) else path + c
            if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths 
                UpdatedPathScore[newpath] += PathScore[path] * y[i + 1]
            else: # Create new path 
                UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition 
                UpdatedPathScore[newpath] = PathScore[path] * y[i + 1]
    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth): 
    PrunedBlankPathScore = {} 
    PrunedPathScore = {} # First gather all the relevant scores 
    scorelist = []
    for p in PathsWithTerminalBlank: 
        scorelist += [BlankPathScore[p]] 

    for p in PathsWithTerminalSymbol:
        scorelist += [PathScore[p]] 
    # Sort and find cutoff score that retains exactly BeamWidth paths 
    scorelist.sort(reverse=True) # In decreasing order 
    cutoff =  scorelist[BeamWidth] if BeamWidth < len(scorelist) else scorelist[-1]
    PrunedPathsWithTerminalBlank = set() 
    for p in PathsWithTerminalBlank: 
        if BlankPathScore[p] > cutoff: 
            PrunedPathsWithTerminalBlank.add(p) # Set addition 
            PrunedBlankPathScore[p] = BlankPathScore[p]
    
    PrunedPathsWithTerminalSymbol = set() 
    for p in PathsWithTerminalSymbol: 
        if PathScore[p] > cutoff: 
            PrunedPathsWithTerminalSymbol.add(p) # Set addition 
            PrunedPathScore[p] = PathScore[p]
    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
# All paths with terminal symbols will remain 
    MergedPaths = PathsWithTerminalSymbol 
    FinalPathScore = PathScore
# Paths with terminal blanks will contribute scores to existing identical paths from 
# PathsWithTerminalSymbol if present, or be included in the final set, otherwise 
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPaths.add(p) # Set addition 
            FinalPathScore[p] = BlankPathScore[p]
    return MergedPaths, FinalPathScore

def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
        # Follow the pseudocode from lecture to complete beam search :-)
    PathScore = {}
    BlankPathScore = {}
    # First time instant: Initialize paths with each of the symbols, 
    # including blank, using score at time t=1 
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:,0,:])
    # Subsequent time steps 
    for t in range(1, y_probs.shape[1]):
        # Prune the collection down to the BeamWidth 
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, BeamWidth)
        # First extend paths by a blank 
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:,t,:], BlankPathScore, PathScore)
        # Next extend paths by a symbol 
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y_probs[:,t,:], BlankPathScore, PathScore)
    
    # Merge identical paths differing only by the final blank 
    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
    # Pick best path 
    BestPath = max(FinalPathScore, key=FinalPathScore.get)
    
    return BestPath, FinalPathScore