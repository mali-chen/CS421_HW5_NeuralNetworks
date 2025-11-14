import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
import numpy as np
import random
import math

# CS 421 - Homework 5: Part B
# Date: Nov. 11, 2025
# Authors: Malissa Chen and Rhiannon McKinley

# step 1:
# network structure
num_input = 10
num_hidden = 8
num_output = 1

# initialize weights
np.random.seed(0)
weight_hidden = np.random.uniform(-1.0, 1.0, (num_hidden, num_input + 1))
weight_output = np.random.uniform(-1.0, 1.0, (num_output, num_hidden + 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_matrix(x, weight_hidden, weight_output):
    # add bias input
    x_with_bias = np.append(x, 1)  # now x has 5 numbers

    # calculate hidden layer signals using matrix multiplication
    hidden_inputs = np.dot(weight_hidden, x_with_bias)  

    # apply sigmoid to get hidden layer outputs 
    hidden_outputs = sigmoid(hidden_inputs)

    # add bias to hidden layer outputs (for the output layer’s bias)
    hidden_with_bias = np.append(hidden_outputs, 1)  # now has 9 numbers

    # Calculate final output
    final_input = np.dot(weight_output, hidden_with_bias)  
    final_output = sigmoid(final_input)             

    return hidden_with_bias, final_output

# Cited from Copilot
def sigmoid_derivative(x):
    return x * (1 - x)

def backpropagation(x, target, weight_hidden, weight_output, learning_rate=0.5):
    # Forward pass
    h_b, y = forward_matrix(x, weight_hidden, weight_output)

    # Compute output error
    error_output = target - y  
    delta_output = error_output * sigmoid_derivative(y)  

    # Compute hidden layer error
    h = h_b[:-1]  # remove bias from hidden output
    weight_output_no_bias = weight_output[:, :-1]  
    error_hidden = delta_output.dot(weight_output_no_bias) 
    delta_hidden = error_hidden * sigmoid_derivative(h) 

    # Update output weights
    weight_output += learning_rate * delta_output.reshape(-1, 1) * h_b.reshape(1, -1)

    # Prepare input with bias
    x_b = np.append(x, 1)  # shape (5,)
    # Update hidden weights
    weight_hidden += learning_rate * delta_hidden.reshape(-1, 1) * x_b.reshape(1, -1)

    return weight_hidden, weight_output

# get features from gamestate
def extract_features(state):
    player_id = state.whoseTurn
    opponent_id = 1 - player_id
    my_inv = state.inventories[player_id]
    opp_inv = state.inventories[opponent_id]

    food_diff = my_inv.foodCount - opp_inv.foodCount
    food_input = (food_diff + 2) / 4.0

    combat_types = (DRONE, SOLDIER, R_SOLDIER)
    
    def effective_health(a): return a.health + 7 if a.type == R_SOLDIER else a.health
    my_army = sum(effective_health(a) for a in my_inv.ants if a.type in combat_types)
    opp_army = sum(effective_health(a) for a in opp_inv.ants if a.type in combat_types)
    army_score = 0.5 + 0.5 * (my_army - opp_army) / max(my_army + opp_army, 1)

    def worker_factor(ants):
        count = len([a for a in ants if a.type == WORKER])
        return 0.1 if count == 0 else 0.6 if count == 1 else 1.0 if count == 2 else 0.5
    my_worker_score = worker_factor(my_inv.ants)
    opp_worker_score = worker_factor(opp_inv.ants)

    task_score = 0.5

    queen = my_inv.getQueen()
    enemy_queen = opp_inv.getQueen()
    my_hill = my_inv.getAnthill()
    opp_hill = opp_inv.getAnthill()
    queen_score = (queen.health / 10.0) if queen else 0
    my_hill_score = min(max((my_hill.captureHealth / 3.0) if my_hill else 0, 0), 1)
    opp_hill_score = min(max((opp_hill.captureHealth / 3.0) if opp_hill else 0, 0), 1)

    def dist_score(ants, targets):
        targets = [t for t in targets if t]
        if not ants or not targets:
            return 0.5
        dists = [approxDist(a.coords, t.coords) for a in ants for t in targets]
        min_d, avg_d = min(dists), sum(dists) / len(dists)
        return (1 / (1 + min_d) + 1 / (1 + avg_d)) / 2

    attack_targets = [enemy_queen, opp_hill] + [a for a in opp_inv.ants if a.type == WORKER]
    threat_targets = [queen, my_hill]
    attack_score = dist_score(getAntList(state, player_id, combat_types), attack_targets)
    threat_score = dist_score(getAntList(state, opponent_id, combat_types), threat_targets)

    return np.array([
        food_input,
        army_score,
        my_worker_score,
        1 - opp_worker_score,
        task_score,
        queen_score,
        1 - my_hill_score,
        1 - opp_hill_score,
        attack_score,
        1 - threat_score
    ])

def evaluate(state):
    # compute a normalized utility score for a given state
    player_id = state.whoseTurn
    opponent_id = 1 - player_id

    my_inv = state.inventories[player_id]
    opp_inv = state.inventories[opponent_id]

    # food comparison
    food_diff = my_inv.foodCount - opp_inv.foodCount
    food_score = 0.5 + 0.5 * food_diff / max(abs(food_diff) + 2, 2)

    # army strength
    combat_types = (DRONE, SOLDIER, R_SOLDIER)

    def effective_health(a):
        return a.health + 7 if a.type == R_SOLDIER else a.health

    my_army = sum(effective_health(a) for a in my_inv.ants if a.type in combat_types)
    opp_army = sum(effective_health(a) for a in opp_inv.ants if a.type in combat_types)
    army_score = 0.5 + 0.5 * (my_army - opp_army) / max(my_army + opp_army, 1)

    # worker evaluation
    def worker_factor(ants):
        count = len([a for a in ants if a.type == WORKER])
        if count == 0:
            return 0.1
        if count == 1:
            return 0.6
        if count == 2:
            return 1.0
        return 0.5

    my_worker_score = worker_factor(my_inv.ants)
    opp_worker_score = worker_factor(opp_inv.ants)
    worker_score = 0.67 * my_worker_score + 0.33 * (1 - opp_worker_score)

    # worker tasks
    food_bonus, pickup_prox, delivery_prox = 0, 0.5, 0.5
    my_workers = [a for a in my_inv.ants if a.type == WORKER]

    if my_workers:
        distances = []
        hill_and_tunnels = [my_inv.getAnthill()] + my_inv.getTunnels()
        foods = getConstrList(state, NEUTRAL, [FOOD])

        for w in my_workers:
            if w.carrying:
                food_bonus += 0.3
                if hill_and_tunnels:
                    distances.append(min(approxDist(w.coords, d.coords) for d in hill_and_tunnels if d))
            elif foods:
                distances.append(min(approxDist(w.coords, f.coords) for f in foods))

        if distances:
            closest, avg_dist = min(distances), sum(distances) / len(distances)
            delivery_prox = 1 / (1 + closest)
            pickup_prox = 1 / (1 + avg_dist)

        deposit_bonus = sum(
            0.5 for w in my_workers if w.carrying and any(w.coords == d.coords for d in [my_inv.getAnthill()] + my_inv.getTunnels() if d)
        )
        task_score = 0.3 * pickup_prox + 0.2 * delivery_prox + food_bonus + deposit_bonus
    else:
        task_score = 0

    # queen and hill factors
    queen = my_inv.getQueen()
    enemy_queen = opp_inv.getQueen()
    my_hill = my_inv.getAnthill()
    opp_hill = opp_inv.getAnthill()

    queen_score = (queen.health / 10.0) if queen else 0
    my_hill_score = min(max((my_hill.captureHealth / 3.0) if my_hill else 0, 0), 1)
    opp_hill_score = min(max((opp_hill.captureHealth / 3.0) if opp_hill else 0, 0), 1)

    def smooth_score(x, scale=6):
        # converts a raw score into value between 0 and 1
        return 1.0 / (1.0 + math.exp(-scale * (x - 0.5)))

    # distances to attack and threats
    def dist_score(ants, targets):
        targets = [t for t in targets if t is not None]
        if not ants or not targets:
            return 0.5
        dists = [approxDist(a.coords, t.coords) for a in ants for t in targets]
        min_d, avg_d = min(dists), sum(dists) / len(dists)
        return (1 / (1 + min_d) + 1 / (1 + avg_d)) / 2

    attack_targets = [enemy_queen, opp_hill] + [a for a in opp_inv.ants if a.type == WORKER]
    threat_targets = [queen, my_hill]
    attack_score = dist_score(getAntList(state, player_id, combat_types), attack_targets)
    threat_score = dist_score(getAntList(state, opponent_id, combat_types), threat_targets)

    # combine scores
    raw_score = (
        0.30 * food_score +
        0.20 * worker_score +
        0.10 * task_score +
        0.15 * army_score +
        0.10 * queen_score +
        0.05 * (1 - my_hill_score) +
        0.05 * (1 - opp_hill_score) +
        0.05 * attack_score -
        0.05 * threat_score
    )

    return max(0.0, min(1.0, smooth_score(raw_score)))

from GameState import GameState

training_states = [GameState.getBasicState() for _ in range(50)]

examples = [(extract_features(state), [evaluate(state)]) for state in training_states]


# training loop
max_epochs = 1000
epoch = 0
average_error = 1.0

while average_error > 0.05 and epoch < max_epochs:
    total_error = 0.0
    # randomly pick 10 examples each epoch
    samples = random.sample(examples, 10)

    # train on each sample
    for x, target in samples:
        x = np.array(x)
        target = np.array(target)

        # forward + backprop update
        weight_hidden, weight_output = backpropagation(x, target, weight_hidden, weight_output, learning_rate= 0.5)

        # calculate network output
        _, output = forward_matrix(x, weight_hidden, weight_output)
        sample_error = (target - output) ** 2
        total_error += sample_error

        # update global weights
        weight_hidden, weight_output = weight_hidden, weight_output

    # compute average error for this epoch
    average_error = total_error.mean()
    epoch += 1
    print(f"Epoch {epoch}: Avg Error = {average_error:.4f}")

    def get_utility_score(state):
        features = extract_features(state)
        _, output = forward_matrix(features, weight_hidden, weight_output)
        return output[0]

##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "Network Web")

        # Variables for utility
        self.anthillBestDist = None
        self.tunnelBestDist = None
        self.bestRet = None
    
    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    # makeNode
    # creates a node dictionary with state, move, depth, and evaluation.
    ##
    def makeNode(self, move, state, depth, parent):
        return {
            "move": move,
            "state": state,
            "depth": depth,
            "eval": self.utility(state) + depth,
            "parent": parent
        }
    
    ##
    # expandNode
    # expands a node into child nodes by simulating all legal moves.
    ##
    def expandNode(self, initNode):
        if initNode is None:
            return []

        moves = listAllLegalMoves(initNode["state"])
        nodes = []

        for m in moves:
            nextState = getNextState(initNode["state"], m)
            node = self.makeNode(m, nextState, initNode["depth"] + 1, initNode)
            nodes.append(node)

        return nodes
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        frontierNodes = []

        rootNode = self.makeNode(None, currentState, 0, None) # root node is depth 0 and has no parent node
        frontierNodes.append(rootNode)

        A_STAR_DEPTH = 3 
        for i in range(0, A_STAR_DEPTH):
            lowestNode = frontierNodes[0]
            for node in frontierNodes:
                if node["eval"] < lowestNode["eval"]:
                    lowestNode = node

            frontierNodes.remove(lowestNode)
            nodeList = self.expandNode(lowestNode)
            for node in nodeList:
                frontierNodes.append(node)
        
        bestList = []
        lowestNode = frontierNodes[0]
        for node in frontierNodes:
            if node["eval"] < lowestNode["eval"]:
                lowestNode = node
                bestList.clear()
            if node["eval"] == lowestNode["eval"]:
                bestList.append(node)
        
        if len(bestList) > 0 :
            bestList.append(lowestNode)
            lowestNode = bestList[random.randint(0, len(bestList) - 1)]

        while(lowestNode["parent"]["parent"] != None):
            lowestNode = lowestNode["parent"]

        return lowestNode["move"]

    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        #method templaste, not implemented
        pass

   