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
import math

# CS 421 - Homework 5: Part B
# Date: Nov. 11, 2025
# Authors: Malissa Chen and Rhiannon McKinley

# step 1:
# network structure
num_input = 10
num_hidden = 16
num_output = 1

# initialize weights
np.random.seed(0)
weight_hidden = np.random.uniform(-1.0, 1.0, (num_hidden, num_input + 1))
weight_output = np.random.uniform(-1.0, 1.0, (num_output, num_hidden + 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_matrix(x, weight_hidden, weight_output):
    # add bias input
    x_with_bias = np.append(x, 1.0) 

    # calculate hidden layer signals using matrix multiplication
    hidden_inputs = np.dot(weight_hidden, x_with_bias)  

    # apply sigmoid to get hidden layer outputs 
    hidden_outputs = sigmoid(hidden_inputs)

    # add bias to hidden layer outputs (for the output layer’s bias)
    hidden_with_bias = np.append(hidden_outputs, 1.0)  

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

# returns the next step from start moving toward goal
# moves one tile in the direction that reduces the Manhattan distance
    
def nearestLocation(start, goal):
    x1, y1 = start
    x2, y2 = goal

    dx = x2 - x1
    dy = y2 - y1

    # move 1 step horizontally if needed
    if dx > 0:
        return (x1 + 1, y1)
    if dx < 0:
        return (x1 - 1, y1)

    # move 1 step vertically if horizontal is aligned
    if dy > 0:
        return (x1, y1 + 1)
    if dy < 0:
        return (x1, y1 - 1)

    # already at goal
    return start

def worker_behavior(state, worker):
    player_id = state.whoseTurn
    inv = state.inventories[player_id]

    # get target list of food
    foods = getConstrList(state, NEUTRAL, [FOOD])
    dropoffs = [inv.getAnthill()] + inv.getTunnels()

   # find nearest targer
    def closest_target(from_coord, targets):
        if not targets:
            return None
        best = min(targets, key=lambda t: approxDist(from_coord, t.coords))
        return best

    # if carrying food, go deliver
    if worker.carrying:
        drop = closest_target(worker.coords, dropoffs)
        if drop is None:
            return None  
        
        # if standing on dropoff, deposit food
        if drop.coords == worker.coords:
            return Move(MOVE_ANT, None, 0)   

        # Move toward dropoff
        path = (worker.coords, drop.coords)
        if path and len(path) > 1:
            return Move(MOVE_ANT, path[1], 1)
        return None

    # not carrying food, find foods
    food = closest_target(worker.coords, foods)
    if food is None:
        return None  # no food on board

    # if standing on food
    if worker.coords == food.coords:
        return Move(END, None, None)

    # move toward food
    path = (worker.coords, food.coords)
    if path and len(path) > 1:
        return Move(MOVE_ANT, nearestLocation(worker.coords, food.coords), 1)

    return None

## utility function
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

# training_states = [GameState.getBasicState() for _ in range(50)]

# examples = [(extract_features(state), [evaluate(state)]) for state in training_states]


# # training loop
# max_epochs = 1000
# epoch = 0
# average_error = 1.0

# while average_error > 0.05 and epoch < max_epochs:
#     total_error = 0.0
#     # randomly pick 10 examples each epoch
#     samples = random.sample(examples, 10)

#     # train on each sample
#     for x, target in samples:
#         x = np.array(x)
#         target = np.array(target)

#         # forward + backprop update
#         weight_hidden, weight_output = backpropagation(x, target, weight_hidden, weight_output, learning_rate= 0.5)

#         # calculate network output
#         _, output = forward_matrix(x, weight_hidden, weight_output)
#         sample_error = (target - output) ** 2
#         total_error += sample_error

#         # update global weights
#         weight_hidden, weight_output = weight_hidden, weight_output

#     # compute average error for this epoch
#     average_error = total_error.mean()
#     epoch += 1
# print(f"Epoch {epoch}: Avg Error = {average_error:.4f}")
# print("Final Hidden Weights:\n", weight_hidden)
# print("Final Output Weights:\n", weight_output)

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

        # hard coded training weights
        self.weight_hidden = np.array([
            [ 0.09862196,  0.43137369,  0.20572574,  0.09155729, -0.15169545,  0.29377814,
            -0.12482558,  0.783546  ,  0.92832048, -0.23212201,  0.58543999],
            [ 0.0536723 ,  0.13197158,  0.85036977, -0.86533946, -0.82985894, -0.96779829,
            0.66523969,  0.5563135 ,  0.73590675,  0.95311914,  0.59008204],
            [-0.0787361 ,  0.55936352, -0.76379011,  0.27679135, -0.71498825,  0.88594818,
            0.04369664, -0.17067612, -0.4725836 ,  0.54677255, -0.09108899],
            [ 0.1417715 , -0.9575168 ,  0.23625171,  0.23301793,  0.2387716 ,  0.89730336,
            0.3636406 , -0.2809842 , -0.12103249,  0.40016599, -0.86974185],
            [ 0.33585274,  0.34359505, -0.57877102, -0.73797265, -0.36682399, -0.26793984,
            0.14039354, -0.12279697,  0.97906699, -0.79359107, -0.57760787],
            [-0.67784455,  0.30575306, -0.49350951, -0.06821291, -0.5116124 , -0.68298801,
            -0.77924972,  0.31265918, -0.72409769, -0.60729886, -0.26347683],
            [ 0.64137126, -0.80641265,  0.67576677, -0.80891055,  0.95230373, -0.063928  ,
            0.95352218,  0.20969104,  0.47791196, -0.92223962, -0.43561648],
            [-0.75909879, -0.40721151, -0.76244294, -0.36311908, -0.17096592, -0.87068883,
            0.38494424,  0.13320291, -0.46871293,  0.0470042 , -0.8111028 ],
            [ 0.14854285,  0.85524225, -0.36353212,  0.32879051, -0.73975442,  0.42595413,
            -0.42118781, -0.63361728,  0.16967573, -0.96313505,  0.65117978],
            [-0.99159792,  0.3546442 , -0.46018183,  0.46860807,  0.92338821, -0.50447147,
            0.15231467,  0.18408386,  0.14351494, -0.55482561,  0.90352027],
            [-0.10597427,  0.69259232,  0.39891355, -0.40553114,  0.62737062, -0.20743856,
            0.76220639,  0.16254575,  0.7632457 ,  0.38483816,  0.45005851],
            [ 0.00185879,  0.9113773 ,  0.2878224 , -0.15371185,  0.21199646, -0.96319354,
            -0.39685037,  0.32034707, -0.42063475,  0.23524089, -0.14404254],
            [-0.73138524, -0.40576872,  0.13946315,  0.17754546,  0.14631713,  0.3017349 ,
            0.30420654, -0.13716313,  0.79075982, -0.26720963, -0.13293688],
            [ 0.78533815,  0.61387942,  0.40807546, -0.79686163,  0.84045667,  0.43146548,
            0.99769401, -0.70110339,  0.73774356, -0.67352269,  0.23410201],
            [-0.75138164,  0.69699485,  0.6148336 ,  0.13996258, -0.18465502, -0.85970923,
            0.39485755, -0.09291463,  0.44508959,  0.73374304,  0.95299979],
            [ 0.71424762, -0.97393089, -0.27951568,  0.46473481, -0.65409971,  0.04735509,
            -0.89132402, -0.60000695, -0.96031547,  0.59003634, -0.54686875]
        ])

        self.weight_output = np.array([
            [-0.3433716 ,  0.84076264,  0.37931765, -0.95803242, -0.67931062,  0.2386689 ,
            0.14316103, -0.5262492 ,  0.83734084,  0.20000303,  0.04008353,  0.16642346,
            0.43821947, -0.40755969, -0.23175306, -0.59352619, -0.66918056]
        ])

        # Variables for utility
        self.anthillBestDist = None
        self.tunnelBestDist = None
        self.bestRet = None
    
    def get_utility_score(self, state):
        features = extract_features(state)
        _, output = forward_matrix(
            features,
            self.weight_hidden,     
            self.weight_output      
        )
        return float(output[0])

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
            "eval": (self.get_utility_score(state)) + depth,
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

        while lowestNode["parent"] is not None and lowestNode["parent"]["parent"] is not None:
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

   