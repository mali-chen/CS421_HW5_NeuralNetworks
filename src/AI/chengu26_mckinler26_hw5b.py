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
from GameState import GameState

import math

# CS 421 - Homework 5: Part B
# Date: Nov. 11, 2025
# Authors: Malissa Chen and Rhiannon McKinley

# -------------------------------------
# Training functions
# -------------------------------------

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

# def backpropagation(x, target, weight_hidden, weight_output, learning_rate=0.5):
#     # Forward pass
#     h_b, y = forward_matrix(x, weight_hidden, weight_output)

#     # Compute output error
#     error_output = target - y  
#     delta_output = error_output * sigmoid_derivative(y)  

#     # Compute hidden layer error
#     h = h_b[:-1]  # remove bias from hidden output
#     weight_output_no_bias = weight_output[:, :-1]  
#     error_hidden = delta_output.dot(weight_output_no_bias) 
#     delta_hidden = error_hidden * sigmoid_derivative(h) 

#     # Update output weights
#     weight_output += learning_rate * delta_output.reshape(-1, 1) * h_b.reshape(1, -1)

#     # Prepare input with bias
#     x_b = np.append(x, 1)  # shape (5,)
#     # Update hidden weights
#     weight_hidden += learning_rate * delta_hidden.reshape(-1, 1) * x_b.reshape(1, -1)

#     return weight_hidden, weight_output

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

####################################################################
## CREDIT: Andrew Asch's utility function, permission from Dr.Nuxoll
####################################################################
#utility
#Description: Determines how (un)/favorable the current game state is to player
#
#Parameters:
#   currentState - A clone of the current state (GameState)
##
def utility(currentState):
    #get pointers
    me = currentState.whoseTurn
    them = 1 - me

    myAttAnts = getAntList(currentState, me, (DRONE, SOLDIER, R_SOLDIER))
    myDrones = getAntList(currentState, me, (DRONE,))
    mySoldiers = getAntList(currentState, me, (SOLDIER,))
    myRSoldiers = getAntList(currentState, me, (R_SOLDIER,))
    myDefAnts = getAntList(currentState, me, (DRONE, SOLDIER, R_SOLDIER, QUEEN))
    myWorkers = getAntList(currentState, me, (WORKER,))

    theirAnts = getAntList(currentState, them, (DRONE, SOLDIER, R_SOLDIER))
    theirWorkers = getAntList(currentState, them, (WORKER,))

    myFood = currentState.inventories[me].foodCount
    myBuildObjs = getConstrList(currentState, me, (ANTHILL, TUNNEL))
    myAnthill = currentState.inventories[me].getAnthill()
    foodObjs = getConstrList(currentState, None, (FOOD,))
    theirFood = currentState.inventories[them].foodCount
    theirAnthill = currentState.inventories[them].getAnthill()
    
    try:
        myQueen = getAntList(currentState, me, (QUEEN,))[0]
        myQueenHealth = myQueen.health
    except IndexError:
        myQueenHealth = 1 #queen is dead
    
    try:  
        theirQueen = getAntList(currentState, them, (QUEEN,))[0]
        theirQueenHealth = theirQueen.health
    except IndexError:
        theirQueenHealth = 1
    

    #NOTE: comparitive advantages
    troopAdv = len(myAttAnts) - len(theirAnts) #incentivises placing troops
    healthAdv = (myAnthill.captureHealth*4 + myQueenHealth) - \
        (theirAnthill.captureHealth*4 + theirQueenHealth)
    foodAdv = (myFood - theirFood)
    
    #NOTE: small movement/productivity utility bumps
    #incentivises soldiers (reap the most benefit with least amount of strat)
    antVal = len(mySoldiers)*2 + len(myRSoldiers) + len(myDrones) 

    #incetivises having 1 worker
    workerVal = min(len(myWorkers), 1) 
    
    #workers should be incentivised to be productive
    stepsFromFood = [0]
    stepsFromBuilding = [0]
    carryingWorkers=0
    for w in myWorkers: 
        if w.carrying:
            carryingWorkers += 1
            #find the closest building to each worker and incentivise moving towards
            stepsMinB = stepsToReach(currentState, w.coords, myBuildObjs[0].coords)
            for b in myBuildObjs:
                if stepsMinB > stepsToReach(currentState, w.coords, b.coords):
                    stepsMinB = stepsToReach(currentState, w.coords, b.coords)
            stepsFromBuilding.append(stepsMinB)
                
        else: 
            #find the closest food to each worker and incentivise moving towards
            stepsMinF = 999
            for f in foodObjs:
                if stepsMinF > stepsToReach(currentState, w.coords, f.coords) \
                    and f.coords[1] < 4:
                    stepsMinF = stepsToReach(currentState, w.coords, f.coords)
            stepsFromFood.append(stepsMinF)
    
    #when workers are close to goal, provide a small bump to utility
    #we also need to incentivise having the worker pick up and set
    #down the food. If the worker is 1 distance away from the food,
    #it knows that picking it up will shift the goal from the food 
    #to the building. When the worker picks up the food it is further
    #from the new goal, and utility is subtracted. To make the worker 
    #actually reach the goals, we incetivised having workers carrying 
    #food over any possible distance incentive, and getting us food. 
    workerMvmt = (1/(sum(stepsFromFood) + sum(stepsFromBuilding) + 1) + \
                    carryingWorkers + myFood*2)

    #drone, soldier, and rsoldier should be insentivised to advance
    moveForward = sum([min(ant.coords[1], 7) for ant in myAttAnts]) - \
                    sum([min(9-ant.coords[1], 7) for ant in theirAnts])
    
    #drone, soldier, rsoldier and queen should be incetivised to defend 
    #when they aren't yet in enemy territory. When they take health off
    #attacking troops, they move is further rewarded. 
    threats = []
    for ant in theirAnts:
        if ant.coords[1] < 6:
            threats.append(ant)
            
    protectQ = 0
    if len(threats) > 0:
        defend = sum([1/stepsToReach(currentState, ant.coords, 
                                        threats[0].coords) for ant in myDefAnts \
                                        if ant.coords[1] < 6])-threats[0].health
        #make sure the queen doesn't defend when she is a one shot
        if myQueenHealth <=5: 
            protectQ = min([stepsToReach(currentState, 
                                            t.coords, 
                                            myQueen.coords) for t in threats])

    else: defend = 0

    #incentivise getting closer to and killing workers
    theirWorkerCount = -len(theirWorkers)
    distFromWorkers = []
    distFromQueen = []
    for ant in myAttAnts:
        try:
            distFromWorkers.append(-min([stepsToReach(currentState, 
                                                        ant.coords, 
                                            w.coords) for w in theirWorkers]))
        except ValueError:
            pass #they have no workers left
        try:
            distFromQueen.append(-abs(ant.coords[0] - theirQueen.coords[0]) + \
                                    -abs(ant.coords[1] - theirQueen.coords[1]))
        except UnboundLocalError:
            pass #queen is dead

    attackWorkers = theirWorkerCount + sum(distFromWorkers)/10
    attackQueen = -theirQueenHealth + sum(distFromQueen)/10
    
    #using arbitary weights so that variables interact with each other in the 
    #way we expect
    realAdv = troopAdv/5 + healthAdv/15 + foodAdv/50
    gameplayIncentives = workerMvmt/1000 + antVal/1000 + moveForward/10000 + \
        defend/100 + workerVal/10 + protectQ/100 + attackWorkers/100 + \
        attackQueen/1000

    #need to scale output (using min/max)
    # NOTE: ran at least 500 against all the agents provided and the ones
    #that we made. We also played against the AI ourselves to 
    # inflate/deflate utility in a utility that would reasonably occur 
    # within gameplay. These were our max and min util values
    # with a bit of safety buffer added on top. 
    # NOTE: when we tested manually, we were able to generate utilities 
    # of less than -10. In these situations the game is so disadvantageous 
    # to us, that we are fine accepting the loss and bounding at a more 
    # reasonable -1.5 in these extremely rare instances.
    minUtil = -1.5
    maxUtil = 1.75
    scaleUtil = ((realAdv + gameplayIncentives) - minUtil) / (maxUtil - minUtil)

    #bite the bullet and bound util if it every leaves range
    #should be incredibly rare 
    scaleUtil = max(scaleUtil, 0)
    scaleUtil = min(scaleUtil, 1)
    
    return scaleUtil

training_states = [GameState.getBasicState() for _ in range(50)]

examples = [(extract_features(state), [utility(state)]) for state in training_states]


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
    
     # hard-coded trained weights
        self.weight_hidden = np.array([
            [ 0.09722334,  0.42997507,  0.20544602,  0.08903977, -0.15309407,  0.29098089,
             -0.12482558,  0.783546,    0.92692185, -0.23352063,  0.58264274],
            [ 0.05960081,  0.13790009,  0.85155547, -0.85466814, -0.82393043, -0.95594127,
              0.66523969,  0.5563135,   0.74183527,  0.95904765,  0.60193907],
            [-0.07627701,  0.56182262, -0.7632983,   0.28121772, -0.71252916,  0.89086636,
              0.04369664, -0.17067612, -0.47012451,  0.54923164, -0.08617081],
            [ 0.13477455, -0.96451374,  0.23485233,  0.22042343,  0.23177465,  0.88330947,
              0.3636406,  -0.2809842,  -0.12802944,  0.39316905, -0.88373574],
            [ 0.33254482,  0.34028712, -0.5794326,  -0.74392691, -0.37013191, -0.27455569,
              0.14039354, -0.12279697,  0.97575906, -0.79689899, -0.58422372],
            [-0.67717783,  0.30641979, -0.49337617, -0.06701281, -0.51094568, -0.68165456,
             -0.77924972,  0.31265918, -0.72343096, -0.60663214, -0.26214339],
            [ 0.64226545, -0.80551845,  0.67594561, -0.80730099,  0.95319792, -0.06213961,
              0.95352218,  0.20969104,  0.47880615, -0.92134542, -0.43382809],
            [-0.75982564, -0.40793837, -0.76258832, -0.36442742, -0.17169278, -0.87214254,
              0.38494424,  0.13320291, -0.46943978,  0.04627734, -0.81255651],
            [ 0.1533625,   0.8600619,  -0.36256819,  0.33746587, -0.73493477,  0.43559342,
             -0.42118781, -0.63361728,  0.17449538, -0.9583154,   0.66081907],
            [-0.99014777,  0.35609436, -0.4598918,   0.47121835,  0.92483837, -0.50157115,
              0.15231467,  0.18408386,  0.14496509, -0.55337545,  0.90642059],
            [-0.10561944,  0.69294715,  0.39898451, -0.40489245,  0.62772545, -0.20672891,
              0.76220639,  0.16254575,  0.76360053,  0.38519299,  0.45076817],
            [ 0.00300733,  0.91252583,  0.28805211, -0.15164449,  0.21314499, -0.96089648,
             -0.39685037,  0.32034707, -0.41948622,  0.23638942, -0.14174547],
            [-0.7280126,  -0.40239608,  0.14013768,  0.18361621,  0.14968977,  0.30848018,
              0.30420654, -0.13716313,  0.79413246, -0.26383699, -0.12619161],
            [ 0.7832314,   0.61177267,  0.4076541,  -0.80065379,  0.83834991,  0.42725197,
              0.99769401, -0.70110339,  0.7356368,  -0.67562944,  0.2298885 ],
            [-0.75274972,  0.69562677,  0.61455998,  0.13750004, -0.18602309, -0.86244539,
              0.39485755, -0.09291463,  0.44372151,  0.73237496,  0.95026363],
            [ 0.71048556, -0.97769295, -0.2802681,   0.45796311, -0.65786177,  0.03983097,
             -0.89132402, -0.60000695, -0.96407753,  0.58627428, -0.55439287]
        ])

        self.weight_output = np.array([
            [-0.29456887,  0.86291247,  0.42162745, -0.9270406,  -0.66688301,  0.24481565,
              0.15935515, -0.52333623,  0.88193364,  0.24003433,  0.08475929,  0.18563125,
              0.46982523, -0.36252937, -0.19137937, -0.57464766, -0.60963243]
        ])
    
    # sigmoid activation
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # neural network evaluation
    def nn_utility(self, currentState):
        features = extract_features(currentState)
        # hidden layer
        x_with_bias = np.append(features, 1.0)
        hidden_input = np.dot(self.weight_hidden, x_with_bias)
        hidden_output = self.sigmoid(hidden_input)
        hidden_output_bias = np.append(hidden_output, 1.0)
        # output layer
        final_input = np.dot(self.weight_output, hidden_output_bias)
        final_output = self.sigmoid(final_input)
        return float(final_output)
    

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
    def makeNode(self, currentState, move, depth):
        nextState = getNextState(currentState, move)
        
        return {"move": move, 
                "state": nextState, 
                "depth": depth, 
                "eval": self.nn_utility(nextState) + depth, 
                "parent": currentState}
    
    ##
    #bestMove
    #Description: Takes a list of all possible nodes from a gameState and
    #   returns the node that will provide the AI with maximum utility. 
    #   When multiple moves provide the same utility, a random one is returned.
    #   This prevents cyclical behavior in the code.
    #
    #Parameters:
    #   nodeList - a list of the nodes that would result from all the possible
    #               moves from this gameState
    ##
    def bestMove(self, nodeList):
        maxScore = []
        maxNode = []
        
        #add all moves to the list that tentatively are the best
        for i in nodeList:
            if len(maxScore) == 0 or i['eval'] == maxScore[0]:
                maxScore.append(i["eval"])
                maxNode.append(i["move"])

            #if there is a new best, clear the lists
            if i['eval'] > maxScore[0]:
                maxScore = []
                maxNode = []
                maxScore.append(i['eval'])
                maxNode.append(i["move"])
        
        return random.choice(maxNode)
    
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
            node = self.makeNode(initNode["state"], m, initNode["depth"] + 1)
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
        moves = listAllLegalMoves(currentState)
        nodeList = [self.makeNode(currentState, m, 1) for m in moves]
        
        return self.bestMove(nodeList)

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

