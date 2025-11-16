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
    
    ####################################################################
    ## CREDIT: Andrew Asch's utility function, permission from Dr.Nuxoll
    ####################################################################
    #utility
    #Description: Determines how (un)/favorable the current game state is to player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    ##
    def utility(self, currentState):
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

   