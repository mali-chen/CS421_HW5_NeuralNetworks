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
def mapStateToInput(gameState):
    myInv = getCurrPlayerInventory(gameState)
    enemyInv = getEnemyInv(None, gameState)

    myFood = myInv.foodCount / 11.0
    enemyFood = enemyInv.foodCount / 11.0
    myWorkers = len(getAntList(gameState, myInv.player, (WORKER,))) / 5.0
    enemyWorkers = len(getAntList(gameState, enemyInv.player, (WORKER,))) / 5.0
    mySoldiers = len(getAntList(gameState, myInv.player, (SOLDIER, DRONE, R_SOLDIER))) / 5.0
    enemySoldiers = len(getAntList(gameState, enemyInv.player, (SOLDIER, DRONE, R_SOLDIER))) / 5.0
    anthillHealth = myInv.getAnthill().captureHealth / 3.0
    enemyAnthillHealth = enemyInv.getAnthill().captureHealth / 3.0
    queenHealth = myInv.getQueen().health / 10.0
    enemyQueenHealth = enemyInv.getQueen().health / 10.0

    return np.array([
        myFood, enemyFood,
        myWorkers, enemyWorkers,
        mySoldiers, enemySoldiers,
        anthillHealth, enemyAnthillHealth,
        queenHealth, enemyQueenHealth
    ])

# step 2:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_matrix(x, w_hidden, w_output):
    """Forward pass for a 2-layer net."""
    x_b = np.append(x, 1) # bias
    h_in = np.dot(w_hidden, x_b)
    h_out = sigmoid(h_in)
    h_b = np.append(h_out, 1) # bias for output
    y_in = np.dot(w_output, h_b)
    y_out = sigmoid(y_in)
    return h_b, y_out

def backpropagation(x, target, w_hidden, w_output, lr=0.3):
    h_b, y = forward_matrix(x, w_hidden, w_output)
    y = y[0]
    error = target - y
    delta_output = error * y * (1 - y)

    h = h_b[:-1]
    w_out_no_bias = w_output[:, :-1]
    delta_hidden = (delta_output * w_out_no_bias.flatten()) * h * (1 - h)

    # update weights
    w_output += lr * delta_output * h_b
    x_b = np.append(x, 1)
    w_hidden += lr * np.outer(delta_hidden, x_b)
    return w_hidden, w_output, error**2


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
        # map the game state to network input
        x = mapStateToInput(state)

        # load or initialize weights (hard-code these later in Part B step 3)
        np.random.seed(0)
        w_hidden = np.random.uniform(-1, 1, (16, 10 + 1))
        w_output = np.random.uniform(-1, 1, (1, 16 + 1))

        _, y = forward_matrix(x, w_hidden, w_output)
        eval_value = float(y[0]) + depth

        return {
            "move": move,
            "state": state,
            "depth": depth,
            "eval": eval_value,
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
        # Part B start 

        # a.
        frontierNodes = []

        # b.
        rootNode = self.makeNode(None, currentState, 0, None) # root node is depth 0 and has no parent node
        frontierNodes.append(rootNode)

        # c.
        A_STAR_DEPTH = 3 # Can't change this (must search depth 3)
        for i in range(0, A_STAR_DEPTH):
            lowestNode = frontierNodes[0]
            for node in frontierNodes:
                if node["eval"] < lowestNode["eval"]:
                    lowestNode = node

            frontierNodes.remove(lowestNode)
            nodeList = self.expandNode(lowestNode)
            for node in nodeList:
                frontierNodes.append(node)
        # d.
        
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

   