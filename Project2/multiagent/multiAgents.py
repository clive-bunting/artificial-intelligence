# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        foodList = currentFood.asList()
        minFoodDistance = None
        for food in foodList:
            distance = manhattanDistance(newPos, food)
            if minFoodDistance == None or minFoodDistance > distance:
                minFoodDistance = distance

        minGhostDistance = None
        for ghostPos in currentGameState.getGhostPositions():
            distance = manhattanDistance(newPos, ghostPos)
            if minGhostDistance == None or minGhostDistance > distance:
                minGhostDistance = distance
        
        if minGhostDistance == 0:
            return 0.0
            
        evaluation = 10.0
        if minFoodDistance > 0:
            evaluation = (10.0 / (minFoodDistance + 1.0))
        if minGhostDistance < 3:
            evaluation -= 10.0 / minGhostDistance
        return  evaluation

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getMaxValue(self, gameState, agentIndex, ply):
        v = -10000000000.0
        actions = gameState.getLegalActions(agentIndex)
        nextAgent = agentIndex + 1
        nextPly = ply
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            nextPly += 1
        actionToTake = None
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            val, a = self.getValue(successor, nextAgent, nextPly)
            if val > v:
                v = val
                actionToTake = action
        return (v, actionToTake)

    def getMinValue(self, gameState, agentIndex, ply):
        v = 10000000000.0
        actions = gameState.getLegalActions(agentIndex)
        nextAgent = agentIndex + 1
        nextPly = ply
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            nextPly += 1
        actionToTake = None
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            val, a = self.getValue(successor, nextAgent, nextPly)
            if val < v:
                v = val
                actionToTake = action
        return (v, actionToTake)

    def getValue(self, gameState, agentIndex, ply):
        if gameState.isLose() or gameState.isWin(): return (self.evaluationFunction(gameState),None)
        if ply > self.depth:
            return (self.evaluationFunction(gameState),None)
        if agentIndex == 0:
            return self.getMaxValue(gameState, agentIndex, ply)
        return self.getMinValue(gameState, agentIndex, ply)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        v, action = self.getValue(gameState, 0, 1)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getMaxValue(self, gameState, alpha, beta, agentIndex, ply):
        v = -10000000000.0
        actions = gameState.getLegalActions(agentIndex)
        nextAgent = agentIndex + 1
        nextPly = ply
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            nextPly += 1
        actionToTake = None
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            val, a = self.getValue(successor, alpha, beta, nextAgent, nextPly)
            if val > v:
                v = val
                actionToTake = action
            if v > beta:
                return (v, actionToTake)
            alpha = max(alpha, v)
        return (v, actionToTake)

    def getMinValue(self, gameState, alpha, beta, agentIndex, ply):
        v = 10000000000.0
        actions = gameState.getLegalActions(agentIndex)
        nextAgent = agentIndex + 1
        nextPly = ply
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            nextPly += 1
        actionToTake = None
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            val, a = self.getValue(successor, alpha, beta, nextAgent, nextPly)
            if val < v:
                v = val
                actionToTake = action
            if v < alpha:
                return  (v, actionToTake)
            beta = min (beta, v)
        return (v, actionToTake)

    def getValue(self, gameState, alpha, beta, agentIndex, ply):
        if gameState.isLose() or gameState.isWin(): return (self.evaluationFunction(gameState),None)
        if ply > self.depth:
            return (self.evaluationFunction(gameState),None)
        if agentIndex == 0:
            return self.getMaxValue(gameState, alpha, beta, agentIndex, ply)
        return self.getMinValue(gameState, alpha, beta, agentIndex, ply)       
        
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        v, action = self.getValue(gameState, -1000000.0, 1000000.0, 0, 1)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getMaxValue(self, gameState, agentIndex, ply):
        v = -10000000000.0
        actions = gameState.getLegalActions(agentIndex)
        nextAgent = agentIndex + 1
        nextPly = ply
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            nextPly += 1
        actionToTake = None
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            val, a = self.getValue(successor, nextAgent, nextPly)
            if val >= v:
                v = val
                actionToTake = action
        return (v, actionToTake)
        
    def getExpectedValue(self, gameState, agentIndex, ply):
        v = 0.0
        actions = gameState.getLegalActions(agentIndex)
        nextAgent = agentIndex + 1
        nextPly = ply
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            nextPly += 1
        
        # Assume each action is equally likely
        probabilityOfAction = (1.0 / float(len(actions)))
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            val, a = self.getValue(successor, nextAgent, nextPly)
            v += val * probabilityOfAction
        return (v, None)

    def getValue(self, gameState, agentIndex, ply):
        if gameState.isLose() or gameState.isWin(): return (self.evaluationFunction(gameState),None)
        if ply > self.depth:
            return (self.evaluationFunction(gameState),None)
        if agentIndex == 0:
            return self.getMaxValue(gameState, agentIndex, ply)
        return self.getExpectedValue(gameState, agentIndex, ply)
        
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        v, action = self.getValue(gameState, 0, 1)
        return action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: 
          Score more if we have less food left
          Score more if the distance to all the remaining food is less
          Score more if the nerest ghost is further away (up to a point, effectively ignore them once they get far enough away)
    """
    food = currentGameState.getFood()
    foodList = food.asList()
    foodDistance = 0.001 # Make sure we don't have divide by zero
    pacmanPos = currentGameState.getPacmanPosition()
    for food in foodList:
        foodDistance += manhattanDistance(pacmanPos, food)
     
    minGhostDistance = None
    for ghostPos in currentGameState.getGhostPositions():
        ghostDistance = manhattanDistance(pacmanPos, ghostPos)
        if minGhostDistance == None or minGhostDistance > ghostDistance:
           minGhostDistance = ghostDistance        
    if minGhostDistance > 3:
        minGhostDistance = 0   

    return currentGameState.getScore() + 10.0 * (1.0 / foodDistance) + 10.0 * (2.0 / (currentGameState.getNumFood() + 1.0)) + 0.1 * minGhostDistance

# Abbreviation
better = betterEvaluationFunction

