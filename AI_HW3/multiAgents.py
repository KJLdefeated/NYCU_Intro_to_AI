from util import manhattanDistance, Queue
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        """
        Implmenting Minmax Tree with recursive function.
        """
        actions = []
        for act in gameState.getLegalActions(0):
            actions.append((act, self.Minimax(gameState.getNextState(0, act), act, self.depth-1, 1, False)))
        action, _ = max(actions, key=lambda x: x[1])
        return action
        # End your code (Part 1)

    def Minimax(self, gameState, action, depth, agentid, maxx):
        if gameState.isWin() or gameState.isLose() or (depth == 0 and agentid == 0):
            return self.evaluationFunction(gameState) #Determine terminal state
        actions = []
        if maxx: #max layer
            for action in gameState.getLegalActions(agentid):
                actions.append(self.Minimax(gameState.getNextState(agentid, action), action, depth-1, agentid+1, False))
            
            return max(actions)
        #min layer
        for action in gameState.getLegalActions(agentid):
            actions.append(self.Minimax(gameState.getNextState(agentid, action), action, depth, 
                                        (0 if agentid == gameState.getNumAgents()-1 else agentid+1), 
                                        (True if agentid == gameState.getNumAgents()-1 else False)))
        return min(actions)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.INF = 1e20
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        """
        Maintain (alpha, beta) to represent the upper bound and lower bound of value function.
        This can pruning the min-max tree.
        """
        action = gameState.getLegalActions(0)[0]
        alpha = -self.INF
        beta = self.INF
        for act in gameState.getLegalActions(0):
            v = self.Alpha_beta(gameState.getNextState(0, act), act, self.depth-1, 1, False, alpha, beta)
            if v > alpha:
                alpha = v
                action = act
        return action
        # End your code (Part 2)

    def Alpha_beta(self, gameState, action, depth, agentid, maxx, alpha, beta):
        if gameState.isWin() or gameState.isLose() or (depth == 0 and agentid == 0):
            return self.evaluationFunction(gameState) #terminal state
        if maxx:
            v = -self.INF
            for action in gameState.getLegalActions(agentid):
                v = max(v, self.Alpha_beta(gameState.getNextState(agentid, action), action, 
                                           depth-1, agentid+1, False, alpha, beta))
                if v > beta:
                    break
                alpha = max(alpha, v)
            return v
        #if min
        v = self.INF
        for action in gameState.getLegalActions(agentid):
            v = min(v, self.Alpha_beta(gameState.getNextState(agentid, action), action, depth, 
                                       agentid=(0 if agentid == gameState.getNumAgents()-1 else agentid+1), 
                                       maxx=(True if agentid == gameState.getNumAgents()-1 else False), alpha=alpha, beta=beta))
            if v < alpha:
                break
            beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        """
        The different part of expectimax and minmax agent is the min layer.
        The min part take the mean value of the child state values.
        """
        actions = []
        for act in gameState.getLegalActions(0):
            next_state = gameState.getNextState(0, act)
            actions.append((act, self.Expectimax(next_state, act, self.depth-1, 1, False)))
        action = ''
        best_v = -10000000
        for act, v in actions:
            if best_v < v and act != 'Stop':
                best_v = v
                action = act
        #action, _ = max(actions, key=lambda x: x[1])
        #print(best_v)
        #print(actions)
        return action
        # End your code (Part 3)

    def Expectimax(self, gameState, action, depth, agentid, maxx):
        if gameState.isWin() or gameState.isLose() or (depth == 0 and agentid == 0):
            return self.evaluationFunction(gameState)*(depth+1)
        actions = gameState.getLegalActions(agentid)
        if maxx:
            values = []
            for action in actions:
                values.append(self.Expectimax(gameState.getNextState(agentid, action), action, depth-1, agentid+1, False))
            return max(values)
        value = 0 
        for action in gameState.getLegalActions(agentid):
            value += self.Expectimax(gameState.getNextState(agentid, action), action, depth, 
                                     (0 if agentid == gameState.getNumAgents()-1 else agentid+1), 
                                     (True if agentid == gameState.getNumAgents()-1 else False))
            # Take mean of values

        return value/len(actions)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    """
    Since eating scared ghost can get 200 points.
    My poolicy is first find capsule and turn the ghost into scare and eat it.
    When the capsule is gone, the agent then eat all the food and hide from ghost.
    When the ghost is scared, the value is high when close to ghost 
    otherwise the value is very low when closing to ghost(near to lose).
    The distance  between agent and food and ghost is computed by bfs. 
    """
    def BFS2(xy1, xy2):
        q = Queue()
        vis = {}
        q.push(xy1)
        vis[xy1] = 0
        while not q.isEmpty():
            pos = q.pop()
            if pos == xy2:
                return vis[pos]
            if not currentGameState.hasWall(pos[0] + 1,pos[1]) and (pos[0] + 1,pos[1]) not in vis:
                q.push((pos[0] + 1,pos[1]))
                vis[(pos[0] + 1,pos[1])] = vis[pos] + 1
            if not currentGameState.hasWall(pos[0],pos[1] + 1) and (pos[0],pos[1] + 1) not in vis:
                q.push((pos[0],pos[1] + 1))
                vis[(pos[0],pos[1] + 1)] = vis[pos] + 1
            if not currentGameState.hasWall(pos[0] - 1,pos[1]) and (pos[0] - 1,pos[1]) not in vis:
                q.push((pos[0] - 1,pos[1]))
                vis[(pos[0] - 1,pos[1])] = vis[pos] + 1
            if not currentGameState.hasWall(pos[0],pos[1] - 1) and (pos[0],pos[1] - 1) not in vis:
                q.push((pos[0],pos[1] - 1))
                vis[(pos[0],pos[1] - 1)] = vis[pos] + 1
        return None
    def BFS(xy1):
        q = Queue()
        vis = {}
        q.push(xy1)
        vis[xy1] = 0
        while not q.isEmpty():
            pos = q.pop()
            if currentGameState.hasFood(pos[0], pos[1]):
                return vis[pos]
            if not currentGameState.hasWall(pos[0] + 1,pos[1]) and (pos[0] + 1,pos[1]) not in vis:
                q.push((pos[0] + 1,pos[1]))
                vis[(pos[0] + 1,pos[1])] = vis[pos] + 1
            if not currentGameState.hasWall(pos[0],pos[1] + 1) and (pos[0],pos[1] + 1) not in vis:
                q.push((pos[0],pos[1] + 1))
                vis[(pos[0],pos[1] + 1)] = vis[pos] + 1
            if not currentGameState.hasWall(pos[0] - 1,pos[1]) and (pos[0] - 1,pos[1]) not in vis:
                q.push((pos[0] - 1,pos[1]))
                vis[(pos[0] - 1,pos[1])] = vis[pos] + 1
            if not currentGameState.hasWall(pos[0],pos[1] - 1) and (pos[0],pos[1] - 1) not in vis:
                q.push((pos[0],pos[1] - 1))
                vis[(pos[0],pos[1] - 1)] = vis[pos] + 1
        return None

    if currentGameState.isLose():
        return -10000000
    
    score = currentGameState.getScore()
    GhostStates = currentGameState.getGhostStates()
    pos = currentGameState.getPacmanPosition()
    ScaredGhosttime = 0
    ScaredGhostdis = 1000000
    for state in GhostStates:
        dis = BFS2(pos, state.getPosition())
        if dis is not None:
            ScaredGhosttime += state.scaredTimer
            ScaredGhostdis = min(dis,ScaredGhostdis)
    nearestFoodDistance = BFS(pos)
    value = score
    if ScaredGhosttime > 2 and dis > 0:
        value += 250/ScaredGhostdis
    if len(currentGameState.getCapsules()):
        NearestCapsuleDistance = min([BFS2(pos, cap) for cap in currentGameState.getCapsules()])
        value += 10/NearestCapsuleDistance+10
    if nearestFoodDistance is not None:
        value += 10/nearestFoodDistance+5
    
    return value
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
