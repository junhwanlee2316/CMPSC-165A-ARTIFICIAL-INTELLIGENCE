# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import copy
from util import Stack

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """
    Search node object for your convenience.

    This object uses the state of the node to compare equality and for its hash function,
    so you can use it in things like sets and priority queues if you want those structures
    to use the state for comparison.

    Example usage:
    >>> S = Node("Start", None, None, 0)
    >>> A1 = Node("A", S, "North", 4)
    >>> B1 = Node("B", S, "South", 3)
    >>> B2 = Node("B", A1, "West", 6)
    >>> B1 == B2
    True
    >>> A1 == B2
    False
    >>> node_list1 = [B1, B2]
    >>> B1 in node_list1
    True
    >>> A1 in node_list1
    False
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this.
    """
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def iterativeDeepeningSearch(problem):
    
    # Iterative Depth Limited Search helper function
    def _dls(depthLimit):
        
        # Stack contains pair of current node and its depth
        stack = util.Stack()
        visited = set()

        # Create a start node and push to stack
        src = Node(problem.getStartState(), None, None, 0)
        stack.push((src, 0)) 
        visited.add(src.state)

        # Begin DFS
        while stack.isEmpty() == 0:
            
            # Get current node and depth
            curr, currDepth = stack.pop()

            # Check for depth limit
            if currDepth > depthLimit:
                continue  
            
            # Find successors
            for action in problem.getActions(curr.state):
                childState = problem.getResult(curr.state,action)
                
                # If the child stat is not visited already
                if childState not in visited:
                    
                    # Create a new node
                    child = Node(childState, curr, action, problem.getCost(curr.state, action))
                    
                    # Check if the goal state reached
                    if problem.goalTest(childState):
                        return child
                    
                    # Add child state to visited and continue DFS with current depth + 1
                    else:
                        stack.push((child, currDepth + 1))
                        visited.add(childState)
                        
        return None
    
    maxDepth = 0
    
    # Begin IDS
    while True:
        
        # Run depth limited search
        goalNode = _dls(maxDepth)
        
        # Backtrack goal node to find the path
        if goalNode: return backtrackPath(goalNode)
        maxDepth += 1

    return None



def aStarSearch(problem, heuristic=nullHeuristic):
    
    # Helper function to update priority queue
    def _updateFrontier(item, priority, pq):
        
        # Convert into a list for easier manipulation
        heap = list(pq.heap)
        
        # Iterate through the items in the heap
        for i, (h_p, h_c, h_i) in enumerate(heap):
            
            # Check if the states of the items match
            if h_i[0].state == item[0].state:
                # Existing priority is less than or equal to new priority
                if h_p <= priority:
                    break
                
                # Remove existing item from the heap
                del heap[i]
                
                # Append the new item with updated priority
                heap.append((priority, h_c, item))
                
                # Sort the heap to make sure it is in order based on updated priority
                heap.sort()
                break
        else:
            # If the loop completes without breaking, add the item with its priority
            heap.append((priority, item[1], item))
             
            # Sort the heap to make sure it is in order based on updated priority
            heap.sort()

        # Update the priority queue with the modified heap
        pq.heap = heap

    pq = util.PriorityQueue()
    visited = set()
    src = Node(problem.getStartState(), None, None, 0)
    pq.push( (src, []), heuristic(problem.getStartState(), problem) )
    visited.add( problem.getStartState() )

    while pq.isEmpty() == 0:
        node, actions = pq.pop()
    
        if problem.goalTest(node.state):
            return backtrackPath(node)

        if node.state not in visited:
            visited.add(node.state)

        for action in problem.getActions(node.state):
            childState = problem.getResult(node.state, action)
            if childState not in visited:
                childNode = Node(childState, node, action, problem.getCost(node.state, action))
                totalCost = problem.getCostOfActions(actions+[action])+heuristic(childState, problem)
                _updateFrontier((childNode, actions + [action]), totalCost, pq)
    
    return None

    
def backtrackPath(node):
    path = []
    while node.parent:
        path.insert(0, node.action)
        node = node.parent
    return path
    
    
# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
